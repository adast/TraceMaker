// SPDX-License-Identifier: GPL-3.0-or-later
// Ready-only tile supply. The caller retains cache publication and deterministic work accounting.
#pragma once
#include "gpu/fixed/context.hpp"
#include "gpu/fixed/geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tmk::gpu::fixed {
struct AsyncContextHash {
  std::size_t operator()(const Context& c) const {
    std::size_t h = std::hash<const void*>{}(c.owner);
    auto mix = [&](std::uint64_t value) {
      h ^= std::hash<std::uint64_t>{}(value) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(c.generation);
    mix(c.flags);
    mix(static_cast<std::uint64_t>(c.hw));
    mix(static_cast<std::uint64_t>(c.margin));
    mix(static_cast<std::uint64_t>(c.via_mask));
    mix(static_cast<std::uint64_t>(c.max_pad));
    mix(static_cast<std::uint64_t>(c.layer));
    mix(static_cast<std::uint64_t>(c.net));
    return h;
  }
};

// Thread-confined to the original CPU search. Fake and CUDA backends share this
// exact state machine; only a backend destructor may wait for pending output.
template <class Backend> class AsyncTileBroker {
  using Point = tmk::gpu::geometry::Point;
  using Clock = std::chrono::steady_clock;

  struct Key {
    std::uint64_t context = 0;
    int x = 0, y = 0;
    bool operator==(const Key&) const = default;
  };

  struct Hash {
    std::size_t operator()(const Key& k) const {
      return std::hash<std::uint64_t>{}(k.context) ^ (static_cast<std::size_t>(k.x) * 0x8da6b343u) ^
             (static_cast<std::size_t>(k.y) * 0xd8163841u);
    }
  };

  struct Pending {
    Key key;
    int width = 0, height = 0;
    std::uint64_t epoch = 0;
  };

  struct Entry {
    Context context;
    std::uint64_t id = 0, age = 0, last_poll = 0;
    std::unique_ptr<Backend> backend;
    Pending pending;
    bool active = false;
  };

  struct Tile {
    int width = 0, height = 0;
    std::vector<std::int32_t> codes;
    std::vector<std::uint64_t> used;
    typename std::list<Key>::iterator age;

    std::size_t bytes() const {
      return codes.capacity() * sizeof(std::int32_t) + used.capacity() * sizeof(std::uint64_t);
    }
  };

public:
  struct Metrics {
    std::uint64_t queries = 0, hits = 0, batches = 0, points = 0, polls = 0, collections = 0, stale_batches = 0,
                  discarded_batches = 0;
    std::uint64_t context_misses = 0, capacity_declines = 0, unsupported_queries = 0, unique_consumed = 0;
    std::uint64_t evicted_tiles = 0, evicted_scenes = 0, grid_changes = 0;
    double drain_s = 0, submit_s = 0, collect_s = 0;
  };

private:
  Metrics metrics_;
  std::unordered_map<Context, std::unique_ptr<Entry>, AsyncContextHash> entries_;
  std::unordered_set<Context, AsyncContextHash> unsupported_;
  std::unordered_map<Key, Tile, Hash> tiles_;
  std::list<Key> ages_;
  Entry* recent_ = nullptr;
  const void* owner_ = nullptr;
  Point origin_{};
  long long pitch_ = 0;
  int width_ = 0, height_ = 0, edge_ = 64;
  bool configured_ = false;
  std::uint64_t epoch_ = 0, next_context_ = 0, clock_ = 0, poll_interval_ = 64, last_admission_poll_ = 0;
  std::size_t tile_bytes_ = 0, scene_bytes_ = 0, pinned_bytes_ = 0;
  std::size_t scene_limit_, scene_byte_limit_, tile_limit_, tile_byte_limit_;

  static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  void clear_tiles() {
    tiles_.clear();
    ages_.clear();
    tile_bytes_ = 0;
  }

  void evict_tile() {
    auto entry = tiles_.find(ages_.front());
    if (entry == tiles_.end())
      throw std::runtime_error("Async tile age accounting");
    tile_bytes_ -= entry->second.bytes();
    tiles_.erase(entry);
    ages_.pop_front();
    ++metrics_.evicted_tiles;
  }

  bool evict_idle_scene() {
    auto oldest = entries_.end();
    for (auto i = entries_.begin(); i != entries_.end(); ++i)
      if (!i->second->active && (oldest == entries_.end() || i->second->age < oldest->second->age))
        oldest = i;
    if (oldest == entries_.end() && metrics_.queries - last_admission_poll_ >= poll_interval_) {
      last_admission_poll_ = metrics_.queries;
      for (auto i = entries_.begin(); i != entries_.end(); ++i) {
        ++metrics_.polls;
        if (i->second->backend->ready()) {
          collect(*i->second, false);
          oldest = i;
          break;
        }
      }
    }
    if (oldest == entries_.end())
      return false;
    const auto id = oldest->second->id;
    for (auto i = tiles_.begin(); i != tiles_.end();) {
      if (i->first.context == id) {
        tile_bytes_ -= i->second.bytes();
        ages_.erase(i->second.age);
        i = tiles_.erase(i);
        ++metrics_.evicted_tiles;
      } else
        ++i;
    }
    scene_bytes_ -= oldest->second->backend->memory_bytes();
    pinned_bytes_ -= oldest->second->backend->pinned_bytes();
    recent_ = nullptr;
    const auto start = Clock::now();
    entries_.erase(oldest);
    metrics_.drain_s += elapsed(start);
    ++metrics_.evicted_scenes;
    return true;
  }

  void collect(Entry& entry, bool retain = true) {
    const auto start = Clock::now();
    auto result = entry.backend->collect();
    metrics_.collect_s += elapsed(start);
    entry.active = false;
    ++metrics_.collections;
    const auto count = static_cast<std::size_t>(entry.pending.width) * static_cast<std::size_t>(entry.pending.height);
    if (result.codes.size() != count ||
        std::any_of(result.codes.begin(), result.codes.end(), [](std::int32_t v) { return v < -2; }))
      throw std::runtime_error("Incomplete async tile output");
    if (entry.pending.epoch != epoch_) {
      ++metrics_.stale_batches;
      return;
    }
    if (!retain) {
      ++metrics_.discarded_batches;
      return;
    }
    Tile tile;
    tile.width = entry.pending.width;
    tile.height = entry.pending.height;
    tile.codes = std::move(result.codes);
    tile.used.resize((count + 63) / 64);
    const auto bytes = tile.bytes();
    if (bytes > tile_byte_limit_ || !tile_limit_) {
      ++metrics_.capacity_declines;
      return;
    }
    while (tiles_.size() >= tile_limit_ || tile_bytes_ > tile_byte_limit_ - bytes)
      evict_tile();
    if (tiles_.contains(entry.pending.key))
      throw std::runtime_error("Duplicate pending async tile");
    ages_.push_back(entry.pending.key);
    tile.age = std::prev(ages_.end());
    tiles_.emplace(entry.pending.key, std::move(tile));
    tile_bytes_ += bytes;
  }

  bool cached(const Key& key, int x, int y, std::int32_t& answer) {
    auto found = tiles_.find(key);
    if (found == tiles_.end())
      return false;
    auto& tile = found->second;
    const auto dx = x - key.x, dy = y - key.y;
    if (dx < 0 || dy < 0 || dx >= tile.width || dy >= tile.height)
      throw std::runtime_error("Async tile indexing");
    const auto index =
        static_cast<std::size_t>(dy) * static_cast<std::size_t>(tile.width) + static_cast<std::size_t>(dx);
    answer = tile.codes[index];
    const auto bit = std::uint64_t{1} << (index % 64);
    if (!(tile.used[index / 64] & bit)) {
      tile.used[index / 64] |= bit;
      ++metrics_.unique_consumed;
    }
    if (std::next(tile.age) != ages_.end())
      ages_.splice(ages_.end(), ages_, tile.age);
    ++metrics_.hits;
    return true;
  }

  template <class Factory> Entry* context_entry(const Context& context, Factory& factory) {
    Entry* entry = recent_;
    if (!entry || !(entry->context == context)) {
      auto found = entries_.find(context);
      if (found == entries_.end()) {
        if (unsupported_.contains(context)) {
          ++metrics_.unsupported_queries;
          return nullptr;
        }
        while (entries_.size() >= scene_limit_ && evict_idle_scene()) {
        }
        if (entries_.size() >= scene_limit_) {
          ++metrics_.capacity_declines;
          return nullptr;
        }
        ++metrics_.context_misses;
        auto backend = factory(context, std::size_t{65536});
        if (!backend) {
          if (unsupported_.size() >= 256)
            unsupported_.clear();
          unsupported_.insert(context);
          ++metrics_.unsupported_queries;
          return nullptr;
        }
        const auto bytes = backend->memory_bytes();
        if (bytes > scene_byte_limit_) {
          ++metrics_.capacity_declines;
          return nullptr;
        }
        while (scene_bytes_ > scene_byte_limit_ - bytes && evict_idle_scene()) {
        }
        if (scene_bytes_ > scene_byte_limit_ - bytes) {
          ++metrics_.capacity_declines;
          return nullptr;
        }
        if (next_context_ == UINT64_MAX)
          throw std::runtime_error("Async context identifier exhausted");
        auto created = std::make_unique<Entry>();
        created->context = context;
        created->id = ++next_context_;
        created->backend = std::move(backend);
        scene_bytes_ += bytes;
        pinned_bytes_ += created->backend->pinned_bytes();
        entry = created.get();
        entries_.emplace(context, std::move(created));
      } else
        entry = found->second.get();
      recent_ = entry;
    }
    return entry;
  }

public:
  explicit AsyncTileBroker(std::size_t scenes = 8, std::size_t scene_bytes = 512 * 1024 * 1024,
                           std::size_t tiles = 4096, std::size_t tile_bytes = 256 * 1024 * 1024)
      : scene_limit_(scenes), scene_byte_limit_(scene_bytes), tile_limit_(tiles), tile_byte_limit_(tile_bytes) {}

  ~AsyncTileBroker() = default;
  AsyncTileBroker(const AsyncTileBroker&) = delete;
  AsyncTileBroker& operator=(const AsyncTileBroker&) = delete;

  bool configure(const void* owner, Point origin, long long pitch, int width, int height, int edge,
                 std::uint64_t poll_interval = 64) {
    configured_ = false;
    if (!owner || pitch <= 0 || width <= 0 || height <= 0 || edge < 8 || edge > 256 || !poll_interval ||
        !scene_limit_ || !scene_byte_limit_ || !tile_limit_ || !tile_byte_limit_ || origin.x < -1000000000 ||
        origin.x > 1000000000 || origin.y < -1000000000 || origin.y > 1000000000)
      return false;
    if (static_cast<long long>(width - 1) > (1000000000 - origin.x) / pitch ||
        static_cast<long long>(height - 1) > (1000000000 - origin.y) / pitch)
      return false;
    if (owner != owner_ || origin.x != origin_.x || origin.y != origin_.y || pitch != pitch_ || width != width_ ||
        height != height_ || edge != edge_) {
      if (epoch_ == UINT64_MAX)
        throw std::runtime_error("Async grid epoch exhausted");
      ++epoch_;
      ++metrics_.grid_changes;
      clear_tiles();
    }
    owner_ = owner;
    origin_ = origin;
    pitch_ = pitch;
    width_ = width;
    height_ = height;
    edge_ = edge;
    poll_interval_ = poll_interval;
    configured_ = true;
    return true;
  }

  template <class Factory> bool predict(const Context& context, Point point, Factory factory, std::int32_t& answer) {
    ++metrics_.queries;
    if (!configured_ || context.owner != owner_ || !context.generation || context.layer < 0 || context.net < 0 ||
        context.hw < 0 || context.margin < 0 || context.hw > 1000000000 || context.margin > 1000000000 - context.hw ||
        (context.flags & kViaProbe) || point.x < -1000000000 || point.x > 1000000000 || point.y < -1000000000 ||
        point.y > 1000000000)
      return false;
    const auto dx = point.x - origin_.x, dy = point.y - origin_.y;
    if (dx < 0 || dy < 0 || dx % pitch_ || dy % pitch_ || dx / pitch_ >= width_ || dy / pitch_ >= height_)
      return false;
    const int x = static_cast<int>(dx / pitch_), y = static_cast<int>(dy / pitch_);
    Entry* entry = context_entry(context, factory);
    if (!entry)
      return false;
    entry->age = ++clock_;
    const Key key{entry->id, (x / edge_) * edge_, (y / edge_) * edge_};
    if (cached(key, x, y, answer))
      return true;
    if (entry->active) {
      if (metrics_.queries - entry->last_poll < poll_interval_)
        return false;
      entry->last_poll = metrics_.queries;
      ++metrics_.polls;
      if (!entry->backend->ready())
        return false;
      collect(*entry);
      if (cached(key, x, y, answer))
        return true;
    }
    const int w = std::min(edge_, width_ - key.x), h = std::min(edge_, height_ - key.y);
    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int yy = 0; yy < h; ++yy)
      for (int xx = 0; xx < w; ++xx)
        points.push_back({origin_.x + static_cast<long long>(key.x + xx) * pitch_,
                          origin_.y + static_cast<long long>(key.y + yy) * pitch_});
    entry->pending = {key, w, h, epoch_};
    const auto start = Clock::now();
    entry->backend->submit(points);
    metrics_.submit_s += elapsed(start);
    entry->active = true;
    entry->last_poll = metrics_.queries;
    ++metrics_.batches;
    metrics_.points += points.size();
    return false;
  }

  void shutdown() {
    const auto start = Clock::now();
    recent_ = nullptr;
    entries_.clear();
    unsupported_.clear();
    clear_tiles();
    scene_bytes_ = 0;
    pinned_bytes_ = 0;
    configured_ = false;
    metrics_.drain_s += elapsed(start);
  }

  const Metrics& metrics() const {
    return metrics_;
  }

  std::size_t retained_tiles() const {
    return tiles_.size();
  }

  std::size_t retained_tile_bytes() const {
    return tile_bytes_;
  }

  std::size_t retained_scenes() const {
    return entries_.size();
  }

  std::size_t retained_device_bytes() const {
    return scene_bytes_;
  }

  std::size_t retained_pinned_bytes() const {
    return pinned_bytes_;
  }
};
}  // namespace tmk::gpu::fixed
