// SPDX-License-Identifier: GPL-3.0-or-later
// Host-only cache of pure geometry indexes, with full equality after hashing.
// No obstacle owner, net, rule, removal or routed-state assumptions are needed.
#pragma once
#include "gpu/fixed/index.hpp"

namespace tmk::gpu::spatial {
class Cache {
  struct Entry {
    std::uint64_t hash = 0, age = 0;
    std::size_t bytes = 0;
    std::vector<tmk::gpu::geometry::Shape> shapes;
    std::vector<tmk::gpu::geometry::Point> points;
    Index index;
  };

  std::vector<Entry> entries_;
  std::size_t bytes_ = 0, byte_limit_, entry_limit_, minimum_points_;
  std::uint64_t clock_ = 0;

  static std::uint64_t fingerprint(const std::vector<tmk::gpu::geometry::Shape>& shapes,
                                   const std::vector<tmk::gpu::geometry::Point>& points) {
    std::uint64_t value = 1469598103934665603ULL;
    auto add = [&](auto n) {
      value ^= static_cast<std::uint64_t>(n);
      value *= 1099511628211ULL;
    };
    add(shapes.size());
    add(points.size());
    for (const auto& s : shapes) {
      add(s.first);
      add(s.count);
      add(s.closed);
      add(s.radius);
      add(s.x0);
      add(s.y0);
      add(s.x1);
      add(s.y1);
    }
    for (auto p : points) {
      add(p.x);
      add(p.y);
    }
    return value;
  }

  static bool equal(const Entry& e, const std::vector<tmk::gpu::geometry::Shape>& shapes,
                    const std::vector<tmk::gpu::geometry::Point>& points) {
    if (e.shapes.size() != shapes.size() || e.points.size() != points.size())
      return false;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      const auto& a = e.shapes[i];
      const auto& b = shapes[i];
      if (a.first != b.first || a.count != b.count || a.closed != b.closed || a.radius != b.radius || a.x0 != b.x0 ||
          a.y0 != b.y0 || a.x1 != b.x1 || a.y1 != b.y1)
        return false;
    }
    for (std::size_t i = 0; i < points.size(); ++i)
      if (e.points[i].x != points[i].x || e.points[i].y != points[i].y)
        return false;
    return true;
  }

public:
  explicit Cache(std::size_t byte_limit = 64 * 1024 * 1024, std::size_t entry_limit = 8,
                 std::size_t minimum_points = 8192)
      : byte_limit_(byte_limit), entry_limit_(entry_limit), minimum_points_(minimum_points) {}

  Index get(const std::vector<tmk::gpu::geometry::Shape>& shapes, const std::vector<tmk::gpu::geometry::Point>& points,
            bool& hit) {
    hit = false;
    if (points.size() < minimum_points_ || !entry_limit_ || !byte_limit_)
      return build(shapes, points);
    const auto hash = fingerprint(shapes, points);
    ++clock_;
    for (auto& e : entries_)
      if (e.hash == hash && equal(e, shapes, points)) {
        e.age = clock_;
        hit = true;
        return e.index;
      }
    auto index = build(shapes, points);
    const auto bytes =
        shapes.size() * sizeof(tmk::gpu::geometry::Shape) + points.size() * sizeof(tmk::gpu::geometry::Point) +
        index.spatial.size() * sizeof(Spatial) +
        (index.row_offsets.size() + index.row_edges.size() + index.cell_offsets.size() + index.cell_edges.size()) *
            sizeof(std::uint32_t);
    if (bytes > byte_limit_)
      return index;
    while (!entries_.empty() && (entries_.size() >= entry_limit_ || bytes_ > byte_limit_ - bytes)) {
      auto oldest = std::min_element(entries_.begin(), entries_.end(),
                                     [](const auto& a, const auto& b) { return a.age < b.age; });
      bytes_ -= oldest->bytes;
      entries_.erase(oldest);
    }
    entries_.push_back({hash, clock_, bytes, shapes, points, index});
    bytes_ += bytes;
    return index;
  }
};

inline Index cached_build(const std::vector<tmk::gpu::geometry::Shape>& shapes,
                          const std::vector<tmk::gpu::geometry::Point>& points, bool& hit) {
  static thread_local Cache cache;
  return cache.get(shapes, points, hit);
}
}  // namespace tmk::gpu::spatial
