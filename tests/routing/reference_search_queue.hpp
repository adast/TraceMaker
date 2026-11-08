// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace tmk::route {
// Exact lexicographic (signed f, unsigned state key) queue. Radix partition f;
// use a normal tie heap for equal-f entries. A lower f rebases every live
// entry, so correctness does not depend on heuristic consistency or monotone
// inputs.
class ReferenceSearchQueue {
public:
  using Entry = std::pair<std::int64_t, std::uint64_t>;

  bool empty() const {
    return size_ == 0;
  }

  std::size_t size() const {
    return size_;
  }

  const Entry& top() {
    refill();
    return buckets_[0].front();
  }

  void pop() {
    refill();
    auto& b = buckets_[0];
    std::pop_heap(b.begin(), b.end(), Tie{});
    b.pop_back();
    --size_;
  }

  void emplace(std::int64_t f, std::uint64_t key) {
    const auto value = ordered(f);
    if (size_ == 0)
      last_ = value;
    else if (value < last_) {
      auto entries = release_entries();
      last_ = value;
      for (const auto& e : entries)
        append(e);
      size_ = entries.size();
    }
    const auto index = bucket(value);
    append({f, key});
    ++size_;
    if (index == 0) {
      auto& b = buckets_[0];
      std::push_heap(b.begin(), b.end(), Tie{});
    }
  }

  std::vector<Entry> release_entries() {
    std::vector<Entry> out;
    out.reserve(size_);
    for (auto& b : buckets_) {
      out.insert(out.end(), b.begin(), b.end());
      b.clear();
    }
    size_ = 0;
    nonempty_ = 0;
    return out;
  }

  void assign(std::vector<Entry> entries) {
    for (auto& b : buckets_)
      b.clear();
    size_ = 0;
    nonempty_ = 0;
    if (entries.empty())
      return;
    last_ = ordered(std::min_element(entries.begin(), entries.end())->first);
    for (const auto& e : entries)
      append(e);
    size_ = entries.size();
    auto& zero = buckets_[0];
    std::make_heap(zero.begin(), zero.end(), Tie{});
  }

private:
  struct Tie {
    bool operator()(const Entry& a, const Entry& b) const {
      return a.second > b.second;
    }
  };

  static std::uint64_t ordered(std::int64_t f) {
    return static_cast<std::uint64_t>(f) ^ (std::uint64_t{1} << 63);
  }

  int bucket(std::uint64_t f) const {
    return static_cast<int>(std::bit_width(f ^ last_));
  }

  void append(Entry e) {
    const int index = bucket(ordered(e.first));
    buckets_[static_cast<std::size_t>(index)].push_back(e);
    if (index)
      nonempty_ |= std::uint64_t{1} << (index - 1);
  }

  void refill() {
    if (!buckets_[0].empty())
      return;
    const int index = 1 + std::countr_zero(nonempty_);
    auto& old = buckets_[static_cast<std::size_t>(index)];
    last_ = ordered(std::min_element(old.begin(), old.end())->first);
    nonempty_ &= ~(std::uint64_t{1} << (index - 1));
    // Repartitioned entries always move to a lower bucket, so old is never
    // modified while it is traversed and keeps its allocation for later use.
    for (const auto& e : old)
      append(e);
    old.clear();
    auto& zero = buckets_[0];
    std::make_heap(zero.begin(), zero.end(), Tie{});
  }

  std::array<std::vector<Entry>, 65> buckets_;
  std::size_t size_ = 0;
  std::uint64_t last_ = 0, nonempty_ = 0;
};
}  // namespace tmk::route
