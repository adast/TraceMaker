// SPDX-License-Identifier: GPL-3.0-or-later
#include "reference_search_queue.hpp"
#include "route/search_queue.hpp"
#include <cstdio>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>

using Entry = tmk::route::ReferenceSearchQueue::Entry;
static std::uint64_t checks = 0, pops = 0, resets = 0;

static void require(bool okay) {
  ++checks;
  if (!okay)
    throw std::runtime_error("Legacy f-only scan changes queue contents/order");
}

int main() {
  std::mt19937_64 random(0xad2395714);
  for (unsigned mode = 0; mode < 9; ++mode) {
    tmk::route::ReferenceSearchQueue original;
    tmk::route::SearchQueue candidate;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> reference;
    auto check = [&] {
      require(original.empty() == reference.empty() && candidate.empty() == reference.empty());
      require(original.size() == reference.size() && candidate.size() == reference.size());
      if (!reference.empty()) {
        require(original.top() == reference.top());
        require(candidate.top() == reference.top());
      }
    };
    for (unsigned i = 0; i < 60000; ++i) {
      if (reference.empty() || (reference.size() < 256 && random() % 3)) {
        std::int64_t f = 0;
        if (mode == 1)
          f = std::int64_t(i);
        if (mode == 2)
          f = std::int64_t(random() % 31) - 15;
        if (mode == 3)
          f = std::bit_cast<std::int64_t>(random());
        if (mode == 4)
          f = std::numeric_limits<std::int64_t>::min() + std::int64_t(random() % 500);
        if (mode == 5)
          f = i % 2 ? std::numeric_limits<std::int64_t>::min() : std::numeric_limits<std::int64_t>::max();
        if (mode == 6)
          f = std::numeric_limits<std::int64_t>::max() - std::int64_t(random() % 500);
        if (mode == 7)
          f = std::int64_t(random() % 1000000);
        if (mode == 8)
          f = -std::int64_t(i);
        const auto key = i % 3 ? random() % 37 : random();
        original.emplace(f, key);
        candidate.emplace(f, key);
        reference.emplace(f, key);
      } else {
        check();
        original.pop();
        candidate.pop();
        reference.pop();
        ++pops;
      }
      check();
      if (i % 271 == 0) {
        auto a = original.release_entries(), b = candidate.release_entries();
        require(a == b);  // The exact physical export order is retained too.
        require(original.empty() && candidate.empty());
        auto sorted = a;
        std::sort(sorted.begin(), sorted.end());
        auto copy = reference;
        for (auto entry : sorted) {
          require(entry == copy.top());
          copy.pop();
        }
        require(copy.empty());
        if (i % 2) {
          for (auto& entry : a)
            entry.first = std::int64_t(random() % 97) - 48;
          b = a;
          reference = decltype(reference)(std::greater<>{}, a);
        }
        original.assign(std::move(a));
        candidate.assign(std::move(b));
        check();
        ++resets;
      }
    }
    while (!reference.empty()) {
      check();
      original.pop();
      candidate.pop();
      reference.pop();
      ++pops;
    }
    check();
  }
  std::printf("{\"all_exact\":true,\"checks\":%llu,\"pops\":%llu,\"release_assign_cases\":%llu,\"modes\":9}\n",
              (unsigned long long)checks, (unsigned long long)pops, (unsigned long long)resets);
}
