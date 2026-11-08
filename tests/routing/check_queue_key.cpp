// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/queue_key.hpp"
#include "route/search_queue.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using I = std::int64_t;
using U = std::uint64_t;
using Queue = tmk::route::SearchQueue;
using Entry = Queue::Entry;
static U checks = 0, pops = 0, proved = 0, exports = 0, rekeys = 0;

static void require(bool condition) {
  ++checks;
  if (!condition)
    std::abort();
}

static U original_key(I g, std::size_t state, bool deep) {
  if (!deep)
    return state;
  constexpr U max = (U{1} << 32) - 1;
  const U q = std::min<U>(static_cast<U>(std::max<I>(g, 0)) >> 8, max);
  return ((max - q) << 31) | state;
}

static Entry canonical(Entry e, const tmk::route::QueueKeyFormat& format) {
  if (format.raw())
    e.second >>= 32;
  return e;
}

int main() {
  const std::array<U, 9> sizes = {
      0, 1, 17, U{1} << 31, (U{1} << 31) + 1, U{1} << 32, (U{1} << 32) + 1, U{1} << 40, std::numeric_limits<U>::max()};
  const std::array<I, 12> values = {std::numeric_limits<I>::min(),
                                    -1000000000000LL,
                                    -1,
                                    0,
                                    1,
                                    255,
                                    256,
                                    257,
                                    1000000,
                                    I(tmk::route::kMaxQuant << 8),
                                    I(tmk::route::kMaxQuant << 8) + 257,
                                    std::numeric_limits<I>::max()};
  for (bool deep : {false, true})
    for (auto size : sizes) {
      tmk::route::QueueKeyFormat format(deep, static_cast<std::size_t>(size));
      for (U state :
           {U{0}, U{1}, (U{1} << 31) - 1, U{1} << 31, (U{1} << 32) - 1, U{1} << 32, std::numeric_limits<U>::max()}) {
        if (size && state >= size)
          continue;
        for (I before : values) {
          auto key = format.encode(before, static_cast<std::size_t>(state));
          if (format.raw())
            require(format.state(key) == state);
          else {
            require(key == original_key(before, static_cast<std::size_t>(state), deep));
            require(format.state(key) == (deep ? (key & tmk::route::kDeepStateMask) : key));
          }
          for (I current : values) {
            const bool prune = format.proves_stale(key, current);
            require(!prune || before > current);
            require(format.snapshot_supported() || !prune);
          }
        }
      }
    }

  std::mt19937_64 random(0x7241a893);
  for (bool deep : {false, true})
    for (unsigned mode = 0; mode < 5; ++mode) {
      constexpr std::size_t states = 512;
      tmk::route::QueueKeyFormat format(deep, states);
      Queue original, packed;
      std::array<I, states> best, h;
      best.fill(std::numeric_limits<I>::max());
      for (auto& value : h)
        value = static_cast<I>(random() % 200001) - 100000;
      bool have_popped_live = false;
      std::size_t popped_state = 0;
      for (unsigned step = 0; step < 90000; ++step) {
        if (original.empty() || (original.size() < 8192 && random() % 3)) {
          const auto state = static_cast<std::size_t>(random() % states);
          I g;
          if (best[state] == std::numeric_limits<I>::max()) {
            g = mode == 1 ? 5000 : mode == 2 ? (I{1} << 41) : 100000000;
            if (mode == 3)
              g = static_cast<I>(random() % 1000000000001ULL) - 500000000000LL;
            if (mode == 4) {
              g = static_cast<I>(random() % 1000000);
              h[state] = 1000000 - g;
            }
          } else
            g = best[state] - static_cast<I>(1 + random() % (mode == 0 ? 16384 : 64));
          best[state] = g;
          original.emplace(g + h[state], original_key(g, state, deep));
          packed.emplace(g + h[state], format.encode(g, state));
          have_popped_live = false;
        } else {
          auto a = original.top(), b = packed.top();
          require(a == canonical(b, format));
          const auto state = deep ? static_cast<std::size_t>(a.second & tmk::route::kDeepStateMask)
                                  : static_cast<std::size_t>(a.second);
          require(format.state(b.second) == state);
          const I insertion_g = a.first - h[state];
          const bool early = format.proves_stale(b.second, best[state]);
          require(!early || insertion_g > best[state]);
          proved += early;
          have_popped_live = insertion_g == best[state];
          popped_state = state;
          original.pop();
          packed.pop();
          ++pops;
        }
        require(original.size() == packed.size());
        require(original.empty() == packed.empty());
        if (step % 257 == 0) {
          auto a = original.release_entries(), b = packed.release_entries();
          require(a.size() == b.size());
          for (std::size_t i = 0; i < a.size(); ++i)
            require(a[i] == canonical(b[i], format));
          ++exports;
          if (step % 3 == 0) {
            auto new_h = h;
            for (auto& value : new_h)
              value = static_cast<I>(random() % 200001) - 100000;
            std::vector<Entry> kept_a, kept_b;
            if (have_popped_live) {
              auto g = best[popped_state];
              kept_a.emplace_back(g + new_h[popped_state], original_key(g, popped_state, deep));
              kept_b.emplace_back(g + new_h[popped_state], format.encode(g, popped_state));
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
              const auto state = deep ? static_cast<std::size_t>(a[i].second & tmk::route::kDeepStateMask)
                                      : static_cast<std::size_t>(a[i].second);
              const auto insertion_g = a[i].first - h[state];
              if (insertion_g > best[state])
                continue;
              require(insertion_g == best[state]);
              a[i].first = b[i].first = best[state] + new_h[state];
              kept_a.push_back(a[i]);
              kept_b.push_back(b[i]);
            }
            h = new_h;
            a = std::move(kept_a);
            b = std::move(kept_b);
            ++rekeys;
          }
          original.assign(std::move(a));
          packed.assign(std::move(b));
        }
      }
      while (!original.empty()) {
        auto a = original.top(), b = packed.top();
        require(a == canonical(b, format));
        auto state = format.state(b.second);
        require(!format.proves_stale(b.second, best[state]) || a.first - h[state] > best[state]);
        original.pop();
        packed.pop();
        ++pops;
      }
      require(packed.empty());
    }
  require(proved > 1000 && exports > 1000 && rekeys > 1000);
  std::cout << "{\"all_exact\":true,\"checks\":" << checks << ",\"pops\":" << pops << ",\"proved_stale\":" << proved
            << ",\"physical_exports\":" << exports << ",\"field_rekeys\":" << rekeys << "}\n";
}
