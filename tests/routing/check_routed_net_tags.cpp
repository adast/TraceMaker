// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/routed_net_tags.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

std::uint64_t checks = 0, predictions = 0;

void require(bool v) {
  ++checks;
  if (!v)
    throw std::runtime_error("Same-net union predicted foreign/unknown membership");
}

int main() {
  std::mt19937_64 rng(0xa95f190d);
  for (unsigned epoch = 0; epoch < 80; ++epoch) {
    std::uint32_t tag = tmk::route::net_tags::empty;
    std::uint16_t count = 0;
    std::vector<std::int32_t> active;
    for (unsigned op = 0; op < 12000; ++op) {
      if (active.empty() || (active.size() < 300 && rng() % 100 < 55)) {
        const std::int32_t net = (epoch % 4 == 0) ? 7 : static_cast<std::int32_t>(rng() % 8);
        tmk::route::net_tags::mark(tag, count, 1, net);
        ++count;
        active.push_back(net);
      } else {
        const auto i = static_cast<std::size_t>(rng() % active.size());
        const auto net = active[i];
        active[i] = active.back();
        active.pop_back();
        tmk::route::net_tags::mark(tag, count, -1, net);
        --count;
      }
      require(count == active.size());
      for (std::int32_t net : {0, 1, 2, 7, INT32_MAX}) {
        const bool same = tmk::route::net_tags::same_positive_net(tag, net);
        if (same) {
          require(!active.empty() && std::all_of(active.begin(), active.end(), [&](auto n) { return n == net; }));
          ++predictions;
        }
        require(net > 0 || !same);
      }
    }
  }
  // Original counter arithmetic wraps. A wrap can never manufacture a new
  // homogeneous union, even after the original count becomes nonzero again.
  for (auto net : {0, 1, INT32_MAX}) {
    std::uint32_t tag = 0;
    std::uint16_t count = 0;
    for (unsigned i = 0; i < 65537; ++i) {
      tmk::route::net_tags::mark(tag, count, 1, net);
      ++count;
    }
    require(count == 1 && tag == tmk::route::net_tags::unsafe && !tmk::route::net_tags::same_positive_net(tag, net));
    for (unsigned i = 0; i < 65537; ++i) {
      tmk::route::net_tags::mark(tag, count, -1, net);
      --count;
      require(tag == tmk::route::net_tags::unsafe);
    }
    require(count == 0);
    tmk::route::net_tags::mark(tag, count, 1, 2);
    require(tag == tmk::route::net_tags::unsafe);
  }
  for (int delta : {-2, -1, 0, 2}) {
    std::uint32_t tag = 0;
    tmk::route::net_tags::mark(tag, 0, delta, 1);
    require(tag == tmk::route::net_tags::unsafe);
  }
  {
    std::uint32_t tag = 0;
    tmk::route::net_tags::mark(tag, 0, 1, -1);
    require(tag == tmk::route::net_tags::unsafe);
  }
  require(predictions > 100000);
  std::printf("{\"all_exact\":true,\"checks\":%llu,\"safe_predictions\":%llu,\"random_updates\":960000,\"original_"
              "counter_wrap_covered\":true}\n",
              static_cast<unsigned long long>(checks), static_cast<unsigned long long>(predictions));
}
