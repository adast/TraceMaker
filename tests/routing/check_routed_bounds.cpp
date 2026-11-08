// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/routed_bounds.hpp"
#include <array>
#include <cstdio>
#include <random>
#include <stdexcept>

std::uint64_t checks = 0, predictions = 0;

void require(bool v) {
  ++checks;
  if (!v)
    throw std::runtime_error("Tight routed union missed a possible foreign collision");
}

struct Item {
  int net;
  std::uint64_t layers;
  long long x0, y0, x1, y1, x, y, drill;
  bool via;
};

void configure(tmk::route::RoutedBounds& bounds, int nx, int ny, int nl, long long ox, long long oy, long long pitch,
               bool eligible, long long radius, long long mask, long long hc, long long clearance) {
  bounds.configure({nx, ny, nl, ox, oy, pitch, eligible, true, true, radius, mask, hc, clearance});
}

int main() {

  constexpr int nx = 32, ny = 32, nl = 4;
  constexpr long long ox = -1000000, oy = -1500000, pitch = 100000, maxr = 125000, mask = 400000, hc = 150000,
                      clearance = 200000;
  tmk::route::RoutedBounds u;
  configure(u, nx, ny, nl, ox, oy, pitch, true, maxr, mask, hc, clearance);
  require(u.enabled());
  std::vector<Item> active;
  std::mt19937_64 rng(0xe478312f);
  auto update = [&](const Item& it, int delta) {
    u.mark_copper(it.x0, it.y0, it.x1, it.y1, it.layers, it.net, delta);
    if (it.via)
      u.mark_hole(it.x, it.y, it.drill, it.net, delta);
  };
  for (unsigned step = 0; step < 6000; ++step) {
    if (active.empty() || (active.size() < 50 && rng() % 100 < 53)) {
      Item it{};
      it.net = static_cast<int>(rng() % 5);
      it.layers = rng() % 16;
      it.via = rng() % 3 == 0;
      it.x = ox + static_cast<long long>(rng() % 4200000) - 500000;
      it.y = oy + static_cast<long long>(rng() % 4200000) - 500000;
      it.drill = static_cast<long long>(rng() % 400001);
      const long long r = static_cast<long long>(rng() % 150001);
      it.x0 = it.x - r;
      it.x1 = it.x + r + static_cast<long long>(rng() % 250001);
      it.y0 = it.y - r;
      it.y1 = it.y + r + static_cast<long long>(rng() % 250001);
      update(it, 1);
      active.push_back(it);
    } else {
      const auto i = static_cast<std::size_t>(rng() % active.size());
      update(active[i], -1);
      active[i] = active.back();
      active.pop_back();
    }
    for (unsigned q = 0; q < 64; ++q) {
      const int gx = static_cast<int>(rng() % nx), gy = static_cast<int>(rng() % ny),
                layer = static_cast<int>(rng() % nl), net = static_cast<int>(rng() % 5);
      const long long radius = static_cast<long long>(rng() % (maxr + 1)),
                      via_mask = static_cast<long long>(rng() % (mask + 1));
      const auto xy = static_cast<std::size_t>(gy * nx + gx), i = static_cast<std::size_t>(layer * nx * ny) + xy;
      const bool answer = u.predict(i, xy, net, radius, 0, via_mask);
      require(net > 0 || !answer);
      if (!answer)
        continue;
      const long long px = ox + gx * pitch, py = oy + gy * pitch;
      for (const auto& it : active) {
        if (it.net == net && net > 0)
          continue;
        const long long req =
            it.via && (layer == 0 || layer == nl - 1) ? std::max(clearance, via_mask + 1000) : clearance;
        const bool copper = (it.layers & (std::uint64_t{1} << layer)) && px >= it.x0 - radius - req &&
                            px <= it.x1 + radius + req && py >= it.y0 - radius - req && py <= it.y1 + radius + req;
        const auto hr = it.drill / 2 + radius + hc;
        const bool hole = it.via && px >= it.x - hr && px <= it.x + hr && py >= it.y - hr && py <= it.y + hr;
        require(!copper && !hole);
      }
      u.verified();
      ++predictions;
    }
  }
  require(predictions > 1000);
  tmk::route::RoutedBounds edge;
  configure(edge, 1, 1, 1, 0, 0, 2, true, 0, 0, 0, 0);
  require(edge.enabled());
  for (unsigned n = 0; n < 65537; ++n)
    edge.mark_copper(0, 0, 0, 0, 1, 7, 1);
  require(!edge.predict(0, 0, 7, 0, 0, 0));
  for (unsigned n = 0; n < 65537; ++n)
    edge.mark_copper(0, 0, 0, 0, 1, 7, -1);
  require(!edge.predict(0, 0, 7, 0, 0, 0));
  configure(edge, 1, 1, 1, 0, 0, 2, true, 0, 0, 2000, 0);
  edge.mark_copper(0, 0, 0, 0, 1, 7, 1);
  for (unsigned n = 0; n < 65537; ++n)
    edge.mark_hole(0, 0, 0, 7, 1);
  require(!edge.predict(0, 0, 7, 0, 0, 0));
  for (unsigned n = 0; n < 65537; ++n)
    edge.mark_hole(0, 0, 0, 7, -1);
  require(!edge.predict(0, 0, 7, 0, 0, 0));
  configure(edge, 1, 1, 1, 0, 0, 2, true, 100, 5, 0, 0);
  require(!edge.predict(0, 0, 7, INT64_MAX, INT64_MAX, 0));
  require(!edge.predict(0, 0, 7, 100, 1, 0));
  require(!edge.predict(0, 0, 7, 0, 0, 6));
  edge.mark_copper(10, 0, -10, 0, 1, 7, 1);
  require(!edge.enabled());
  configure(edge, 1, 1, 1, 0, 0, 2, true, 0, 0, 2000, 0);
  edge.mark_hole(0, 0, -1, 7, 1);
  require(!edge.enabled());
  configure(edge, 1, 1, 1, 0, 0, 1, true, 0, 0, 0, 0);
  require(!edge.enabled());
  configure(edge, 1, 1, 1, 0, 0, 2, false, 0, 0, 0, 0);
  require(!edge.enabled());
  std::printf("{\"all_exact\":true,\"checks\":%llu,\"safe_predictions\":%llu,\"random_spatial_updates\":6000,\"random_"
              "queries\":384000,\"continuous_bbox_oracle\":true,\"negative_origins_and_outside_items\":true,\"cross_"
              "layer_holes\":true,\"counter_wrap\":true}\n",
              static_cast<unsigned long long>(checks), static_cast<unsigned long long>(predictions));
}
