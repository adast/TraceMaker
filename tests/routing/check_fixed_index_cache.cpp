// SPDX-License-Identifier: GPL-3.0-or-later
// Cache equivalence, invalidation, bounded retention and integer predicates.
#include "gpu/fixed/index_cache.hpp"
#include <cassert>
#include <cstdio>

int main() {
  std::vector<tmk::gpu::geometry::Point> points;
  for (int i = 0; i < 64; ++i)
    points.push_back({(i % 8) * 100000, (i / 8) * 100000});
  points.push_back({0, 0});
  std::vector<tmk::gpu::geometry::Shape> shapes{{0, 64, 1, 10, -10, -10, 700010, 700010}};
  tmk::gpu::spatial::Cache cache(1024 * 1024, 2, 0), disabled(0, 0, 0);
  bool hit = false;
  unsigned checks = 0;
  auto first = cache.get(shapes, points, hit);
  assert(!hit);
  auto again = cache.get(shapes, points, hit);
  assert(hit);
  ++checks;
  for (long long y = -10000; y <= 710000; y += 10000)
    for (long long x = -10000; x <= 710000; x += 10000) {
      tmk::gpu::geometry::Probe q{0, x, y, 1000, 100};
      assert(tmk::gpu::spatial::closer(first.view(), shapes.data(), points.data(), q) ==
             tmk::gpu::spatial::closer(again.view(), shapes.data(), points.data(), q));
      ++checks;
    }
  // Every field participating in scene identity must force a fresh index.
  for (unsigned field = 0; field < 9; ++field) {
    auto changed = shapes;
    auto& s = changed[0];
    switch (field) {
    case 0:
      s.first = 1;
      break;
    case 1:
      --s.count;
      break;
    case 2:
      s.closed = 0;
      break;
    case 3:
      ++s.radius;
      break;
    case 4:
      --s.x0;
      break;
    case 5:
      --s.y0;
      break;
    case 6:
      ++s.x1;
      break;
    case 7:
      ++s.y1;
      break;
    default:
      changed.push_back(shapes[0]);
      break;
    }
    auto index = cache.get(changed, points, hit);
    assert(!hit);
    ++checks;
    index = cache.get(changed, points, hit);
    assert(hit);
    ++checks;
  }
  auto changed_points = points;
  ++changed_points[0].x;
  auto changed = cache.get(shapes, changed_points, hit);
  assert(!hit);
  ++checks;
  changed = cache.get(shapes, changed_points, hit);
  assert(hit);
  ++checks;
  ++changed_points[0].y;
  changed = cache.get(shapes, changed_points, hit);
  assert(!hit);
  ++checks;
  auto evicted = cache.get(shapes, points, hit);
  assert(!hit);
  ++checks;
  auto unused = disabled.get(shapes, points, hit);
  assert(!hit);
  unused = disabled.get(shapes, points, hit);
  assert(!hit);
  ++checks;
  tmk::gpu::spatial::Cache tiny(1, 2, 0);
  unused = tiny.get(shapes, points, hit);
  assert(!hit);
  unused = tiny.get(shapes, points, hit);
  assert(!hit);
  ++checks;
  // A returned index owns its storage and survives cache replacement/eviction.
  for (long long x = -10000; x <= 710000; x += 10000) {
    tmk::gpu::geometry::Probe q{0, x, 300000, 1000, 100};
    assert(tmk::gpu::spatial::closer(first.view(), shapes.data(), points.data(), q) ==
           tmk::gpu::geometry::closer(shapes[0], points.data(), q));
    ++checks;
  }
  std::printf("{\"checks\":%u,\"complete_key_invalidation\":true,\"bounded_cache\":true,\"returned_index_owns_"
              "storage\":true}\n",
              checks);
}
