// SPDX-License-Identifier: GPL-3.0-or-later
// Independent transformation oracle against the unchanged CPU geometry library.
#include "geom/shape.hpp"
#include "gpu/fixed/index.hpp"
#include <cassert>
#include <cstdio>
#include <random>

int main() {
  std::mt19937_64 random(0x3fa029591);
  std::vector<tmk::geom::Shape> original;
  std::vector<tmk::gpu::geometry::Shape> shapes;
  std::vector<tmk::gpu::geometry::Point> vertices;
  auto add = [&](tmk::geom::Shape shape) {
    shapes.push_back({static_cast<std::uint32_t>(vertices.size()), static_cast<std::uint32_t>(shape.pts.size()),
                      shape.closed, shape.r, shape.box.x0, shape.box.y0, shape.box.x1, shape.box.y1});
    for (auto p : shape.pts)
      vertices.push_back({p.x, p.y});
    original.push_back(std::move(shape));
  };
  for (unsigned n : {1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 511})
    for (bool closed : {false, true})
      for (bool wide : {false, true}) {
        std::vector<tmk::geom::Point> points;
        for (unsigned i = 0; i < n; ++i) {
          const long long x =
              wide ? ((i % 4 < 2) ? -999999999 : 999999999) : static_cast<long long>(random() % 4000001) - 2000000;
          const long long y = wide ? ((i % 4 == 0 || i % 4 == 3) ? -999999999 : 999999999)
                                   : static_cast<long long>(random() % 4000001) - 2000000;
          points.push_back(i && i % 7 == 0 ? points.back() : tmk::geom::Point{x, y});
        }
        add(closed ? tmk::geom::Shape::polygon(std::move(points), 3)
                   : tmk::geom::Shape::polyline(std::move(points), 3));
      }
  const auto indexed = tmk::gpu::spatial::build(shapes, vertices, 20000000, 4000000, 1);
  const auto limited = tmk::gpu::spatial::build(shapes, vertices, 0, 0, 1);
  const auto mixed = tmk::gpu::spatial::build(shapes, vertices, 64, 64, 1);
  std::uint64_t checked = 0;
  std::size_t indexed_shapes = 0;
  for (auto meta : indexed.spatial)
    indexed_shapes += meta.nx != 0;
  auto check = [&](unsigned shape, tmk::geom::Point point, long long radius, long long clearance) {
    const auto& reference = original[shape];
    const tmk::gpu::geometry::Probe query{shape, point.x, point.y, radius, clearance};
    const bool expected = tmk::geom::closer_than_disk(reference, point, radius, clearance);
    for (const auto* index : {&indexed, &limited, &mixed}) {
      assert(tmk::gpu::spatial::closer(index->view(), shapes.data(), vertices.data(), query) == expected);
      const bool contains = tmk::geom::point_in_polygon(point, reference.pts);
      assert(tmk::gpu::spatial::inside(index->view(), shape, shapes.data(), vertices.data(), {point.x, point.y}) ==
             contains);
    }
    ++checked;
  };
  for (unsigned shape = 0; shape < shapes.size(); ++shape) {
    for (unsigned k = 0; k < 4096; ++k) {
      const auto& reference = original[shape];
      auto point = reference.pts[random() % reference.pts.size()];
      if (k % 2) {
        point.x = static_cast<long long>(random() % 2000000001) - 1000000000;
        point.y = static_cast<long long>(random() % 2000000001) - 1000000000;
      } else {
        point.x = std::clamp<tmk::Coord>(point.x + static_cast<int>(k % 5) - 2, -1000000000, 1000000000);
        point.y = std::clamp<tmk::Coord>(point.y + static_cast<int>((k / 5) % 5) - 2, -1000000000, 1000000000);
      }
      const long long radius = k % 3 ? random() % 100000001 : 0,
                      clearance = k % 7 ? static_cast<long long>(random() % 400001) - 100000 : -(radius + reference.r);
      check(shape, point, radius, clearance);
    }
    const auto& data = indexed.spatial[shape];
    // Exact bucket seams, bbox extremes and strict-distance thresholds. These
    // probes use the original geometry library as their independent oracle.
    if (data.nx)
      for (int x : {0, data.nx / 2, data.nx - 1})
        for (int y : {0, data.ny / 2, data.ny - 1})
          for (int delta : {-1, 0, 1})
            for (long long radius : {0, 1, 99997, 100000}) {
              tmk::geom::Point point{data.x0 + x * data.cell + delta, data.y0 + y * data.cell + delta};
              if (std::abs(point.x) <= 1000000000 && std::abs(point.y) <= 1000000000)
                for (long long clearance : {-3, 0, 1})
                  check(shape, point, radius, clearance);
            }
  }
  const auto mixed_shapes = std::count_if(mixed.spatial.begin(), mixed.spatial.end(), [](auto v) { return v.nx != 0; });
  assert(mixed_shapes > 0 && static_cast<std::size_t>(mixed_shapes) < shapes.size());
  std::printf("{\"queries\":%llu,\"shapes\":%zu,\"indexed_shapes\":%zu,\"mixed_budget_indexed_shapes\":%zu,\"all_"
              "exact\":true,\"zero_budget_fallback_exact\":true,\"bucket_boundaries_checked\":true}\n",
              static_cast<unsigned long long>(checked), shapes.size(), indexed_shapes,
              static_cast<std::size_t>(mixed_shapes));
}
