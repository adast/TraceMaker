// SPDX-License-Identifier: GPL-3.0-or-later
// Owned immutable fixed geometry. SceneView borrows these arrays only for the owner's lifetime.
#pragma once
#include "gpu/fixed/index.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace tmk::gpu::fixed {
struct Feature {
  std::uint32_t shape;
  std::int32_t owner;
  int x0, y0, x1, y1;
  long long guard_reach, clearance;
};

struct Meta {
  long long x0 = 0, y0 = 0, cell = 1000000, radius = 0, max_reach = 0, raster_cell = 250000;
  int nx = 0, ny = 0, raster_w = 0, raster_h = 0, loop_count = 0;
};

struct Scene {
  Meta meta;
  bool supported = false;
  bool index_cache_hit = false;
  std::string reason;
  std::vector<tmk::gpu::geometry::Shape> shapes;
  std::vector<tmk::gpu::geometry::Point> points;
  std::vector<Feature> features;
  std::vector<std::uint32_t> offsets, indices, loops;
  std::vector<std::uint8_t> raster;
  tmk::gpu::spatial::Index geometry_index;
};

struct SceneView {
  Meta meta;
  const tmk::gpu::geometry::Shape* shapes;
  const tmk::gpu::geometry::Point* points;
  const Feature* features;
  const std::uint32_t *offsets, *indices, *loops;
  const std::uint8_t* raster;
  tmk::gpu::spatial::View geometry_index;
};
}  // namespace tmk::gpu::fixed
