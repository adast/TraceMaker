// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "gpu/fixed/scene.hpp"

namespace tmk::gpu::fixed {
#ifdef __CUDACC__
#define FX_HD __host__ __device__
#else
#define FX_HD
#endif
FX_HD inline int coordinate(long long v, long long origin, long long cell, int length) {
  const auto value = static_cast<int>((v - origin) / cell);
  return value < 0 ? 0 : value >= length ? length - 1 : value;
}

FX_HD inline bool inside(const SceneView& view, tmk::gpu::geometry::Point p) {
  const auto& m = view.meta;
  if (!m.loop_count)
    return true;
  if (view.raster) {
    const auto x = (p.x - m.x0) / m.raster_cell, y = (p.y - m.y0) / m.raster_cell;
    if (x < 0 || y < 0 || x >= m.raster_w || y >= m.raster_h)
      return false;
    const auto state = view.raster[y * m.raster_w + x];
    if (!state)
      return false;
    if (state != 2)
      return true;
  }
  for (int i = 0; i < m.loop_count; ++i) {
    const bool in = tmk::gpu::spatial::inside(view.geometry_index, view.loops[i], view.shapes, view.points, p);
    if (i == 0 ? !in : in)
      return false;
  }
  return true;
}

FX_HD inline std::int32_t code_at(const SceneView& view, tmk::gpu::geometry::Point p) {
  if (!inside(view, p))
    return -2;
  const auto& m = view.meta;
  const auto reach = m.radius + m.max_reach;
  const int x0 = coordinate(p.x - reach, m.x0, m.cell, m.nx), x1 = coordinate(p.x + reach, m.x0, m.cell, m.nx);
  const int y0 = coordinate(p.y - reach, m.y0, m.cell, m.ny), y1 = coordinate(p.y + reach, m.y0, m.cell, m.ny);
  std::int32_t code = -1;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      const auto cell = y * m.nx + x;
      for (auto k = view.offsets[cell]; k < view.offsets[cell + 1]; ++k) {
        const auto& feature = view.features[view.indices[k]];
        if (feature.owner > 0 && feature.owner == code)
          continue;
        if (feature.guard_reach >= 0) {
          const auto r = m.radius + feature.guard_reach;
          const int qx0 = coordinate(p.x - r, m.x0, m.cell, m.nx), qx1 = coordinate(p.x + r, m.x0, m.cell, m.nx);
          const int qy0 = coordinate(p.y - r, m.y0, m.cell, m.ny), qy1 = coordinate(p.y + r, m.y0, m.cell, m.ny);
          if (qx0 > feature.x1 || qx1 < feature.x0 || qy0 > feature.y1 || qy1 < feature.y0)
            continue;
        }
        const tmk::gpu::geometry::Probe query{feature.shape, p.x, p.y, m.radius, feature.clearance};
        if (!tmk::gpu::spatial::closer(view.geometry_index, view.shapes, view.points, query))
          continue;
        if (feature.owner == 0 || (code != -1 && code != feature.owner))
          return -2;
        code = feature.owner;
      }
    }
  return code;
}

#undef FX_HD

inline SceneView host_view(const Scene& scene) {
  return {scene.meta,
          scene.shapes.data(),
          scene.points.data(),
          scene.features.data(),
          scene.offsets.data(),
          scene.indices.data(),
          scene.loops.data(),
          scene.raster.empty() ? nullptr : scene.raster.data(),
          scene.geometry_index.view()};
}
}  // namespace tmk::gpu::fixed
