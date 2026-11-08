// SPDX-License-Identifier: GPL-3.0-or-later
// Exact edge/row lookup for existing rounded-core predicates; no batching policy.
#pragma once
#include "gpu/fixed/geometry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tmk::gpu::spatial {
struct Spatial {
  long long x0 = 0, y0 = 0, x1 = 0, y1 = 0, cell = 1;
  int nx = 0, ny = 0;
  std::uint32_t rows = 0, cells = 0;
};

struct View {
  const Spatial* spatial = nullptr;
  const std::uint32_t *row_offsets = nullptr, *row_edges = nullptr, *cell_offsets = nullptr, *cell_edges = nullptr;
};

struct Index {
  std::vector<Spatial> spatial;
  std::vector<std::uint32_t> row_offsets, row_edges, cell_offsets, cell_edges;

  View view() const {
    return {spatial.data(), row_offsets.data(), row_edges.data(), cell_offsets.data(), cell_edges.data()};
  }
};
#ifdef __CUDACC__
#define IG_HD __host__ __device__
#else
#define IG_HD
#endif
IG_HD inline int coordinate(long long v, long long origin, long long cell, int length) {
  const auto value = static_cast<int>((v - origin) / cell);
  return value < 0 ? 0 : value >= length ? length - 1 : value;
}

IG_HD inline bool inside(View view, std::uint32_t shape_id, const tmk::gpu::geometry::Shape* shapes,
                         const tmk::gpu::geometry::Point* points, tmk::gpu::geometry::Point p) {
  const auto& shape = shapes[shape_id];
  points += shape.first;
  if (!view.spatial || !view.spatial[shape_id].nx)
    return tmk::gpu::geometry::inside(p, points, shape.count);
  const auto& data = view.spatial[shape_id];
  if (p.x < data.x0 || p.x > data.x1 || p.y < data.y0 || p.y > data.y1)
    return false;
  const auto row = static_cast<std::uint32_t>(coordinate(p.y, data.y0, data.cell, data.ny));
  bool parity = false;
  for (auto k = view.row_offsets[data.rows + row]; k < view.row_offsets[data.rows + row + 1]; ++k) {
    const auto edge = view.row_edges[k];
    const auto a = points[edge], b = points[edge + 1 < shape.count ? edge + 1 : 0];
    if (tmk::gpu::geometry::orientation(a, b, p) == 0 && tmk::gpu::geometry::on_segment(a, b, p))
      return true;
    if ((a.y > p.y) != (b.y > p.y)) {
      const tmk::gpu::geometry::Wide lhs = tmk::gpu::geometry::Wide(p.x - a.x) * (b.y - a.y),
                                     rhs = tmk::gpu::geometry::Wide(p.y - a.y) * (b.x - a.x);
      if (b.y > a.y ? lhs < rhs : lhs > rhs)
        parity = !parity;
    }
  }
  return parity;
}

IG_HD inline bool closer(View view, const tmk::gpu::geometry::Shape* shapes, const tmk::gpu::geometry::Point* points,
                         const tmk::gpu::geometry::Probe& query) {
  const auto& shape = shapes[query.shape];
  if (!view.spatial || !view.spatial[query.shape].nx)
    return tmk::gpu::geometry::closer(shape, points, query);
  if (shape.x0 - query.clearance > query.x + query.radius || shape.x1 + query.clearance < query.x - query.radius ||
      shape.y0 - query.clearance > query.y + query.radius || shape.y1 + query.clearance < query.y - query.radius)
    return false;
  const long long threshold = query.clearance + shape.radius + query.radius;
  if (threshold <= 0)
    return false;
  const tmk::gpu::geometry::Point p{query.x, query.y};
  if (shape.closed && inside(view, query.shape, shapes, points, p))
    return true;
  const auto& data = view.spatial[query.shape];
  points += shape.first;
  const int x0 = coordinate(p.x - threshold, data.x0, data.cell, data.nx),
            x1 = coordinate(p.x + threshold, data.x0, data.cell, data.nx);
  const int y0 = coordinate(p.y - threshold, data.y0, data.cell, data.ny),
            y1 = coordinate(p.y + threshold, data.y0, data.cell, data.ny);
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      const auto cell = static_cast<std::uint32_t>(y * data.nx + x);
      for (auto k = view.cell_offsets[data.cells + cell]; k < view.cell_offsets[data.cells + cell + 1]; ++k) {
        const auto edge = view.cell_edges[k];
        if (tmk::gpu::geometry::point_segment(p, points[edge], points[edge + 1 < shape.count ? edge + 1 : 0],
                                              threshold))
          return true;
      }
    }
  return false;
}

#undef IG_HD

// Any conservative index resolution preserves the exact predicate. The budget
// only selects the original linear fallback when a shape is costly to index.
inline Index build(const std::vector<tmk::gpu::geometry::Shape>& shapes,
                   const std::vector<tmk::gpu::geometry::Point>& points, std::size_t entry_budget = 20000000,
                   std::size_t cell_budget = 4000000, unsigned minimum_points = 16) {
  // GPU offsets and cell addressing use 32-bit types even if a caller supplies
  // larger budgets. Falling back leaves the original exact predicate intact.
  entry_budget = std::min<std::size_t>(entry_budget, UINT32_MAX);
  cell_budget = std::min<std::size_t>(cell_budget, INT32_MAX);
  Index out;
  out.spatial.resize(shapes.size());
  std::size_t cells_used = 0;
  for (std::size_t id = 0; id < shapes.size(); ++id) {
    const auto& shape = shapes[id];
    if (!shape.count || static_cast<std::uint64_t>(shape.first) + shape.count > points.size())
      throw std::runtime_error("Geometry index shape dimensions");
    if (shape.count < minimum_points)
      continue;
    const auto* vertices = points.data() + shape.first;
    Spatial data;
    data.x0 = data.x1 = vertices[0].x;
    data.y0 = data.y1 = vertices[0].y;
    for (std::uint32_t i = 1; i < shape.count; ++i) {
      data.x0 = std::min(data.x0, vertices[i].x);
      data.x1 = std::max(data.x1, vertices[i].x);
      data.y0 = std::min(data.y0, vertices[i].y);
      data.y1 = std::max(data.y1, vertices[i].y);
    }
    const long double area =
        (static_cast<long double>(data.x1 - data.x0) + 1) * (static_cast<long double>(data.y1 - data.y0) + 1);
    data.cell = std::max<long long>(
        100000,
        static_cast<long long>(std::sqrt(area / std::max<long double>(1, static_cast<long double>(shape.count) / 8))));
    const auto nx = (data.x1 - data.x0) / data.cell + 1, ny = (data.y1 - data.y0) / data.cell + 1;
    if (nx < 1 || ny < 1 || static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) > cell_budget - cells_used)
      continue;
    data.nx = static_cast<int>(nx);
    data.ny = static_cast<int>(ny);
    const auto cells = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const auto distance_edges = shape.count == 1 ? 1 : shape.closed ? shape.count : shape.count - 1;
    std::vector<std::array<int, 4>> rectangles;
    rectangles.reserve(shape.count);
    std::uint64_t row_entries = 0, cell_entries = 0;
    for (std::uint32_t edge = 0; edge < shape.count; ++edge) {
      const auto a = vertices[edge], b = vertices[edge + 1 < shape.count ? edge + 1 : 0];
      const int x0 = coordinate(std::min(a.x, b.x), data.x0, data.cell, data.nx),
                x1 = coordinate(std::max(a.x, b.x), data.x0, data.cell, data.nx);
      const int y0 = coordinate(std::min(a.y, b.y), data.y0, data.cell, data.ny),
                y1 = coordinate(std::max(a.y, b.y), data.y0, data.cell, data.ny);
      rectangles.push_back({x0, y0, x1, y1});
      row_entries += static_cast<std::uint64_t>(y1 - y0 + 1);
      if (edge < distance_edges)
        cell_entries += static_cast<std::uint64_t>(x1 - x0 + 1) * static_cast<std::uint64_t>(y1 - y0 + 1);
    }
    if (row_entries + cell_entries + out.row_edges.size() + out.cell_edges.size() > entry_budget ||
        static_cast<std::uint64_t>(out.row_offsets.size()) + static_cast<std::uint64_t>(ny) + 1 > UINT32_MAX ||
        static_cast<std::uint64_t>(out.cell_offsets.size()) + cells + 1 > UINT32_MAX)
      continue;
    std::vector<std::uint32_t> rows(static_cast<std::size_t>(ny) + 1, 0), buckets(cells + 1, 0);
    for (std::uint32_t edge = 0; edge < shape.count; ++edge) {
      const auto r = rectangles[edge];
      for (int y = r[1]; y <= r[3]; ++y) {
        ++rows[static_cast<std::size_t>(y) + 1];
        if (edge < distance_edges)
          for (int x = r[0]; x <= r[2]; ++x)
            ++buckets[static_cast<std::size_t>(y) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x) + 1];
      }
    }
    rows[0] = static_cast<std::uint32_t>(out.row_edges.size());
    buckets[0] = static_cast<std::uint32_t>(out.cell_edges.size());
    for (std::size_t i = 1; i < rows.size(); ++i)
      rows[i] += rows[i - 1];
    for (std::size_t i = 1; i < buckets.size(); ++i)
      buckets[i] += buckets[i - 1];
    data.rows = static_cast<std::uint32_t>(out.row_offsets.size());
    data.cells = static_cast<std::uint32_t>(out.cell_offsets.size());
    out.row_offsets.insert(out.row_offsets.end(), rows.begin(), rows.end());
    out.cell_offsets.insert(out.cell_offsets.end(), buckets.begin(), buckets.end());
    out.row_edges.resize(out.row_edges.size() + row_entries);
    out.cell_edges.resize(out.cell_edges.size() + cell_entries);
    for (std::uint32_t edge = 0; edge < shape.count; ++edge) {
      const auto r = rectangles[edge];
      for (int y = r[1]; y <= r[3]; ++y) {
        out.row_edges[rows[static_cast<std::size_t>(y)]++] = edge;
        if (edge < distance_edges)
          for (int x = r[0]; x <= r[2]; ++x)
            out.cell_edges[buckets[static_cast<std::size_t>(y) * static_cast<std::size_t>(nx) +
                                   static_cast<std::size_t>(x)]++] = edge;
      }
    }
    out.spatial[id] = data;
    cells_used += cells;
  }
  return out;
}
}  // namespace tmk::gpu::spatial
