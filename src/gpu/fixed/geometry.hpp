// SPDX-License-Identifier: GPL-3.0-or-later
// Private exact disk/rounded-core predicates. No floating-point approximation.
#pragma once
#include <cstdint>

namespace tmk::gpu::geometry {
struct Point {
  long long x, y;
};

struct Shape {
  std::uint32_t first, count;
  int closed;
  long long radius, x0, y0, x1, y1;
};

struct Probe {
  std::uint32_t shape;
  long long x, y, radius, clearance;
};
#ifdef __CUDACC__
#define EG_HD __host__ __device__
#else
#define EG_HD
#endif
using Wide = __int128_t;

EG_HD inline Wide orientation(Point a, Point b, Point p) {
  return Wide(b.x - a.x) * (p.y - a.y) - Wide(b.y - a.y) * (p.x - a.x);
}

EG_HD inline bool on_segment(Point a, Point b, Point p) {
  return p.x >= (a.x < b.x ? a.x : b.x) && p.x <= (a.x > b.x ? a.x : b.x) && p.y >= (a.y < b.y ? a.y : b.y) &&
         p.y <= (a.y > b.y ? a.y : b.y);
}

EG_HD inline bool point_segment(Point p, Point a, Point b, long long threshold) {
  const Wide dx = b.x - a.x, dy = b.y - a.y, px = p.x - a.x, py = p.y - a.y;
  const Wide squared = Wide(threshold) * threshold, length = dx * dx + dy * dy;
  if (length == 0)
    return px * px + py * py < squared;
  const Wide projection = dx * px + dy * py;
  if (projection <= 0)
    return px * px + py * py < squared;
  if (projection >= length) {
    const Wide qx = p.x - b.x, qy = p.y - b.y;
    return qx * qx + qy * qy < squared;
  }
  const Wide cross = dx * py - dy * px;
  return cross * cross < squared * length;
}

EG_HD inline bool inside(Point p, const Point* points, std::uint32_t count) {
  bool result = false;
  for (std::uint32_t i = 0, j = count - 1; i < count; j = i++) {
    const auto a = points[j], b = points[i];
    if (orientation(a, b, p) == 0 && on_segment(a, b, p))
      return true;
    if ((a.y > p.y) != (b.y > p.y)) {
      const Wide lhs = Wide(p.x - a.x) * (b.y - a.y), rhs = Wide(p.y - a.y) * (b.x - a.x);
      if (b.y > a.y ? lhs < rhs : lhs > rhs)
        result = !result;
    }
  }
  return result;
}

EG_HD inline bool closer(const Shape& shape, const Point* points, const Probe& query) {
  if (shape.x0 - query.clearance > query.x + query.radius || shape.x1 + query.clearance < query.x - query.radius ||
      shape.y0 - query.clearance > query.y + query.radius || shape.y1 + query.clearance < query.y - query.radius)
    return false;
  const long long threshold = query.clearance + shape.radius + query.radius;
  if (threshold <= 0)
    return false;
  const Point p{query.x, query.y};
  points += shape.first;
  if (shape.closed && inside(p, points, shape.count))
    return true;
  if (shape.count == 1)
    return point_segment(p, points[0], points[0], threshold);
  for (std::uint32_t i = 0; i + 1 < shape.count; ++i)
    if (point_segment(p, points[i], points[i + 1], threshold))
      return true;
  return shape.closed && shape.count >= 3 && point_segment(p, points[shape.count - 1], points[0], threshold);
}

#undef EG_HD
}  // namespace tmk::gpu::geometry
