// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/index_cache.hpp"
#include "gpu/fixed/scene_query.hpp"
#include "route/obstacles.hpp"
#include <algorithm>
#include <stdexcept>

namespace tmk::route {
namespace {
void require_scene(bool condition, const char* reason) {
  if (!condition)
    throw std::runtime_error(reason);
}

struct SceneBuilder {
  gpu::fixed::Scene& out;

  std::uint32_t append_shape(const geom::Shape& geometry) {

    require_scene(!geometry.pts.empty() && geometry.r >= 0 && geometry.r <= 1000000000,
                  "Unsupported empty/radius shape");
    require_scene(out.points.size() + geometry.pts.size() < UINT32_MAX, "Fixed scene vertex count");
    const auto index = static_cast<std::uint32_t>(out.shapes.size());
    out.shapes.push_back({static_cast<std::uint32_t>(out.points.size()),
                          static_cast<std::uint32_t>(geometry.pts.size()), geometry.closed, geometry.r, geometry.box.x0,
                          geometry.box.y0, geometry.box.x1, geometry.box.y1});
    for (auto vertex : geometry.pts) {
      require_scene(vertex.x >= -1000000000 && vertex.x <= 1000000000 && vertex.y >= -1000000000 &&
                        vertex.y <= 1000000000,
                    "Fixed scene coordinate domain");
      out.points.push_back({vertex.x, vertex.y});
    }
    return index;
  }

  void add_feature(const geom::Shape& geometry, const geom::Box& original_box, Coord clearance, model::NetId owner,
                   Coord guard) {
    auto& meta = out.meta;

    if (geometry.pts.empty())
      return;
    const auto threshold = meta.radius + geometry.r + clearance;
    require_scene(threshold >= -1000000000 && threshold <= 1000000000 && owner >= 0,
                  "Fixed feature arithmetic/owner domain");
    const int x0 = tmk::gpu::fixed::coordinate(original_box.x0, meta.x0, meta.cell, meta.nx),
              x1 = tmk::gpu::fixed::coordinate(original_box.x1, meta.x0, meta.cell, meta.nx);
    const int y0 = tmk::gpu::fixed::coordinate(original_box.y0, meta.y0, meta.cell, meta.ny),
              y1 = tmk::gpu::fixed::coordinate(original_box.y1, meta.y0, meta.cell, meta.ny);
    out.features.push_back({append_shape(geometry), owner, x0, y0, x1, y1, guard, clearance});
    meta.max_reach = std::max<long long>({meta.max_reach, clearance, guard});
  }

  void build_indices(bool use_cache) {
    auto& meta = out.meta;
    const auto cells = static_cast<std::size_t>(meta.nx) * static_cast<std::size_t>(meta.ny);
    out.offsets.assign(cells + 1, 0);
    std::uint64_t entries = 0;
    for (const auto& f : out.features)
      for (int y = f.y0; y <= f.y1; ++y)
        for (int x = f.x0; x <= f.x1; ++x) {
          require_scene(++entries <= 20000000, "Fixed scene index capacity");
          ++out.offsets[static_cast<std::size_t>(y) * static_cast<std::size_t>(meta.nx) + static_cast<std::size_t>(x) +
                        1];
        }
    for (std::size_t i = 1; i < out.offsets.size(); ++i)
      out.offsets[i] += out.offsets[i - 1];
    out.indices.resize(entries);
    auto cursor = out.offsets;
    for (std::size_t i = 0; i < out.features.size(); ++i) {
      const auto& f = out.features[i];
      for (int y = f.y0; y <= f.y1; ++y)
        for (int x = f.x0; x <= f.x1; ++x)
          out.indices[cursor[static_cast<std::size_t>(y) * static_cast<std::size_t>(meta.nx) +
                             static_cast<std::size_t>(x)]++] = static_cast<std::uint32_t>(i);
    }
    if (use_cache)
      out.geometry_index = tmk::gpu::spatial::cached_build(out.shapes, out.points, out.index_cache_hit);
    else
      out.geometry_index = tmk::gpu::spatial::build(out.shapes, out.points);
  }
};
}  // namespace

gpu::fixed::Context Obstacles::fixed_context(int layer, Coord hw, Coord margin, model::NetId probe_net,
                                             bool via_probe) const {
  gpu::fixed::Context context;
  // Include every rule-sensitive setting in the immutable-scene key.
  context.owner = this;
  context.generation = fixed_revision_.value();
  context.layer = layer;
  context.net = probe_net;
  context.hw = hw;
  context.margin = margin;
  context.via_mask = via_mask_;
  context.max_pad = vias_off_pads_max_;
  context.flags = static_cast<std::uint64_t>(soft_zones_) | (static_cast<std::uint64_t>(soft_mask_graphics_) << 1) |
                  (static_cast<std::uint64_t>(soft_copper_graphics_) << 2) |
                  (static_cast<std::uint64_t>(vias_off_pads_) << 3) |
                  (static_cast<std::uint64_t>(pad_via_exempt_) << 4) |
                  (static_cast<std::uint64_t>(linear_zone_tests_) << 5) | (static_cast<std::uint64_t>(via_probe) << 6);
  return context;
}

gpu::fixed::Scene Obstacles::export_fixed_scene(geom::Point p, const gpu::fixed::Context& context,
                                                bool use_index_cache) const {
  gpu::fixed::Scene out;
  const auto layer = context.layer, probe_net = context.net;
  const Coord hw = context.hw, margin = context.margin;
  const bool via_probe = (context.flags & gpu::fixed::kViaProbe) != 0;
  if (!(fixed_context(layer, hw, margin, probe_net, via_probe) == context))
    throw std::invalid_argument("Fixed-scene context is no longer current");
  // Only track-point scenes are consumed by the selected GPU path. Via checks stay scalar.
  if (via_probe) {
    out.reason = "Via probes use the scalar reference";
    return out;
  }
  SceneBuilder builder{out};
  if (!r_.custom.empty() || re_->any_physical_hole_clearance() || re_->any_physical_clearance()) {
    out.reason = "Custom rules require a position-dependent scene";
    return out;
  }
  if (hw < 0 || margin < 0 || hw > 1000000000 || margin > 1000000000 - hw || layer < 0 || layer >= b_.copper_count()) {
    out.reason = "Probe parameters outside exact batch domain";
    return out;
  }
  auto& meta = out.meta;
  meta.x0 = bounds_.x0;
  meta.y0 = bounds_.y0;
  meta.radius = hw + margin;
  meta.nx = static_cast<int>((bounds_.x1 - bounds_.x0) / meta.cell) + 1;
  meta.ny = static_cast<int>((bounds_.y1 - bounds_.y0) / meta.cell) + 1;
  require_scene(meta.nx > 0 && meta.ny > 0 &&
                    static_cast<std::uint64_t>(meta.nx) * static_cast<std::uint64_t>(meta.ny) <= 4000000,
                "Fixed scene grid dimensions");
  if (!outline_.empty()) {
    out.loops.push_back(builder.append_shape(geom::Shape::polygon(outline_)));
    for (const auto& cutout : cutouts_)
      out.loops.push_back(builder.append_shape(geom::Shape::polygon(cutout)));
    out.raster = inside_raster_;
    meta.raster_w = ir_w_;
    meta.raster_h = ir_h_;
    meta.raster_cell = ir_cell_;
  }
  meta.loop_count = static_cast<int>(out.loops.size());
  const auto disk = geom::Shape::point(p, hw + margin);
  drc::CopperItem sample;
  sample.kind = drc::ItemKind::Track;
  sample.index = -1;
  sample.sub = 0;
  sample.net = probe_net;
  sample.layers = model::layer_bit(layer);
  sample.shapes = {disk};
  sample.box = disk.box;
  sample.footprint = -1;
  sample.pos = p;
  sample.width = 2 * hw;
  sample.owner = -1;
  sample.removed = false;
  sample.free_via = false;
  // Only immutable copper participates; mutable routed items remain on the scalar path.
  for (const auto& item : cm_.items) {
    if (item.owner >= 0 || item.removed || !(item.layers & model::layer_bit(layer)) || soft_item(item))
      continue;
    const Coord base = re_->clearance(sample, item, layer);
    Coord clearance = base;
    if (via_mask_ > 0 && (layer == 0 || layer == b_.copper_count() - 1) && item.kind != drc::ItemKind::Zone &&
        item.kind != drc::ItemKind::Pad) {
      if (item.kind == drc::ItemKind::Via)
        clearance = std::max(clearance, via_mask_ + 1000);
    }
    for (const auto& geometry : item.shapes) {
      builder.add_feature(geometry, item.box, clearance, item.net, re_->max_clearance() + 1);
    }
  }
  const int side = layer == 0 ? 0 : layer == b_.copper_count() - 1 ? 1 : -1;
  if (side >= 0) {
    const Coord extra = 1000;
    for (const auto& aperture : apertures_[side])
      for (const auto& geometry : aperture.shapes)
        builder.add_feature(geometry, aperture.box, aperture.margin + extra, aperture.net, extra + 1);
  }
  const Coord text_clearance = std::max(r_.minimums.clearance, r_.default_class().clearance);
  for (const auto& [l, geometry] : texts_)
    if (l == layer)
      builder.add_feature(geometry, geometry.box, text_clearance, 0, -1);
  // Fixed holes and edges keep their independent clearances and ownership semantics.
  const Coord hole_clearance = re_->router_hole_clearance();
  for (const auto& hole : cm_.holes) {
    if (hole.removed || (hole.item >= 0 && cm_.items[static_cast<std::size_t>(hole.item)].owner >= 0))
      continue;
    builder.add_feature(hole.shape, hole.shape.box, std::max(hole_clearance, hole.clearance),
                        hole.plated ? hole.net : 0, std::max(hole_clearance, max_hole_local_) + 1);
  }
  const Coord edge_clearance = re_->router_edge_clearance();
  for (const auto& geometry : edge_segs_)
    builder.add_feature(geometry, geometry.box, edge_clearance, 0, edge_clearance + 1);
  for (const auto& [geometry, zone] : keepouts_)
    if ((zone->copper & model::layer_bit(layer)) && zone->keepout_tracks)
      builder.add_feature(geometry, geometry.box, 1, 0, -1);
  if (side >= 0 && !soft_mask_graphics_)
    for (const auto& geometry : mask_open_[side])
      builder.add_feature(geometry, geometry.box, 100000, 0, -1);
  builder.build_indices(use_index_cache);
  out.supported = true;
  return out;
}
}  // namespace tmk::route
