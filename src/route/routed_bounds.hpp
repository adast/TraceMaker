// SPDX-License-Identifier: GPL-3.0-or-later
// Incremental conservative bounding-box unions for disk/copper and disk/holes.
// This is only an early-out. Uncertain cells always use original geometry.
#pragma once
#include "route/routed_net_tags.hpp"
#include <algorithm>
#include <limits>
#include <vector>

namespace tmk::route {
// Parameters are fixed for one routing lattice; uncertain geometry disables the shortcut.
struct RoutedBoundsConfig {
  int width = 0, height = 0, layers = 0;
  long long origin_x = 0, origin_y = 0, pitch = 0;
  bool eligible = false, enabled = true, verify = false;
  long long max_radius = 0, mask_expansion = 0, hole_clearance = 0, copper_clearance = 0;
};

class RoutedBounds {
  std::vector<std::uint16_t> copper_count_, hole_count_;
  std::vector<std::uint32_t> copper_, holes_;
  bool enabled_ = false, eligible_ = false, verify_ = false;
  int nx_ = 0, ny_ = 0, nl_ = 0;
  std::size_t plane_ = 0;
  long long ox_ = 0, oy_ = 0, pitch_ = 0, max_radius_ = 0, max_mask_ = 0, hc_ = 0, copper_reach_ = 0, hole_reach_ = 0;
  std::uint64_t queries_ = 0, early_ = 0, verified_ = 0, empty_hits_ = 0, own_hits_ = 0, copper_marks_ = 0,
                hole_marks_ = 0;

  static bool coordinate(long long v) {
    return v >= -1000000000 && v <= 1000000000;
  }

  static void update(std::uint32_t& tag, std::uint16_t& count, int delta, int net) {
    net_tags::mark(tag, count, delta, net);
    count = static_cast<std::uint16_t>(count + delta);
  }

  template <class F> void rectangle(long long x0, long long y0, long long x1, long long y1, long long reach, F f) {
    if (!coordinate(x0) || !coordinate(x1) || !coordinate(y0) || !coordinate(y1) || x0 > x1 || y0 > y1) {
      enabled_ = false;
      return;
    }
    // All arithmetic stays within +/-3e9 before division. Clamp before narrowing.
    const int a = static_cast<int>(std::clamp<long long>((x0 - reach - ox_) / pitch_ - 1, 0, nx_));
    const int b = static_cast<int>(std::clamp<long long>((x1 + reach - ox_) / pitch_ + 1, -1, nx_ - 1));
    const int c = static_cast<int>(std::clamp<long long>((y0 - reach - oy_) / pitch_ - 1, 0, ny_));
    const int d = static_cast<int>(std::clamp<long long>((y1 + reach - oy_) / pitch_ + 1, -1, ny_ - 1));
    for (int y = c; y <= d; ++y)
      for (int x = a; x <= b; ++x)
        f(static_cast<std::size_t>(y) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(x));
  }

public:
  void configure(const RoutedBoundsConfig& config) {
    const auto nx = config.width, ny = config.height, nl = config.layers;
    const auto ox = config.origin_x, oy = config.origin_y, pitch = config.pitch;
    const auto radius = config.max_radius, mask = config.mask_expansion;
    const auto hc = config.hole_clearance, clearance = config.copper_clearance;
    const bool eligible = config.eligible;
    enabled_ = false;
    verify_ = config.verify;
    nx_ = nx;
    ny_ = ny;
    nl_ = nl;
    ox_ = ox;
    oy_ = oy;
    pitch_ = pitch;
    max_radius_ = radius;
    max_mask_ = mask;
    hc_ = hc;
    eligible_ = eligible && nx > 0 && ny > 0 && nl > 0 && nl <= 64 && pitch >= 2 && coordinate(ox) && coordinate(oy) &&
                radius >= 0 && radius <= 1000000000 && mask >= 0 && mask <= 1000000000 && hc >= 0 && hc <= 1000000000 &&
                clearance >= 0 && clearance <= 1000000000;
    copper_.clear();
    copper_count_.clear();
    holes_.clear();
    hole_count_.clear();
    if (!eligible_ || !config.enabled)
      return;
    copper_reach_ = radius + std::max(clearance, mask + 1000) + 1;
    hole_reach_ = radius + hc + 1;
    if (copper_reach_ > 1000000000 || hole_reach_ > 1000000000) {
      eligible_ = false;
      return;
    }
    plane_ = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (plane_ > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(nl)) {
      eligible_ = false;
      return;
    }
    try {
      const auto cells = plane_ * static_cast<std::size_t>(nl);
      copper_.assign(cells, net_tags::empty);
      copper_count_.assign(cells, 0);
      if (hc > 0) {
        holes_.assign(plane_, net_tags::empty);
        hole_count_.assign(plane_, 0);
      }
      enabled_ = true;
    } catch (const std::exception&) {
      copper_.clear();
      copper_count_.clear();
      holes_.clear();
      hole_count_.clear();
    }
  }

  bool enabled() const {
    return enabled_;
  }

  bool verify() const {
    return verify_;
  }

  void deactivate() {
    enabled_ = false;
  }

  // Diagnostic comparisons are counted separately from deterministic routing work.
  std::uint64_t verified_predictions() const {
    return verified_;
  }

  void verified() {
    ++verified_;
  }

  void mark_copper(long long x0, long long y0, long long x1, long long y1, std::uint64_t layers, int net, int delta) {
    if (!enabled_)
      return;
    rectangle(x0, y0, x1, y1, copper_reach_, [&](std::size_t xy) {
      for (int l = 0; l < nl_; ++l)
        if (layers & (std::uint64_t{1} << l)) {
          const auto i = static_cast<std::size_t>(l) * plane_ + xy;
          update(copper_[i], copper_count_[i], delta, net);
          ++copper_marks_;
        }
    });
  }

  void mark_hole(long long x, long long y, long long drill, int net, int delta) {
    if (!enabled_ || hc_ <= 0)
      return;
    if (!coordinate(x) || !coordinate(y) || drill < 0 || drill > 1000000000) {
      deactivate();
      return;
    }
    const auto r = drill / 2;
    rectangle(x - r, y - r, x + r, y + r, hole_reach_, [&](std::size_t i) {
      update(holes_[i], hole_count_[i], delta, net);
      ++hole_marks_;
    });
  }

  bool predict(std::size_t i, std::size_t xy, int net, long long hw, long long margin, long long mask) {
    if (!enabled_)
      return false;
    ++queries_;
    if (net <= 0 || hw < 0 || margin < 0 || margin > max_radius_ || hw > max_radius_ - margin || mask < 0 ||
        mask > max_mask_)
      return false;
    const auto c = copper_[i], h = hc_ > 0 ? holes_[xy] : net_tags::empty;
    if ((c != net_tags::empty && !net_tags::same_positive_net(c, net)) ||
        (h != net_tags::empty && !net_tags::same_positive_net(h, net)))
      return false;
    ++early_;
    if (c == net_tags::empty && h == net_tags::empty)
      ++empty_hits_;
    else
      ++own_hits_;
    return true;
  }
};
}  // namespace tmk::route
