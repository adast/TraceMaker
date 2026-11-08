// SPDX-License-Identifier: GPL-3.0-or-later
// Immutable per-search costs; invalid/overflowing combinations use the original expression.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tmk::route {
class PlanarCostTable {
public:
  using I = std::int64_t;
  using Layer = std::array<I, 9 * 8>;

  PlanarCostTable(int layers, I step, I diag, I dir_pen, bool swap, const std::vector<I>& weights)
      : costs_(static_cast<std::size_t>(layers)) {
    for (int l = 0; l < layers; ++l) {
      auto& layer = costs_[static_cast<std::size_t>(l)];
      layer.fill(-1);
      if (step < 0 || diag < 0 ||
          (!weights.empty() &&
           (static_cast<std::size_t>(l) >= weights.size() || weights[static_cast<std::size_t>(l)] < 0)))
        continue;
      for (int dir = 0; dir < 9; ++dir)
        for (int d = 0; d < 8; ++d) {
          I value = (d & 1) ? diag : step;
          if (!weights.empty()) {
            if (__builtin_mul_overflow(value, weights[static_cast<std::size_t>(l)], &value))
              continue;
            value /= 1000;
          }
          if (dir_pen > 0 && !(d & 1) && ((d == 0 || d == 4) != ((l % 2 == 0) != swap)) &&
              __builtin_add_overflow(value, dir_pen, &value))
            continue;
          if (dir != 8 && d != dir) {
            const int a = (d - dir + 8) % 8, b = (dir - d + 8) % 8;
            I turn;
            if ((a < b ? a : b) == 1)
              turn = step / 2;
            else if (__builtin_mul_overflow(step, I{2}, &turn))
              continue;
            if (__builtin_add_overflow(value, turn, &value))
              continue;
          }
          layer[static_cast<std::size_t>(dir * 8 + d)] = value;
        }
    }
  }

  const I* row(int layer, int dir) const {
    return costs_[static_cast<std::size_t>(layer)].data() + dir * 8;
  }

  static bool add(I prefix, I extra, I& value) {
    I sum;
    if (prefix < 0 || __builtin_add_overflow(prefix, extra, &sum))
      return false;
    value = sum;
    return true;
  }

private:
  std::vector<Layer> costs_;
};
}  // namespace tmk::route
