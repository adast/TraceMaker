// SPDX-License-Identifier: GPL-3.0-or-later
// Exact ascending direction enumeration for the original eight planar moves.
#pragma once
#include <array>
#include <bit>
#include <cstdint>

namespace tmk::route {
inline constexpr auto kPlanarMoveMasks = [] {
  std::array<unsigned, 9> result{};
  for (int dir = 0; dir < 9; ++dir)
    for (int d = 0; d < 8; ++d) {
      const int a = (d - dir + 8) % 8, b = (dir - d + 8) % 8;
      if (dir == 8 || (a < b ? a : b) < 3)
        result[static_cast<unsigned>(dir)] |= 1u << d;
    }
  return result;
}();
}  // namespace tmk::route
