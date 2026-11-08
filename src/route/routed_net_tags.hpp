// SPDX-License-Identifier: GPL-3.0-or-later
// Conservative same-net union over the ORIGINAL uint16 proximity contributors.
// A mixed union stays mixed until a proven empty count. Any count wrap or
// malformed update permanently disables this cell until a complete reset.
#pragma once
#include <cstdint>
#include <limits>

namespace tmk::route::net_tags {
inline constexpr std::uint32_t empty = 0, mixed = UINT32_MAX, unsafe = UINT32_MAX - 1;

inline void mark(std::uint32_t& tag, std::uint16_t before, int delta, std::int32_t net) {
  if (tag == unsafe)
    return;
  if (delta == 1) {
    if (before == UINT16_MAX || net < 0 || (before == 0 && tag != empty)) {
      tag = unsafe;
      return;
    }
    const auto code = static_cast<std::uint32_t>(net) + 1;
    if (before == 0)
      tag = code;
    else if (tag != code)
      tag = mixed;
  } else if (delta == -1) {
    if (before == 0) {
      tag = unsafe;
      return;
    }
    if (before == 1)
      tag = empty;
  } else
    tag = unsafe;
}

inline bool same_positive_net(std::uint32_t tag, std::int32_t net) {
  return net > 0 && tag == static_cast<std::uint32_t>(net) + 1;
}
}  // namespace tmk::route::net_tags
