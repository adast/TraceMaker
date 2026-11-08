// SPDX-License-Identifier: GPL-3.0-or-later
// Keep 16-byte (f,key) entries. Raw keys order by state before snapshot bits.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tmk::route {
inline constexpr std::uint64_t kMaxQuant = (std::uint64_t{1} << 32) - 1;
inline constexpr std::uint64_t kDeepStateMask = (std::uint64_t{1} << 31) - 1;

inline constexpr std::uint64_t quantize(std::int64_t g) {
  return std::min<std::uint64_t>(static_cast<std::uint64_t>(std::max<std::int64_t>(g, 0)) >> 8, kMaxQuant);
}

class QueueKeyFormat {
public:
  constexpr QueueKeyFormat(bool deep, std::size_t states)
      : deep_(deep), raw_(!deep && states > 0 && states <= (std::uint64_t{1} << 32)),
        snapshot_(raw_ || (deep && states > 0 && states <= (std::uint64_t{1} << 31))) {}

  constexpr std::uint64_t encode(std::int64_t g, std::size_t state) const {
    if (!deep_ && !raw_)
      return state;
    const auto q = quantize(g);
    if (raw_)
      return (static_cast<std::uint64_t>(state) << 32) | q;
    return ((kMaxQuant - q) << 31) | state;  // Literal original deep key.
  }

  constexpr std::size_t state(std::uint64_t key) const {
    if (raw_)
      return static_cast<std::size_t>(key >> 32);
    return deep_ ? static_cast<std::size_t>(key & kDeepStateMask) : static_cast<std::size_t>(key);
  }

  constexpr bool proves_stale(std::uint64_t key, std::int64_t current_g) const {
    if (!snapshot_)
      return false;
    const auto q = raw_ ? (key & kMaxQuant) : kMaxQuant - (key >> 31);
    // q>0 proves that insertion g was nonnegative and at least q*256.
    // Saturation/rounding cause missed opportunities, never false positives.
    return q != 0 && current_g < static_cast<std::int64_t>(q << 8);
  }

  constexpr bool raw() const {
    return raw_;
  }

  constexpr bool snapshot_supported() const {
    return snapshot_;
  }

private:
  bool deep_, raw_, snapshot_;
};

}  // namespace tmk::route
