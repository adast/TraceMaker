// SPDX-License-Identifier: GPL-3.0-or-later
// A complete fixed-scene key. Registry implementation is host C++ only.
#pragma once
#include <cstdint>
#include <tuple>

namespace tmk::gpu::fixed {
struct Context {
  const void* owner = nullptr;
  std::uint64_t generation = 0, flags = 0;
  long long hw = 0, margin = 0, via_mask = 0, max_pad = 0;
  int layer = 0, net = 0;

  bool operator==(const Context& b) const {
    return std::tie(owner, generation, flags, hw, margin, via_mask, max_pad, layer, net) ==
           std::tie(b.owner, b.generation, b.flags, b.hw, b.margin, b.via_mask, b.max_pad, b.layer, b.net);
  }
};

// Every lifetime and fixed-copper mutation gets a globally unique revision.
class SceneRevision {
  std::uint64_t value_;

public:
  SceneRevision();
  SceneRevision(const SceneRevision&) = delete;
  SceneRevision& operator=(const SceneRevision&) = delete;

  std::uint64_t value() const {
    return value_;
  }

  void invalidate();
};

inline constexpr std::uint64_t kViaProbe = std::uint64_t{1} << 6;
}  // namespace tmk::gpu::fixed
