// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>

namespace tmk::gpu::fixed {
// The selected configuration supplies ready track answers; via legality stays on CPU.
struct PrefetchOptions {
  bool enabled = true;
  bool index_cache = true;
  std::size_t scene_capacity = 64;
  int tile_edge = 64;
  std::uint64_t poll_interval = 1;
};

// Runtime observations, not deterministic search-work counters.
struct PrefetchStatistics {
  std::uint64_t calls = 0, hits = 0, batches = 0, points = 0, verified = 0, failures = 0;
};
}  // namespace tmk::gpu::fixed
