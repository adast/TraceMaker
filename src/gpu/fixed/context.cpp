// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/context.hpp"
#include <atomic>
#include <limits>
#include <stdexcept>

namespace tmk::gpu::fixed {
namespace {
std::atomic<std::uint64_t> next_revision{1};

std::uint64_t fresh_revision() {
  auto value = next_revision.load(std::memory_order_relaxed);
  for (;;) {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Fixed-scene revisions exhausted");
    if (next_revision.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
      return value;
  }
}
}  // namespace

SceneRevision::SceneRevision() : value_(fresh_revision()) {}

void SceneRevision::invalidate() {
  value_ = fresh_revision();
}
}  // namespace tmk::gpu::fixed
