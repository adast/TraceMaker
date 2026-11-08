// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "gpu/device.hpp"
#include "gpu/fixed/async_session.hpp"
#include "gpu/fixed/options.hpp"
#include "gpu/fixed/scene.hpp"
#include "gpu/fixed/tile_broker.hpp"

namespace tmk::gpu::fixed {
// One router owns this supply for its entire lifetime. Lookup never waits for a pending tile.
class Prefetch {
  PrefetchOptions options_;
  int device_;
  bool verify_;
  AsyncTileBroker<AsyncSession> broker_;
  const void* owner_ = nullptr;
  bool configured_ = false, failed_ = false;
  std::uint64_t calls_ = 0, verified_ = 0, failures_ = 0;

public:
  Prefetch(PrefetchOptions options, int device, bool verify = false)
      : options_(options), device_(device), verify_(verify), broker_(options.scene_capacity) {
    if (options.tile_edge < 8 || options.tile_edge > 256 || !options.scene_capacity || options.scene_capacity > 128 ||
        !options.poll_interval || options.poll_interval > 4096)
      throw std::invalid_argument("Invalid fixed-geometry prefetch configuration");
    options_.enabled = options_.enabled && device >= 0 && compiled_backend() == Backend::Cuda;
  }

  bool configure(const void* owner, geometry::Point origin, long long pitch, int width, int height) {
    owner_ = owner;
    configured_ = options_.enabled && !failed_ &&
                  broker_.configure(owner, origin, pitch, width, height, options_.tile_edge, options_.poll_interval);
    return configured_;
  }

  bool verify() const {
    return verify_;
  }

  bool use_index_cache() const {
    return options_.index_cache;
  }

  void record_verified_hit() {
    ++verified_;
  }

  template <class BuildScene>
  bool predict(const Context& context, geometry::Point point, BuildScene build_scene, std::int32_t& answer) {
    ++calls_;
    if (!configured_ || failed_ || context.owner != owner_ || (context.flags & kViaProbe))
      return false;
    try {
      return broker_.predict(
          context, point,
          [&](const Context&, std::size_t capacity) {
            auto scene = build_scene();
            if (!scene.supported)
              return std::unique_ptr<AsyncSession>{};
            return std::make_unique<AsyncSession>(scene, capacity, device_);
          },
          answer);
    } catch (const std::exception&) {
      // A partial submission still owns its buffers. Shutdown drains before CPU fallback can continue.
      ++failures_;
      failed_ = true;
      configured_ = false;
      broker_.shutdown();
      return false;
    }
  }

  PrefetchStatistics statistics() const {
    const auto& m = broker_.metrics();
    return {calls_, m.hits, m.batches, m.points, verified_, failures_};
  }
};
}  // namespace tmk::gpu::fixed
