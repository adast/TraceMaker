// SPDX-License-Identifier: GPL-3.0-or-later
// Private asynchronous transport for the already verified fixed-code kernel.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tmk::gpu::geometry {
struct Point;
}

namespace tmk::gpu::fixed {
struct Scene;

struct AsyncTiming {
  double submit_host_s = 0;
  double elapsed_until_collect_s = 0;
  double collect_host_s = 0;
  double kernel_s = 0;
};

struct AsyncBatch {
  AsyncTiming timing;
  std::vector<std::int32_t> codes;
};

class AsyncSession {
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  explicit AsyncSession(const Scene&, std::size_t capacity = 16384, int device = 0);
  ~AsyncSession();
  AsyncSession(const AsyncSession&) = delete;
  AsyncSession& operator=(const AsyncSession&) = delete;
  void submit(const std::vector<tmk::gpu::geometry::Point>&);
  bool pending() const;
  bool ready() const;
  AsyncBatch collect();  // Requires ready(); never waits for unfinished work.
  void wait();           // Explicit component-test/lifecycle synchronization.
  double setup_s() const;
  std::size_t memory_bytes() const;
  std::size_t pinned_bytes() const;
};
}  // namespace tmk::gpu::fixed
