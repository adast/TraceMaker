// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/async_session.hpp"
#include <stdexcept>

namespace tmk::gpu::fixed {
struct AsyncSession::Impl {};

namespace {
[[noreturn]] void unavailable() {
  throw std::runtime_error("CUDA fixed geometry is unavailable");
}
}  // namespace

AsyncSession::AsyncSession(const Scene&, std::size_t, int) {
  unavailable();
}

AsyncSession::~AsyncSession() = default;

void AsyncSession::submit(const std::vector<geometry::Point>&) {
  unavailable();
}

bool AsyncSession::pending() const {
  return false;
}

bool AsyncSession::ready() const {
  return false;
}

AsyncBatch AsyncSession::collect() {
  unavailable();
}

void AsyncSession::wait() {
  unavailable();
}

double AsyncSession::setup_s() const {
  return 0;
}

std::size_t AsyncSession::memory_bytes() const {
  return 0;
}

std::size_t AsyncSession::pinned_bytes() const {
  return 0;
}
}  // namespace tmk::gpu::fixed
