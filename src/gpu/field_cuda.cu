// SPDX-License-Identifier: GPL-3.0-or-later
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

#include "gpu/device.hpp"
#include "gpu/field.hpp"

namespace tmk::gpu {
namespace {
constexpr unsigned kMask = 0xffffffffu;
constexpr int kThreads = 128;
constexpr int kMaxRounds = 100000;
constexpr long long kKeyInf = 0x7fffffffffffffffLL;

// A line sweep is a segmented prefix-min of d[i] - cost*i. A warp handles
// 32 cells at once; blocked cells reset the scan, including its inter-tile carry.
// Signed 64-bit keys are essential even when all final distances fit in int32.
__device__ bool sweep_line(const std::uint8_t* pass, std::int32_t* d, int start, int stride, int len, int cost) {
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int blocks = (len + 31) / 32;
  bool changed = false;
  long long carry = kKeyInf;
  for (int block = 0; block < blocks; ++block) {
    const int i = block * 32 + lane;
    const int idx = start + i * stride;
    const bool open = i < len && pass[idx];
    const int value = open ? d[idx] : kFieldInf;
    const long long at = static_cast<long long>(cost) * i;
    long long key = open && value < kFieldInf ? value - at : kKeyInf;
    unsigned reset = open ? 0u : 1u;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const long long other = __shfl_up_sync(kMask, key, off);
      const unsigned other_reset = __shfl_up_sync(kMask, reset, off);
      if (lane >= off) {
        if (!reset)
          key = min(key, other);
        reset |= other_reset;
      }
    }
    const long long run = reset ? key : min(carry, key);
    if (open && run != kKeyInf && run + at < value) {
      d[idx] = static_cast<int>(run + at);
      changed = true;
    }
    carry = __shfl_sync(kMask, run, 31);
  }
  // Every lane reloads only its own cells, so the two passes share no
  // cross-lane memory dependency: shuffles exchange register values instead.
  carry = kKeyInf;
  for (int block = blocks - 1; block >= 0; --block) {
    const int i = block * 32 + lane;
    const int idx = start + i * stride;
    const bool open = i < len && pass[idx];
    const int value = open ? d[idx] : kFieldInf;
    const long long at = static_cast<long long>(cost) * i;
    long long key = open && value < kFieldInf ? value + at : kKeyInf;
    unsigned reset = open ? 0u : 1u;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const long long other = __shfl_down_sync(kMask, key, off);
      const unsigned other_reset = __shfl_down_sync(kMask, reset, off);
      if (lane + off < 32) {
        if (!reset)
          key = min(key, other);
        reset |= other_reset;
      }
    }
    const long long run = reset ? key : min(carry, key);
    if (open && run != kKeyInf && run - at < value) {
      d[idx] = static_cast<int>(run - at);
      changed = true;
    }
    carry = __shfl_sync(kMask, run, 0);
  }
  return __any_sync(kMask, changed);
}

__device__ int line_count(int w, int h, int direction) {
  return direction == 0 ? h : direction == 1 ? w : w + h - 1;
}

// Forward and reverse sweeps may be paired per line because the lines and
// layers are disjoint. Four directions leave the CPU's exact per-round state.
__device__ bool sweep(const std::uint8_t* pass, std::int32_t* d, int w, int h, int layers, int direction, int cost) {
  const int lines = line_count(w, h, direction);
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int warps = static_cast<int>(gridDim.x * blockDim.x / 32);
  bool changed = false;
  for (int line = warp; line < lines * layers; line += warps) {
    const int layer = line / lines, t = line % lines;
    int x, y, dx, dy;
    if (direction == 0) {
      x = 0;
      y = t;
      dx = 1;
      dy = 0;
    } else if (direction == 1) {
      x = t;
      y = 0;
      dx = 0;
      dy = 1;
    } else {
      dx = 1;
      dy = direction == 2 ? 1 : -1;
      const int sy = dy > 0 ? 0 : h - 1;
      if (t < w) {
        x = t;
        y = sy;
      } else {
        x = 0;
        const int k = t - w;
        y = k < sy ? k : k + 1;
      }
    }
    int len = w - x;
    if (dy > 0)
      len = min(len, h - y);
    if (dy < 0)
      len = min(len, y + 1);
    if (direction == 1)
      len = h;
    changed |= sweep_line(pass, d, layer * w * h + y * w + x, dy * w + dx, len, cost);
  }
  return changed;
}

__device__ bool relax_vias(const std::uint8_t* pass, const std::uint8_t* via_pass, std::int32_t* d, int n, int layers,
                           int cost) {
  bool changed = false;
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += static_cast<int>(gridDim.x * blockDim.x)) {
    if (!via_pass[i])
      continue;
    int best = kFieldInf;
    for (int l = 0; l < layers; ++l)
      best = min(best, d[l * n + i]);
    if (best >= kFieldInf)
      continue;
    for (int l = 0; l < layers; ++l)
      if (pass[l * n + i] && best + cost < d[l * n + i]) {
        d[l * n + i] = best + cost;
        changed = true;
      }
  }
  return changed;
}

__global__ void init_kernel(const std::uint8_t* target, std::int32_t* d, int total) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < total)
    d[i] = target[i] ? 0 : kFieldInf;
}

__global__ void sweep_kernel(const std::uint8_t* pass, std::int32_t* d, int w, int h, int layers, int direction,
                             int cost, int* changed) {
  const bool local = sweep(pass, d, w, h, layers, direction, cost);
  if (local && (threadIdx.x & 31) == 0)
    atomicExch(changed, 1);
}

__global__ void via_kernel(const std::uint8_t* pass, const std::uint8_t* via_pass, std::int32_t* d, int n, int layers,
                           int cost, int* changed) {
  const bool local = relax_vias(pass, via_pass, d, n, layers, cost);
  if (__any_sync(kMask, local) && (threadIdx.x & 31) == 0)
    atomicExch(changed, 1);
}

// All stages and convergence checks stay on this GPU. A cooperative launch
// bounds the grid to resident blocks, making each grid-wide barrier valid.
__global__ void solve_kernel(const std::uint8_t* pass, const std::uint8_t* via_pass, std::int32_t* d, int w, int h,
                             int layers, int step, int diag, int via, int* changed) {
  const auto grid = cooperative_groups::this_grid();
  for (int round = 0; round < kMaxRounds; ++round) {
    if (grid.thread_rank() == 0)
      *changed = 0;
    grid.sync();
    for (int direction = 0; direction < 4; ++direction) {
      const bool local = sweep(pass, d, w, h, layers, direction, direction < 2 ? step : diag);
      if (local && (threadIdx.x & 31) == 0)
        atomicExch(changed, 1);
      grid.sync();
    }
    if (layers > 1) {
      const bool local = relax_vias(pass, via_pass, d, w * h, layers, via);
      if (__any_sync(kMask, local) && (threadIdx.x & 31) == 0)
        atomicExch(changed, 1);
      grid.sync();
    }
    if (*changed == 0)
      return;
    // Keep all threads in this round until they have read the flag. Otherwise
    // thread zero could reset it while another block is deciding to return.
    grid.sync();
  }
}

cudaError_t launch_solver(int device, const std::uint8_t* pass, const std::uint8_t* via_pass, std::int32_t* d,
                          const FieldProblem& p, int* changed, cudaStream_t stream) {
  int cooperative = 0, sms = 0, active = 0;
  cudaError_t error = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
  if (error != cudaSuccess)
    return error;
  if (cooperative) {
    error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess)
      return error;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, solve_kernel, kThreads, 0);
    if (error != cudaSuccess)
      return error;
    const int useful = ((p.w + p.h - 1) * p.layers + kThreads / 32 - 1) / (kThreads / 32);
    const int blocks = std::max(1, std::min(useful, active * sms));
    int w = p.w, h = p.h, layers = p.layers, step = p.step, diag = p.diag, via = p.via;
    void* args[] = {&pass, &via_pass, &d, &w, &h, &layers, &step, &diag, &via, &changed};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(solve_kernel), blocks, kThreads, args, 0, stream);
  }
  // Noncooperative devices still use the exact parallel scans, with stream
  // kernel boundaries providing the grid barriers instead.
  const int n = p.w * p.h;
  for (int round = 0; round < kMaxRounds; ++round) {
    error = cudaMemsetAsync(changed, 0, sizeof(int), stream);
    if (error != cudaSuccess)
      return error;
    for (int direction = 0; direction < 4; ++direction) {
      const int lines = p.layers * (direction == 0 ? p.h : direction == 1 ? p.w : p.w + p.h - 1);
      sweep_kernel<<<(lines * 32 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          pass, d, p.w, p.h, p.layers, direction, direction < 2 ? p.step : p.diag, changed);
    }
    if (p.layers > 1)
      via_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(pass, via_pass, d, n, p.layers, p.via,
                                                                         changed);
    int host_changed = 0;
    error = cudaMemcpyAsync(&host_changed, changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess)
      return error;
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess || !host_changed)
      return error;
  }
  return cudaErrorNotReady;  // never publish a field that has not converged
}
}  // namespace

GpuStatus field_gpu(int device, const FieldProblem& p, std::vector<std::int32_t>& out) {
  out.clear();
  if (p.w <= 0 || p.h <= 0 || p.layers <= 0 || p.step < 0 || p.diag < 0 || p.via < 0 || p.via > 1000000000)
    return {false, "invalid field dimensions or costs"};
  const std::size_t n = static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h);
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) / static_cast<std::size_t>(p.layers))
    return {false, "field dimensions exceed device indexing"};
  const std::size_t total = n * static_cast<std::size_t>(p.layers);
  const std::size_t bytes = total * (sizeof(std::int32_t) + 2) + n + sizeof(int);
  if (!has_free_memory(device, bytes))
    return {false, "not enough free device memory"};
  if (cudaSetDevice(device) != cudaSuccess)
    return {false, "cudaSetDevice failed"};
  const cudaStream_t stream = cudaStreamPerThread;
  std::uint8_t *pass = nullptr, *via_pass = nullptr, *target = nullptr;
  std::int32_t* distance = nullptr;
  int* changed = nullptr;
  auto cleanup = [&] {
    cudaFreeAsync(pass, stream);
    cudaFreeAsync(via_pass, stream);
    cudaFreeAsync(target, stream);
    cudaFreeAsync(distance, stream);
    cudaFreeAsync(changed, stream);
    cudaStreamSynchronize(stream);
  };
  cudaError_t error = cudaMallocAsync(&pass, total, stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&via_pass, n, stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&target, total, stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&distance, total * sizeof(std::int32_t), stream);
  if (error == cudaSuccess)
    error = cudaMallocAsync(&changed, sizeof(int), stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(pass, p.pass, total, cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(via_pass, p.via_pass, n, cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(target, p.target, total, cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    init_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        target, distance, static_cast<int>(total));
    error = cudaGetLastError();
  }
  if (error == cudaSuccess)
    error = launch_solver(device, pass, via_pass, distance, p, changed, stream);
  int host_changed = 1;
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&host_changed, changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess)
    error = cudaStreamSynchronize(stream);
  if (error == cudaSuccess && !host_changed) {
    out.resize(total);
    error = cudaMemcpyAsync(out.data(), distance, total * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess)
      error = cudaStreamSynchronize(stream);
  }
  cleanup();
  if (error != cudaSuccess || host_changed) {
    cudaGetLastError();
    out.clear();
    return {false, "CUDA field allocation, execution or convergence failed"};
  }
  return {true, {}};
}
}  // namespace tmk::gpu
