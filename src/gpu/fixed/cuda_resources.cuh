// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace tmk::gpu::fixed::detail {
using Clock = std::chrono::steady_clock;

inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

inline void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct DeviceArray {
  T* data = nullptr;

  explicit DeviceArray(std::size_t size) {
    if (size)
      check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
  }

  ~DeviceArray() {
    if (data)
      cudaFree(data);
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  void upload(const std::vector<T>& values) {
    if (!values.empty())
      check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

struct CudaStream {
  cudaStream_t value = nullptr;

  CudaStream() {
    check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  ~CudaStream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
};

struct CudaEvent {
  cudaEvent_t value = nullptr;

  explicit CudaEvent(unsigned flags) {
    check(cudaEventCreateWithFlags(&value, flags));
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() {
    if (value)
      cudaEventDestroy(value);
  }
};

template <class T> struct PinnedHostArray {
  T* data = nullptr;

  explicit PinnedHostArray(std::size_t size) {
    check(cudaMallocHost(reinterpret_cast<void**>(&data), size * sizeof(T)));
  }

  PinnedHostArray(const PinnedHostArray&) = delete;
  PinnedHostArray& operator=(const PinnedHostArray&) = delete;

  ~PinnedHostArray() {
    if (data)
      cudaFreeHost(data);
  }
};

template <class T> struct StreamArray {
  T* data = nullptr;
  cudaStream_t stream;

  StreamArray(std::size_t size, cudaStream_t value) : stream(value) {
    check(cudaMallocAsync(reinterpret_cast<void**>(&data), size * sizeof(T), stream));
  }

  StreamArray(const StreamArray&) = delete;
  StreamArray& operator=(const StreamArray&) = delete;

  ~StreamArray() {
    if (data)
      cudaFreeAsync(data, stream);
  }
};
}  // namespace tmk::gpu::fixed::detail
