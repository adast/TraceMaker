// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/async_session.hpp"
#include "gpu/fixed/cuda_resources.cuh"
#include "gpu/fixed/scene_query.hpp"
#include <algorithm>

namespace tmk::gpu::fixed {
using detail::check;
using detail::Clock;
using detail::CudaEvent;
using detail::CudaStream;
using detail::DeviceArray;
using detail::PinnedHostArray;
using detail::seconds;
using detail::StreamArray;

namespace {
__global__ void evaluate(SceneView view, const geometry::Point* queries, std::int32_t* output, int count) {
  for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x * gridDim.x))
    output[i] = code_at(view, queries[i]);
}
}  // namespace

struct AsyncSession::Impl {
  // Stream is destroyed last, draining asynchronous buffer frees as well.
  CudaStream stream;
  CudaEvent kernel_begin{cudaEventDefault}, kernel_end{cudaEventDefault};
  CudaEvent done{cudaEventDisableTiming};
  PinnedHostArray<geometry::Point> host_input;
  PinnedHostArray<std::int32_t> host_output;
  StreamArray<geometry::Point> input;
  StreamArray<std::int32_t> output;
  DeviceArray<geometry::Shape> shapes;
  DeviceArray<geometry::Point> points;
  DeviceArray<Feature> features;
  DeviceArray<std::uint32_t> offsets, indices, loops;
  DeviceArray<std::uint8_t> raster;
  DeviceArray<spatial::Spatial> spatial;
  DeviceArray<std::uint32_t> row_offsets, row_edges, cell_offsets, cell_edges;
  SceneView view{};
  std::size_t capacity, count = 0, scene_bytes = 0;
  bool active = false, failed = false;
  double setup = 0, submit_s = 0;
  Clock::time_point submitted{};

  Impl(const Scene& s, std::size_t limit)
      : host_input(limit), host_output(limit), input(limit, stream.value), output(limit, stream.value),
        shapes(s.shapes.size()), points(s.points.size()), features(s.features.size()), offsets(s.offsets.size()),
        indices(s.indices.size()), loops(s.loops.size()), raster(s.raster.size()),
        spatial(s.geometry_index.spatial.size()), row_offsets(s.geometry_index.row_offsets.size()),
        row_edges(s.geometry_index.row_edges.size()), cell_offsets(s.geometry_index.cell_offsets.size()),
        cell_edges(s.geometry_index.cell_edges.size()), capacity(limit) {
    shapes.upload(s.shapes);
    points.upload(s.points);
    features.upload(s.features);
    offsets.upload(s.offsets);
    indices.upload(s.indices);
    loops.upload(s.loops);
    raster.upload(s.raster);
    const auto& g = s.geometry_index;
    spatial.upload(g.spatial);
    row_offsets.upload(g.row_offsets);
    row_edges.upload(g.row_edges);
    cell_offsets.upload(g.cell_offsets);
    cell_edges.upload(g.cell_edges);
    // Legacy upload ordering does not automatically reach a nonblocking stream.
    check(cudaStreamSynchronize(cudaStreamLegacy));
    view = {s.meta,
            shapes.data,
            points.data,
            features.data,
            offsets.data,
            indices.data,
            loops.data,
            raster.data,
            {spatial.data, row_offsets.data, row_edges.data, cell_offsets.data, cell_edges.data}};
    scene_bytes = s.shapes.size() * sizeof(geometry::Shape) + s.points.size() * sizeof(geometry::Point) +
                  s.features.size() * sizeof(Feature) + s.raster.size() + g.spatial.size() * sizeof(spatial::Spatial) +
                  (s.offsets.size() + s.indices.size() + s.loops.size() + g.row_offsets.size() + g.row_edges.size() +
                   g.cell_offsets.size() + g.cell_edges.size()) *
                      sizeof(std::uint32_t);
  }

  ~Impl() {
    // All pinned/device scene storage must outlive every pending copy/kernel.
    cudaStreamSynchronize(stream.value);
  }
};

AsyncSession::AsyncSession(const Scene& scene, std::size_t capacity, int device) {
  if (!scene.supported || capacity == 0 || capacity > 65536)
    throw std::runtime_error("Invalid asynchronous fixed-scene request");
  int devices = 0;
  check(cudaGetDeviceCount(&devices));
  if (device < 0 || device >= devices)
    throw std::runtime_error("Fixed-geometry GPU is not visible");
  check(cudaSetDevice(device));
  const auto start = Clock::now();
  cudaFuncAttributes attributes{};
  check(cudaFuncGetAttributes(&attributes, tmk::gpu::fixed::evaluate));
  impl_ = std::make_unique<Impl>(scene, capacity);
  impl_->setup = seconds(start);
}

AsyncSession::~AsyncSession() = default;

bool AsyncSession::pending() const {
  return impl_->active;
}

double AsyncSession::setup_s() const {
  return impl_->setup;
}

std::size_t AsyncSession::memory_bytes() const {
  return impl_->scene_bytes + impl_->capacity * (sizeof(geometry::Point) + sizeof(std::int32_t));
}

std::size_t AsyncSession::pinned_bytes() const {
  return impl_->capacity * (sizeof(geometry::Point) + sizeof(std::int32_t));
}

void AsyncSession::submit(const std::vector<geometry::Point>& queries) {
  auto& s = *impl_;
  if (s.failed || s.active || queries.empty() || queries.size() > s.capacity)
    throw std::runtime_error("Asynchronous fixed session is failed, busy or outside capacity");
  for (auto p : queries)
    if (p.x < -1000000000 || p.x > 1000000000 || p.y < -1000000000 || p.y > 1000000000)
      throw std::runtime_error("Asynchronous fixed query coordinate domain");
  s.submitted = Clock::now();
  std::copy(queries.begin(), queries.end(), s.host_input.data);
  s.count = queries.size();
  try {
    check(cudaMemcpyAsync(s.input.data, s.host_input.data, s.count * sizeof(queries[0]), cudaMemcpyHostToDevice,
                          s.stream.value));
    check(cudaMemsetAsync(s.output.data, 0xa5, s.count * sizeof(std::int32_t), s.stream.value));
    check(cudaEventRecord(s.kernel_begin.value, s.stream.value));
    tmk::gpu::fixed::evaluate<<<std::min<std::size_t>((s.count + 127) / 128, 1024), 128, 0, s.stream.value>>>(
        s.view, s.input.data, s.output.data, static_cast<int>(s.count));
    check(cudaGetLastError());
    check(cudaEventRecord(s.kernel_end.value, s.stream.value));
    check(cudaMemcpyAsync(s.host_output.data, s.output.data, s.count * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                          s.stream.value));
    check(cudaEventRecord(s.done.value, s.stream.value));
    s.active = true;
    s.submit_s = seconds(s.submitted);
  } catch (...) {
    // Partial submission retains its storage until destruction; never reuse it.
    s.failed = true;
    throw;
  }
}

bool AsyncSession::ready() const {
  auto& s = *impl_;
  if (s.failed)
    throw std::runtime_error("Asynchronous fixed session failed");
  if (!s.active)
    return false;
  const auto error = cudaEventQuery(s.done.value);
  if (error == cudaErrorNotReady)
    return false;
  if (error != cudaSuccess) {
    s.failed = true;
    check(error);
  }
  return true;
}

AsyncBatch AsyncSession::collect() {
  auto& s = *impl_;
  if (!ready())
    throw std::runtime_error("Asynchronous fixed result is not ready");
  AsyncBatch result;
  result.timing.submit_host_s = s.submit_s;
  result.timing.elapsed_until_collect_s = seconds(s.submitted);
  const auto start = Clock::now();
  result.codes.assign(s.host_output.data, s.host_output.data + s.count);
  if (std::any_of(result.codes.begin(), result.codes.end(), [](std::int32_t code) { return code < -2; })) {
    s.failed = true;
    throw std::runtime_error("Incomplete asynchronous fixed-code output");
  }
  float milliseconds = 0;
  const auto error = cudaEventElapsedTime(&milliseconds, s.kernel_begin.value, s.kernel_end.value);
  if (error != cudaSuccess) {
    s.failed = true;
    check(error);
  }
  result.timing.kernel_s = milliseconds / 1000.0;
  result.timing.collect_host_s = seconds(start);
  s.active = false;
  return result;
}

void AsyncSession::wait() {
  auto& s = *impl_;
  if (s.failed)
    throw std::runtime_error("Asynchronous fixed session failed");
  if (s.active) {
    const auto error = cudaEventSynchronize(s.done.value);
    if (error != cudaSuccess) {
      s.failed = true;
      check(error);
    }
  }
}
}  // namespace tmk::gpu::fixed
