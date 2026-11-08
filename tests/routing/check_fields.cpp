// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/device.hpp"
#include "gpu/field.hpp"
#include <cassert>
#include <random>

int main() {
  std::mt19937 random(0x8a731u);
  const bool gpu_available = !tmk::gpu::list_devices().empty();
  for (int test = 0; test < 32; ++test) {
    tmk::gpu::FieldProblem problem;
    problem.w = 3 + static_cast<int>(random() % 65);
    problem.h = 3 + static_cast<int>(random() % 49);
    problem.layers = test % 3 == 0 ? 4 : test % 3 == 1 ? 2 : 1;
    problem.step = test % 7 == 0 ? 0 : 100;
    problem.diag = 141;
    problem.via = test % 5 == 0 ? 0 : 500;
    const auto cells = static_cast<std::size_t>(problem.w * problem.h);
    const auto states = cells * static_cast<std::size_t>(problem.layers);
    std::vector<std::uint8_t> pass(states), vias(cells), targets(states);
    for (auto& value : pass)
      value = random() % 5 != 0;
    for (auto& value : vias)
      value = random() % 3 != 0;
    for (std::size_t i = 0; i < states; ++i)
      targets[i] = pass[i] && random() % 31 == 0;
    problem.pass = pass.data();
    problem.via_pass = vias.data();
    problem.target = targets.data();
    std::vector<std::int32_t> expected, actual;
    tmk::gpu::field_cpu_reference(problem, expected);
    tmk::gpu::field_cpu(problem, actual);
    assert(actual == expected);
    if (gpu_available) {
      assert(tmk::gpu::field_gpu(0, problem, actual).ok);
      assert(actual == expected);
    }
  }
}
