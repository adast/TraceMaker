// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/device.hpp"
#include "gpu/fixed/async_session.hpp"
#include "gpu/fixed/scene.hpp"
#include "io/kicad/board_reader.hpp"
#include "io/kicad/project_reader.hpp"
#include "route/obstacles.hpp"
#include <cassert>
#include <random>

int main() {
  if (tmk::gpu::compiled_backend() != tmk::gpu::Backend::Cuda || tmk::gpu::list_devices().empty())
    return 77;
  const std::string path = std::string(TM_TEST_SOURCE_DIR) + "/tests/boards/plane_smd/plane_smd.kicad_pcb";
  auto board = tmk::io::read_board_file(path).board;
  auto rules = tmk::io::read_design_rules(path);
  rules.custom.clear();  // Supported-class fixtures; refusal is checked separately by check_fixed_scene.
  tmk::route::Obstacles obstacles(board, rules);
  const auto bounds = obstacles.bounds();
  std::mt19937_64 random(0x7a981u);
  for (int layer = 0; layer < board.copper_count(); ++layer) {
    const auto context = obstacles.fixed_context(layer, 50'000, 70'000, 1);
    const auto scene = obstacles.export_fixed_scene({bounds.x0, bounds.y0}, context);
    assert(scene.supported);
    std::vector<tmk::gpu::geometry::Point> queries;
    std::vector<std::int32_t> expected;
    for (int i = 0; i < 1000; ++i) {
      const tmk::geom::Point point{
          bounds.x0 + static_cast<tmk::Coord>(random() % static_cast<std::uint64_t>(bounds.x1 - bounds.x0 + 1)),
          bounds.y0 + static_cast<tmk::Coord>(random() % static_cast<std::uint64_t>(bounds.y1 - bounds.y0 + 1))};
      queries.push_back({point.x, point.y});
      expected.push_back(obstacles.fixed_code_reference(point, layer, 50'000, 70'000, 1));
    }
    tmk::gpu::fixed::AsyncSession session(scene, queries.size(), 0);
    session.submit(queries);
    assert(session.pending());
    session.wait();
    assert(session.ready());
    assert(session.collect().codes == expected);
    assert(!session.pending());
  }
}
