// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/scene_query.hpp"
#include "io/kicad/board_reader.hpp"
#include "io/kicad/project_reader.hpp"
#include "route/obstacles.hpp"
#include <cassert>
#include <random>

int main() {
  const std::string path = std::string(TM_TEST_SOURCE_DIR) + "/tests/boards/plane_smd/plane_smd.kicad_pcb";
  auto board = tmk::io::read_board_file(path).board;
  auto rules = tmk::io::read_design_rules(path);
  // Exercise supported class-only scenes separately from conservative custom-rule refusal.
  const auto custom = rules.custom;
  rules.custom.clear();
  tmk::route::Obstacles obstacles(board, rules);
  const auto bounds = obstacles.bounds();
  std::mt19937_64 random(0xc3219a41u);
  std::size_t compared = 0;
  for (bool soft : {false, true}) {
    obstacles.set_soft_zones(soft);
    for (int layer = 0; layer < board.copper_count(); ++layer) {
      for (tmk::model::NetId net : {1, 2}) {
        for (tmk::Coord margin : {tmk::Coord{0}, tmk::Coord{70'000}}) {
          const auto context = obstacles.fixed_context(layer, 50'000, margin, net);
          const auto scene = obstacles.export_fixed_scene({bounds.x0, bounds.y0}, context);
          assert(scene.supported);
          for (int i = 0; i < 500; ++i) {
            const tmk::geom::Point point{
                bounds.x0 + static_cast<tmk::Coord>(random() % static_cast<std::uint64_t>(bounds.x1 - bounds.x0 + 1)),
                bounds.y0 + static_cast<tmk::Coord>(random() % static_cast<std::uint64_t>(bounds.y1 - bounds.y0 + 1))};
            const auto expected = obstacles.fixed_code(point, layer, 50'000, margin, net);
            const auto actual = tmk::gpu::fixed::code_at(tmk::gpu::fixed::host_view(scene), {point.x, point.y});
            assert(actual == expected);
            ++compared;
          }
        }
      }
    }
  }
  assert(compared >= 4000);
  const auto via_context = obstacles.fixed_context(0, 50'000, 0, 1, true);
  assert(!obstacles.export_fixed_scene({bounds.x0, bounds.y0}, via_context).supported);
  if (!custom.empty()) {
    rules.custom = custom;
    tmk::route::Obstacles unsupported(board, rules);
    const auto context = unsupported.fixed_context(0, 50'000, 0, 1);
    assert(!unsupported.export_fixed_scene({bounds.x0, bounds.y0}, context).supported);
  }
}
