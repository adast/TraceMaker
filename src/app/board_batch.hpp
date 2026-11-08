// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <filesystem>

namespace tmk::app {
struct BoardBatchOptions {
  std::size_t workers = 2;
  int device = 0;
  bool no_gpu = false;
  bool rescue = true;
  bool why_open = true;
  bool verify_acceleration = false;
};

struct BoardBatchResult {
  std::size_t boards = 0, errors = 0, incomplete = 0;
  double seconds = 0;

  int exit_code() const {
    return errors ? 1 : incomplete ? 3 : 0;
  }
};

// Validate the entire manifest before writing anything. Each worker owns one route and its output family.
BoardBatchResult run_board_batch(const std::filesystem::path& manifest, const std::filesystem::path& report,
                                 const BoardBatchOptions& options);
}  // namespace tmk::app
