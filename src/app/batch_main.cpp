// SPDX-License-Identifier: GPL-3.0-or-later
#include "app/board_batch.hpp"
#include <CLI/CLI.hpp>
#include <cstdio>
#include <string>

int main(int argc, char** argv) try {
  CLI::App cli("Route independent boards concurrently with one selected GPU");
  std::string manifest, report;
  tmk::app::BoardBatchOptions options;
  cli.add_option("manifest", manifest, "JSON array of {id,input,output,work,seed}")
      ->required()
      ->check(CLI::ExistingFile);
  cli.add_option("--report", report, "Combined result JSON")->required();
  cli.add_option("--workers", options.workers, "Concurrent boards; one routing thread per board")
      ->check(CLI::Range(1, 256));
  cli.add_option("--device", options.device, "Visible GPU device index")->check(CLI::NonNegativeNumber);
  cli.add_flag("--no-gpu", options.no_gpu, "Use CPU reference fields");
  cli.add_flag("!--no-rescue", options.rescue, "Skip additional rescue passes after the requested work budget");
  cli.add_flag("!--no-why-open", options.why_open, "Skip the open-connection explanation pass");
  cli.add_flag("--verify-acceleration", options.verify_acceleration, "Check accelerated geometry with CPU references");
  try {
    cli.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    return cli.exit(error);
  }
  const auto result = tmk::app::run_board_batch(manifest, report, options);
  std::printf("%zu boards, %.3f s, %zu errors, %zu incomplete boards\n", result.boards, result.seconds, result.errors,
              result.incomplete);
  return result.exit_code();
} catch (const std::exception& error) {
  std::fprintf(stderr, "Batch failed: %s\n", error.what());
  return 1;
}
