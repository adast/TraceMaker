// SPDX-License-Identifier: GPL-3.0-or-later
#include "app/board_batch.hpp"
#include "app/route_job.hpp"
#include "gpu/device.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace tmk::app {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct BoardJob {
  std::string id;
  fs::path input, output;
  long work = 0;
  std::uint64_t seed = 1;
};

struct BatchPlan {
  std::vector<BoardJob> jobs;
  std::set<fs::path> destinations;
  fs::path report, journal;
};

void reserve_destination(BatchPlan& plan, const fs::path& path) {
  if (!plan.destinations.insert(path).second)
    throw std::invalid_argument("Overlapping batch output: " + path.string());
}

BatchPlan read_plan(const fs::path& manifest, const fs::path& report) {
  std::ifstream input(manifest);
  if (!input)
    throw std::runtime_error("Cannot read batch manifest: " + manifest.string());
  const auto definitions = Json::parse(input);
  if (!definitions.is_array() || definitions.empty())
    throw std::invalid_argument("Batch manifest must be a nonempty array");
  BatchPlan plan;
  std::set<std::string> identifiers;
  std::set<fs::path> inputs{fs::canonical(manifest)};
  for (const auto& entry : definitions) {
    BoardJob job{entry.at("id").get<std::string>(), fs::canonical(entry.at("input").get<std::string>()),
                 fs::weakly_canonical(fs::absolute(entry.at("output").get<std::string>())),
                 entry.at("work").get<long>(), entry.value("seed", std::uint64_t{1})};
    if (job.id.empty() || !identifiers.insert(job.id).second || job.work <= 0 ||
        job.output.extension() != ".kicad_pcb" || !fs::is_regular_file(job.input))
      throw std::invalid_argument("Invalid job ID, board, work budget or output extension");
    for (const auto* extension : {".kicad_pcb", ".kicad_pro", ".kicad_dru"}) {
      inputs.insert(fs::weakly_canonical(fs::path(job.input).replace_extension(extension)));
      reserve_destination(plan, fs::path(job.output).replace_extension(extension));
    }
    for (const auto* suffix : {".json", ".log"})
      reserve_destination(plan, job.output.string() + suffix);
    plan.jobs.push_back(std::move(job));
  }
  plan.report = fs::weakly_canonical(fs::absolute(report));
  plan.journal = plan.report.string() + ".progress.jsonl";
  reserve_destination(plan, plan.report);
  reserve_destination(plan, plan.journal);
  for (const auto& destination : plan.destinations) {
    if (inputs.contains(destination) || fs::exists(destination) || fs::is_symlink(fs::symlink_status(destination)))
      throw std::invalid_argument("Output exists or overlaps an input: " + destination.string());
  }
  return plan;
}

class ActiveRoutes {
  std::atomic<std::size_t> active_{0}, peak_{0};

public:
  void enter() {
    const auto active = active_.fetch_add(1);
    auto peak = peak_.load();
    while (peak < active + 1 && !peak_.compare_exchange_weak(peak, active + 1)) {
    }
  }

  void leave() {
    active_.fetch_sub(1);
  }

  std::size_t peak() const {
    return peak_.load();
  }
};

class ActiveRoute {
  ActiveRoutes& routes_;

public:
  explicit ActiveRoute(ActiveRoutes& routes) : routes_(routes) {
    routes_.enter();
  }

  ~ActiveRoute() {
    routes_.leave();
  }

  ActiveRoute(const ActiveRoute&) = delete;
  ActiveRoute& operator=(const ActiveRoute&) = delete;
};

Json route_one(const BoardJob& definition, std::size_t worker, const BoardBatchOptions& options,
               Clock::time_point start, ActiveRoutes& active) {
  Json outcome{{"id", definition.id},
               {"input", definition.input.string()},
               {"output", definition.output.string()},
               {"work", definition.work},
               {"worker_index", worker}};
  std::ofstream log(definition.output.string() + ".log");
  if (!log)
    throw std::runtime_error("Cannot open board log");
  RouteJob job;
  job.in = definition.input.string();
  job.out = definition.output.string();
  job.json_out = definition.output.string() + ".json";
  job.threads = 1;
  job.variants = 1;
  job.opt.seed = definition.seed;
  job.opt.work_budget = definition.work;
  job.opt.time_limit_s = 3600;
  job.opt.gpu_device = options.no_gpu ? -1 : options.device;
  job.opt.verify_acceleration = options.verify_acceleration;
  job.rescue = options.rescue;
  job.why_open = options.why_open;
  job.log = [&](const std::string& line) { log << line << '\n'; };
  const auto result = [&] {
    ActiveRoute interval(active);
    outcome["route_started_s"] = seconds_since(start);
    auto value = run_route_job(std::move(job));
    outcome["route_finished_s"] = seconds_since(start);
    return value;
  }();
  outcome["processed"] = true;
  outcome["exit_code"] = result.exit_code();
  outcome["summary"] = result.summary;
  return outcome;
}

void write_report(const fs::path& path, const Json& report) {
  std::ofstream output(path);
  if (!output || !(output << report.dump(2) << '\n'))
    throw std::runtime_error("Cannot write batch report: " + path.string());
}
}  // namespace

BoardBatchResult run_board_batch(const fs::path& manifest, const fs::path& report, const BoardBatchOptions& options) {
  if (!options.workers || options.workers > 256 || (!options.no_gpu && options.device < 0))
    throw std::invalid_argument("Invalid board-worker count or GPU device");
  std::size_t visible_gpu_count = 0;
  if (!options.no_gpu) {
    const auto devices = gpu::list_devices();
    visible_gpu_count = devices.size();
    if (std::none_of(devices.begin(), devices.end(),
                     [&](const auto& device) { return device.index == options.device; }))
      throw std::runtime_error("Requested GPU is not visible");
  }
  const auto plan = read_plan(manifest, report);
  for (const auto& destination : plan.destinations)
    fs::create_directories(destination.parent_path());
  std::ofstream journal(plan.journal);
  if (!journal)
    throw std::runtime_error("Cannot open batch progress journal");
  std::mutex journal_mutex;
  std::atomic<std::size_t> next{0}, errors{0}, incomplete{0};
  ActiveRoutes active;
  std::vector<Json> outcomes(plan.jobs.size());
  const auto workers = std::min(options.workers, plan.jobs.size());
  const auto start = Clock::now();
  std::vector<std::jthread> pool;
  pool.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker)
    pool.emplace_back([&, worker] {
      for (;;) {
        const auto index = next.fetch_add(1);
        if (index >= plan.jobs.size())
          return;
        const auto began = Clock::now();
        Json outcome;
        try {
          outcome = route_one(plan.jobs[index], worker, options, start, active);
          if (outcome["exit_code"].get<int>() != 0)
            ++incomplete;
        } catch (const std::exception& error) {
          outcome = {{"id", plan.jobs[index].id}, {"processed", false}, {"error", error.what()}};
          ++errors;
        }
        outcome["wall_s"] = seconds_since(began);
        outcomes[index] = std::move(outcome);
        const std::lock_guard lock(journal_mutex);
        journal << Json{{"index", index},
                        {"id", plan.jobs[index].id},
                        {"processed", outcomes[index]["processed"]},
                        {"elapsed_s", seconds_since(start)}}
                       .dump()
                << '\n';
        journal.flush();
      }
    });
  pool.clear();  // Every route and its private log must finish before the final report.
  if (!journal)
    throw std::runtime_error("Cannot write batch progress journal");
  const auto seconds = seconds_since(start);
  write_report(plan.report, {{"complete", errors == 0},
                             {"field_backend", options.no_gpu ? "cpu" : "gpu"},
                             {"workers", workers},
                             {"visible_gpu_count", visible_gpu_count},
                             {"gpu_device", options.no_gpu ? -1 : options.device},
                             {"peak_active_routes", active.peak()},
                             {"wall_s", seconds},
                             {"boards", outcomes},
                             {"boards_per_second", static_cast<double>(plan.jobs.size()) / seconds}});
  return {plan.jobs.size(), errors.load(), incomplete.load(), seconds};
}
}  // namespace tmk::app
