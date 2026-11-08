// SPDX-License-Identifier: GPL-3.0-or-later
// Deterministic delayed/failing backend, independent scalar oracle, no CUDA.
#include "gpu/fixed/tile_broker.hpp"
#include <cstdio>
#include <functional>
#include <random>
using tmk::gpu::fixed::Context;
using Point = tmk::gpu::geometry::Point;
static std::uint64_t checks = 0;

static void require(bool value, const char* message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}

static std::int32_t oracle(const Context& c, Point p) {
  auto h = c.generation * 0x923fabc781ull + c.flags * 113ull + static_cast<std::uint64_t>(c.hw) * 137ull +
           static_cast<std::uint64_t>(c.margin) * 157ull + static_cast<std::uint64_t>(c.via_mask) * 191ull +
           static_cast<std::uint64_t>(c.max_pad) * 197ull + static_cast<std::uint64_t>(c.layer) * 211ull +
           static_cast<std::uint64_t>(c.net) * 223ull + reinterpret_cast<std::uintptr_t>(c.owner) * 227ull +
           static_cast<std::uint64_t>(p.x) * 229ull + static_cast<std::uint64_t>(p.y) * 233ull;
  return h % 4 == 0 ? -2 : h % 4 == 1 ? -1 : static_cast<std::int32_t>(h % 10007);
}

struct Control {
  bool ready = false;
  int fault = 0;
};

struct Shared {
  unsigned live = 0, drains = 0;
  std::vector<std::shared_ptr<Control>> controls;
};

struct FakeBackend {
  struct Batch {
    std::vector<std::int32_t> codes;
  };

  Shared& shared;
  Context context;
  std::shared_ptr<Control> control;
  std::vector<Point> input;
  bool active = false;

  FakeBackend(Shared& s, Context c) : shared(s), context(c), control(std::make_shared<Control>()) {
    ++shared.live;
    shared.controls.push_back(control);
  }

  ~FakeBackend() {
    if (active) {
      ++shared.drains;
      active = false;
    }
    --shared.live;
  }

  std::size_t memory_bytes() const {
    return 128;
  }

  std::size_t pinned_bytes() const {
    return 64;
  }

  void submit(const std::vector<Point>& points) {
    if (control->fault == 1)
      throw std::runtime_error("Injected submit failure");
    require(!active && !points.empty(), "Invalid fake submit");
    input = points;
    active = true;
  }

  bool ready() {
    if (control->fault == 2)
      throw std::runtime_error("Injected query failure");
    return active && control->ready;
  }

  Batch collect() {
    require(ready(), "Collected unfinished fake output");
    if (control->fault == 3)
      throw std::runtime_error("Injected collect failure");
    Batch result;
    for (auto p : input)
      result.codes.push_back(oracle(context, p));
    active = false;
    if (control->fault == 4)
      result.codes.pop_back();
    if (control->fault == 5)
      result.codes[0] = INT32_MIN;
    return result;
  }
};

using Broker = tmk::gpu::fixed::AsyncTileBroker<FakeBackend>;

static auto factory(Shared& shared) {
  return [&shared](const Context& c, std::size_t capacity) {
    require(capacity == 65536, "Wrong fixed backend capacity");
    return std::make_unique<FakeBackend>(shared, c);
  };
}

static void complete(Shared& shared) {
  for (auto& c : shared.controls)
    c->ready = true;
}

static bool query(Broker& b, Shared& s, const Context& c, Point p) {
  std::int32_t answer = INT32_MIN;
  const auto hit = b.predict(c, p, factory(s), answer);
  if (hit)
    require(answer == oracle(c, p), "Stale/wrong private GPU answer");
  else
    require(answer == INT32_MIN, "Unavailable answer was published");
  return hit;
}

static void obtain(Broker& b, Shared& s, const Context& c, Point p) {
  for (unsigned attempt = 0; attempt < 1000; ++attempt) {
    if (query(b, s, c, p))
      return;
    complete(s);
  }
  throw std::runtime_error("Ready result never consumed");
}

int main() {
  try {
    int owners[3]{};
    Context original;
    original.owner = &owners[0];
    original.generation = 1;
    original.hw = 125000;
    original.margin = 35501;
    original.via_mask = 17;
    original.max_pad = 2000000;
    original.net = 1;
    {
      Shared s;
      Broker b;
      require(b.configure(original.owner, {-1000, -2000}, 10, 18, 19, 8, 4), "Valid grid rejected");
      require(!query(b, s, original, {-1000, -2000}), "Initial query did not fall back");
      for (unsigned i = 0; i < 100; ++i)
        require(!query(b, s, original, {-1000, -2000}), "Unready GPU result exposed");
      require(b.metrics().polls <= 26 && b.metrics().collections == 0 && b.metrics().batches == 1,
              "Unbounded polling or resubmission while pending");
      complete(s);
      obtain(b, s, original, {-1000, -2000});
      for (int y = 0; y < 19; ++y)
        for (int x = 0; x < 18; ++x)
          obtain(b, s, original, {-1000 + x * 10, -2000 + y * 10});
      require(b.metrics().points == 18 * 19 && b.metrics().unique_consumed == 18 * 19,
              "Partial boundary tiles missing/duplicated");
      const auto before = b.metrics().batches;
      require(!query(b, s, original, {-999, -2000}) && !query(b, s, original, {-1010, -2000}) &&
                  !query(b, s, original, {1000000001, 0}),
              "Off-grid/domain query admitted");
      require(b.metrics().batches == before, "Invalid query enqueued work");
      auto changed = original;
      changed.margin++;
      require(!query(b, s, changed, {-1000, -2000}), "New context reused cached output");
      require(b.configure(original.owner, {-900, -2000}, 10, 18, 19, 8, 4), "Changed grid rejected");
      complete(s);
      obtain(b, s, changed, {-900, -2000});
      require(b.metrics().stale_batches == 1, "Old-grid batch was retained");
      b.shutdown();
      require(!s.live && !b.retained_tile_bytes() && !b.retained_device_bytes() && !b.retained_pinned_bytes(),
              "Shutdown leaked retained resources");
    }
    {
      Shared s;
      Broker b(32);
      require(b.configure(original.owner, {0, 0}, 10, 8, 8, 8, 1), "Mutation grid rejected");
      obtain(b, s, original, {0, 0});
      std::vector<Context> contexts;
      auto add = [&](const std::function<void(Context&)>& mutate) {
        auto c = original;
        mutate(c);
        contexts.push_back(c);
      };
      add([](Context& c) { ++c.generation; });
      for (unsigned bit = 0; bit < 6; ++bit)
        add([bit](Context& c) { c.flags ^= std::uint64_t{1} << bit; });
      add([](Context& c) { ++c.hw; });
      add([](Context& c) { ++c.margin; });
      add([](Context& c) { ++c.via_mask; });
      add([](Context& c) { ++c.max_pad; });
      add([](Context& c) { ++c.layer; });
      add([](Context& c) { ++c.net; });
      for (const auto& c : contexts) {
        const auto count = b.metrics().context_misses;
        require(!query(b, s, c, {0, 0}), "Changed complete context reused output");
        require(b.metrics().context_misses == count + 1, "Context field omitted from key");
        obtain(b, s, c, {0, 0});
        obtain(b, s, original, {0, 0});
      }
      auto via = original;
      via.flags |= std::uint64_t{1} << 6;
      const auto count = b.metrics().context_misses;
      for (unsigned i = 0; i < 100; ++i)
        require(!query(b, s, via, {0, 0}), "Via incorrectly used track GPU scene");
      require(b.metrics().context_misses == count, "Via built a scene");
      auto reused = original;
      reused.generation += 100;
      require(!query(b, s, reused, {0, 0}), "Reused owner address retained generation");
      obtain(b, s, reused, {0, 0});
      auto other = original;
      other.owner = &owners[1];
      require(!query(b, s, other, {0, 0}), "Other owner admitted without configuration");
      require(b.configure(other.owner, {0, 0}, 10, 8, 8, 8, 1), "New owner grid rejected");
      obtain(b, s, other, {0, 0});
      b.shutdown();
      require(!s.live, "Owner mutation cleanup failed");
    }
    {
      Shared s;
      Broker b(1);
      auto second = original;
      second.margin++;
      require(b.configure(original.owner, {0, 0}, 1, 8, 8, 8, 4), "Bounded grid rejected");
      query(b, s, original, {0, 0});
      for (unsigned i = 0; i < 8; ++i)
        require(!query(b, s, second, {0, 0}), "Pending scene evicted or stale context exposed");
      require(s.live == 1 && !s.drains && b.metrics().capacity_declines > 0, "Pending context admission not bounded");
      complete(s);
      obtain(b, s, second, {0, 0});
      require(b.metrics().evicted_scenes == 1 && !s.drains && s.live == 1,
              "Completed abandoned scene not reclaimed safely");
      b.shutdown();
      require(!s.live, "Scene capacity cleanup failed");
    }
    {
      Shared s;
      Broker b(8, 1024, 2, 528);
      require(b.configure(original.owner, {0, 0}, 1, 32, 8, 8, 1), "Small tile budget rejected");
      for (int x : {0, 8, 16, 24, 0}) {
        obtain(b, s, original, {x, 0});
        require(b.retained_tiles() <= 2 && b.retained_tile_bytes() <= 528,
                "Actual vector capacity exceeded tile budget");
      }
      require(b.metrics().evicted_tiles >= 3, "Tile eviction not exercised");
      b.shutdown();
    }
    {
      Shared s;
      Broker b(1, 127);
      require(b.configure(original.owner, {0, 0}, 1, 8, 8, 8, 1), "Small scene budget grid rejected");
      require(!query(b, s, original, {0, 0}) && !s.live && b.retained_scenes() == 0 &&
                  b.metrics().capacity_declines == 1,
              "Oversized scene retained");
    }
    {
      Broker b;
      require(b.configure(original.owner, {0, 0}, 1, 8, 8, 8, 1), "Unsupported scene grid rejected");
      unsigned attempts = 0;
      std::int32_t answer = INT32_MIN;
      auto unavailable = [&](const Context&, std::size_t) -> std::unique_ptr<FakeBackend> {
        ++attempts;
        return {};
      };
      for (unsigned i = 0; i < 100; ++i) {
        require(!b.predict(original, {0, 0}, unavailable, answer), "Unsupported scene returned output");
      }
      require(attempts == 1, "Unsupported scene repeatedly exported");
    }
    for (int fault = 1; fault <= 5; ++fault) {
      Shared s;
      Broker b;
      require(b.configure(original.owner, {0, 0}, 1, 8, 8, 8, 1), "Failure grid rejected");
      bool rejected = false;
      try {
        if (fault == 1) {
          std::int32_t answer = INT32_MIN;
          b.predict(
              original, {0, 0},
              [&](const Context& c, std::size_t) {
                auto result = std::make_unique<FakeBackend>(s, c);
                result->control->fault = 1;
                return result;
              },
              answer);
        } else {
          query(b, s, original, {0, 0});
          s.controls.back()->fault = fault;
          complete(s);
          query(b, s, original, {0, 0});
        }
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      require(rejected && b.metrics().hits == 0, "Backend failure published a private answer");
      b.shutdown();
      require(!s.live, "Failed pending backend escaped cleanup");
    }
    {
      Shared s;
      Broker b(4, 512, 4, 2048);
      std::mt19937_64 rng(0xd73288f1);
      Context c = original;
      require(b.configure(c.owner, {-100, -100}, 2, 24, 24, 8, 8), "Fuzz grid rejected");
      for (unsigned i = 0; i < 100000; ++i) {
        if (i % 109 == 0) {
          c.generation = 1 + rng() % 4;
          c.flags = rng() % 4;
          c.net = 1 + static_cast<int>(rng() % 2);
          c.layer = static_cast<int>(rng() % 2);
        }
        if (i % 991 == 0)
          require(b.configure(c.owner, {-100 + static_cast<long long>(i % 2), -100}, 2, 24, 24, 8, 8),
                  "Fuzz reconfiguration rejected");
        if (rng() % 3 == 0)
          complete(s);
        const Point p{-100 + static_cast<long long>(rng() % 24) * 2, -100 + static_cast<long long>(rng() % 24) * 2};
        query(b, s, c, p);
        require(b.retained_scenes() <= 4 && b.retained_device_bytes() <= 512 && b.retained_tile_bytes() <= 2048 &&
                    b.retained_tiles() <= 4,
                "Random mutation exceeded retained budget");
      }
      require(b.metrics().hits > 1000 && b.metrics().stale_batches > 0 && b.metrics().evicted_scenes > 0,
              "Fuzz failed to exercise useful answers/mutations/eviction");
      b.shutdown();
      require(!s.live, "Fuzz cleanup failed");
    }
    std::printf("{\"complete\":true,\"all_exact\":true,\"checks\":%llu,\"random_queries\":100000,\"context_fields\":13,"
                "\"injected_failures\":5}\n",
                static_cast<unsigned long long>(checks));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
