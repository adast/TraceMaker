// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu/fixed/context.hpp"
#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>

int main() {
  std::vector<std::vector<std::uint64_t>> revisions(8);
  std::vector<std::jthread> workers;
  for (auto& row : revisions)
    workers.emplace_back([&row] {
      tmk::gpu::fixed::SceneRevision revision;
      for (int i = 0; i < 1000; ++i) {
        row.push_back(revision.value());
        revision.invalidate();
      }
    });
  workers.clear();
  std::vector<std::uint64_t> all;
  for (const auto& row : revisions)
    all.insert(all.end(), row.begin(), row.end());
  std::sort(all.begin(), all.end());
  assert(all.size() == 8000 && all.front() > 0);
  assert(std::adjacent_find(all.begin(), all.end()) == all.end());
  tmk::gpu::fixed::Context a, b;
  assert(a == b);
  b.generation = all.front();
  assert(!(a == b));
}
