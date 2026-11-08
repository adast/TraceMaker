// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/planar_moves.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static std::uint64_t checks = 0;

static void require(bool okay) {
  ++checks;
  if (!okay)
    throw std::runtime_error("Planar move order differs");
}

static std::vector<int> original(int dir) {
  std::vector<int> result;
  for (int d = 0; d < 8; ++d) {
    if (dir != 8 && std::min((d - dir + 8) % 8, (dir - d + 8) % 8) >= 3)
      continue;
    result.push_back(d);
  }
  return result;
}

static std::vector<int> candidate(int dir) {
  std::vector<int> result;
  for (unsigned mask = tmk::route::kPlanarMoveMasks[static_cast<unsigned>(dir)]; mask; mask &= mask - 1)
    result.push_back(std::countr_zero(mask));
  return result;
}

static volatile std::uint64_t sink = 0;

int main(int argc, char**) {
  constexpr int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1}, dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  for (int dir = 0; dir < 9; ++dir) {
    const auto a = original(dir), b = candidate(dir);
    require(a == b);
    require(a.size() == (dir == 8 ? 8 : 5));
    for (int w = 1; w <= 31; ++w)
      for (int h = 1; h <= 27; ++h)
        for (int cx : {0, w / 2, w - 1})
          for (int cy : {0, h / 2, h - 1}) {
            std::vector<int> x, y;
            for (int d : a)
              if (cx + dx[d] >= 0 && cx + dx[d] < w && cy + dy[d] >= 0 && cy + dy[d] < h)
                x.push_back(d);
            for (int d : b)
              if (cx + dx[d] >= 0 && cx + dx[d] < w && cy + dy[d] >= 0 && cy + dy[d] < h)
                y.push_back(d);
            require(x == y);
          }
  }
  std::printf("{\"all_exact\":true,\"checks\":%llu}\n", static_cast<unsigned long long>(checks));
  if (argc == 1)
    return 0;
  std::vector<unsigned> directions(1u << 18);
  std::uint64_t random = 0x125d8b794a63ULL;
  for (auto& d : directions) {
    random ^= random << 13;
    random ^= random >> 7;
    random ^= random << 17;
    d = static_cast<unsigned>(random % 9);
  }
  for (int repeat = 0; repeat < 12; ++repeat)
    for (int phase = 0; phase < 2; ++phase) {
      const bool use_candidate = (repeat & 1) ? phase == 0 : phase == 1;
      const auto start = std::chrono::steady_clock::now();
      std::uint64_t sum = 0;
      for (int batch = 0; batch < 8; ++batch)
        for (unsigned dir : directions) {
          if (use_candidate) {
            for (unsigned mask = tmk::route::kPlanarMoveMasks[dir]; mask; mask &= mask - 1) {
              const auto d = std::countr_zero(mask);
              sum += (d + 1) * (dir + 3);
            }
          } else {
            for (unsigned d = 0; d < 8; ++d) {
              if (dir != 8 && std::min((d - dir + 8) % 8, (dir - d + 8) % 8) >= 3)
                continue;
              sum += (d + 1) * (dir + 3);
            }
          }
        }
      sink = sum;
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::printf("{\"repeat\":%d,\"candidate\":%s,\"seconds\":%.9f,\"checksum\":%llu}\n", repeat,
                  use_candidate ? "true" : "false", seconds, static_cast<unsigned long long>(sum));
    }
}
