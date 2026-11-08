// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/planar_cost_table.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

using I = std::int64_t;
using W = __int128_t;
constexpr I kMax = std::numeric_limits<I>::max();
constexpr I kMin = std::numeric_limits<I>::min();

// Follow each original signed operation, rejecting it before any undefined overflow.
bool literal(I step, I diag, I pen, bool swap, const std::vector<I>& weights, int l, int dir, int d, I extra, I& out) {
  auto fits = [](W v) { return v >= kMin && v <= kMax; };
  W value = (d & 1) ? diag : step;
  if (!weights.empty()) {
    value *= weights[static_cast<std::size_t>(l)];
    if (!fits(value))
      return false;
    value /= 1000;
  }
  value += extra;
  if (!fits(value))
    return false;
  if (pen > 0 && !(d & 1) && ((d == 0 || d == 4) != ((l % 2 == 0) != swap))) {
    value += pen;
    if (!fits(value))
      return false;
  }
  if (dir != 8 && d != dir) {
    W turn = std::min((d - dir + 8) % 8, (dir - d + 8) % 8) == 1 ? W(step / 2) : W(step) * 2;
    if (!fits(turn))
      return false;
    value += turn;
    if (!fits(value))
      return false;
  }
  out = static_cast<I>(value);
  return true;
}

int main() {
  std::mt19937_64 rng(0x37a472ULL);
  std::uint64_t checks = 0, accepted = 0, rejected = 0;
  const I edges[] = {kMin, kMin + 1, -1000, -1, 0, 1, 3, 1000, kMax / 1000, kMax / 2, kMax - 1, kMax};
  auto test = [&](I step, I diag, I pen, bool swap, const std::vector<I>& weights, I extra) {
    tmk::route::PlanarCostTable table(4, step, diag, pen, swap, weights);
    for (int l = 0; l < 4; ++l)
      for (int dir = 0; dir < 9; ++dir)
        for (int d = 0; d < 8; ++d) {
          I actual = -77, expected = -88;
          bool fast = tmk::route::PlanarCostTable::add(table.row(l, dir)[d], extra, actual);
          bool defined = literal(step, diag, pen, swap, weights, l, dir, d, extra, expected);
          ++checks;
          if (fast) {
            ++accepted;
            if (!defined || actual != expected)
              std::abort();
          } else {
            ++rejected;
            if (actual != -77)
              std::abort();
          }
        }
  };
  for (I step : edges)
    for (I diag : edges)
      for (I pen : edges)
        for (I extra : edges)
          for (bool swap : {false, true}) {
            test(step, diag, pen, swap, {}, extra);
            test(step, diag, pen, swap, {1000, 1001, 1500, 2000}, extra);
          }
  for (int n = 0; n < 24000; ++n) {
    const bool normal = n % 2 == 0;
    I step = normal ? I(rng() % 1000000) : I(rng());
    I diag = normal ? step * 1414 / 1000 : I(rng());
    I pen = normal ? I(rng() % 1000000) : I(rng());
    I extra = normal ? I(rng() % 1000000000) : I(rng());
    std::vector<I> weights;
    if (n % 3)
      for (int l = 0; l < 4; ++l)
        weights.push_back(normal ? I(1000 + rng() % 5000) : I(rng()));
    test(step, diag, pen, n & 1, weights, extra);
  }
  if (accepted < 4000000 || rejected < 1000000)
    std::abort();
  std::cout << "{\"all_exact\":true,\"checks\":" << checks << ",\"accepted\":" << accepted
            << ",\"rejected\":" << rejected << "}\n";
}
