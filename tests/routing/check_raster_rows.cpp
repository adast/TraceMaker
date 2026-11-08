// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/raster_rows.hpp"
#include <cassert>

void check_false(bool value) {
  assert(!value);
}

#include <array>
#include <limits>
#include <random>

namespace {
using Crossings = std::vector<std::vector<double>>;

// Original constructor scan, independent of the interval/prefix implementation.
void original_scan(const Crossings& crossings, std::int64_t x0, std::int64_t cell, std::vector<std::uint8_t>& row) {
  std::vector<std::size_t> at(crossings.size());
  for (std::size_t x = 0; x < row.size(); ++x) {
    const double centre = static_cast<double>(x0 + static_cast<std::int64_t>(x) * cell + cell / 2);
    bool inside = true;
    for (std::size_t loop = 0; loop < crossings.size(); ++loop) {
      auto& k = at[loop];
      while (k < crossings[loop].size() && crossings[loop][k] <= centre)
        ++k;
      const bool odd = ((crossings[loop].size() - k) & 1u) != 0;
      if (loop == 0 ? !odd : odd)
        inside = false;
    }
    if (row[x] == 255)
      row[x] = inside ? 1 : 0;
  }
}
}  // namespace

void test_boundaries() {
  tmk::route::RasterRows classifier;
  const Crossings crossings{{0, 20}, {3, 11}, {8, 15}, {10, 10}};
  std::vector<std::uint8_t> row(21, 255);
  row[5] = 2;
  row[17] = 0;
  row[20] = 1;
  auto expected = row;
  original_scan(crossings, 0, 1, expected);
  assert(classifier.classify(crossings, 0, 1, row.data(), static_cast<int>(row.size())));
  assert(row == expected);
  assert(row[5] == 2);
  assert(row[17] == 0);
  assert(row[20] == 1);
}

void test_seeded_rows() {
  tmk::route::RasterRows classifier;
  std::mt19937_64 random(0x732ef98a);
  for (unsigned test = 0; test < 5000; ++test) {
    const int width = 1 + static_cast<int>(random() % 257);
    const auto cell = 1 + static_cast<std::int64_t>(random() % 1000000);
    auto x0 = static_cast<std::int64_t>(random() % 2000000000) - 1000000000;
    if (test % 31 == 0)
      x0 = std::numeric_limits<std::int64_t>::min() + cell;
    if (test % 37 == 0)
      x0 = std::numeric_limits<std::int64_t>::max() - (static_cast<std::int64_t>(width) + 2) * cell;
    Crossings crossings(random() % 21);
    for (auto& values : crossings) {
      const auto count = random() % 21;
      for (std::uint64_t i = 0; i < count; ++i) {
        const auto x = static_cast<std::int64_t>(random() % static_cast<unsigned>(width));
        double value = static_cast<double>(x0 + x * cell + cell / 2);
        if (i % 3 == 1)
          value = std::nextafter(value, std::numeric_limits<double>::infinity());
        if (i % 3 == 2)
          value = std::nextafter(value, -std::numeric_limits<double>::infinity());
        values.push_back(value);
        if (i % 5 == 0)
          values.push_back(value);
      }
      std::sort(values.begin(), values.end());
    }
    std::vector<std::uint8_t> row(static_cast<std::size_t>(width));
    for (auto& value : row) {
      const auto choice = random() % 7;
      value = choice < 4 ? 255 : static_cast<std::uint8_t>(choice - 4);
    }
    auto expected = row;
    original_scan(crossings, x0, cell, expected);
    assert(classifier.classify(crossings, x0, cell, row.data(), width));
    assert(row == expected);
  }
}

void test_fallback() {
  tmk::route::RasterRows classifier;
  constexpr double infinity = std::numeric_limits<double>::infinity();
  const std::array<Crossings, 3> invalid{{{{std::numeric_limits<double>::quiet_NaN()}}, {{infinity}}, {{2, 1}}}};
  for (const auto& crossings : invalid) {
    std::vector<std::uint8_t> row(17, 255);
    const auto expected = row;
    check_false(classifier.classify(crossings, 0, 1, row.data(), 17));
    assert(row == expected);
  }
  std::vector<std::uint8_t> row(17, 255);
  const auto expected = row;
  check_false(classifier.classify({}, 0, 0, row.data(), 17));
  check_false(classifier.classify({}, 0, -1, row.data(), 17));
  check_false(classifier.classify({}, 0, 1, row.data(), 0));
  check_false(classifier.classify({}, std::numeric_limits<std::int64_t>::max(), 1, row.data(), 2));
  check_false(classifier.classify({}, 0, std::numeric_limits<std::int64_t>::max(), row.data(), 3));
  assert(row == expected);
}

int main() {
  test_boundaries();
  test_seeded_rows();
  test_fallback();
}
