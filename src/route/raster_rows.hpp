// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace tmk::route {

// The outer loop adds coverage; cutouts subtract it. Positive coverage is
// inside the outline and outside every cutout, including overlapping cutouts.
class RasterRows {
public:
  bool classify(const std::vector<std::vector<double>>& crossings, std::int64_t x0, std::int64_t cell,
                std::uint8_t* row, int width) {
    if (!row || !coordinates_valid(x0, cell, width) ||
        crossings.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - 1))
      return false;
    for (const auto& values : crossings) {
      for (double value : values)
        if (!std::isfinite(value))
          return false;
      if (!std::is_sorted(values.begin(), values.end()))
        return false;
    }
    // Failure leaves output untouched, so the caller can use its original scan.
    try {
      difference_.assign(static_cast<std::size_t>(width) + 1, 0);
    } catch (const std::bad_alloc&) {
      return false;
    }
    auto interval = [&](int begin, int end, std::int64_t sign) {
      if (begin < end) {
        difference_[static_cast<std::size_t>(begin)] += sign;
        difference_[static_cast<std::size_t>(end)] -= sign;
      }
    };
    if (crossings.empty())
      interval(0, width, 1);
    for (std::size_t loop = 0; loop < crossings.size(); ++loop) {
      const auto& values = crossings[loop];
      const std::int64_t sign = loop == 0 ? 1 : -1;
      bool inside = (values.size() & 1u) != 0;
      int begin = 0;
      for (double crossing : values) {
        int lo = begin, hi = width;
        // Consume crossings <= the exact same centre as the original scan.
        // Avoid rounding a floating crossing to an integer column index.
        while (lo < hi) {
          const int mid = lo + (hi - lo) / 2;
          const double centre = static_cast<double>(x0 + static_cast<std::int64_t>(mid) * cell + cell / 2);
          if (crossing <= centre)
            hi = mid;
          else
            lo = mid + 1;
        }
        if (inside)
          interval(begin, lo, sign);
        begin = lo;
        inside = !inside;
      }
      if (inside)
        interval(begin, width, sign);
    }
    std::int64_t coverage = 0;
    for (int x = 0; x < width; ++x) {
      coverage += difference_[static_cast<std::size_t>(x)];
      if (row[x] == 255)
        row[x] = coverage > 0 ? 1 : 0;
    }
    return true;
  }

private:
  static bool coordinates_valid(std::int64_t x0, std::int64_t cell, int width) {
    if (cell <= 0 || width <= 0)
      return false;
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    const auto span = static_cast<std::int64_t>(width - 1);
    if (span && cell > maximum / span)
      return false;
    const auto distance = span * cell;
    if (x0 > maximum - distance)
      return false;
    return x0 + distance <= maximum - cell / 2;
  }

  std::vector<std::int64_t> difference_;
};

}  // namespace tmk::route
