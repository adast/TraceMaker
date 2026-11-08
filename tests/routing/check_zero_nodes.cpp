// SPDX-License-Identifier: GPL-3.0-or-later
#include "route/zero_node_array.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

struct Node {
  std::int64_t g;
  std::uint32_t tag;
  std::int32_t parent;
};

template <bool zero> struct Buffers {
  std::conditional_t<zero, tmk::route::ZeroNodeArray<Node>, std::vector<Node>> sn;
  std::vector<std::uint32_t> cstamp, vstamp;
  std::vector<std::int64_t> cell, via;

  void grow(std::size_t cells, std::size_t layers, std::size_t directions, bool /*candidate*/) {
    const auto states = cells * layers * directions;
    if (sn.size() < states) {
      if constexpr (zero)
        sn.reset_zero(states);
      else
        sn.assign(states, Node{0, 0, -1});
    }
    if (cstamp.size() < cells * layers) {
      cstamp.assign(cells * layers, 0);
      cell.resize(cells * layers);
    }
    if (vstamp.size() < cells) {
      vstamp.assign(cells, 0);
      via.resize(cells);
    }
  }

  void wrap() {
    std::fill(sn.begin(), sn.end(), Node{0, 0, -1});
    std::fill(cstamp.begin(), cstamp.end(), 0);
    std::fill(vstamp.begin(), vstamp.end(), 0);
  }
};

static std::uint64_t checks = 0;

static void require(bool yes) {
  ++checks;
  if (!yes)
    throw std::runtime_error("Zero-tagged allocation changes live state");
}

int main() {
  std::mt19937_64 random(0xe4192ac77);
  std::uint64_t wraps = 0, parents = 0;
  for (const std::uint32_t mask : {31u, 0x0fffffffu}) {
    Buffers<false> a;
    Buffers<true> b;
    std::uint32_t gen = mask == 31 ? 0 : mask - 2;
    for (unsigned search = 0; search < 3200; ++search) {
      const std::size_t cells = 1 + random() % 193, layers = 1 + random() % 4, directions = (search % 3) ? 9 : 1,
                        states = cells * layers * directions;
      a.grow(cells, layers, directions, false);
      b.grow(cells, layers, directions, true);
      require(a.sn.size() == b.sn.size() && a.cstamp.size() == b.cstamp.size() && a.vstamp.size() == b.vstamp.size());
      if (++gen > mask) {
        a.wrap();
        b.wrap();
        gen = 1;
        ++wraps;
      }
      std::vector<std::size_t> touched;
      for (unsigned query = 0; query < 240; ++query) {
        const auto s = random() % states;
        const bool av = (a.sn[s].tag & mask) == gen, bv = (b.sn[s].tag & mask) == gen;
        require(av == bv);
        const auto threshold = static_cast<std::int64_t>(random());
        require((av && a.sn[s].g <= threshold) == (bv && b.sn[s].g <= threshold));
        require((!av || a.sn[s].g > threshold) == (!bv || b.sn[s].g > threshold));
        if (av)
          require(a.sn[s].g == b.sn[s].g && a.sn[s].tag == b.sn[s].tag && a.sn[s].parent == b.sn[s].parent);
        if (!av || random() % 3 == 0) {
          const auto parent = touched.empty() ? -1 : static_cast<std::int32_t>(touched[random() % touched.size()]);
          const auto dir = static_cast<std::uint32_t>(random() % 9);
          const Node value{threshold, gen | (dir << 28), parent};
          a.sn[s] = value;
          b.sn[s] = value;
          touched.push_back(s);
        }
        // Validate paths through current-generation nodes. Repeated updates can
        // create cycles in this synthetic stream, so traversal is bounded.
        for (std::int32_t p = static_cast<std::int32_t>(s), depth = 0; p >= 0 && depth < 16; ++depth) {
          const auto index = static_cast<std::size_t>(p);
          require((a.sn[index].tag & mask) == gen && (b.sn[index].tag & mask) == gen);
          require(a.sn[index].g == b.sn[index].g && a.sn[index].parent == b.sn[index].parent &&
                  a.sn[index].tag == b.sn[index].tag);
          p = a.sn[index].parent;
          ++parents;
        }
        const auto ci = random() % (cells * layers), vi = random() % cells;
        const bool ac = a.cstamp[ci] == gen, bc = b.cstamp[ci] == gen, avc = a.vstamp[vi] == gen,
                   bvc = b.vstamp[vi] == gen;
        require(ac == bc && avc == bvc);
        if (ac)
          require(a.cell[ci] == b.cell[ci]);
        else {
          a.cstamp[ci] = b.cstamp[ci] = gen;
          a.cell[ci] = b.cell[ci] = threshold;
        }
        if (avc)
          require(a.via[vi] == b.via[vi]);
        else {
          a.vstamp[vi] = b.vstamp[vi] = gen;
          a.via[vi] = b.via[vi] = threshold;
        }
      }
    }
  }
  tmk::route::ZeroNodeArray<Node> first;
  require(first.size() == 0 && first.begin() == first.end());
  first.reset_zero(17);
  for (std::size_t i = 0; i < first.size(); ++i)
    require(first[i].tag == 0 && first[i].g == 0 && first[i].parent == 0);
  first[3] = {91, 1, -1};
  tmk::route::ZeroNodeArray<Node> moved(std::move(first));
  require(first.size() == 0 && moved.size() == 17 && moved[3].g == 91 && moved[3].parent == -1);
  first = std::move(moved);
  require(moved.size() == 0 && first.size() == 17 && first[3].tag == 1);
  try {
    first.reset_zero(std::numeric_limits<std::size_t>::max());
    require(false);
  } catch (const std::bad_array_new_length&) {
    require(first.size() == 17 && first[3].g == 91);
  }
  first.reset_zero(0);
  require(first.size() == 0 && first.begin() == first.end());
  std::printf("{\"all_exact\":true,\"checks\":%llu,\"wraps\":%llu,\"parent_steps\":%llu,\"searches\":6400}\n",
              (unsigned long long)checks, (unsigned long long)wraps, (unsigned long long)parents);
}
