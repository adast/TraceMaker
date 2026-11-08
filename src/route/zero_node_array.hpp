// SPDX-License-Identifier: GPL-3.0-or-later
// Zero-tagged, implicit-lifetime search states. Invalid parents are not read.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace tmk::route {
template <class T> class ZeroNodeArray {
  static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>);
  static_assert(alignof(T) <= alignof(std::max_align_t));
  T* data_ = nullptr;
  std::size_t size_ = 0;

public:
  ZeroNodeArray() = default;

  ~ZeroNodeArray() {
    std::free(data_);
  }

  ZeroNodeArray(const ZeroNodeArray&) = delete;
  ZeroNodeArray& operator=(const ZeroNodeArray&) = delete;

  ZeroNodeArray(ZeroNodeArray&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

  ZeroNodeArray& operator=(ZeroNodeArray&& other) noexcept {
    if (this != &other) {
      std::free(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  void reset_zero(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    // Allocate before releasing the old state to preserve exception safety.
    T* next = count ? static_cast<T*>(std::calloc(count, sizeof(T))) : nullptr;
    if (count && !next)
      throw std::bad_alloc();
    std::free(data_);
    data_ = next;
    size_ = count;
  }

  std::size_t size() const {
    return size_;
  }

  T& operator[](std::size_t index) {
    return data_[index];
  }

  const T& operator[](std::size_t index) const {
    return data_[index];
  }

  T* begin() {
    return data_;
  }

  T* end() {
    return size_ ? data_ + size_ : data_;
  }

  const T* begin() const {
    return data_;
  }

  const T* end() const {
    return size_ ? data_ + size_ : data_;
  }
};
}  // namespace tmk::route
