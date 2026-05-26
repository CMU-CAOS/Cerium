// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "memory_manager.h"

namespace Cerium {
namespace Runtime {

/** Wrapper Class around std::shared_ptr for Cerium. */
template <typename T> class Pointer {
public:
  Pointer() {}

  Pointer(const MemoryManagerHandle &memory_handle, const size_t n)
      : memory_handle_(memory_handle), n_(n) {
    if (memory_handle_ == nullptr) {
      *this = Pointer<T>(n);
      return;
    }
    ptr_.reset((T *)memory_handle_->allocate(sizeof(T) * n),
               [memory_handle](T *p) {});
  }

  Pointer(const size_t n) : n_(n) {
    ptr_.reset(new T[n], [](T *p) { delete[] p; });
  }

  Pointer(Pointer<T> &&assign) noexcept : n_(assign.n_) {
    memory_handle_ = std::move(assign.memory_handle_);
    ptr_ = std::move(assign.ptr_);
  }

  inline auto &operator=(Pointer<T> &&assign) noexcept {
    memory_handle_ = std::move(assign.memory_handle_);
    ptr_ = std::move(assign.ptr_);
    return *this;
  }

  ~Pointer() {
    n_ = 0;
    ptr_ = nullptr;
    memory_handle_ = nullptr;
  }

  inline T *get() { return ptr_.get(); }

  inline T *get() const { return ptr_.get(); }

  inline T *operator->() const noexcept { return ptr_.get(); }

  inline T &operator*() const { return *ptr_; }

  inline T &operator[](size_t index) { return ptr_.get()[index]; }

  inline const T &operator[](size_t index) const { return ptr_.get()[index]; }

  inline auto n() const { return n_; }

  void copy(const std::vector<T> &src, const std::size_t count) {
    for (size_t i = 0; i < count; i++) {
      ptr_[i] = src[i];
    }
  }

  Pointer<T> move_to_host() { return std::move(*this); }

private:
  MemoryManagerHandle memory_handle_{nullptr};
  std::shared_ptr<T> ptr_;
  std::size_t n_;
  Pointer(Pointer<T> &assign) = delete;
  inline auto &operator=(Pointer<T> &assign) = delete;
};

template <
    typename T_,
    typename = std::enable_if_t<std::is_standard_layout<typename std::remove_cv<
        typename std::remove_reference<T_>::type>::type>::value>>
inline auto allocate(const size_t count) {
  using T =
      typename std::remove_cv<typename std::remove_reference<T_>::type>::type;
  return Pointer<T>(count);
}

} // namespace Runtime
} // namespace Cerium