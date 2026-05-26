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
#include <stdio.h>

#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/memory_manager_cuda.h"
#include "cerium/runtime/memory/pointer.h"

#define DATATYPE_TEMPLATE(T)                                                   \
  template <typename T,                                                        \
            typename = std::enable_if_t<                                       \
                std::is_same<std::remove_cv_t<T>, std::uint64_t>::value ||     \
                std::is_same<std::remove_cv_t<T>, std::uint32_t>::value>>

namespace Cerium {
namespace Runtime {

/** Wrapper Class around std::shared_ptr for Cerium. */
template <typename T> class DevicePointer {

public:
  DevicePointer() {}

  DevicePointer(const MemoryManagerCUDAHandle &memory_handle, const size_t n)
      : memory_handle_(memory_handle), n_(n) {
    if (memory_handle_ == nullptr) {
      *this = DevicePointer<T>(n);
      return;
    }
    ptr_.reset((T *)memory_handle_->allocate(sizeof(T) * n),
               [memory_handle](T *p) {});
  }

  DevicePointer(const size_t n) : n_(n), is_uvm_(false) {
    allocate_memory_on_device(n_);
  }

  DevicePointer(const size_t n, const bool is_uvm) : n_(n), is_uvm_(is_uvm) {
    if (is_uvm_) {
      allocate_memory_uvm(n_);
    } else {
      allocate_memory_on_device(n_);
    }
  }

  DevicePointer(DevicePointer<T> &&assign) noexcept {
    memory_handle_ = std::move(assign.memory_handle_);
    ptr_ = std::move(assign.ptr_);
    n_ = assign.n_;
    is_uvm_ = assign.is_uvm_;
  }

  inline auto &operator=(DevicePointer<T> &&assign) noexcept {
    memory_handle_ = std::move(assign.memory_handle_);
    ptr_ = std::move(assign.ptr_);
    n_ = assign.n_;
    is_uvm_ = assign.is_uvm_;
    return *this;
  }

  ~DevicePointer() {
    n_ = 0;
    ptr_ = nullptr;
    memory_handle_ = nullptr;
    is_uvm_ = false;
  }

  inline T *get() { return ptr_.get(); }

  inline const T *get() const { return ptr_.get(); }

  inline T *operator->() const noexcept { return ptr_.get(); }

  inline T &operator*() const { return *ptr_; }

  inline T &operator[](size_t index) {
    return ptr_.get()[index];
    // return ptr_[index];
  }

  inline const T &operator[](size_t index) const {
    return ptr_.get()[index];
    // return ptr_[index];
  }

  inline auto n() const { return n_; }

  // TODO: Fix These Functions
  void copy(const std::vector<T> &src, const std::size_t count) {
    if (is_uvm_) {
      CUDA::memcpyUVM((void *)(ptr_.get()), (void *)(&src[0]),
                      count * sizeof(T));
    } else {
      CUDA::memcpyHostToDevice((void *)(ptr_.get()), (void *)(&src[0]),
                               count * sizeof(T));
    }
    // CUDA::memcpyHostToDevice((void *)(ptr_.get()),(void *)(&src[0]),count*sizeof(T));
  }

  void copy(const T *src, const std::size_t count) {
    if (is_uvm_) {
      CUDA::memcpyUVM((void *)(ptr_.get()), (void *)(src), count * sizeof(T));
    } else {
      CUDA::memcpyHostToDevice((void *)(ptr_.get()), (void *)(src),
                               count * sizeof(T));
    }
  }

  Pointer<T> move_to_host() {
    Pointer<T> host_ptr(n_);
    if (is_uvm_) {
      CUDA::memcpyUVM((void *)host_ptr.get(), (void *)ptr_.get(),
                      sizeof(T) * n_);
    } else {
      CUDA::memcpyDeviceToHost((void *)host_ptr.get(), (void *)ptr_.get(),
                               sizeof(T) * n_);
    }
    return host_ptr;
  }

  explicit operator bool() const { return ptr_ != nullptr; }

private:
  MemoryManagerCUDAHandle memory_handle_{nullptr};
  size_t n_{0};
  std::shared_ptr<T> ptr_{nullptr};
  CUDA::StreamPtr stream_{nullptr};
  bool is_uvm_{false};
  DevicePointer(DevicePointer<T> &assign) = delete;
  inline auto &operator=(DevicePointer<T> &assign) = delete;

  void allocate_memory_on_device(const size_t n) {
    T *d_alloc = nullptr;
    CUDA::allocate_memory_on_device((void **)&d_alloc, n * sizeof(T));
    ptr_.reset(d_alloc, [](void *d_ptr) { CUDA::free(d_ptr); });
  }

  void allocate_memory_on_device_async(const size_t n) {
    T *d_alloc = nullptr;
    CUDA::allocate_memory_on_device_async((void **)&d_alloc, n * sizeof(T),
                                          stream_);
    ptr_.reset(d_alloc, [](void *d_ptr) { CUDA::free(d_ptr); });
  }

  void allocate_memory_uvm(const size_t n) {
    T *d_alloc = nullptr;
    CUDA::mallocManaged((void **)&d_alloc, n * sizeof(T));
    ptr_.reset(d_alloc, [](void *d_ptr) { CUDA::free(d_ptr); });
  }
};

template <
    typename T_,
    typename = std::enable_if_t<std::is_standard_layout<typename std::remove_cv<
        typename std::remove_reference<T_>::type>::type>::value>>
inline auto allocate_device(const size_t count, const bool is_uvm = false) {

  using T =
      typename std::remove_cv<typename std::remove_reference<T_>::type>::type;
  return DevicePointer<T>(count, is_uvm);
}
} // namespace Runtime
} // namespace Cerium