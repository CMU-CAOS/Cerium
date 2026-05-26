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

#include <cstdint>
#include <vector_types.h>
#include "cerium/runtime/memory/pointer.h"


class Context;

namespace Cerium {
namespace Runtime {
template <template <typename> typename T> class LimbBase {

public:
  enum Form { NTT, COEF };
  using Element_t = std::uint32_t;
  using Element_t2 = uint2;

  LimbBase() = delete;
  LimbBase(const LimbBase &limb) = delete;
  LimbBase(LimbBase &&limb) = default;
  LimbBase(const std::vector<Element_t> &values, std::uint64_t rns_base_id,
           bool is_ntt_form)
      : rns_base_id_(rns_base_id) {
    size_ = values.size();
    data_ = T<Element_t>(size_);
    data_.copy(values, size_);

    if (is_ntt_form) {
      form_ = Form::NTT;
    } else {
      form_ = Form::COEF;
    }
  }
  LimbBase(T<Element_t> &&data, std::size_t size, std::uint64_t rns_base_id,
           bool is_ntt_form)
      : rns_base_id_(rns_base_id), data_(std::move(data)), size_(size) {
    if (is_ntt_form) {
      form_ = Form::NTT;
    } else {
      form_ = Form::COEF;
    }
  }

  std::uint64_t rns_base_id() const { return rns_base_id_; }

  bool is_ntt_form() const { return form_ == Form::NTT; }

  std::size_t size() const { return size_; }

  std::size_t size() { return size_; }

  const Element_t *data() const { return data_.get(); }

  Element_t *data() { return data_.get(); }

  LimbBase<Pointer> move_to_host() {
    Pointer host_pointer = data_.move_to_host();
    return LimbBase<Pointer>(std::move(host_pointer), size_, rns_base_id_,
                             form_ == Form::NTT);
  }

  std::shared_ptr<LimbBase<Pointer>> move_to_host_ptr() {
    Pointer host_pointer = data_.move_to_host();
    return std::make_shared<LimbBase<Pointer>>(
        std::move(host_pointer), size_, rns_base_id_, form_ == Form::NTT);
  }

private:
  std::uint64_t rns_base_id_;
  Form form_;
  T<Element_t> data_;
  std::size_t size_;
};

using Limb = LimbBase<Pointer>;
using LimbPtr = std::shared_ptr<Limb>;

inline auto allocate_limb_elts(const size_t count) {
  return allocate<Limb::Element_t>(count);
}

} // namespace Runtime
} // namespace Cerium
