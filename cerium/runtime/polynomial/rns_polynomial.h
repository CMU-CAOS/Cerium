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

#include "cerium/runtime/polynomial/limb.h"

namespace Cerium {
namespace Runtime {
class RnsPolynomial {

public:
  RnsPolynomial() {}

  void clear() { limb_index_map_.clear(); }

  void write_limb(const std::shared_ptr<Limb> &limb,
                  std::uint64_t rns_base_id) {
    if (limb_index_map_.empty()) {
      limb_size_ = limb->size();
    } else if (limb_size_ != limb->size()) {
      throw std::logic_error(
          "RnsPolynomial: all limbs must have the same size");
    }
    limb_index_map_[rns_base_id] = limb;
  }

  const std::shared_ptr<Limb> &at(std::uint64_t rns_base_id) const {
    return limb_index_map_.at(rns_base_id);
  }

  const auto &limb_index_map() const { return limb_index_map_; }

  auto limb_size() const { return limb_size_; }

  auto num_limbs() const { return limb_index_map_.size(); }

  auto size() const { return num_limbs() * limb_size(); }

  auto empty() const { return limb_index_map_.empty(); }

  auto begin() { return limb_index_map_.begin(); }
  auto begin() const { return limb_index_map_.begin(); }
  auto end() { return limb_index_map_.end(); }
  auto end() const { return limb_index_map_.end(); }

private:
  std::map<std::uint64_t, std::shared_ptr<Limb>> limb_index_map_;
  std::size_t limb_size_ = 0;
};

using RnsPolynomialPtr = std::shared_ptr<RnsPolynomial>;

} // namespace Runtime
} // namespace Cerium