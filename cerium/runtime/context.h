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

#include <seal/memorymanager.h>
#include <seal/seal.h>
#include <seal/util/mempool.h>
#include <seal/util/rns.h>

#include <vector>

#include "math/galois.h"
#include "math/ntt.h"

namespace Cerium {
namespace Runtime {
constexpr std::uint64_t SLOTS = 32768;
class Context {
public:
  Context() = delete;

  Context(const Context &copy) = delete;

  Context(Context &&move) = default;

  Context &operator=(Context &&move) = default;

  Context(const std::vector<std::uint64_t> &rns_bases);

  inline std::size_t slots() const { return slots_; }

  inline std::size_t n() const { return n_; }

  inline std::size_t num_rns_bases() const { return num_rns_bases_; }

  inline const auto *ntt_tables() const { return ntt_tables_.get(); }

  inline const seal::util::NTTTables *seal_ntt_tables() const {
    return seal_ntt_tables_.get();
  }

  inline const std::vector<seal::Modulus> &rns_bases() const {
    return rns_bases_;
  }

  inline const seal::Modulus &get_rns_modulus(std::size_t rns_base_id) const {
    return rns_bases_.at(rns_base_id);
  }

  inline auto *galois_tool() const noexcept { return galois_tool_.get(); }

  std::uint32_t rns_base_bit_count(std::uint64_t num_bases) const {
    // TODO: Fix this...
    return num_bases * 28;
  }

private:
  std::size_t slots_;
  std::size_t n_;
  std::vector<seal::Modulus> rns_bases_;
  std::uint64_t num_rns_bases_;
  seal::MemoryPoolHandle pool_ = seal::MemoryManager::GetPool();
  seal::util::Pointer<seal::util::NTTTables> seal_ntt_tables_;
  seal::util::Pointer<Math::NTTTables<Limb::Element_t>> ntt_tables_;
  seal::util::Pointer<Math::GaloisTool> galois_tool_;
};
} // namespace Runtime
} // namespace Cerium