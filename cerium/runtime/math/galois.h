// Copyright contributors to the SEAL project
// Licensed under the MIT License.
// Original source: https://github.com/microsoft/SEAL
//
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

#include "util.h"

namespace Cerium {
namespace Runtime {
namespace Math {
class GaloisTool {
public:
  GaloisTool(int coeff_count_power, seal::MemoryPoolHandle pool)
      : pool_(std::move(pool)) {
    if (!pool_) {
      throw std::invalid_argument("pool is uninitialized");
    }

    initialize(coeff_count_power);
  }

  DATATYPE_TEMPLATE(T)
  void apply_galois_ntt(const T *operand, std::size_t size,
                        std::uint32_t galois_elt, T *result) const {
#ifdef CERIUM_RUNTIME_DEBUG
    if (!operand) {
      throw std::invalid_argument("operand");
    }
    if (!result) {
      throw std::invalid_argument("result");
    }
    if (operand == result) {
      throw std::invalid_argument(
          "result cannot point to the same value as operand");
    }
    // Verify coprime conditions.
    if (!(galois_elt & 1) ||
        (galois_elt >= 2 * (uint64_t(1) << coeff_count_power_))) {
      throw std::invalid_argument("Galois element is not valid");
    }
#endif
    generate_table_ntt(galois_elt,
                       permutation_tables_[GetIndexFromElt(galois_elt)]);
    auto table = iter(permutation_tables_[GetIndexFromElt(galois_elt)]);

    // Perform permutation.
    SEAL_ITERATE(iter(table, result), coeff_count_,
                 [&](auto I) { std::get<1>(I) = operand[std::get<0>(I)]; });
  }

  auto get_galois_table(std::uint32_t galois_elt) {
    generate_table_ntt(galois_elt,
                       permutation_tables_[GetIndexFromElt(galois_elt)]);
    return iter(permutation_tables_[GetIndexFromElt(galois_elt)]);
  }

  auto get_galois_table_inverse(std::uint32_t galois_elt) {
    generate_table_ntt_inverse(
        galois_elt, permutation_tables_inverse_[GetIndexFromElt(galois_elt)]);
    return iter(permutation_tables_inverse_[GetIndexFromElt(galois_elt)]);
  }

  /**
            Compute the Galois element corresponding to a given rotation step.
            */
  SEAL_NODISCARD std::uint32_t get_elt_from_step(int step) const;

  /**
            Compute the index in the range of 0 to (coeff_count_ - 1) of a given Galois element.
            */
  SEAL_NODISCARD static inline std::size_t
  GetIndexFromElt(std::uint32_t galois_elt) {
#ifdef CERIUM_RUNTIME_DEBUG
    if (!(galois_elt & 1)) {
      throw std::invalid_argument("galois_elt is not valid");
    }
#endif
    return seal::util::safe_cast<std::size_t>((galois_elt - 1) >> 1);
  }

private:
  GaloisTool(const GaloisTool &copy) = delete;

  GaloisTool(GaloisTool &&source) = delete;

  GaloisTool &operator=(const GaloisTool &assign) = delete;

  GaloisTool &operator=(GaloisTool &&assign) = delete;

  void initialize(int coeff_count_power);

  void generate_table_ntt(std::uint32_t galois_elt,
                          seal::util::Pointer<std::uint32_t> &result,
                          bool inverse = false) const;
  void generate_table_ntt_inverse(std::uint32_t galois_elt,
                                  seal::util::Pointer<std::uint32_t> &result) {
    return generate_table_ntt(galois_elt, result, true);
  }

  seal::MemoryPoolHandle pool_;

  int coeff_count_power_ = 0;

  std::size_t coeff_count_ = 0;

  static constexpr std::uint32_t generator_ = 5;

  mutable seal::util::Pointer<seal::util::Pointer<std::uint32_t>>
      permutation_tables_;
  mutable seal::util::Pointer<seal::util::Pointer<std::uint32_t>>
      permutation_tables_inverse_;

  mutable seal::util::ReaderWriterLocker permutation_tables_locker_;
};
} // namespace Math
} // namespace Runtime
} // namespace Cerium
