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

#include <seal/seal.h>
#include <seal/util/dwthandler.h>
#include <seal/util/mempool.h>
#include <seal/util/rns.h>

#include "util.h"

namespace seal {
namespace util {

template <>
class Arithmetic<std::uint32_t, seal::util::MultiplyUIntModOperand,
                 seal::util::MultiplyUIntModOperand> {
public:
  Arithmetic() {}

  Arithmetic(const seal::Modulus &modulus)
      : modulus_(modulus),
        two_times_modulus_(safe_cast<std::uint32_t>(modulus.value() << 1)) {}

  inline std::uint32_t add(const std::uint32_t &a,
                           const std::uint32_t &b) const {
    return a + b;
  }

  inline std::uint32_t sub(const std::uint32_t &a,
                           const std::uint32_t &b) const {
    return a + two_times_modulus_ - b;
  }

  inline std::uint32_t mul_root(const std::uint32_t &a,
                                const MultiplyUIntModOperand &r) const {
    std::uint64_t a64 = static_cast<uint64_t>(a);
    return Cerium::Runtime::Math::safe_cast<std::uint32_t>(
        multiply_uint_mod_lazy(a64, r, modulus_));
  }

  inline std::uint32_t mul_scalar(const std::uint64_t &a,
                                  const MultiplyUIntModOperand &s) const {
    std::uint64_t a64 = static_cast<uint64_t>(a);
    return Cerium::Runtime::Math::safe_cast<std::uint32_t>(
        multiply_uint_mod_lazy(a64, s, modulus_));
  }

  inline MultiplyUIntModOperand
  mul_root_scalar(const MultiplyUIntModOperand &r,
                  const MultiplyUIntModOperand &s) const {
    MultiplyUIntModOperand result;
    result.set(multiply_uint_mod(r.operand, s, modulus_), modulus_);
    return result;
  }

  inline std::uint32_t guard(const std::uint32_t &a) const {
    return SEAL_COND_SELECT(a >= two_times_modulus_, a - two_times_modulus_, a);
  }

private:
  seal::Modulus modulus_;
  std::uint32_t two_times_modulus_;
};
} // namespace util

} // namespace seal