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
#include <seal/util/rns.h>

#include "cerium/runtime/memory/pointer.h"
#include "cerium/runtime/polynomial/limb.h"

#define DATATYPE_TEMPLATE(T)                                                   \
  template <typename T,                                                        \
            typename = std::enable_if_t<                                       \
                std::is_same<std::remove_cv_t<T>, std::uint64_t>::value ||     \
                std::is_same<std::remove_cv_t<T>, std::uint32_t>::value>>

#define DATATYPE_TEMPLATE2(T) template <typename T, typename>

namespace Cerium {
namespace Runtime {

/** Wrapper Class around seal::util for Cerium.
     * Performs appropriate typecasting
     */
namespace Math {

template <typename T, typename S,
          typename = std::enable_if_t<std::is_arithmetic<T>::value>,
          typename = std::enable_if_t<std::is_arithmetic<S>::value>>
inline T safe_cast(S value) {
#ifdef CERIUM_RUNTIME_DEBUG
  return seal::util::safe_cast<T>(value);
#else
  return static_cast<T>(value);
#endif
}

/**
        Returns ((operand1*operand2) + operand3) mod modulus
        */
DATATYPE_TEMPLATE(T)
inline T multiply_add_uint_mod(const T &operand1, const T &operand2,
                               const T &operand3,
                               const seal::Modulus &modulus) {
  if constexpr (std::is_same<T, std::uint64_t>::value) {
    return seal::util::multiply_add_uint_mod(operand1, operand2, operand3,
                                             modulus);
  } else {
    std::uint64_t operand1_64 = static_cast<uint64_t>(operand1);
    std::uint64_t operand2_64 = static_cast<uint64_t>(operand2);
    std::uint64_t operand3_64 = static_cast<uint64_t>(operand3);
    auto result = seal::util::multiply_add_uint_mod(operand1_64, operand2_64,
                                                    operand3_64, modulus);
    return Math::safe_cast<T>(result);
  }
}

/**
        Returns (operand1+operand2) mod modulus
        */
DATATYPE_TEMPLATE(T)
inline T add_uint_mod(T operand1, T operand2, const seal::Modulus &modulus) {
  operand1 += operand2;
  auto modulus_value = Math::safe_cast<T>(modulus.value());
  return operand1 >= modulus_value ? operand1 - modulus_value : operand1;
}

/**
        Returns (operand1-operand2) mod modulus
        */
DATATYPE_TEMPLATE(T)
inline T subtract_uint_mod(const T &operand1, const T &operand2,
                           const seal::Modulus &modulus) {
  if constexpr (std::is_same<T, std::uint64_t>::value) {
    return seal::util::sub_uint_mod(operand1, operand2, modulus);
  } else {
    std::uint64_t operand1_64 = static_cast<uint64_t>(operand1);
    std::uint64_t operand2_64 = static_cast<uint64_t>(operand2);
    auto result = seal::util::sub_uint_mod(operand1_64, operand2_64, modulus);
    return Math::safe_cast<T>(result);
  }
}

/**
        Returns (operand1*operand2) mod modulus
        */
DATATYPE_TEMPLATE(T)
inline T multiply_uint_mod(T operand1, T operand2,
                           const seal::Modulus &modulus) {
  if constexpr (std::is_same<T, std::uint64_t>::value) {
    return seal::util::multiply_uint_mod(operand1, operand2, modulus);
  } else if constexpr (std::is_same<T, std::uint32_t>::value) {
    std::uint64_t operand1_64 = static_cast<uint64_t>(operand1);
    std::uint64_t operand2_64 = static_cast<uint64_t>(operand2);
    std::uint64_t product = operand1_64 * operand2_64;
    auto result = seal::util::barrett_reduce_64(product, modulus);
    return Math::safe_cast<T>(result);
  }
}

/**
        Returns (-1*operand1) mod modulus
        */
DATATYPE_TEMPLATE(T)
inline T negate_uint_mod(T operand, const seal::Modulus &modulus) {
  if constexpr (std::is_same<T, std::uint64_t>::value) {
    return seal::util::negate_uint_mod(operand, modulus);
  } else if constexpr (std::is_same<T, std::uint32_t>::value) {
    auto modulus32 = Math::safe_cast<std::uint32_t>(modulus.value());
    std::int32_t non_zero = static_cast<std::int32_t>(operand != 0);
    return (modulus32 - operand) & static_cast<std::uint32_t>(-non_zero);
  }
}

DATATYPE_TEMPLATE(T)
inline auto set_uint(const T *value, const size_t count, T *result) {
  if (result == value) {
    return;
  }
  std::memcpy(result, value, count * sizeof(T));
}

inline std::uint32_t Shoup(std::uint32_t in, std::uint32_t prime) {
  std::uint64_t temp = static_cast<std::uint64_t>(in) << 32;
  return static_cast<std::uint32_t>(temp / prime);
}

} // namespace Math
} // namespace Runtime
} // namespace Cerium