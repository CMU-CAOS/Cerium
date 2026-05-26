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

#include <seal/memorymanager.h>
#include <seal/seal.h>
#include <seal/util/mempool.h>
#include <seal/util/rns.h>
#include <seal/util/uintcore.h>

#include "cerium/runtime/context.h"
#include "cerium/runtime/math/util.h"
#include "cerium/runtime/polynomial/limb.h"
#include "cerium/runtime/polynomial/rns_polynomial.h"
#include <cassert>

namespace Cerium {
namespace Runtime {

class CKKSEncoder {

  using ComplexArith = seal::util::Arithmetic<std::complex<double>,
                                              std::complex<double>, double>;
  using FFTHandler = seal::util::DWTHandler<std::complex<double>,
                                            std::complex<double>, double>;

public:
  CKKSEncoder(const Context &context);
  CKKSEncoder(const Context &context, const seal::prng_seed_type &prng_seed);

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::shared_ptr<RnsPolynomial>
  encode(const std::vector<T> &values, const double scale,
         const std::vector<std::uint32_t> &rns_base_ids,
         const bool perform_ntt = true,
         seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    return encode_internal(values.data(), values.size(), scale, rns_base_ids,
                           slots_, perform_ntt, std::move(pool));
  }

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::shared_ptr<RnsPolynomial>
  encode(const std::vector<T> &values, const double scale,
         const std::vector<std::uint32_t> &rns_base_ids, const size_t elt_size,
         const bool perform_ntt = true,
         seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    return encode_internal(values.data(), values.size(), scale, rns_base_ids,
                           elt_size, perform_ntt, std::move(pool));
  }

  inline std::shared_ptr<RnsPolynomial>
  encode(const double value, const double scale,
         const std::vector<std::uint32_t> &rns_base_ids,
         seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    return encode_internal(value, scale, rns_base_ids, std::move(pool));
  }

  inline std::map<uint32_t, Limb::Element_t>
  encode_scalar(const double value, const double scale,
                const std::vector<std::uint32_t> &rns_base_ids,
                seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    return encode_internal_scalar(value, scale, rns_base_ids, std::move(pool));
  }

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::vector<T>
  decode(const std::shared_ptr<RnsPolynomial> &plain, const double scale,
         seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    std::vector<T> destination;
    destination.resize(slots_);
    decode_internal(plain, scale, destination.data(), std::move(pool));
    return std::move(destination);
  }

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::vector<double> encode_as_polynomial(
      const std::vector<T> &values, const double scale,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    return encode_as_polynomial_internal(values.data(), values.size(), scale,
                                         context_.slots(), std::move(pool));
  }

private:
  const Context &context_;

  seal::MemoryPoolHandle pool_;

  std::size_t slots_;

  std::shared_ptr<seal::util::ComplexRoots> complex_roots_;

  // Holds 1~(n-1)-th powers of root in bit-reversed order, the 0-th power is left unset.
  seal::util::Pointer<std::complex<double>> root_powers_;

  // Holds 1~(n-1)-th powers of inverse root in scrambled order, the 0-th power is left unset.
  seal::util::Pointer<std::complex<double>> inv_root_powers_;

  seal::util::Pointer<std::size_t> matrix_reps_index_map_;
  seal::util::Pointer<std::size_t> matrix_reps_index_map_inverse_;

  ComplexArith complex_arith_;

  FFTHandler fft_handler_;

  bool is_seeded_;
  seal::prng_seed_type prng_seed_;
  std::shared_ptr<seal::UniformRandomGenerator> prng_;

  void initialize();
  void initialize_matrix_reps_index_map();

  inline double coordinate_wise_random_rounding(const double coeff) {
    double coeff_floor = std::floor(coeff);
    double fractional_part = coeff - std::floor(coeff);
    double probability_round_up = fractional_part;

    uint32_t random_int;
    uint32_t MAX_UINT_32 = 0xffffffff;
    prng_->generate(sizeof(random_int),
                    reinterpret_cast<seal::seal_byte *>(&random_int));
    double random_sample = double(random_int) / double(MAX_UINT_32);
    double rounded_value;
    if (random_sample < probability_round_up) {
      rounded_value = coeff_floor + double(1.0);
    } else {
      rounded_value = coeff_floor;
    }
    return rounded_value;
  };

  std::shared_ptr<RnsPolynomial>
  encode_internal(double value, double scale,
                  const std::vector<std::uint32_t> &rns_base_ids,
                  seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

  std::map<uint32_t, Limb::Element_t> encode_internal_scalar(
      double value, double scale,
      const std::vector<std::uint32_t> &rns_base_ids,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>

  std::shared_ptr<RnsPolynomial> encode_internal(

      const T *values_, std::size_t values_size, const double scale,
      const std::vector<std::uint32_t> &rns_base_ids, size_t elt_size,
      const bool perform_ntt = true,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    // Verify parameters.
    if (!values_ && values_size > 0) {
      throw std::invalid_argument("values cannot be null");
    }
    if (values_size != context_.slots() && values_size != elt_size) {
      throw std::invalid_argument(
          "values_size must be equal to context slots or elt_size");
    }
    if (!pool) {
      throw std::invalid_argument("pool is uninitialized");
    }
    if (rns_base_ids.empty()) {
      throw std::invalid_argument("rns_base_ids is empty");
    }
    if (elt_size > context_.slots()) {
      throw std::invalid_argument(
          "elt_size must be less than or equal to context slots");
    }
    if ((elt_size & (elt_size - 1)) != 0) {
      throw std::invalid_argument("elt_size must be a power of two");
    }
    if (elt_size == 0) {
      throw std::invalid_argument("elt_sizemust be greater than 0");
    }

    std::size_t num_limbs = rns_base_ids.size();
    std::size_t coeff_count = context_.n();

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >=
                       context_.rns_base_bit_count(num_limbs))) {
      throw std::invalid_argument("scale out of bounds");
    }

    auto polynomial = encode_as_polynomial_internal(values_, values_size, scale,
                                                    elt_size, pool);

    std::size_t repeat_count = context_.slots() / elt_size;
    uint32_t repeat_count_log = std::log2(repeat_count);
    assert((1 << repeat_count_log) == repeat_count);

    auto ntt_tables = context_.ntt_tables();

    // values_size is guaranteed to be no bigger than slots_
    auto slots_ = context_.slots();
    auto n_ = coeff_count / repeat_count;

    double max_coeff = 0;

    for (std::size_t i = 0; i < n_; i++) {
      max_coeff = std::max<>(max_coeff, std::fabs(polynomial[i]));
    }
    // Verify that the values are not too large to fit in coeff_modulus
    // Note that we have an extra + 1 for the sign bit
    // Don't compute logarithmis of numbers less than 1
    int max_coeff_bit_count =
        static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
    if (max_coeff_bit_count >= context_.rns_base_bit_count(num_limbs)) {
      throw std::invalid_argument("encoded values are too large");
    }

    double two_pow_64 = std::pow(2.0, 64);

    std::vector<seal::Modulus> rns_modulii;
    rns_modulii.reserve(num_limbs);
    for (std::size_t i = 0; i < num_limbs; i++) {
      rns_modulii.push_back(context_.get_rns_modulus(rns_base_ids.at(i)));
    }

    auto destination_data = allocate<Pointer<Limb::Element_t>>(num_limbs);
    std::for_each_n(seal::util::iter(destination_data.get()), num_limbs,
                    [&](auto &I) { I = allocate_limb_elts(n_); });

    // Use faster decomposition methods when possible
    if (max_coeff_bit_count <= 64) {
      for (std::size_t i = 0; i < n_; i++) {
        double coeffd = polynomial[i];
        bool is_negative = std::signbit(coeffd);

        std::uint64_t coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

        if (is_negative) {
          for (std::size_t j = 0; j < num_limbs; j++) {
            auto temp = seal::util::negate_uint_mod(
                seal::util::barrett_reduce_64(coeffu, rns_modulii.at(j)),
                rns_modulii.at(j));
            destination_data[j][i] = Math::safe_cast<Limb::Element_t>(temp);
          }
        } else {
          for (std::size_t j = 0; j < num_limbs; j++) {
            auto temp = seal::util::barrett_reduce_64(coeffu, rns_modulii[j]);
            destination_data[j][i] = Math::safe_cast<Limb::Element_t>(temp);
          }
        }
      }
    } else if (max_coeff_bit_count <= 128) {
      for (std::size_t i = 0; i < n_; i++) {
        double coeffd = polynomial[i];
        bool is_negative = std::signbit(coeffd);
        coeffd = std::fabs(coeffd);

        std::uint64_t coeffu[2]{
            static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64)),
            static_cast<std::uint64_t>(coeffd / two_pow_64)};

        if (is_negative) {
          for (std::size_t j = 0; j < num_limbs; j++) {
            auto temp = seal::util::negate_uint_mod(
                seal::util::barrett_reduce_128(coeffu, rns_modulii.at(j)),
                rns_modulii.at(j));
            destination_data[j][i] = Math::safe_cast<Limb::Element_t>(temp);
          }
        } else {
          for (std::size_t j = 0; j < num_limbs; j++) {
            auto temp =
                seal::util::barrett_reduce_128(coeffu, rns_modulii.at(j));
            destination_data[j][i] = Math::safe_cast<Limb::Element_t>(temp);
          }
        }
      }
    } else {
      throw std::logic_error("unimplemented");
#if 0
                // TODO: Fit this
                // Slow case
                auto coeffu(seal::util::allocate_uint(num_limbs, pool));
                for (std::size_t i = 0; i < n_; i++)
                {
                    double coeffd = polynomial[i];
                    bool is_negative = std::signbit(coeffd);
                    coeffd = std::fabs(coeffd);

                    // We are at this point guaranteed to fit in the allocated space
                    seal::util::set_zero_uint(num_limbs, coeffu.get());
                    auto coeffu_ptr = coeffu.get();
                    while (coeffd >= 1)
                    {
                        *coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
                        coeffd /= two_pow_64;
                    }

                    // Next decompose this coefficient
                    context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);

                    // Finally replace the sign if necessary
                    if (is_negative)
                    {
                        for (std::size_t j = 0; j < coeff_modulus_size; j++)
                        {
                            destination[i + (j * coeff_count)] = util::negate_uint_mod(coeffu[j], coeff_modulus[j]);
                        }
                    }
                    else
                    {
                        for (std::size_t j = 0; j < coeff_modulus_size; j++)
                        {
                            destination[i + (j * coeff_count)] = coeffu[j];
                        }
                    }
                }
#endif
    }

    auto destination = std::make_shared<RnsPolynomial>();
    size_t log2_coeff_count_by_repeat_count =
        (size_t)std::log2(coeff_count / repeat_count);
    assert((1 << log2_coeff_count_by_repeat_count) ==
           (coeff_count / repeat_count));
    for (std::size_t i = 0; i < num_limbs; i++) {
      if (perform_ntt) {
        // Transform to NTT domain
        Math::ntt_negacyclic_harvey(destination_data[i].get(),
                                    ntt_tables[rns_base_ids.at(i)],
                                    log2_coeff_count_by_repeat_count);
      }
      std::shared_ptr<Limb> dest;
      dest = std::make_shared<Limb>(std::move(destination_data[i]),
                                    coeff_count / repeat_count,
                                    rns_base_ids.at(i), perform_ntt);
      destination->write_limb(std::move(dest), rns_base_ids.at(i));
    }

    return destination;
  }
  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>

  inline std::vector<double> encode_as_polynomial_internal(

      const T *values_, std::size_t values_size, const double scale,
      size_t elt_size,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    // Verify parameters.
    if (!values_ && values_size > 0) {
      throw std::invalid_argument("values cannot be null");
    }
    if (values_size != context_.slots() && values_size != elt_size) {
      throw std::invalid_argument(
          "values_size must be equal to context slots or elt_size");
    }
    if (!pool) {
      throw std::invalid_argument("pool is uninitialized");
    }
    if (elt_size > context_.slots()) {
      throw std::invalid_argument(
          "elt_size must be less than or equal to context slots");
    }
    if ((elt_size & (elt_size - 1)) != 0) {
      throw std::invalid_argument("elt_size must be a power of two");
    }
    if (elt_size == 0) {
      throw std::invalid_argument("elt_sizemust be greater than 0");
    }

    std::size_t coeff_count = context_.n();

    std::size_t repeat_count = context_.slots() / elt_size;

    // values_size is guaranteed to be no bigger than slots_
    auto slots_ = context_.slots();

    auto n_ = coeff_count / repeat_count;
    auto conj_values = allocate<std::complex<double>>(n_);

    for (std::size_t i = 0; i < elt_size; i++) {
      conj_values[i] =
          values_[matrix_reps_index_map_inverse_[i * repeat_count] &
                  (values_size - 1)];
      conj_values[n_ - 1 - i] = std::conj(conj_values[i]);
    }

    uint32_t inv_roots_offset = n_ * (repeat_count - 1);

    double fix = scale / static_cast<double>(n_);
    fft_handler_.transform_from_rev(
        conj_values.get(), seal::util::get_power_of_two(n_),
        inv_root_powers_.get() + inv_roots_offset, &fix);

    std::vector<double> polynomial(n_);
    for (std::size_t i = 0; i < n_; i++) {
      double coeffd = coordinate_wise_random_rounding(conj_values[i].real());
      // double coeffd = std::round(conj_values[i].real());
      polynomial[i] = coeffd;
    }

    return polynomial;
  }

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  void decode_internal(const std::shared_ptr<RnsPolynomial> &input,
                       const double scale, T *destination,
                       seal::MemoryPoolHandle pool) {
    // Verify parameters.
    if (input->empty()) {
      throw std::invalid_argument("input can't be empty");
    }
    for (const auto &[rns_id, limb] : *input) {
      if (!limb->is_ntt_form()) {
        throw std::invalid_argument("input is not in NTT form");
      }
    }
    if (!destination) {
      throw std::invalid_argument("destination cannot be null");
    }
    if (!pool) {
      throw std::invalid_argument("pool is uninitialized");
    }

    std::size_t num_limbs = input->num_limbs();
    std::vector<std::uint32_t> rns_base_ids;
    std::vector<seal::Modulus> rns_modulii;
    rns_base_ids.reserve(num_limbs);
    rns_modulii.reserve(num_limbs);
    for (auto &[rns_base_id, limb] : *input) {
      rns_base_ids.push_back(rns_base_id);
      rns_modulii.push_back(context_.get_rns_modulus(rns_base_id));
    }

    std::size_t coeff_count = context_.n();
    std::size_t rns_poly_uint64_count =
        seal::util::mul_safe(coeff_count, num_limbs);

    auto ntt_tables = context_.seal_ntt_tables();

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) >=
                       context_.rns_base_bit_count(num_limbs))) {
      throw std::invalid_argument("scale out of bounds");
    }

    seal::util::RNSBase rns_base(rns_modulii, pool);
    auto rns_base_prod = rns_base.base_prod();

    // upper_half_threshold is (rns_base_prod + 1)//2
    auto upper_half_threshold = seal::util::allocate_uint(num_limbs, pool);
    seal::util::increment_uint(rns_base_prod, num_limbs,
                               upper_half_threshold.get());
    seal::util::right_shift_uint(upper_half_threshold.get(), 1, num_limbs,
                                 upper_half_threshold.get());
    int logn = seal::util::get_power_of_two(coeff_count);

    double inv_scale = double(1.0) / scale;

    // Create mutable copy of input
    auto plain_copy(seal::util::allocate_uint(rns_poly_uint64_count, pool));

    // Transform each polynomial from NTT domain
    for (std::size_t i = 0; i < num_limbs; i++) {
      for (size_t j = 0; j < coeff_count; j++) {
        plain_copy.get()[i * coeff_count + j] = Math::safe_cast<std::uint64_t>(
            input->at(rns_base_ids.at(i))->data()[j]);
      }
      seal::util::inverse_ntt_negacyclic_harvey(
          plain_copy.get() + (i * coeff_count), ntt_tables[rns_base_ids.at(i)]);
    }

    // CRT-compose the polynomial
    rns_base.compose_array(plain_copy.get(), coeff_count, pool);

    // Create floating-point representations of the multi-precision integer coefficients
    double two_pow_64 = std::pow(2.0, 64);
    auto res = allocate<std::complex<double>>(coeff_count);
    for (std::size_t i = 0; i < coeff_count; i++) {
      res[i] = 0.0;
      if (seal::util::is_greater_than_or_equal_uint(
              plain_copy.get() + (i * num_limbs), upper_half_threshold.get(),
              num_limbs)) {
        double scaled_two_pow_64 = inv_scale;
        for (std::size_t j = 0; j < num_limbs;
             j++, scaled_two_pow_64 *= two_pow_64) {
          if (plain_copy[i * num_limbs + j] > rns_base_prod[j]) {
            auto diff = plain_copy[i * num_limbs + j] - rns_base_prod[j];
            res[i] +=
                diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
          } else {
            auto diff = rns_base_prod[j] - plain_copy[i * num_limbs + j];
            res[i] -=
                diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
          }
        }
      } else {
        double scaled_two_pow_64 = inv_scale;
        for (std::size_t j = 0; j < num_limbs;
             j++, scaled_two_pow_64 *= two_pow_64) {
          auto curr_coeff = plain_copy[i * num_limbs + j];
          res[i] += curr_coeff
                        ? static_cast<double>(curr_coeff) * scaled_two_pow_64
                        : 0.0;
        }
      }

      // Scaling instead incorporated above; this can help in cases
      // where otherwise pow(two_pow_64, j) would overflow due to very
      // large coeff_modulus_size and very large scale
      // res[i] = res_accum * inv_scale;
    }

    fft_handler_.transform_to_rev(res.get(), logn, root_powers_.get());

    for (std::size_t i = 0; i < slots_; i++) {
      destination[i] = seal::from_complex<T>(
          res[static_cast<std::size_t>(matrix_reps_index_map_[i])]);
    }
  }
};
} // namespace Runtime
} // namespace Cerium