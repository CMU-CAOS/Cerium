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

#include "ntt.h"

#include <iterator>

namespace Cerium {
namespace Runtime {

namespace Math {

DATATYPE_TEMPLATE(T)
class NTTTablesCreateIter {
public:
  using value_type = NTTTables<T>;
  using pointer = void;
  using reference = value_type;
  using difference_type = ptrdiff_t;

  // LegacyInputIterator allows reference to be equal to value_type so we can construct
  // the return objects on the fly and return by value.
  using iterator_category = std::input_iterator_tag;

  // Require default constructor
  NTTTablesCreateIter() {}

  // Other constructors
  NTTTablesCreateIter(int coeff_count_power, std::vector<seal::Modulus> modulus,
                      seal::MemoryPoolHandle pool)
      : coeff_count_power_(coeff_count_power), modulus_(modulus),
        pool_(std::move(pool)) {}

  // Require copy and move constructors and assignments
  NTTTablesCreateIter(const NTTTablesCreateIter &copy) = default;

  NTTTablesCreateIter(NTTTablesCreateIter &&source) = default;

  NTTTablesCreateIter &operator=(const NTTTablesCreateIter &assign) = default;

  NTTTablesCreateIter &operator=(NTTTablesCreateIter &&assign) = default;

  // Dereferencing creates NTTTables and returns by value
  inline value_type operator*() const {
    return {coeff_count_power_, modulus_[index_], pool_};
  }

  // Pre-increment
  inline NTTTablesCreateIter &operator++() noexcept {
    index_++;
    return *this;
  }

  // Post-increment
  inline NTTTablesCreateIter operator++(int) noexcept {
    NTTTablesCreateIter result(*this);
    index_++;
    return result;
  }

  // Must be EqualityComparable
  inline bool operator==(const NTTTablesCreateIter &compare) const noexcept {
    return (compare.index_ == index_) &&
           (coeff_count_power_ == compare.coeff_count_power_);
  }

  inline bool operator!=(const NTTTablesCreateIter &compare) const noexcept {
    return !operator==(compare);
  }

  // Arrow operator must be defined
  value_type operator->() const { return **this; }

private:
  size_t index_ = 0;
  int coeff_count_power_ = 0;
  std::vector<seal::Modulus> modulus_;
  seal::MemoryPoolHandle pool_;
};

// DATATYPE_TEMPLATE(T)
template <typename T>
void NTTTables<T>::initialize(int coeff_count_power,
                              const seal::Modulus &modulus) {
  coeff_count_power_ = coeff_count_power;
  coeff_count_ = size_t(1) << coeff_count_power_;
  modulus_ = modulus;
  // We defer parameter checking to try_minimal_primitive_root(...)
  if (!seal::util::try_minimal_primitive_root(2 * coeff_count_, modulus_,
                                              root_)) {
    throw std::invalid_argument("invalid modulus");
  }
  if (!seal::util::try_invert_uint_mod(root_, modulus_, inv_root_)) {
    throw std::invalid_argument("invalid modulus");
  }

  // Populate tables with powers of root in specific orders.
  root_powers_ = seal::util::allocate<seal::util::MultiplyUIntModOperand>(
      coeff_count_, pool_);
  seal::util::MultiplyUIntModOperand root;
  root.set(root_, modulus_);
  uint64_t power = root_;
  for (size_t i = 1; i < coeff_count_; i++) {
    root_powers_[seal::util::reverse_bits(i, coeff_count_power_)].set(power,
                                                                      modulus_);
    power = seal::util::multiply_uint_mod(power, root, modulus_);
  }
  root_powers_[0].set(static_cast<uint64_t>(1), modulus_);

  inv_root_powers_ = seal::util::allocate<seal::util::MultiplyUIntModOperand>(
      coeff_count_, pool_);
  root.set(inv_root_, modulus_);
  power = inv_root_;
  for (size_t i = 1; i < coeff_count_; i++) {
    inv_root_powers_[seal::util::reverse_bits(i - 1, coeff_count_power_) + 1]
        .set(power, modulus_);
    power = seal::util::multiply_uint_mod(power, root, modulus_);
  }
  inv_root_powers_[0].set(static_cast<uint64_t>(1), modulus_);

  // Compute n^(-1) modulo q.
  uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
  if (!seal::util::try_invert_uint_mod(degree_uint, modulus_,
                                       inv_degree_modulo_.operand)) {
    throw std::invalid_argument("invalid modulus");
  }
  inv_degree_modulo_.set_quotient(modulus_);

  mod_arith_lazy_ = ModArithLazy(modulus_);
  ntt_handler_ = NTTHandler(mod_arith_lazy_);

  set_is_inv_roots_initialized(true);
}

DATATYPE_TEMPLATE2(T)
void CreateNTTTables(int coeff_count_power,
                     const std::vector<seal::Modulus> &modulus,
                     seal::util::Pointer<NTTTables<T>> &tables,
                     seal::MemoryPoolHandle pool) {
  if (!pool) {
    throw std::invalid_argument("pool is uninitialized");
  }
  if (!modulus.size()) {
    throw std::invalid_argument("invalid modulus");
  }
  // coeff_count_power and modulus will be validated by "allocate"
  NTTTablesCreateIter<T> iter(coeff_count_power, modulus, pool);
  tables = seal::util::allocate(iter, modulus.size(), pool);

  // tables = seal::util::allocate<NTTTables<T>>(modulus.size());
  // for(size_t i = 0; i < modulus.size(); i++) {
  //     tables[i] = NTTTables<T>(coeff_count_power,modulus[i]);
  // }
}

DATATYPE_TEMPLATE2(T)
void ntt_negacyclic_harvey_lazy(T *operand, const NTTTables<T> &tables) {
  tables.ntt_handler().transform_to_rev(operand, tables.coeff_count_power(),
                                        tables.get_from_root_powers());
}

DATATYPE_TEMPLATE2(T)
void ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables) {

  ntt_negacyclic_harvey_lazy(operand, tables);
  // Finally maybe we need to reduce every coefficient modulo q, but we
  // know that they are in the range [0, 4q).
  // Since word size is controlled this is fast.
  T modulus = seal::util::safe_cast<T>(tables.modulus().value());
  T two_times_modulus = modulus * 2;
  std::size_t n = std::size_t(1) << tables.coeff_count_power();

  for (size_t i = 0; i < n; i++) {
    if (operand[i] >= two_times_modulus) {
      operand[i] -= two_times_modulus;
    }
    if (operand[i] >= modulus) {
      operand[i] -= modulus;
    }
  }
}

DATATYPE_TEMPLATE2(T)
void ntt_negacyclic_harvey_lazy(T *operand, const NTTTables<T> &tables,
                                const size_t logN) {
  tables.ntt_handler().transform_to_rev(operand, logN,
                                        tables.get_from_root_powers());
}

DATATYPE_TEMPLATE2(T)
void ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables,
                           const size_t logN) {

  ntt_negacyclic_harvey_lazy(operand, tables, logN);
  // Finally maybe we need to reduce every coefficient modulo q, but we
  // know that they are in the range [0, 4q).
  // Since word size is controlled this is fast.
  T modulus = seal::util::safe_cast<T>(tables.modulus().value());
  T two_times_modulus = modulus * 2;
  std::size_t n = std::size_t(1) << logN;

  for (size_t i = 0; i < n; i++) {
    if (operand[i] >= two_times_modulus) {
      operand[i] -= two_times_modulus;
    }
    if (operand[i] >= modulus) {
      operand[i] -= modulus;
    }
  }
}

DATATYPE_TEMPLATE2(T)
void inverse_ntt_negacyclic_harvey_lazy(T *operand,
                                        const NTTTables<T> &tables) {
  auto inv_degree_modulo = tables.inv_degree_modulo();
  tables.ntt_handler().transform_from_rev(operand, tables.coeff_count_power(),
                                          tables.get_from_inv_root_powers(),
                                          &inv_degree_modulo);
}

DATATYPE_TEMPLATE2(T)
void inverse_ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables) {
  inverse_ntt_negacyclic_harvey_lazy(operand, tables);
  T modulus = seal::util::safe_cast<T>(tables.modulus().value());
  std::size_t n = std::size_t(1) << tables.coeff_count_power();

  // Final adjustments; compute a[j] = a[j] * n^{-1} mod q.
  // We incorporated the final adjustment in the butterfly. Only need to reduce here.
  for (size_t i = 0; i < n; i++) {
    if (operand[i] >= modulus) {
      operand[i] -= modulus;
    }
  }
}

template void
CreateNTTTables<uint32_t>(int coeff_count_power,
                          const std::vector<seal::Modulus> &modulus,
                          seal::util::Pointer<NTTTables<uint32_t>> &tables,
                          seal::MemoryPoolHandle pool);

template void
ntt_negacyclic_harvey<uint32_t>(uint32_t *operand,
                                const NTTTables<uint32_t> &tables);
template void ntt_negacyclic_harvey<uint32_t>(uint32_t *operand,
                                              const NTTTables<uint32_t> &tables,
                                              const size_t logN);
template void
inverse_ntt_negacyclic_harvey<uint32_t>(uint32_t *operand,
                                        const NTTTables<uint32_t> &tables);

} // namespace Math
} // namespace Runtime
} // namespace Cerium