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

#include "arithmetic32.h"
#include "util.h"

namespace Cerium {
namespace Runtime {
namespace Math {
template <typename T>
// DATATYPE_TEMPLATE(T)
class NTTTables {
  using ModArithLazy =
      seal::util::Arithmetic<T, seal::util::MultiplyUIntModOperand,
                             seal::util::MultiplyUIntModOperand>;
  using NTTHandler =
      seal::util::DWTHandler<T, seal::util::MultiplyUIntModOperand,
                             seal::util::MultiplyUIntModOperand>;

public:
  NTTTables(NTTTables &&source) = default;

  NTTTables(NTTTables &copy)
      : pool_(copy.pool_), root_(copy.root_),
        coeff_count_power_(copy.coeff_count_power_),
        coeff_count_(copy.coeff_count_), modulus_(copy.modulus_),
        inv_degree_modulo_(copy.inv_degree_modulo_) {
    root_powers_ = seal::util::allocate<seal::util::MultiplyUIntModOperand>(
        coeff_count_, pool_);
    inv_root_powers_ = seal::util::allocate<seal::util::MultiplyUIntModOperand>(
        coeff_count_, pool_);

    std::copy_n(copy.root_powers_.get(), coeff_count_, root_powers_.get());
    std::copy_n(copy.inv_root_powers_.get(), coeff_count_,
                inv_root_powers_.get());
  }

  NTTTables(int coeff_count_power, const seal::Modulus &modulus,
            seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool())
      : pool_(std::move(pool)) {
#ifdef SEAL_DEBUG
    if (!pool_) {
      throw std::invalid_argument("pool is uninitialized");
    }
#endif
    initialize(coeff_count_power, modulus);
  }

  SEAL_NODISCARD inline std::uint64_t get_root() const { return root_; }

  SEAL_NODISCARD inline const seal::util::MultiplyUIntModOperand *
  get_from_root_powers() const {
    return root_powers_.get();
  }

  SEAL_NODISCARD inline const seal::util::MultiplyUIntModOperand *
  get_from_inv_root_powers() const {
    return inv_root_powers_.get();
  }

  SEAL_NODISCARD inline seal::util::MultiplyUIntModOperand
  get_from_root_powers(std::size_t index) const {
#ifdef SEAL_DEBUG
    if (index >= coeff_count_) {
      throw std::out_of_range("index");
    }
#endif
    return root_powers_[index];
  }

  SEAL_NODISCARD inline seal::util::MultiplyUIntModOperand
  get_from_inv_root_powers(std::size_t index) const {
#ifdef SEAL_DEBUG
    if (index >= coeff_count_) {
      throw std::out_of_range("index");
    }
#endif
    return inv_root_powers_[index];
  }

  SEAL_NODISCARD inline const seal::util::MultiplyUIntModOperand &
  inv_degree_modulo() const {
    return inv_degree_modulo_;
  }

  SEAL_NODISCARD inline const seal::Modulus &modulus() const {
    return modulus_;
  }

  SEAL_NODISCARD inline int coeff_count_power() const {
    return coeff_count_power_;
  }

  SEAL_NODISCARD inline std::size_t coeff_count() const { return coeff_count_; }

  const NTTHandler &ntt_handler() const { return ntt_handler_; }

  bool get_is_roots_initialized() const { return is_roots_initialized_; }

  void set_is_roots_initialized(bool is_roots_initialized) {
    is_roots_initialized_ = is_roots_initialized;
  }

  bool get_is_inv_roots_initialized() const {
    return is_inv_roots_initialized_;
  }

  void set_is_inv_roots_initialized(bool is_inv_roots_initialized) {
    is_inv_roots_initialized_ = is_inv_roots_initialized;
  }

#if 0
            void set_root_powers(uint32_t* root_powers, size_t n) {
                std::uint32_t *d_alloc = nullptr;
                CUDA::allocate_memory_on_device((void**)&d_alloc, n*sizeof(uint32_t));
                CUDA::memcpyHostToDevice(d_alloc, root_powers, n*sizeof(uint32_t));
                root_powers_ptr.reset(d_alloc, [](void *d_ptr) {CUDA::free(d_ptr);});
            }

            void set_inv_root_powers(uint32_t* inv_root_powers, size_t n) {
                std::uint32_t *d_alloc = nullptr;
                CUDA::allocate_memory_on_device((void**)&d_alloc, n*sizeof(uint32_t));
                CUDA::memcpyHostToDevice(d_alloc, inv_root_powers, n*sizeof(uint32_t));
                inv_root_powers_ptr.reset(d_alloc, [](void *d_ptr) {CUDA::free(d_ptr);});
            }

            uint32_t *get_root_powers() const{
                return root_powers_ptr.get();
            }

            uint32_t *get_inv_root_powers() const{
                return inv_root_powers_ptr.get();
            }

            void set_root_powers_shoup(uint32_t* root_powers_shoup, size_t n) {
                std::uint32_t *d_alloc = nullptr;
                CUDA::allocate_memory_on_device((void**)&d_alloc, n*sizeof(uint32_t));
                CUDA::memcpyHostToDevice(d_alloc, root_powers_shoup, n*sizeof(uint32_t));
                root_powers_shoup_ptr.reset(d_alloc, [](void *d_ptr) {CUDA::free(d_ptr);});
            }

            void set_inv_root_powers_shoup(uint32_t* inv_root_powers_shoup, size_t n) {
                std::uint32_t *d_alloc = nullptr;
                CUDA::allocate_memory_on_device((void**)&d_alloc, n*sizeof(uint32_t));
                CUDA::memcpyHostToDevice(d_alloc, inv_root_powers_shoup, n*sizeof(uint32_t));
                inv_root_powers_shoup_ptr.reset(d_alloc, [](void *d_ptr) {CUDA::free(d_ptr);});
            }
#endif

  uint32_t *get_root_powers_shoup() const {
    return root_powers_shoup_ptr.get();
  }

  uint32_t *get_inv_root_powers_shoup() const {
    return inv_root_powers_shoup_ptr.get();
  }

private:
  NTTTables &operator=(const NTTTables &assign) = delete;

  NTTTables &operator=(NTTTables &&assign) = delete;

  void initialize(int coeff_count_power, const seal::Modulus &modulus);

  seal::MemoryPoolHandle pool_;

  std::uint64_t root_ = 0;

  std::uint64_t inv_root_ = 0;

  int coeff_count_power_ = 0;

  std::size_t coeff_count_ = 0;

  seal::Modulus modulus_;

  // Inverse of coeff_count_ modulo modulus_.
  seal::util::MultiplyUIntModOperand inv_degree_modulo_;

  // Holds 1~(n-1)-th powers of root_ in bit-reversed order, the 0-th power is
  // left unset.
  seal::util::Pointer<seal::util::MultiplyUIntModOperand> root_powers_;

  // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power
  // is left unset.
  seal::util::Pointer<seal::util::MultiplyUIntModOperand> inv_root_powers_;

  // Holds self-computed roots_powers for device-side NTT/INTT
  std::shared_ptr<std::uint32_t> root_powers_ptr = nullptr;
  std::shared_ptr<std::uint32_t> inv_root_powers_ptr = nullptr;
  std::shared_ptr<std::uint32_t> root_powers_shoup_ptr = nullptr;
  std::shared_ptr<std::uint32_t> inv_root_powers_shoup_ptr = nullptr;
  bool is_roots_initialized_ = false;
  bool is_inv_roots_initialized_ = false;

  ModArithLazy mod_arith_lazy_;

  NTTHandler ntt_handler_;
};

/**
        Allocate and construct an array of NTTTables each with different a modulus.

        @throws std::invalid_argument if modulus is empty, modulus does not support NTT,
        coeff_count_power is invalid, or pool is uninitialized.
        */

DATATYPE_TEMPLATE(T)
void CreateNTTTables(int coeff_count_power,
                     const std::vector<seal::Modulus> &modulus,
                     seal::util::Pointer<NTTTables<T>> &tables,
                     seal::MemoryPoolHandle pool);

DATATYPE_TEMPLATE(T)
void ntt_negacyclic_harvey_lazy(T *operand, const NTTTables<T> &tables);

DATATYPE_TEMPLATE(T)
void ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables);

DATATYPE_TEMPLATE(T)
void ntt_negacyclic_harvey_lazy(T *operand, const NTTTables<T> &tables,
                                const size_t logN);

DATATYPE_TEMPLATE(T)
void ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables,
                           const size_t logN);

DATATYPE_TEMPLATE(T)
void inverse_ntt_negacyclic_harvey_lazy(T *operand, const NTTTables<T> &tables);

DATATYPE_TEMPLATE(T)
void inverse_ntt_negacyclic_harvey(T *operand, const NTTTables<T> &tables);

} // namespace Math
} // namespace Runtime
} // namespace Cerium