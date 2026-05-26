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
#include "cerium/runtime/io/ckks-encoder.h"
#include "cerium/runtime/math/util.h"
#include "cerium/runtime/polynomial/limb.h"
#include "cerium/runtime/polynomial/rns_polynomial.h"

namespace Cerium {
namespace Runtime {

class CKKSEncryptor;

class CKKSEncryptorContext {
public:
  CKKSEncryptorContext(const Context &context,
                       const std::vector<std::int64_t> &secret_key,
                       const std::vector<std::int64_t> &ephemeral_key);
  CKKSEncryptorContext(const Context &context,
                       const std::vector<std::int64_t> &secret_key,
                       const std::vector<std::int64_t> &ephemeral_key,
                       const seal::prng_seed_type &prng_seed);

  CKKSEncryptorContext() = delete;
  CKKSEncryptorContext(const CKKSEncryptorContext &) = delete;
  CKKSEncryptorContext(CKKSEncryptorContext &&) = default;
  CKKSEncryptorContext &operator=(CKKSEncryptorContext &&) = delete;

  std::vector<CKKSEncryptor> create_encryptors(size_t num_encryptors);
  std::vector<CKKSEncoder> create_encoders(size_t num_encoders);

private:
  const Context &context_;
  const std::vector<std::int64_t> secret_key_;
  const std::vector<std::int64_t> ephemeral_key_;
  std::vector<std::shared_ptr<Limb>> secret_key_rns_;
  std::vector<std::shared_ptr<Limb>> secret_key_squared_rns_;
  std::vector<std::shared_ptr<Limb>> ephemeral_key_rns_;
  bool is_seeded_;
  seal::prng_seed_type prng_seed_;
  std::shared_ptr<seal::UniformRandomGenerator> prng_;
  bool save_ciphertext_random_seed_ = false;

  void compute_secret_key_rns();
  void compute_ephemeral_key_rns();

  friend class CKKSEncryptor;

  // RnsPolynomialPtr sample_poly_uniform(std::shared_ptr<seal::UniformRandomGenerator> &prng);
  // RnsPolynomialPtr sample_poly_cbd(std::shared_ptr<seal::UniformRandomGenerator> &prng);
};

class CKKSEncryptor {
  using LimbVector = std::vector<std::shared_ptr<Limb>>;

public:
  CKKSEncryptor(const CKKSEncryptorContext &encryptor_context);
  CKKSEncryptor(const CKKSEncryptorContext &encryptor_context,
                const seal::prng_seed_type &prng_seed);

  CKKSEncryptor() = delete;
  CKKSEncryptor(const CKKSEncryptor &) = delete;
  CKKSEncryptor(CKKSEncryptor &&) = default;
  CKKSEncryptor &operator=(CKKSEncryptor &&) = delete;

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::pair<std::shared_ptr<RnsPolynomial>,
                   std::shared_ptr<RnsPolynomial>>
  encode_and_encrypt(
      const std::vector<T> &values, const double scale,
      const std::vector<std::uint32_t> &rns_base_ids,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {

    auto encoded_value =
        encoder_.encode(values, scale, rns_base_ids, true, pool);
    return encrypt_internal(encoded_value, rns_base_ids, pool);
  }

  inline std::pair<std::shared_ptr<RnsPolynomial>,
                   std::shared_ptr<RnsPolynomial>>
  encode_and_encrypt(
      const double &value, const double scale,
      const std::vector<std::uint32_t> &rns_base_ids,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {

    auto encoded_value = encoder_.encode(value, scale, rns_base_ids, pool);
    return encrypt_internal(encoded_value, rns_base_ids, pool);
  }

  template <typename T,
            typename = std::enable_if_t<
                std::is_same<std::remove_cv_t<T>, double>::value ||
                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
  inline std::vector<T> decrypt_and_decode(
      const std::pair<RnsPolynomialPtr, RnsPolynomialPtr> &ciphertext,
      const double scale,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool()) {
    auto decrypt = decrypt_internal(ciphertext, pool);
    return encoder_.decode<T>(decrypt, scale, std::move(pool));
  }

  std::pair<RnsPolynomialPtr, RnsPolynomialPtr>
  encrypt_zero(const LimbVector &key_rns,
               const std::vector<std::uint32_t> &rns_base_ids,
               seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr>
  encrypt_zero(const std::vector<std::uint32_t> &rns_base_ids,
               seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool(),
               bool use_ephemeral_key = false);

  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_relin_evalkey(
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_rotation_evalkey(
      const std::uint32_t rotation_amout, const std::uint64_t level,
      const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_rotation_inv_evalkey(
      const std::uint32_t rotation_amout, const std::uint64_t level,
      const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_conjugation_evalkey(
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_bootstrap_evalkey(
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_ephemeral_evalkey(
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_bootstrap_evalkey2(
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

private:
  const Context &context_;
  const CKKSEncryptorContext &encryptor_context_;
  CKKSEncoder encoder_;
  bool is_seeded_;
  seal::prng_seed_type prng_seed_;
  std::shared_ptr<seal::UniformRandomGenerator> prng_;
  bool save_ciphertext_random_seed_ = false;

  RnsPolynomialPtr
  sample_poly_uniform(std::shared_ptr<seal::UniformRandomGenerator> &prng,
                      const std::vector<std::uint32_t> &rns_base_ids);
  RnsPolynomialPtr
  sample_poly_cbd(std::shared_ptr<seal::UniformRandomGenerator> &prng,
                  const std::vector<std::uint32_t> &rns_base_ids);

  std::pair<std::shared_ptr<RnsPolynomial>, std::shared_ptr<RnsPolynomial>>
  encrypt_internal(
      const std::shared_ptr<RnsPolynomial> &message,
      const std::vector<std::uint32_t> &rns_base_ids,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  RnsPolynomialPtr decrypt_internal(
      const std::pair<RnsPolynomialPtr, RnsPolynomialPtr> &ciphertext,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());

  std::pair<RnsPolynomialPtr, RnsPolynomialPtr> generate_evalkey(
      const LimbVector &old_key, const LimbVector &new_key,
      const std::uint64_t level, const std::uint64_t extension_size,
      const std::vector<std::uint64_t> &digit_partition,
      seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool());
  std::pair<RnsPolynomialPtr, RnsPolynomialPtr>
  generate_evalkey(const LimbVector &new_key, const std::uint64_t level,
                   const std::uint64_t extension_size,
                   const std::vector<std::uint64_t> &digit_partition,
                   seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool(),
                   bool use_ephemeral_key = false);
};
} // namespace Runtime
} // namespace Cerium