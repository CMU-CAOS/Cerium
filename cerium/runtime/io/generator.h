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

#include "cerium/runtime/context.h"
#include "cerium/runtime/io/ckks-encoder.h"
#include "cerium/runtime/io/ckks-encryptor.h"

#include "cerium/runtime/memory/memory_manager.h"
#include "cerium/runtime/utils/logger.h"
#include "cerium/runtime/utils/map_wrapper.h"

#include <string>
#include <vector>

namespace Cerium {
namespace Runtime {

class IOGenerator {

public:
  struct EvalkeyInfo {
    enum KeyType {
      Mul,
      Rot,
      Con,
      Boot,
      Ephemeral,
      Boot2,
      RotInv,
    } key_type;
    uint32_t level;
    uint32_t extension_size;
    std::vector<uint64_t> digit_partition;
    int32_t rotation_amount;
    uint8_t ct_number;
    std::string id;
    std::vector<uint32_t> rns_bases;

    EvalkeyInfo(std::string &&key_info, std::string &&rns_bases);
  };

  using RnsPolynomialPair = std::pair<RnsPolynomialPtr, RnsPolynomialPtr>;
  using EvalKeyEntryType = std::tuple<EvalkeyInfo, RnsPolynomialPair>;

  std::pair<std::string, EvalKeyEntryType> parse_evalkey(std::string &line);

  struct PlaintextInfo {
    std::string var;
    std::size_t elt_size;
    std::vector<uint32_t> rns_bases;
    PlaintextInfo(std::string &&info, std::string &&rns_bases);
  };
  using PlaintextEntryType = std::tuple<PlaintextInfo, RnsPolynomialPtr>;
  std::pair<std::string, PlaintextEntryType> parse_plaintext(std::string &line);

  IOGenerator(const Context &context) : context(context) {}

  void generate_and_serialize_evalkeys(const std::string &outputs_file_name,
                                       const std::string &inputs_file_name,
                                       CKKSEncryptorContext &encryptor_context);
  void generate_and_serialize_plaintexts(
      const std::string &output_file_name, const std::string &inputs_file_name,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
      const std::string &stream_name = "Plaintext Stream");

  void generate_and_serialize_ciphertexts(
      const std::string &output_file_name, const std::string &inputs_file_name,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
      CKKSEncryptorContext &encryptor_context);

  std::unordered_map<std::string, RnsPolynomialPair>
  generate_ciphertext_inputs(const std::string &inputs_file_name,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
                    CKKSEncryptorContext &encryptor_context);

  std::pair<std::unordered_map<std::string, RnsPolynomialPair>, size_t>
  generate_evalkeys(const std::string &inputs_file_name,
                    CKKSEncryptorContext &encryptor_context);
  std::pair<std::unordered_map<std::string, RnsPolynomialPtr>, size_t>
  generate_plaintexts(
      const std::string &inputs_file_name,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr,
      const std::string &stream_name);
  // void generate_ciphertexts();

  void serialize_evalkeys(
      const std::string &output_file_name,
      const std::unordered_map<std::string, RnsPolynomialPair> &evalkeys,
      size_t evalkeys_size);
  void serialize_plaintexts(
      const std::string &output_file_name,
      const std::unordered_map<std::string, RnsPolynomialPtr> &plaintexts,
      size_t plaintexts_size);
  void serialize_ciphertexts(
      const std::string &output_file_name,
      const std::unordered_map<std::string, RnsPolynomialPair> &ciphertexts,
      size_t ciphertexts_size);

  inline static const std::string evalkey_header = "Cerium::Evalkeys\n";
  inline static const std::string plaintext_header = "Cerium::Plaintexts\n";
  inline static const std::string ciphertext_header = "Cerium::Ciphertexts\n";

  static std::unordered_map<std::string,
                            std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
  deserialize_evalkeys(
      const Context &context, const std::string &evalkeys_file_name,
      std::function<MemoryManagerHandle(size_t size)>
          memory_pool_create_callback = [](size_t size) { return nullptr; });
  static std::unordered_map<std::string, RnsPolynomialPtr>
  deserialize_plaintexts(
      const Context &context, const std::string &plaintexts_file_name,
      std::function<MemoryManagerHandle(size_t size)>
          memory_pool_create_callback = [](size_t size) { return nullptr; });
  static std::unordered_map<std::string, RnsPolynomialPtr>
  deserialize_plaintexts_no_copy(
      const Context &context, const std::string &plaintexts_file_name,
      std::function<MemoryManagerHandle(size_t size)>
          memory_pool_create_callback = [](size_t size) { return nullptr; });
  static std::pair<std::unordered_map<std::string, RnsPolynomialPtr>, size_t>
  deserialize_plaintexts_old_format(const Context &context,
                                    const std::string &plaintexts_file_name);

  static std::unordered_map<std::string, RnsPolynomialPair>
  deserialize_ciphertexts(
      const Context &context, const std::string &ciphertexts_file_name,
      std::function<MemoryManagerHandle(size_t size)>
          memory_pool_create_callback = [](size_t size) { return nullptr; });

private:
  const Context &context;
  seal::MemoryPoolHandle pool_glo = seal::MemoryManager::GetPool();
  Cerium::Runtime::Utils::Logger logger{"[IOGenerator]"};
};

} // namespace Runtime
} // namespace Cerium
