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

#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term_map.h"
#include "cerium/compiler/util/logging.h"
#include "cerium/compiler/util/overloaded.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "cerium/compiler/backend/cuda_kernel.h"
#include "cerium/compiler/backend/kernel_group.h"
#include "cerium/compiler/backend/keyswitch_digits.h"
#include "cerium/compiler/backend/logger.h"
#include "cerium/compiler/backend/terms.h"

#include <tuple>

namespace Cerium {
namespace Backend {

class CeriumCompiler {

  void validate_partition_state(const PartitionInfo &partition) const;

  Backend::Polynomial
  make_term(uint16_t level,
            const std::function<void(Backend::Term &)> &configure_term);
  Backend::Polynomial make_input(uint16_t level);
  Backend::Polynomial make_function_input(uint16_t level);
  Backend::Polynomial make_plaintext(uint16_t level);
  Backend::Polynomial make_scalar(uint16_t level);

  std::shared_ptr<KernelGroup>
  create_kernel_group(const PartitionInfo &partition,
                      KernelGroup::Type type = KernelGroup::Type::Ewi);
  std::shared_ptr<KernelGroup> create_kernel_group();
  std::shared_ptr<KernelGroup>
  create_kernel_group(std::shared_ptr<KernelGroup> kg_other);
  std::shared_ptr<KernelGroup>
  create_kernel_group_intt(const PartitionInfo &partition);
  std::shared_ptr<KernelGroup> create_kernel_group_intt();
  std::shared_ptr<KernelGroup> create_kernel_group_bconv();
  std::shared_ptr<KernelGroup> create_kernel_group_ntt();
  std::shared_ptr<KernelGroup> create_kernel_group_sud2();
  std::shared_ptr<KernelGroup> create_kernel_group_pmul();
  std::shared_ptr<KernelGroup> create_kernel_group_dist();
  std::shared_ptr<KernelGroup> create_kernel_group_agg();
  std::shared_ptr<KernelGroup> create_kernel_group_ard();
  std::shared_ptr<KernelGroup> create_kernel_group_call();

  // Eval-key metadata
  struct EvalKeyIndexType {
    KeySwitch::KeySwitchType key_switch_type;
    Backend::EvalKeyType evk_type;
    uint8_t level;
    uint8_t extension_size;
    uint8_t partition_size;
    uint8_t partition_id;
    int32_t rot_idx;

    EvalKeyIndexType(KeySwitch::KeySwitchType key_switch_type,
                     Backend::EvalKeyType evk_type, uint8_t level,
                     uint8_t extension_size, uint8_t partition_size,
                     uint8_t partition_id, int32_t rot_idx)
        : key_switch_type(key_switch_type), evk_type(evk_type), level(level),
          extension_size(extension_size), partition_size(partition_size),
          partition_id(partition_id), rot_idx(rot_idx) {}
  };

  struct EvalKeyIndexTypeHash {
    std::size_t operator()(const EvalKeyIndexType &evk_index) const {
      return std::hash<KeySwitch::KeySwitchType>()(evk_index.key_switch_type) ^
             std::hash<Backend::EvalKeyType>()(evk_index.evk_type) ^
             std::hash<uint8_t>()(evk_index.level) ^
             std::hash<uint8_t>()(evk_index.extension_size) ^
             std::hash<uint8_t>()(evk_index.partition_size) ^
             std::hash<uint8_t>()(evk_index.partition_id) ^
             std::hash<int32_t>()(evk_index.rot_idx);
    }
  };

  struct EvalKeyIndexTypeCompare {
    bool operator()(const EvalKeyIndexType &lhs,
                    const EvalKeyIndexType &rhs) const {
      return lhs.key_switch_type == rhs.key_switch_type &&
             lhs.evk_type == rhs.evk_type && lhs.level == rhs.level &&
             lhs.extension_size == rhs.extension_size &&
             lhs.partition_size == rhs.partition_size &&
             lhs.partition_id == rhs.partition_id && lhs.rot_idx == rhs.rot_idx;
    }
  };

  template <typename T>
  using EvalKeyIndexTypeMap =
      std::unordered_map<EvalKeyIndexType, T, EvalKeyIndexTypeHash,
                         EvalKeyIndexTypeCompare>;

  EvalKeyIndexTypeMap<uint64_t> evalkey_input_indices;
  using PolyPair = std::pair<Backend::Polynomial, Backend::Polynomial>;

  std::pair<PolyPair, PolyPair>
  get_evalkey(const Backend::Polynomial &input, Backend::EvalKeyType evk_type,
              int32_t rot_idx, KeySwitch::KeySwitchType key_switch_type,
              uint16_t extension_size);

  Backend::Polynomial make_treg(uint16_t level);
  Backend::Polynomial make_zero_treg();
  Backend::Polynomial make_treg_from(const Backend::Polynomial &other);
  Backend::Polynomial make_copy_treg_from(const Backend::Polynomial &other);
  Backend::Polynomial make_rescale_treg_from(const Backend::Polynomial &other,
                                             const uint32_t rescale_levels = 1);
  Backend::Polynomial make_bcor_from(const Backend::Polynomial &other);
  void set_output(Frontend::Term::Ptr &args1, const std::string &name);
  Backend::Polynomial
  make_new_term_share_from(const Backend::Polynomial *other);
  Backend::Polynomial
  make_new_term_share_from(const Backend::Polynomial &other);
  Backend::Polynomial make_modswitch_treg_from(Backend::Polynomial &other);
  Backend::Polynomial
  make_double_rescale_treg_from(const Backend::Polynomial &other);

  // Runtime value plumbing
  using RuntimeValue =
      std::variant<Backend::Ciphertext, Backend::Plaintext,
                   Backend::CiphertextVector, Backend::PlaintextVector>;

  template <std::size_t I, std::size_t J>
  using RuntimeValuePairInternal =
      std::pair<std::variant_alternative_t<I, RuntimeValue>,
                std::variant_alternative_t<J, RuntimeValue>>;

  using RuntimeValuePair = std::variant<
      RuntimeValuePairInternal<0, 0>, RuntimeValuePairInternal<0, 1>,
      RuntimeValuePairInternal<0, 2>, RuntimeValuePairInternal<0, 3>,
      RuntimeValuePairInternal<1, 0>, RuntimeValuePairInternal<1, 1>,
      RuntimeValuePairInternal<1, 2>, RuntimeValuePairInternal<1, 3>,
      RuntimeValuePairInternal<2, 0>, RuntimeValuePairInternal<2, 1>,
      RuntimeValuePairInternal<2, 2>, RuntimeValuePairInternal<2, 3>,
      RuntimeValuePairInternal<3, 0>, RuntimeValuePairInternal<3, 1>,
      RuntimeValuePairInternal<3, 2>, RuntimeValuePairInternal<3, 3>>;

  RuntimeValuePair get_runtime_value_pair(const Frontend::Term::Ptr &args1,
                                          const Frontend::Term::Ptr &args2);

  Frontend::TermMapOptional<RuntimeValue> Objects;

  std::string output_prefix;

  uint64_t nextTermIndex = 1;
  uint64_t nextTermShareIndex = 1;

  uint8_t mod_partitions = 1;
  uint8_t current_partition_size = mod_partitions;
  uint8_t current_partition_id = 0;

  uint64_t num_vregs = 1024;
  uint64_t num_bcus = 128;

  std::vector<std::shared_ptr<KernelGroup>> kernel_groups;
  std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_split;

  std::string function_name;

  Backend::Ciphertext getCiphertext(const Frontend::Term::Ptr &t) {
    return std::get<Backend::Ciphertext>(Objects.at(t));
  }

  bool isCipher(const Frontend::Term::Ptr &t) {
    return std::holds_alternative<Backend::Ciphertext>(Objects.at(t));
  }

  bool isPlain(const Frontend::Term::Ptr &t) {
    return std::holds_alternative<Backend::Plaintext>(Objects.at(t));
  }

  bool isPlain2(const Frontend::Term::Ptr &t) {
    return std::holds_alternative<Backend::Plaintext>(Objects.at(t)) ||
           std::holds_alternative<Backend::PlaintextVector>(Objects.at(t));
  }

  bool isCiphertextVector(const Frontend::Term::Ptr &t) {
    return std::holds_alternative<Backend::CiphertextVector>(Objects.at(t));
  }

  bool isPlaintextVector(const Frontend::Term::Ptr &t) {
    return std::holds_alternative<Backend::PlaintextVector>(Objects.at(t));
  }

  bool isVector(const Frontend::Term::Ptr &t) {
    return isCiphertextVector(t) || isPlaintextVector(t);
  }

  std::map<Backend::TermIndexType, std::shared_ptr<Backend::Term>> terms;
  std::map<Backend::TermIndexType, std::shared_ptr<Backend::TermShare>>
      term_shares;

  // Internal pipeline helpers
private:
  template <typename T> T &initValue(const Frontend::Term::Ptr &term) {
    return std::get<T>(Objects[term] = T{});
  }

  // Partitioning and I/O
  void partition(const PartitionInfo &partition);
  void receive(const Frontend::Term::Ptr &term,
                   const Frontend::Term::Ptr &args1);
  void receive2(const Frontend::Term::Ptr &term,
                const Frontend::Term::Ptr &args1);

  void reduce(const Frontend::Term::Ptr &term,
              const std::vector<Frontend::Term::Ptr> &args);

  // Arithmetic and transforms
  void process_input(const Frontend::Term::Ptr &term);
  void add(const Frontend::Term::Ptr &term,
               const Frontend::Term::Ptr &args1,
               const Frontend::Term::Ptr &args2);
  void mul(const Frontend::Term::Ptr &term,
               const Frontend::Term::Ptr &args1,
               const Frontend::Term::Ptr &args2);
  void sub(const Frontend::Term::Ptr &term,
               const Frontend::Term::Ptr &args1,
               const Frontend::Term::Ptr &args2);

  void rotate(const Frontend::Term::Ptr &term,
                  const Frontend::Term::Ptr &args1, int32_t rot_idx,
                  const bool use_keyswitch2 = false);
  void conjugate(const Frontend::Term::Ptr &term,
                     const Frontend::Term::Ptr &args1,
                     const bool use_keyswitch2 = false);

  void relinearize(const Frontend::Term::Ptr &term,
                       const Frontend::Term::Ptr &args1);

  void relinearize2(const Frontend::Term::Ptr &term,
                        const Frontend::Term::Ptr &args1);

  void relinearize3(const Frontend::Term::Ptr &term,
                        const Frontend::Term::Ptr &args1);
  void rescale_internal(Backend::Ciphertext &output,
                        const Backend::Ciphertext &input1);
  void rescale(const Frontend::Term::Ptr &term,
                   const Frontend::Term::Ptr &args1);
  void double_rescale(const Frontend::Term::Ptr &term,
                          const Frontend::Term::Ptr &args1);
  Backend::Ciphertext keyswitch(Backend::Ciphertext &inp,
                                Backend::EvalKeyType evk_type,
                                int32_t rot_idx = 0,
                                const uint32_t rescale_levels = 0);
  Backend::Ciphertext keyswitch2(Backend::Ciphertext &inp,
                                 Backend::EvalKeyType evk_type,
                                 int32_t rot_idx = 0,
                                 const uint32_t rescale_levels = 0);
  Backend::Ciphertext keyswitch_ciphertext_impl(Backend::Ciphertext &input,
                                                Backend::EvalKeyType evk_type,
                                                int32_t rot_idx,
                                                const uint32_t rescale_levels,
                                                bool use_keyswitch2_split);
  Backend::Ciphertext keyswitch3(Backend::Ciphertext &inp,
                                 Backend::EvalKeyType evk_type,
                                 int32_t rot_idx = 0,
                                 const uint32_t rescale_levels = 0);
  Backend::CiphertextVector keyswitch_vec(Backend::CiphertextVector &inp,
                                          Backend::EvalKeyType evk_type,
                                          int32_t rot_idx = 0,
                                          const uint32_t rescale_levels = 0);
  void rotate(Backend::Ciphertext &output, const Frontend::Term::Ptr &args1,
              int32_t rot_idx, const bool use_keyswitch2 = false);
  void rotate3(Backend::Ciphertext &output, const Frontend::Term::Ptr &args1,
               int32_t rot_idx, const bool use_keyswitch2 = false);
  void rotate_internal(Backend::Ciphertext &output, Backend::Ciphertext &input1,
                       int32_t rot_idx, const bool use_keyswitch2 = false);
  void rotate3_internal(Backend::Ciphertext &output,
                        Backend::Ciphertext &input1, int32_t rot_idx,
                        const bool use_keyswitch2 = false);
  void to_ephemeral(Backend::Ciphertext &output,
                    const Frontend::Term::Ptr &args1);
  void bootstrap_mod_raise(Backend::Ciphertext &output,
                           const Frontend::Term::Ptr &args1,
                           const uint16_t raise_to_level,
                           bool use_ephemeral_key = false);

  void mod_switch(const Frontend::Term::Ptr &term,
                      const Frontend::Term::Ptr &args1);

  void
  bsgs_multiply_accumulate(const Frontend::Term::Ptr &output,
                               const std::vector<Frontend::Term::Ptr> &args,
                               std::vector<int32_t> babystep_rotation_indices,
                               std::vector<int32_t> giantstep_rotation_indices,
                               const uint32_t rescale_levels = 0);

  // Dispatch and entry points
  void process_function_call(const Frontend::Term::Ptr &term);
  void process_function_input(const Frontend::Term::Ptr &term);
  void process_function_output(const Frontend::Term::Ptr &term);
  void initialize_cipher_input(
      const Frontend::Term::Ptr &term, uint16_t limbs, const std::string &name,
      Backend::Polynomial (CeriumCompiler::*make_poly)(uint16_t));

  void hoisted_input_broadcast(const Frontend::Term::Ptr &term,
                               std::vector<int32_t> rotation_indices,
                               const uint32_t rescale_levels);
  void rotate_accumulate(const Frontend::Term::Ptr &term,
                         std::vector<int32_t> rotation_indices,
                         const uint32_t rescale_levels);
  Backend::CiphertextVector rotate_accumulate_internal(
      const std::vector<Backend::CiphertextVector> &inputs,
      const std::vector<int32_t> rotation_indices, const uint32_t dnum_glo,
      const uint8_t rescale_levels);
  void rotate_multiply_accumulate_common(
      const Frontend::Term::Ptr &term,
      const std::vector<Frontend::Term::Ptr> &args,
      std::vector<int32_t> rotation_indices, const uint32_t rescale_levels,
      const std::function<Backend::CiphertextVector(
          const Backend::CiphertextVector &input1,
          const std::vector<Backend::Plaintext> &plaintexts,
          const std::vector<int32_t> &rotation_indices,
          const uint32_t rescale_levels)> &build_output);
  void rotate_multiply_accumulate(const Frontend::Term::Ptr &output,
                                  const std::vector<Frontend::Term::Ptr> &args,
                                  std::vector<int32_t> rotation_indices,
                                  const uint32_t rescale_levels = 0);
  void multiply_rotate_accumulate(const Frontend::Term::Ptr &output,
                                  const std::vector<Frontend::Term::Ptr> &args,
                                  std::vector<int32_t> rotation_indices,
                                  const uint32_t rescale_levels = 0);

  std::tuple<std::vector<Backend::Polynomial>, std::vector<Backend::Polynomial>>
  hoisted_input_broadcast_keyswitching_internal(
      const Backend::Ciphertext &input,
      const std::vector<int32_t> rotation_indices);
  std::vector<Backend::CiphertextVector>
  hoisted_input_broadcast_keyswitching_internal_vec(
      const Backend::CiphertextVector &input,
      const std::vector<int32_t> rotation_indices);
  std::tuple<std::vector<Backend::Polynomial>, std::vector<Backend::Polynomial>>
  hoisted_basic_parallel_keyswitching_internal(
      const Backend::Ciphertext &input,
      const std::vector<int32_t> rotation_indices);

  PolyPair bsgs_giantstep_accumulate_keyswitch_iterations(
      const std::tuple<std::vector<Backend::Polynomial>,
                       std::vector<Backend::Polynomial>> &babysteps,
      const std::vector<Backend::Plaintext> &plaintexts,
      const std::vector<int32_t> giantstep_rotation_indices,
      const KernelGroup::SplitType &split, const uint32_t extension_size,
      const uint32_t dnum_glo, const uint8_t rescale_count);
  Backend::CiphertextVector bsgs_giantstep_accumulate_keyswitch_iterations_vec(
      const std::vector<Backend::CiphertextVector> &babysteps,
      const std::vector<Backend::Plaintext> &plaintexts,
      const std::vector<int32_t> giantstep_rotation_indices,
      const KernelGroup::SplitType &split, const uint32_t extension_size,
      const uint32_t dnum_glo, const uint8_t rescale_count);

  Backend::CiphertextVector rotate_multiply_accumulate_internal_vec(
      const Backend::CiphertextVector &input1,
      const std::vector<Backend::Plaintext> &plaintexts,
      const std::vector<int32_t> rotation_indices, const uint8_t rescale_count);
  Backend::CiphertextVector multiply_rotate_accumulate_internal_vec(
      const Backend::CiphertextVector &input1,
      const std::vector<Backend::Plaintext> &plaintexts,
      const std::vector<int32_t> rotation_indices, const uint8_t rescale_count);

  std::tuple<uint64_t, KernelGroup::SplitType>
  compute_keyswitch_split(KeySwitch::KeySwitchType key_switch_type,
                          uint64_t level);
  std::tuple<uint64_t, KernelGroup::SplitType>
  compute_keyswitch_split_aggregation(KeySwitch::KeySwitchType key_switch_type,
                                      uint64_t level);
  std::tuple<uint64_t, KernelGroup::SplitType>
  compute_keyswitch2_split(KeySwitch::KeySwitchType key_switch_type,
                           uint64_t level);

  // Fusion, scheduling, and diagnostics
  void cross_chip_communication_optimization_pass();
  void make_new_kernels();
  void split_horizontal_fusion();
  void fuse_kernels_horizontal();
  void topologically_sort_kernels();

  void print_graph_simple2(const std::string &name);

  void set_instruction_limbs();
  void fuse_kernels();
  void fuse_kernels_vertical();

  void print_limb_instructions();

  void set_temporary_values();

  void write_program_inputs();
  void write_istream_inputs(const std::string &inputs_file_name,
                            uint32_t partition_id);

  void do_next_use_analysis(Backend::KernelFusionContext &kernel_factory,
                            const uint32_t partition_id);

  void split_kernels_limbwise();
#ifdef CERIUM_USE_SPLIT_KERNELS_LIMBWISE_V2
  void split_kernels_limbwise_v2();
#endif
  void print_graph_split();

  // Digit splitting
  using DigitType = std::pair<Backend::TermIndexType, std::set<uint16_t>>;

  struct DigitHash {
    std::size_t operator()(const DigitType &digit) const {
      auto hash = std::hash<Backend::TermIndexType>()(digit.first);
      for (auto &d : digit.second) {
        hash ^= std::hash<uint16_t>()(d);
      }
      return hash;
    }
  };

  struct DigitCompare {
    std::size_t operator()(const DigitType &lhs, const DigitType &rhs) const {
      if (lhs.first != rhs.first) {
        return false;
      }
      return lhs.second == rhs.second;
    }
  };

  template <typename T>
  using DigitMap = std::unordered_map<DigitType, T, DigitHash, DigitCompare>;

  DigitMap<Backend::Polynomial> evalkey_digits_to_term;

  Backend::TermPtrMap<std::set<Backend::LimbIndexType>> program_io_local;
  Backend::TermPtrMap<std::set<Backend::LimbIndexType>> program_io;

  void split_digit_wise_internal(const KernelGroup::SplitType &split,
                                 std::shared_ptr<KernelGroup> &kg,
                                 DigitMap<Backend::Polynomial> &digit_map);
  void split_digit_wise_internal_mad(const KernelGroup::SplitType &split,
                                     std::shared_ptr<KernelGroup> &kg,
                                     DigitMap<Backend::Polynomial> &digit_map,
                                     bool dont_split_modular = true);

  void split_digit_wise(const KernelGroup::SplitType &,
                        std::shared_ptr<KernelGroup> &intt,
                        std::shared_ptr<KernelGroup> &drm,
                        std::shared_ptr<KernelGroup> &bconv,
                        std::shared_ptr<KernelGroup> &ntt,
                        std::shared_ptr<KernelGroup> &evkmul,
                        std::shared_ptr<KernelGroup> &evkmul_ext);
  void split_digit_wise(const KernelGroup::SplitType &,
                        std::shared_ptr<KernelGroup> &intt,
                        std::shared_ptr<KernelGroup> &bconv,
                        std::shared_ptr<KernelGroup> &ntt,
                        std::shared_ptr<KernelGroup> &evkmul,
                        std::shared_ptr<KernelGroup> &evkmul_ext,
                        const bool set_dont_split_modular = true);
  template <typename ReceiveCipherFn>
  Backend::Ciphertext
  receive_ciphertext_common(const Backend::Ciphertext &input1,
                            ReceiveCipherFn &&receive_cipher_operands);
  template <typename ReceiveFn>
  void receive_visit(const Frontend::Term::Ptr &term,
                     const Frontend::Term::Ptr &args1,
                     ReceiveFn &&receive_cipher);

  uint32_t SLOTS = 32768;

public:
  CeriumCompiler(Frontend::Function &f,
                   const uint8_t mod_partitions, const uint64_t num_vregs,
                   const std::string &output_prefix, const uint64_t num_bcus)
      : function_name(f.getName()), Objects(f),
        mod_partitions(mod_partitions), current_partition_size(mod_partitions),
        current_partition_id(0), num_vregs(num_vregs),
        output_prefix(output_prefix), num_bcus(num_bcus) {
    validate_partition_state(
        PartitionInfo{current_partition_id, current_partition_size});
    create_kernel_group();
  }

  void init(Frontend::Function &f) {}

  void operator()(Frontend::Function &f, const Frontend::Term::Ptr &term);

  void finish();
};

} // namespace Backend
} // namespace Cerium
