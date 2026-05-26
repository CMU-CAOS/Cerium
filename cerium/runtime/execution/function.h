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

#include <chrono>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "cerium/runtime/context.h"
#include "cerium/runtime/io/ckks-encoder.h"
#include "cerium/runtime/io/ckks-encryptor.h"
#include "cerium/runtime/polynomial/limb.h"

#include "cerium/runtime/cuda/device_limb.h"
#include "cerium/runtime/execution/evaluator.h"
#include "cerium/runtime/utils/dlhandler.h"
#include "cerium/runtime/utils/logger.h"
#include "cerium/runtime/utils/map_wrapper.h"
#include "cerium/runtime/utils/string.h"

namespace Cerium {
namespace Runtime {

std::vector<std::string> split_string(const std::string &str,
                                      const std::string &delimiter);

class CeriumFunction {

public:
  using LimbT = DeviceLimb;
  using LimbPtrT = DeviceLimbPtr;

  using RegisterFileType = std::vector<std::shared_ptr<LimbT>>;
  using ScalarRegisterFileType = std::vector<LimbT::Element_t>;

  struct RuntimeOptions {
    bool use_uvm_everything{false};
    bool skip_copy_remapables{false};
    bool no_check_remapables{false};
    std::optional<int> num_remapable_input_threads;
    bool remapables_use_explicit_copy{false};
    bool perform_uvm_prefetching{true};
    bool perform_uvm_prefetching_back_to_host{false};
    uint64_t num_iters{1};
    bool use_prefetch_stream{true};
    // When using one explicit-copy staging buffer per called function, wait
    // only for that function's previous consumer before overwriting it.
    bool prefetch_wait_per_function{false};
    // Queue one additional prefetch when the first copy is below this size.
    // Zero disables the extra lookahead.
    uint64_t small_prefetch_lookahead_bytes{0};
    bool all_ops_in_same_stream{false};
    bool no_cross_chip_comm{false};
    bool no_cross_chip_drm{false};
    bool no_cross_chip_ags{false};
    bool no_cross_chip_ard{false};
    bool no_comm_false_deps{false};
    bool print_cuda_graph{false};

    static RuntimeOptions FromEnvironment();
  };

  struct RuntimeStats {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    uint64_t iterations{0};
    double total_ms{0};
    double average_ms{0};
  };

  CeriumFunction(const Context &context, const std::string &function_name,
              const bool use_uvm)
      : context_(context), coeff_count(context_.n()),
        function_name_(function_name),
        logger("[CeriumFunction:" + function_name + "]") {
    runtime_options_ = RuntimeOptions::FromEnvironment();
    use_uvm_ = use_uvm || runtime_options_.use_uvm_everything;
  }
  void generate_inputs(const std::string &inputs_file_name,
                       const std::string &evalkeys_file_name,
                       const std::string &plaintexts_file_name,
                       const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs);
  void copy_inputs_to_device(const std::string &local_inputs_file_base,
                             size_t num_partitons);
  void
  copy_ciphertext_inputs_to_gpu(
    const std::string &local_inputs_file_base, size_t num_partitions,
    const std::unordered_map<std::string, std::pair<RnsPolynomialPtr,RnsPolynomialPtr>> & encrypted_inputs);


  void set_program_rf(const std::string &instruction_file_,
                      const std::string &code_file_, const uint8_t partitions_,
                      const uint32_t num_gpus_compiled_,
                      const uint32_t registers_,
                      const std::vector<LimbT::Element_t *> &register_files_,
                      const uint32_t num_bcus_,
                      const std::vector<std::vector<DeviceBaseConverter *>>
                          base_conversion_units_,
                      const std::vector<std::shared_ptr<EvaluatorContext>>
                          &evaluator_contexts) {
    instruction_file_base = instruction_file_;
    code_file_base = code_file_;
    partitions = partitions_;
    if (num_gpus_compiled_ == 0) {
      num_gpus_compiled = partitions_;
    } else {
      num_gpus_compiled = num_gpus_compiled_;
    }
    registers = registers_;
    register_files = register_files_;
    assert(register_files.size() == partitions);

    base_conversion_units = base_conversion_units_;
    assert(base_conversion_units.size() == partitions);
    evaluator_contexts_ = evaluator_contexts;
    remapable_program_memory_local_.resize(partitions);
    num_bcus = num_bcus_;
  }
  void run_program_multithread_threadfn(const size_t tid,
                                        const CUDA::StreamPtr &stream,
                                        const CUDA::StreamPtr &prefetch_stream);

  void prefetch_threadfn(const std::size_t tid, const CUDA::StreamPtr &stream,
                         const CUDA::StreamPtr &prefetch_stream,
                         size_t prefetch_idx,
                         const CUDA::EventPtr &wait_for_event);

  void run_program_multithread();

  void create_program_multithread();
  void create_cuda_graph();
  void run_program_multithread_cugraph();
  const RuntimeStats &runtime_stats() const { return runtime_stats_; }

  void generate_remapable_inputs(const std::string &remapables_file_name,
                                 const std::string &plaintexts_file_name,
                                 const std::string &map_key,
                                 bool remapables_use_uvm = false);
  void generate_remapable_inputs_and_copy_to_device_batch(
      const std::string &remapables_file_name,
      const std::vector<std::string> &plaintexts_file_names,
      const std::vector<std::string> &map_keys,
      const std::string &local_remapable_file_base, size_t num_partitions,
      bool remapables_use_uvm);

  void copy_remapable_inputs_to_device(
      const std::string &local_remapable_file_base, size_t num_partitions,
      const std::string &map_key, bool remapables_use_uvm = false);

  void create_remapable_offsets(const std::string &local_remapable_file_base,
                                size_t num_gpus);

  void add_function_cerium_function(
      const std::string &function_name,
      std::shared_ptr<CeriumFunction> cerium_function) {
    called_cerium_functions_[function_name] = cerium_function;
  }

  std::map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
  get_program_outputs();

  auto function_name() { return function_name_; }

private:
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

    EvalkeyInfo(std::string key_info);
  };
  struct EvalKeyEntry {
    std::string term;
    EvalkeyInfo info;
    std::vector<uint32_t> rns_base_ids;
  };


  struct Term {
    enum Type { Mem, Reg, Bcu, Arg, Zero } type;
    std::string string;
    bool dead{false};
    Term(std::string &term);
    Term(std::string &&term);
  };

  enum KernelType {
    Generic,
    Int,
    Ntt,
    Sud,
    Rsv,
    Bco,
    Mov,
    Pmu,
    Drm,
    Ags,
    Ard,
    Call,
  };

  struct FunctionMetaData {

    bool valid{false};
    size_t function_id;
    KernelType kernel_type;
    std::vector<Term> outputs;
    std::vector<Term> inputs;
    std::vector<Term> scalars;
    std::vector<Term> remapables;
    std::string base_conv_string;
    std::vector<int32_t> rotations;
    std::vector<uint32_t> divide_bases;
    std::vector<uint32_t> pmu_bases;
    std::vector<uint32_t> rsv_bases;

    size_t comm_size;
    size_t num_comm_limbs;

    std::vector<Term> scatter;
    std::vector<Term> gather;

    std::string function_name;
    std::unordered_map<std::string, Term> function_args_map;

    std::vector<uint32_t> bases;

    CeriumFunction *cerium_function{nullptr};

    FunctionMetaData(const std::string &instruction, CeriumFunction *cerium_function)
        : cerium_function(cerium_function) {

      auto split = split_string(instruction, "\n");
      if (split.empty()) {
        valid = false;
        return;
      }
      const auto required_items = static_cast<size_t>(FunctionItems::bases) + 1;
      if (split.size() < required_items) {
        throw std::runtime_error(
            "Invalid function metadata: expected at least " +
            std::to_string(required_items) + " fields, got " +
            std::to_string(split.size()) + " in instruction: " + instruction);
      }

      function_id =
          std::stoul(split[static_cast<size_t>(FunctionItems::function_id)]);
      kernel_type = cerium_function->get_kernel_type(
          split[static_cast<size_t>(FunctionItems::kernel_type)]);
      outputs = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::outputs)]));
      inputs = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::inputs)]));
      scalars = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::scalars)]));
      remapables = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::remapables)]));
      rotations = cerium_function->parse_rotations_csv(
          std::string(split[static_cast<size_t>(FunctionItems::rotations)]));
      base_conv_string = std::string(
          split[static_cast<size_t>(FunctionItems::base_conv_string)]);
      divide_bases = cerium_function->parse_divide_bases_csv(
          std::string(split[static_cast<size_t>(FunctionItems::divide_bases)]));
      pmu_bases = cerium_function->parse_divide_bases_csv(
          std::string(split[static_cast<size_t>(FunctionItems::pmu_bases)]));
      rsv_bases = cerium_function->parse_divide_bases_csv(
          std::string(split[static_cast<size_t>(FunctionItems::rsv_bases)]));

      comm_size = cerium_function->parse_num_comm_limbs(
          std::string(split[static_cast<size_t>(FunctionItems::comm_size)]));
      num_comm_limbs = cerium_function->parse_num_comm_limbs(std::string(
          split[static_cast<size_t>(FunctionItems::num_comm_limbs)]));
      scatter = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::scatter)]));
      gather = cerium_function->parse_term_csv(
          std::string(split[static_cast<size_t>(FunctionItems::gather)]));

      function_name = cerium_function->parse_function_name(std::string(
          split[static_cast<size_t>(FunctionItems::function_name)]));
      function_args_map = cerium_function->parse_function_args_map(std::string(
          split[static_cast<size_t>(FunctionItems::function_args_map)]));
      bases = cerium_function->parse_divide_bases_csv(
          std::string(split[static_cast<size_t>(FunctionItems::bases)]));

      valid = true;
    }

  private:
    enum class FunctionItems {
      function_id = 1,
      kernel_type,
      outputs,
      inputs,
      scalars,
      remapables,
      base_conv_string,
      rotations,
      divide_bases,
      pmu_bases,
      rsv_bases,
      comm_size,
      num_comm_limbs,
      scatter,
      gather,
      function_name,
      function_args_map,
      bases,
    };
  };

  struct KernelArgs {
    KernelType kernel_type;
    uint32_t function_id;
    std::vector<void *> args;
    CUDA::EventPtr event;
    CUDA::EventPtr event_end;
    CUDA::EventPtr wait_for_event;
  };

  std::vector<std::vector<KernelArgs>> kernel_args_;

  struct PrefetchArgs {
    CeriumFunction *cerium_function;
    void *pointer;
    size_t size;
    CUDA::EventPtr event;
    CUDA::EventPtr wait_for_event;
  };

  std::vector<std::vector<PrefetchArgs>> prefetch_args_;

  // The most recent main-stream event that protects each called function's
  // explicit-copy staging buffer, indexed by partition.
  std::vector<std::unordered_map<CeriumFunction *, CUDA::EventPtr>>
      prefetch_last_use_events_;
  // Functions with a copy queued but not yet consumed. A function has one
  // explicit-copy staging buffer, so it cannot be queued twice concurrently.
  std::vector<std::unordered_set<CeriumFunction *>> prefetch_queued_functions_;

  std::vector<CUDA::StreamPtr> streams_;
  std::vector<CUDA::StreamPtr> prefetch_streams_;
  std::vector<CUDA::GraphPtr> graphs_;
  std::vector<CUDA::GraphExecPtr> graph_exec_;
  std::vector<uint8_t> graphs_init_;
  // Whether each partition's CUDA graph contains an NCCL collective node.
  std::vector<uint8_t> graphs_have_collectives_;

  const Context &context_;
  std::size_t coeff_count;
  std::vector<std::shared_ptr<Runtime::EvaluatorContext>>
      evaluator_contexts_;
  std::vector<std::shared_ptr<Runtime::Evaluator>> evaluators_;
  bool use_cugraph{false};

  std::string function_name_;

  std::vector<std::unordered_map<std::string, std::shared_ptr<LimbT>>>
      program_memory_local_;
  std::unordered_map<std::string, std::shared_ptr<Limb>> program_memory_;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, RnsPolynomialPtr>>
      remapable_program_memory_;
  std::unordered_map<std::string, LimbT::Element_t> program_memory_scalar_;

  std::vector<std::unordered_map<std::string, std::shared_ptr<LimbT>>>
      function_args_input_local_;
  std::vector<std::unordered_map<std::string, std::shared_ptr<LimbT>>>
      function_args_output_local_;
  std::vector<std::unordered_map<std::string, std::shared_ptr<LimbT>>>
      function_args_local_;
  std::vector<std::unordered_map<std::string, std::string>>
      function_args_name_map_;
  std::vector<std::unordered_map<std::string, size_t>>
      remapable_program_memory_local_;

  std::vector<DevicePointer<LimbT::Element_t *>> remapable_base_;
  std::unordered_map<std::string, std::vector<DevicePointer<LimbT::Element_t>>>
      remapable_pointers_;

  std::vector<DevicePointer<LimbT::Element_t>>
      remapable_pointers_explicit_copy_;

  std::vector<std::unordered_set<std::string>> program_outputs_local_;
  struct ProgramOutputEntry {
    std::string c0_term;
    std::string c1_term;
    std::vector<uint32_t> rns_base_ids;
  };

  std::unordered_map<std::string, ProgramOutputEntry> program_outputs_;
  RuntimeStats runtime_stats_;

  std::unordered_map<std::string, std::shared_ptr<CeriumFunction>>
      called_cerium_functions_;

  std::mutex sync_lock;
  std::mutex mem_lock;

  std::string instruction_file_base;
  std::string code_file_base;
  uint32_t partitions;
  uint32_t num_gpus_compiled;
  uint32_t registers;
  uint32_t num_bcus;

  void handle_evalkey_stream(std::ifstream &input_file,
                             const std::string &evalkeys_file_name);
  EvalKeyEntry parse_evalkey(std::string &line);
  std::string evalkey_header = "Cerium::Evalkeys\n";
  std::string plaintext_header = "Cerium::Plaintexts\n";

  bool get_boolean_env_variable(const std::string &var);

  // These functions could be made as object independent utils
  KernelType get_kernel_type(const std::string &kernel_type_string);
  std::vector<Term> parse_term_csv(std::string &&csv_string);
  std::vector<uint32_t> parse_term_base_conv_vec(std::string &csv_string,
                                                 KernelType type,
                                                 const size_t tid);
  std::vector<int32_t> parse_rotations_csv(std::string &&csv_string);
  std::vector<uint32_t> parse_divide_bases_csv(std::string &&csv_string);
  size_t parse_num_comm_limbs(const std::string &str);

  std::string parse_function_name(std::string &&csv_string);
  std::unordered_map<std::string, Term>
  parse_function_args_map(std::string &&csv_string);
  LimbT::Element_t *
  get_function_arg_memory_location_from_name(const size_t partition_id,
                                             const std::string &);

  std::vector<DevicePointer<uint32_t>> rotate_permutations_device_;

  LimbT::Element_t get_scalar_from_term(const Term &term);
  LimbT::Element_t *get_pointer_from_term(const Term &term, bool create,
                                          const size_t tid);
  size_t get_remapable_offset_from_term(const Term &term, const size_t tid);
  void erase_term(const Term &term, const size_t tid);

  std::vector<DevicePointer<LimbT::Element_t>> register_files_memory_;
  std::vector<LimbT::Element_t *> register_files;
  std::vector<std::vector<std::unique_ptr<Runtime::DeviceBaseConverter>>>
      base_conversion_units_memory_;
  std::vector<std::vector<Runtime::DeviceBaseConverter *>>
      base_conversion_units;

  void init_streams(const std::size_t num_partitions);
  void init_prefetch_streams(const std::size_t num_partitions);
  void init_register_file(const std::size_t num_partitions,
                          const std::size_t num_registers);
  void init_base_conversion_units(const size_t num_partitions,
                                  const size_t num_bcu);

  void init_evaluators(const size_t num_partitions);
  void init_graphs(const size_t num_partitions, const size_t num_registers);

  void init_comm_records(const size_t num_partitions,
                         const size_t num_registers);

  using FunctionMapType = std::unordered_map<std::uint32_t, void *>;
  std::vector<Cerium::Runtime::Utils::DLHandler> dl_handle;
  std::vector<FunctionMapType> function_maps;
  void load_function_map(std::string filename, const size_t num_partitions);

  using LimbDataType = LimbT::Element_t;
  std::vector<LimbDataType>
  compute_sud_factors(const size_t &tid, const std::optional<uint32_t> &bcu_id,
                      const std::vector<std::uint32_t> &divide_bases);
  std::vector<LimbDataType>
  compute_pmu_factors(const std::vector<std::uint32_t> &divide_bases);
  auto compute_rsv_factors(const std::vector<uint32_t> &dest_rns_base_ids);

  struct DependencyHash {
    std::size_t
    operator()(const std::pair<CUDA::NodePtr, CUDA::NodePtr> &p) const {
      auto hash1 = std::hash<const void *>{}(p.first.get());
      auto hash2 = std::hash<const void *>{}(p.second.get());
      return hash1 ^ (hash2 << 1);
    }
  };
  struct DependencyEqual {
    bool operator()(const std::pair<CUDA::NodePtr, CUDA::NodePtr> &p1,
                    const std::pair<CUDA::NodePtr, CUDA::NodePtr> &p2) const {
      return (p1.first.get() == p2.first.get()) &&
             (p1.second.get() == p2.second.get());
    }
  };
  using DependencySet =
      std::unordered_set<std::pair<CUDA::NodePtr, CUDA::NodePtr>,
                         DependencyHash, DependencyEqual>;

  std::vector<std::unordered_map<size_t, std::vector<CUDA::NodePtr>>>
      reg_last_reads;
  std::vector<std::unordered_map<size_t, CUDA::NodePtr>> reg_last_write;
  std::vector<std::vector<std::vector<CUDA::NodePtr>>> bcu_last_read;
  std::vector<std::vector<CUDA::NodePtr>> bcu_last_write;

  std::vector<std::unordered_map<size_t, std::vector<size_t>>>
      reg_last_reads_main_event;
  std::vector<std::unordered_map<size_t, size_t>> reg_last_write_main_event;

  std::vector<std::unordered_map<size_t, std::vector<size_t>>>
      reg_last_reads_comm_event;
  std::vector<std::unordered_map<size_t, size_t>> reg_last_write_comm_event;

  std::vector<std::unordered_map<std::string, CUDA::NodePtr>> fn_last_use;

  std::vector<std::unordered_map<std::string, std::vector<CUDA::NodePtr>>>
      mem_last_read;
  std::vector<std::unordered_map<std::string, CUDA::NodePtr>> mem_last_write;

  std::vector<std::unordered_map<std::string, std::vector<CUDA::NodePtr>>>
      farg_last_read;
  std::vector<std::unordered_map<std::string, CUDA::NodePtr>> farg_last_write;

  std::vector<CUDA::NodePtr> get_true_deps_from_term(const Term &term,
                                                     const size_t tid);
  std::vector<CUDA::NodePtr> get_false_deps_from_term(const Term &term,
                                                      const size_t tid);
  void update_term_reads(const Term &term, const size_t tid,
                         const CUDA::NodePtr &node);
  void update_term_writes(const Term &term, const size_t tid,
                          const CUDA::NodePtr &node);

  void update_runtime_stats();
  void print_runtime_stats() const;

  uint64_t NUM_ITERS = 1;

  seal::MemoryPoolHandle pool_glo = seal::MemoryManager::GetPool();

  size_t NUM_BASE_CONV_UNITS = 128;

  enum class RemapablesType { Device, UVM, ExplicitCopy };

  bool remapable_offsets_set_ = false;
  bool use_uvm_ = false;
  RuntimeOptions runtime_options_;
  RemapablesType remapables_type_ = RemapablesType::Device;
  std::vector<size_t> remapables_size_;

  Cerium::Runtime::Utils::Logger logger;
};

} // namespace Runtime
} // namespace Cerium
