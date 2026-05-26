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

#include "cerium/runtime/execution/function.h"
#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/execution/device_utils.h"
#include "cerium/runtime/io/generator.h"
#include "cerium/runtime/utils/logger.h"

#include <fstream>
#include <iomanip>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <locale>
#include <regex>
#include <stdexcept>
#include <string>

#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <execution>
#include <optional>
#include <thread>

#include <dlfcn.h>

#include "cerium/runtime/cuda/memory_manager_cuda.h"
#include "cerium/runtime/utils/overloaded.h"
#include "cerium/runtime/utils/string.h"

namespace Cerium {
namespace Runtime {

CeriumFunction::RuntimeOptions CeriumFunction::RuntimeOptions::FromEnvironment() {
  RuntimeOptions options;
  options.use_uvm_everything = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_USE_UVM_EVERYTHING", 0);
  options.skip_copy_remapables = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_SKIP_COPY_REMAPABLES", 0);
  options.no_check_remapables = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_NO_CHECK_REMAPABLES", 0);
  if (const char *env_p =
          std::getenv("CERIUM_RUNTIME_NUM_REMAPABLE_INPUT_THREADS")) {
    options.num_remapable_input_threads = std::stoi(env_p);
  }
  options.remapables_use_explicit_copy = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_REMAPABLES_USE_EXPLICIT_COPY", 0);
  options.perform_uvm_prefetching = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_PERFORM_UVM_PREFETCHING", 1);
  options.perform_uvm_prefetching_back_to_host =
      Cerium::Runtime::Utils::get_uint_env_variable(
          "CERIUM_RUNTIME_PERFORM_UVM_PREFETCHING_BACK_TO_HOST", 0);
  options.num_iters =
      Cerium::Runtime::Utils::get_uint_env_variable("CERIUM_RUNTIME_NUM_ITERS", 1);
  options.use_prefetch_stream = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_USE_PREFETCH_STREAM", 1);
  options.prefetch_wait_per_function =
      Cerium::Runtime::Utils::get_uint_env_variable(
          "CERIUM_RUNTIME_PREFETCH_WAIT_PER_FUNCTION", 1);
  options.small_prefetch_lookahead_bytes =
      Cerium::Runtime::Utils::get_uint_env_variable(
          "CERIUM_RUNTIME_SMALL_PREFETCH_LOOKAHEAD_BYTES", 1024*1024*1024);
  options.all_ops_in_same_stream = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_ALL_OPS_IN_STREAM", 0);
  options.no_cross_chip_comm =
      Cerium::Runtime::Utils::get_uint_env_variable("CERIUM_RUNTIME_NO_COMM", 0);
  options.no_cross_chip_drm =
      Cerium::Runtime::Utils::get_uint_env_variable("CERIUM_RUNTIME_NO_DRM", 0);
  options.no_cross_chip_ags =
      Cerium::Runtime::Utils::get_uint_env_variable("CERIUM_RUNTIME_NO_AGS", 0);
  options.no_cross_chip_ard =
      Cerium::Runtime::Utils::get_uint_env_variable("CERIUM_RUNTIME_NO_ARD", 0);
  options.no_comm_false_deps = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_NO_COMM_FALSE_DEPS", 0);
  options.print_cuda_graph = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_PRINT_CUDA_GRAPH", 0);
  return options;
}

void CeriumFunction::update_runtime_stats() {
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            runtime_stats_.end - runtime_stats_.start)
                            .count();
  runtime_stats_.iterations = NUM_ITERS;
  runtime_stats_.total_ms = static_cast<double>(total_ms);
  runtime_stats_.average_ms =
      NUM_ITERS == 0 ? 0
                     : runtime_stats_.total_ms / static_cast<double>(NUM_ITERS);
}

void CeriumFunction::print_runtime_stats() const {
  std::cout << "------------------ Completed -------------------------\n";
  std::cout << "Iterations Run = " << runtime_stats_.iterations << "\n";
  std::cout << "Total Execution Time = " << runtime_stats_.total_ms
            << " milliseconds" << std::endl;
  std::cout << "Average Execution Time = " << runtime_stats_.average_ms
            << " milliseconds" << std::endl;
  std::cout << "------------------------------------------------------\n";
}

std::vector<std::string> split_string(const std::string &str,
                                      const std::string &delimiter) {
  std::vector<std::string> strings =
      Cerium::Runtime::Utils::split_string(str, delimiter);

  for (auto &str : strings) {
    auto pos = str.find(": ");
    if (pos != std::string::npos) {
      str = str.substr(pos + 2);
    }
  }
  if (!strings.empty()) {
    assert(strings[0][0] == 'f');
  }
  return strings;
}

CeriumFunction::LimbT::Element_t *
CeriumFunction::get_function_arg_memory_location_from_name(
    const size_t partition_id, const std::string &name) {
  return function_args_local_[partition_id]
      .at(function_args_name_map_[partition_id].at(name))
      ->data();
}

void CeriumFunction::load_function_map(std::string basename,
                                    const size_t num_partitions) {
  typedef const FunctionMapType &(*map_function_t)();
  auto get_function_map =
      [&](const Cerium::Runtime::Utils::DLHandler &handle) -> const FunctionMapType {
    std::string function_map_symbol;

    if (use_cugraph) {
      function_map_symbol = "__GET_FUNCTION_MAP_CUGRAPH__";
    } else {
      function_map_symbol = "__GET_FUNCTION_MAP__";
    }
    map_function_t map_function_ =
        (map_function_t)dlsym(handle.get(), function_map_symbol.c_str());
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      std::cerr << "Cannot load symbol '" << function_map_symbol
                << "': " << dlsym_error << '\n';
      throw std::runtime_error("DL Sym Error");
    }
    return (map_function_());
  };

  function_maps.resize(num_partitions);
  dl_handle.reserve(num_partitions);
  for (size_t tid = 0; tid < num_partitions; tid++) {
    CUDA::setDevice(tid);

    std::string filename =
        basename + std::to_string(tid % num_gpus_compiled) + ".so";
    dl_handle.emplace_back(filename);
    function_maps[tid] = get_function_map(dl_handle[tid]);
  }
}

void CeriumFunction::init_streams(const std::size_t num_partitions) {
  if (streams_.size() > 0) {
    return;
  }
  streams_.resize(num_partitions);
  for (size_t tid = 0; tid < num_partitions; tid++) {
    CUDA::setDevice(tid);
    streams_[tid] = CUDA::createStream();
  }
}

void CeriumFunction::init_prefetch_streams(const std::size_t num_partitions) {
  if (prefetch_streams_.size() > 0) {
    return;
  }
  prefetch_streams_.resize(num_partitions);
  for (size_t tid = 0; tid < num_partitions; tid++) {
    CUDA::setDevice(tid);
    prefetch_streams_[tid] = CUDA::createStream();
  }
}

void CeriumFunction::init_register_file(const std::size_t num_partitions,
                                     const std::size_t num_registers) {
  if (register_files.size() > 0) {
    return;
  }
  register_files.resize(num_partitions);
  register_files_memory_.resize(num_partitions);
  for (size_t tid = 0; tid < num_partitions; tid++) {
    CUDA::setDevice(tid);
    register_files_memory_[tid] = allocate_uint_device(
        context_.n() * num_registers, false /* These are never on uvm*/);
    register_files[tid] = register_files_memory_[tid].get();
  }
}

void CeriumFunction::init_base_conversion_units(const size_t num_partitions,
                                             const size_t num_bcu) {
  if (base_conversion_units.size() > 0) {
    return;
  }
  base_conversion_units.resize(num_partitions);
  base_conversion_units_memory_.resize(num_partitions);
  for (size_t tid = 0; tid < num_partitions; tid++) {
    base_conversion_units_memory_[tid].resize(num_bcu);
    base_conversion_units[tid].resize(num_bcu);
    CUDA::setDevice(tid);
    for (size_t j = 0; j < num_bcu; j++) {
      base_conversion_units_memory_[tid][j] =
          std::make_unique<Runtime::DeviceBaseConverter>(context_);
      base_conversion_units[tid][j] =
          base_conversion_units_memory_[tid][j].get();
    }
  }
}

CeriumFunction::Term::Term(std::string &&s) {
  auto pos = s.find("[X]");
  dead = false;
  if (pos != std::string::npos) {
    dead = true;
    s.erase(pos);
  }
  string = s;
  if (string[0] == 'b') {
    type = Type::Bcu;
  } else if (string[0] == 'r') {
    type = Type::Reg;
  } else if (string[0] == 'f') {
    type = Type::Arg;
  } else if (string[0] == 'Z') {
    type = Type::Zero;
  } else {
    type = Type::Mem;
  }
}

CeriumFunction::Term::Term(std::string &s) { *this = Term(std::move(s)); }

std::vector<CeriumFunction::Term>
CeriumFunction::parse_term_csv(std::string &&csv_string) {
  csv_string.erase(
      std::remove_if(csv_string.begin(), csv_string.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      csv_string.end());

  std::vector<Term> split;
  if (csv_string == "") {
    return split;
  }
  if (csv_string.back() == ',') {
    csv_string.pop_back();
  }
  if (csv_string == "") {
    return split;
  }
  size_t lpos = 0, rpos = 0;
  while ((rpos = csv_string.find(",", lpos)) != std::string::npos) {
    split.push_back(Term(csv_string.substr(lpos, rpos - lpos)));
    lpos = rpos + 1;
  }
  split.push_back(Term(csv_string.substr(lpos)));
  return split;
}

std::vector<int32_t>
CeriumFunction::parse_rotations_csv(std::string &&csv_string) {
  csv_string.erase(
      std::remove_if(csv_string.begin(), csv_string.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      csv_string.end());

  std::vector<int32_t> rotation_indices;
  if (csv_string == "") {
    return rotation_indices;
  }
  if (csv_string.back() == ',') {
    csv_string.pop_back();
  }
  if (csv_string == "") {
    return rotation_indices;
  }
  size_t lpos = 0, rpos = 0;
  while ((rpos = csv_string.find(",", lpos)) != std::string::npos) {
    rotation_indices.push_back(std::stoi(csv_string.substr(lpos, rpos - lpos)));
    lpos = rpos + 1;
  }
  rotation_indices.push_back(std::stoi(csv_string.substr(lpos)));
  return rotation_indices;
}
std::vector<uint32_t>
CeriumFunction::parse_divide_bases_csv(std::string &&csv_string) {
  csv_string.erase(
      std::remove_if(csv_string.begin(), csv_string.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      csv_string.end());

  std::vector<uint32_t> divide_bases;
  if (csv_string == "") {
    return divide_bases;
  }
  if (csv_string.back() == ',') {
    csv_string.pop_back();
  }
  if (csv_string == "") {
    return divide_bases;
  }
  size_t lpos = 0, rpos = 0;
  while ((rpos = csv_string.find(",", lpos)) != std::string::npos) {
    divide_bases.push_back(std::stoi(csv_string.substr(lpos, rpos - lpos)));
    lpos = rpos + 1;
  }
  divide_bases.push_back(std::stoi(csv_string.substr(lpos)));
  return divide_bases;
}

std::string CeriumFunction::parse_function_name(std::string &&csv_string) {
  csv_string.erase(
      std::remove_if(csv_string.begin(), csv_string.end(),
                     [](unsigned char ch) { return ch == ' ' || ch == '\n'; }),
      csv_string.end());
  if (!csv_string.empty() && csv_string.back() == ',') {
    csv_string.pop_back();
  }

  return csv_string;
}

std::unordered_map<std::string, CeriumFunction::Term>
CeriumFunction::parse_function_args_map(std::string &&csv_string) {
  csv_string.erase(std::remove_if(csv_string.begin(), csv_string.end(),
                                  [](unsigned char ch) {
                                    return ch == ' ' || ch == '\n' ||
                                           ch == '<' || ch == '>';
                                  }),
                   csv_string.end());
  std::unordered_map<std::string, CeriumFunction::Term> map;
  if (csv_string == "") {
    return map;
  }
  if (csv_string.back() == ',') {
    csv_string.pop_back();
  }
  if (csv_string == "") {
    return map;
  }
  auto add_entry = [&](const std::string &value) {
    if (value.empty()) {
      throw std::runtime_error("Invalid function args map entry: " +
                               csv_string);
    }
    const auto separator = value.find("~");
    if (separator == std::string::npos) {
      throw std::runtime_error("Invalid function args map format: " + value);
    }
    auto k = value.substr(0, separator);
    auto v = value.substr(separator + 1);
    if (k.empty() || v.empty()) {
      throw std::runtime_error("Invalid function args map format: " + value);
    }
    map.insert({k, CeriumFunction::Term(v)});
  };
  size_t lpos = 0, rpos = 0;
  while ((rpos = csv_string.find(",", lpos)) != std::string::npos) {
    auto value = csv_string.substr(lpos, rpos - lpos);
    lpos = rpos + 1;
    add_entry(value);
  }
  add_entry(csv_string.substr(lpos));
  return map;
}

std::vector<std::uint32_t>
CeriumFunction::parse_term_base_conv_vec(std::string &base_conv_string_multi,
                                      KernelType kernel_type,
                                      const size_t tid) {
  std::vector<uint32_t> ret;
  size_t pos;
  while ((pos = base_conv_string_multi.find(">")) != std::string::npos) {
    auto base_conv_string = base_conv_string_multi.substr(0, pos);
    base_conv_string_multi = base_conv_string_multi.substr(pos + 1);
    base_conv_string.erase(
        std::remove_if(base_conv_string.begin(), base_conv_string.end(),
                       [](unsigned char ch) {
                         return ch == ' ' || ch == '\n' || ch == '<';
                       }),
        base_conv_string.end());
    if (base_conv_string.empty()) {
      continue;
    }
    if (base_conv_string.back() == ',') {
      base_conv_string.pop_back();
    }

    if (base_conv_string == "") {
      continue;
    }
    auto strings = Cerium::Runtime::Utils::split_string(base_conv_string, ":");
    auto base_conv_index_str = strings[0];
    assert(base_conv_index_str[0] == 'b');
    base_conv_index_str[0] = '0';
    auto base_conv_index = std::stol(base_conv_index_str);
    std::vector<std::vector<size_t>> bases;
    for (size_t i = 1; i < strings.size(); i++) {
      std::vector<size_t> bases_;
      auto csv_string = strings[i];
      auto csv_values = Cerium::Runtime::Utils::split_string(csv_string, ",");
      for (auto &csv_value : csv_values) {
        bases_.push_back(std::stoul(csv_value));
      }
      bases.push_back(bases_);
    }
    if (kernel_type == KernelType::Generic || kernel_type == KernelType::Int ||
        kernel_type == KernelType::Drm) {
      base_conversion_units[tid]
          .at(base_conv_index)
          ->append_input_bases(bases[0]);
    } else if (kernel_type == KernelType::Bco) {
      base_conversion_units[tid]
          .at(base_conv_index)
          ->set_output_bases(bases[1]);
    }
    ret.push_back(base_conv_index);
  }
  return ret;
}

CeriumFunction::LimbT::Element_t
CeriumFunction::get_scalar_from_term(const Term &term) {
  return program_memory_scalar_.at(term.string);
}

size_t CeriumFunction::get_remapable_offset_from_term(const Term &term,
                                                   const size_t tid) {
  if (term.type == Term::Type::Mem) {
    auto it = remapable_program_memory_local_[tid].find(term.string);
    if (it != remapable_program_memory_local_[tid].end()) {
      return it->second;
    }
  }
  throw std::runtime_error("Invalid Term: " + term.string);
  return -1;
}

CeriumFunction::LimbT::Element_t *
CeriumFunction::get_pointer_from_term(const Term &term, bool create,
                                   const size_t tid) {
  if (term.type == Term::Type::Mem) {
    if (!create) {
      return program_memory_local_[tid].at(term.string)->data();
    }
    auto it = program_memory_local_[tid].find(term.string);
    if (it != program_memory_local_[tid].end()) {
      return it->second->data();
    }
    auto coeff_count_ = context_.n();
    auto destination_data = allocate_uint_device(context_.n(), use_uvm_);
    seal::Modulus modulus = context_.get_rns_modulus(0);
    auto destination = std::make_shared<LimbT>(std::move(destination_data),
                                               coeff_count_, 0, true);
    program_memory_local_[tid][term.string] = destination;
    return destination->data();
  } else if (term.type == Term::Type::Bcu) {
    size_t lpos = term.string.find("(");
    size_t rpos = term.string.find(")");
    auto limb_idx = std::stoul(term.string.substr(lpos + 1, rpos - lpos - 1));
    auto index = std::stoul(term.string.substr(1, lpos));
    return base_conversion_units[tid].at(index)->get_pointer(limb_idx);
  } else if (term.type == Term::Type::Reg) {
    auto reg = std::stoul(term.string.substr(1));
    if (reg >= registers) {
      throw std::runtime_error(
          "[" + function_name_ + "]: " + "Register: " + std::to_string(reg) +
          "exceeds register file size: " + std::to_string(registers));
    }
    return register_files[tid] + reg * coeff_count;
  } else if (term.type == Term::Type::Arg) {
    return function_args_local_[tid].at(term.string)->data();
  } else if (term.type == Term::Type::Zero) {
    return nullptr;
  }
  throw std::runtime_error("Invalid Term: " + term.string);
  return nullptr;
}

std::vector<CUDA::NodePtr>
CeriumFunction::get_true_deps_from_term(const Term &term, const size_t tid) {
  std::vector<CUDA::NodePtr> deps;
  if (term.type == Term::Type::Mem) {
    auto it_w = mem_last_write[tid].find(term.string);
    if (it_w != mem_last_write[tid].end()) {
      auto &dep = it_w->second;
      deps.push_back(dep);
    }
  } else if (term.type == Term::Type::Bcu) {
    size_t lpos = term.string.find("(");
    size_t rpos = term.string.find(")");
    auto limb_idx = std::stoul(term.string.substr(lpos + 1, rpos - lpos - 1));
    auto index = std::stoul(term.string.substr(1, lpos));
    auto &dep = bcu_last_write[tid][index];
    deps.push_back(dep);
  } else if (term.type == Term::Type::Reg) {
    auto reg = std::stoul(term.string.substr(1));
    const auto &dep = reg_last_write[tid][reg];
    deps.push_back(dep);
  } else if (term.type == Term::Type::Arg) {
    auto it_w = farg_last_write[tid].find(term.string);
    if (it_w != farg_last_write[tid].end()) {
      auto &dep = it_w->second;
      deps.push_back(dep);
    }
  }
  return deps;
}

std::vector<CUDA::NodePtr>
CeriumFunction::get_false_deps_from_term(const Term &term, const size_t tid) {
  std::vector<CUDA::NodePtr> deps;
  if (term.type == Term::Type::Mem) {
    auto it_w = mem_last_write[tid].find(term.string);
    if (it_w != mem_last_write[tid].end()) {
      deps.push_back(it_w->second);
    }
    auto it_r = mem_last_read[tid].find(term.string);
    if (it_r != mem_last_read[tid].end()) {
      deps.insert(deps.end(), it_r->second.begin(), it_r->second.end());
    }
  } else if (term.type == Term::Type::Bcu) {
    size_t lpos = term.string.find("(");
    size_t rpos = term.string.find(")");
    auto limb_idx = std::stoul(term.string.substr(lpos + 1, rpos - lpos - 1));
    auto index = std::stoul(term.string.substr(1, lpos));
    auto &dep_w = bcu_last_write[tid][index];
    deps.push_back(dep_w);
    auto &dep_r = bcu_last_read[tid][index];
    deps.insert(deps.end(), dep_r.begin(), dep_r.end());
  } else if (term.type == Term::Type::Reg) {
    auto reg = std::stoul(term.string.substr(1));
    const auto &dep_w = reg_last_write[tid][reg];
    deps.push_back(dep_w);
    auto &dep_r = reg_last_reads[tid][reg];
    deps.insert(deps.end(), dep_r.begin(), dep_r.end());
  } else if (term.type == Term::Type::Arg) {
    auto it_w = farg_last_write[tid].find(term.string);
    if (it_w != farg_last_write[tid].end()) {
      deps.push_back(it_w->second);
    }
    auto it_r = farg_last_read[tid].find(term.string);
    if (it_r != farg_last_read[tid].end()) {
      deps.insert(deps.end(), it_r->second.begin(), it_r->second.end());
    }
  }
  return deps;
}

void CeriumFunction::update_term_reads(const Term &term, const size_t tid,
                                    const CUDA::NodePtr &node) {
  if (node == nullptr) {
    return;
  }
  if (term.type == Term::Type::Mem) {
    mem_last_read[tid][term.string].push_back(node);
  } else if (term.type == Term::Type::Bcu) {
    size_t lpos = term.string.find("(");
    size_t rpos = term.string.find(")");
    auto limb_idx = std::stoul(term.string.substr(lpos + 1, rpos - lpos - 1));
    auto index = std::stoul(term.string.substr(1, lpos));
    bcu_last_read[tid][index].push_back(node);
  } else if (term.type == Term::Type::Reg) {
    auto reg = std::stoul(term.string.substr(1));
    reg_last_reads[tid][reg].push_back(node);
  } else if (term.type == Term::Type::Arg) {
    farg_last_read[tid][term.string].push_back(node);
  }
}

void CeriumFunction::update_term_writes(const Term &term, const size_t tid,
                                     const CUDA::NodePtr &node) {
  if (node == nullptr) {
    return;
  }
  if (term.type == Term::Type::Mem) {
    mem_last_write[tid][term.string] = node;
    mem_last_read[tid].erase(term.string);
  } else if (term.type == Term::Type::Bcu) {
    size_t lpos = term.string.find("(");
    size_t rpos = term.string.find(")");
    auto limb_idx = std::stoul(term.string.substr(lpos + 1, rpos - lpos - 1));
    auto index = std::stoul(term.string.substr(1, lpos));
    bcu_last_write[tid][index] = node;
    bcu_last_read[tid][index].clear();
  } else if (term.type == Term::Type::Reg) {
    auto reg = std::stoul(term.string.substr(1));
    reg_last_write[tid][reg] = node;
    reg_last_reads[tid][reg].clear();
  } else if (term.type == Term::Type::Arg) {
    farg_last_write[tid][term.string] = node;
    farg_last_read[tid].erase(term.string);
  }
}

void CeriumFunction::erase_term(const Term &term, const size_t tid) {
  if (!term.dead) {
    return;
  }
  if (term.type == Term::Type::Mem) {
    program_memory_local_[tid][term.string] = nullptr;
  } else if (term.type == Term::Type::Bcu) {
  }
}

void CeriumFunction::init_evaluators(const size_t num_partitions) {
  if (evaluators_.size() == num_partitions) {
    return;
  }
  if (evaluator_contexts_.size() != num_partitions) {
    evaluator_contexts_.clear();
    evaluator_contexts_.resize(num_partitions);
  }
  evaluators_.clear();
  LOG(logger, INFO) << "Creating Evaluators\n" << std::flush;
  for (size_t tid = 0; tid < num_partitions; tid++) {
    CUDA::setDevice(tid);
    if (!evaluator_contexts_[tid]) {
      evaluator_contexts_[tid] =
          std::make_shared<EvaluatorContext>(context_, tid, num_partitions);
    }
    evaluators_.emplace_back(
        std::make_shared<Evaluator>(context_, evaluator_contexts_[tid]));
  }
  LOG(logger, INFO) << "Creating Evaluators\n" << std::flush;
}

void CeriumFunction::init_graphs(const size_t num_partitions,
                              const size_t num_registers) {
  graphs_.clear();
  graphs_init_.clear();
  graph_exec_.resize(num_partitions);
  for (size_t i = 0; i < num_partitions; i++) {
    graphs_.push_back(CUDA::createGraph());
    graphs_init_.push_back(false);
  }
  reg_last_reads.resize(num_partitions);
  reg_last_write.resize(num_partitions);
  bcu_last_read.resize(num_partitions);
  bcu_last_write.resize(num_partitions);
  mem_last_read.resize(num_partitions);
  mem_last_write.resize(num_partitions);
  fn_last_use.resize(num_partitions);
  farg_last_read.resize(num_partitions);
  farg_last_write.resize(num_partitions);
  for (size_t i = 0; i < num_partitions; i++) {
    bcu_last_read[i].resize(num_bcus);
    bcu_last_write[i].resize(num_bcus);
  }
}
void CeriumFunction::init_comm_records(const size_t num_partitions,
                                    const size_t num_registers) {
  reg_last_reads_main_event.resize(num_partitions);
  reg_last_write_main_event.resize(num_partitions);

  reg_last_reads_comm_event.resize(num_partitions);
  reg_last_write_comm_event.resize(num_partitions);
}

std::vector<CeriumFunction::LimbDataType> CeriumFunction::compute_sud_factors(
    const size_t &tid, const std::optional<uint32_t> &bcu_id,
    const std::vector<std::uint32_t> &divide_bases) {

  auto rns_bases = context_.rns_bases();
  std::vector<LimbDataType> sud_factors(rns_bases.size());
  if (false && bcu_id.has_value()) {
    auto &bcu = base_conversion_units[tid].at(bcu_id.value());
    auto input_bases_prod = bcu->input_bases_product();
    auto input_bases_size = bcu->input_bases_size();
    const auto &output_base_ids = bcu->output_base_ids();
    for (auto i = 0; i < output_base_ids.size(); i++) {
      seal::Modulus modulus1 = context_.get_rns_modulus(output_base_ids[i]);
      auto input_bases_prod_mod =
          seal::util::modulo_uint(input_bases_prod, input_bases_size, modulus1);
      std::uint64_t input_bases_prod_mod_inv = 0;
      if (!seal::util::try_invert_uint_mod(input_bases_prod_mod, modulus1,
                                           input_bases_prod_mod_inv)) {
        throw std::invalid_argument("modular inverse failed");
      }
      sud_factors[output_base_ids[i]] = input_bases_prod_mod_inv;
    }
  } else {
    seal::util::Pointer<seal::util::RNSBase> input_rns_bases_;
    std::vector<seal::Modulus> input_rns_modulii;
    for (auto &id : divide_bases) {
      input_rns_modulii.push_back(context_.get_rns_modulus(id));
    }
    input_rns_bases_ = seal::util::allocate<seal::util::RNSBase>(
        pool_glo, input_rns_modulii, pool_glo);

    auto input_bases_prod = input_rns_bases_->base_prod();
    auto input_bases_size = input_rns_bases_->size();

    for (auto i = 0; i < rns_bases.size(); i++) {
      seal::Modulus modulus1 = context_.get_rns_modulus(i);
      auto input_bases_prod_mod =
          seal::util::modulo_uint(input_bases_prod, input_bases_size, modulus1);
      std::uint64_t input_bases_prod_mod_inv = 0;
      if (!seal::util::try_invert_uint_mod(input_bases_prod_mod, modulus1,
                                           input_bases_prod_mod_inv)) {
        continue;
      }
      sud_factors[i] = input_bases_prod_mod_inv;
    }
  }
  return sud_factors;
}
std::vector<CeriumFunction::LimbDataType>
CeriumFunction::compute_pmu_factors(const std::vector<std::uint32_t> &pmu_bases) {

  auto rns_bases = context_.rns_bases();
  std::vector<LimbDataType> pmu_factors(rns_bases.size());
  std::vector<seal::Modulus> pmu_rns_modulii;
  for (auto &id : pmu_bases) {
    pmu_rns_modulii.push_back(context_.get_rns_modulus(id));
  }
  auto pmu_rns_bases_ = seal::util::RNSBase(pmu_rns_modulii, pool_glo);

  auto pmu_bases_prod = pmu_rns_bases_.base_prod();
  auto pmu_bases_size = pmu_rns_bases_.size();

  for (auto i = 0; i < rns_bases.size(); i++) {
    seal::Modulus modulus = context_.get_rns_modulus(i);
    auto pmu_bases_prod_mod =
        seal::util::modulo_uint(pmu_bases_prod, pmu_bases_size, modulus);
    pmu_factors[i] = pmu_bases_prod_mod;
  }

  return pmu_factors;
}

CeriumFunction::KernelType
CeriumFunction::get_kernel_type(const std::string &kernel_type_string) {
  if (kernel_type_string == "generic") {
    return KernelType::Generic;
  } else if (kernel_type_string == "mov") {
    return KernelType::Mov;
  } else if (kernel_type_string == "int") {
    return KernelType::Int;
  } else if (kernel_type_string == "ntt") {
    return KernelType::Ntt;
  } else if (kernel_type_string == "sud") {
    return KernelType::Sud;
  } else if (kernel_type_string == "rsv") {
    return KernelType::Rsv;
  } else if (kernel_type_string == "bco") {
    return KernelType::Bco;
  } else if (kernel_type_string == "pmu") {
    return KernelType::Pmu;
  } else if (kernel_type_string == "drm") {
    return KernelType::Drm;
  } else if (kernel_type_string == "ags") {
    return KernelType::Ags;
  } else if (kernel_type_string == "ard") {
    return KernelType::Ard;
  } else if (kernel_type_string == "call") {
    return KernelType::Call;
  }
  throw std::runtime_error("Invalid Kernel Type");
  return KernelType::Generic;
}

size_t CeriumFunction::parse_num_comm_limbs(const std::string &str) {
  if (str.empty()) {
    return 0;
  }
  return std::stoul(str);
}

} // namespace Runtime
} // namespace Cerium
