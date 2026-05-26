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

#include "cerium/runtime/execution/program.h"
#include "cerium/runtime/utils/string.h"

#include <cctype>
#include <fstream>

namespace Cerium {
namespace Runtime {

namespace {

std::vector<std::pair<std::string, std::string>> parse_remapable_keys(
    const std::string &serialized_keys) {
  std::vector<std::pair<std::string, std::string>> result;
  size_t pos = 0;
  auto skip_whitespace = [&]() {
    while (pos < serialized_keys.size() &&
           std::isspace(static_cast<unsigned char>(serialized_keys[pos]))) {
      ++pos;
    }
  };
  auto invalid_config = []() {
    throw std::runtime_error("Invalid remapable_keys config");
  };
  auto parse_string = [&]() {
    skip_whitespace();
    if (pos >= serialized_keys.size() ||
        (serialized_keys[pos] != '\'' && serialized_keys[pos] != '"')) {
      invalid_config();
    }
    const auto quote = serialized_keys[pos++];
    const auto start = pos;
    const auto end = serialized_keys.find(quote, pos);
    if (end == std::string::npos) {
      invalid_config();
    }
    pos = end + 1;
    return serialized_keys.substr(start, end - start);
  };

  skip_whitespace();
  if (pos >= serialized_keys.size() || serialized_keys[pos++] != '{') {
    invalid_config();
  }
  while (true) {
    skip_whitespace();
    if (pos < serialized_keys.size() && serialized_keys[pos] == '}') {
      ++pos;
      return result;
    }
    auto map_key = parse_string();
    skip_whitespace();
    if (pos >= serialized_keys.size() || serialized_keys[pos++] != ':') {
      invalid_config();
    }
    auto plaintexts_file_name = parse_string();
    result.emplace_back(std::move(map_key), std::move(plaintexts_file_name));
    skip_whitespace();
    if (pos >= serialized_keys.size()) {
      invalid_config();
    }
    if (serialized_keys[pos] == ',') {
      ++pos;
    } else if (serialized_keys[pos] == '}') {
      ++pos;
      return result;
    } else {
      invalid_config();
    }
  }
}

} // namespace

std::vector<std::string> Program::get_calls(const std::string &function_name,
                                            const std::string &function_dir) {

  std::string file_name = directory_base + "/" + function_dir + "/calls";
  std::ifstream file(file_name);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + file_name);
  }
  std::string line;
  bool completed = false;
  std::vector<std::string> calls;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
    line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
    if (line == "{") {
      continue;
    }
    if (line == "}") {
      completed = true;
      break;
    }
    calls.push_back(line);
  }
  return calls;
}

std::map<std::string, std::string>
Program::get_config(const std::string &function_name,
                    const std::string &function_dir) {

  std::string file_name =
      directory_base + "/" + function_dir + "/compile_config";
  std::ifstream file(file_name);
  if (!file.is_open()) {
    LOG(logger, WARN) << "Cannot open file: " << file_name << "\n"
                      << std::flush;
    return {};
  }
  std::map<std::string, std::string> config;
  std::string line;
  bool completed = false;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
    line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
    if (line == "{") {
      continue;
    }
    if (line == "}") {
      completed = true;
      break;
    }
    auto parse = Cerium::Runtime::Utils::split_string(line, ":");
    config[parse[0]] = parse[1];
  }
  return config;
}

std::map<std::string, std::shared_ptr<CeriumFunction>> Program::make_cerium_functions(
    uint32_t num_gpus,
    const std::map<std::string, std::map<std::string, std::string>> &config,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_wrapper) {

  std::vector<std::string> functions;
  functions.emplace_back(top_level_function);

  struct FunctionNode {
    size_t depth{0};
    size_t depth_vregs{0};
    size_t vregs{0};
    size_t num_gpus_compiled{0};
    size_t depth_bcu{0};
    size_t num_bcus{0};
    std::vector<std::string> calls;

    FunctionNode(){};
    FunctionNode(size_t depth, size_t depth_vregs, size_t depth_bcu)
        : depth(depth), depth_vregs(depth_vregs), depth_bcu(depth_bcu){};
  };

  std::map<std::string, FunctionNode> function_metadata;
  function_metadata[top_level_function] = FunctionNode(0, 0, 0);

  while (!functions.empty()) {
    auto function_name = functions.back();
    functions.pop_back();
    auto calls = get_calls(function_name, function_name);
    auto function_config_map = get_config(function_name, function_name);
    call_graph[function_name] = calls;
    size_t register_file_size = 4096;
    size_t num_baseconversion_units = 128;
    size_t num_gpus_compiled = num_gpus;

    auto function_config = config.find(function_name);
    if (function_config != config.end()) {
      for (auto &[it_key, it_value] : function_config->second) {
        function_config_map[it_key] = it_value;
      }
    }

    auto it = function_config_map.find("vregs");
    if (it != function_config_map.end()) {
      register_file_size = static_cast<size_t>(std::stoul(it->second));
    }
    it = function_config_map.find("bcus");
    if (it != function_config_map.end()) {
      num_baseconversion_units = static_cast<size_t>(std::stoul(it->second));
    }
    it = function_config_map.find("gpus");
    if (it != function_config_map.end()) {
      num_gpus_compiled = static_cast<size_t>(std::stoul(it->second));
    }

    LOG(logger, INFO) << "Making CeriumFunction: " << function_name
                      << ": vregs=" << register_file_size
                      << " bcus=" << num_baseconversion_units
                      << " num_gpus_compiled=" << num_gpus_compiled << "\n"
                      << std::flush;
    auto &metadata = function_metadata.at(function_name);
    metadata.vregs = register_file_size;
    metadata.num_bcus = num_baseconversion_units;
    metadata.num_gpus_compiled = num_gpus_compiled;
    auto fn_depth = metadata.depth;
    auto fn_depth_vregs = metadata.depth_vregs;
    auto fn_depth_bcu = metadata.depth_bcu;
    for (auto &c : calls) {
      functions.emplace_back(c);
      if (function_metadata.find(c) == function_metadata.end()) {
        function_metadata[c].depth = fn_depth + 1;
        function_metadata[c].depth_vregs = fn_depth_vregs + register_file_size;
        function_metadata[c].depth_bcu =
            fn_depth_bcu + num_baseconversion_units;
      } else {
        function_metadata[c].depth =
            std::max(fn_depth + 1, function_metadata.at(c).depth);
        function_metadata[c].depth_vregs =
            std::max(fn_depth_vregs + register_file_size,
                     function_metadata.at(c).depth_vregs);
        function_metadata[c].depth_bcu =
            std::max(fn_depth_bcu + num_baseconversion_units,
                     function_metadata.at(c).depth_bcu);
      }
    }
    metadata.calls = calls;
  }

  size_t max_depth_vregs = 0;
  size_t max_depth_bcu = 0;
  for (auto &[function_name, metadata] : function_metadata) {
    max_depth_vregs =
        std::max(metadata.depth_vregs + metadata.vregs, max_depth_vregs);
    max_depth_bcu =
        std::max(metadata.depth_bcu + metadata.num_bcus, max_depth_bcu);
    call_graph_by_depth[metadata.depth].push_back(function_name);
  }

  register_files_memory_.resize(num_gpus);
  base_conversion_units_memory_.resize(num_gpus);

  bool use_uvm_everything = Cerium::Runtime::Utils::get_uint_env_variable(
      "CERIUM_RUNTIME_USE_UVM_EVERYTHING", 0);

  CUDA::NCCL::ncclInit(num_gpus);
  evaluator_contexts.resize(num_gpus);
  auto thread_fn = [&](size_t tid) {
    CUDA::setDevice(tid);
    LOG(logger, INFO) << "[" << tid << "]"
                      << ": Allocating Register File Memory: size = "
                      << max_depth_vregs << " vregs\n"
                      << std::flush;
    LOG(logger, INFO) << "[" << tid << "]"
                      << ": Allocating BCUs: size = " << max_depth_bcu
                      << " bcus\n"
                      << std::flush;
    register_files_memory_[tid] =
        allocate_uint_device(context_.n() * max_depth_vregs,
                             use_uvm_everything /* These are never on uvm*/);
    base_conversion_units_memory_[tid].resize(max_depth_bcu);
    for (int j = 0; j < max_depth_bcu; j++) {
      base_conversion_units_memory_[tid][j] =
          std::make_unique<Runtime::DeviceBaseConverter>(context_);
    }
    evaluator_contexts[tid] =
        std::make_shared<EvaluatorContext>(context_, tid, num_gpus);
  };
  std::vector<std::thread> threads;
  for (auto tid = 0; tid < num_gpus; tid++) {
    threads.emplace_back(thread_fn, tid);
  }
  for (auto tid = 0; tid < num_gpus; tid++) {
    threads[tid].join();
  }

  for (auto &[function_name, metadata] : function_metadata) {
    bool use_uvm = false;
    bool generate_plaintexts = false;
    bool generate_evalkeys = false;
    auto cerium_function =
        std::make_shared<CeriumFunction>(context_, function_name, false);
    auto function_base_dir = directory_base + "/" + function_name;
    std::vector<LimbT::Element_t *> register_files;
    std::vector<std::vector<DeviceBaseConverter *>> base_conversion_units;
    register_files.resize(num_gpus);
    base_conversion_units.resize(num_gpus);
    for (auto i = 0; i < num_gpus; i++) {
      register_files[i] =
          register_files_memory_[i].get() + context_.n() * metadata.depth_vregs;
      base_conversion_units[i].resize(metadata.num_bcus);
      for (int j = 0; j < metadata.num_bcus; j++) {
        base_conversion_units[i][j] =
            base_conversion_units_memory_[i][j + metadata.depth_bcu].get();
      }
    }
    cerium_function->generate_inputs(
        function_base_dir + "/program_inputs", function_base_dir + "/evalkeys",
        function_base_dir + "/plaintexts", raw_inputs_wrapper);
    cerium_function->set_program_rf(
        function_base_dir + "/functions", function_base_dir + "/function_map_",
        num_gpus, metadata.num_gpus_compiled, metadata.vregs, register_files,
        metadata.num_bcus, base_conversion_units, evaluator_contexts);
    cerium_functions[function_name] = cerium_function;
  }
  for (auto &[function_name, metadata] : function_metadata) {
    for (auto &c : metadata.calls) {
      cerium_functions.at(function_name)
          ->add_function_cerium_function(std::move(std::string(c)),
                                     cerium_functions.at(c));
    }
  }

  return cerium_functions;
}
std::map<std::string, std::shared_ptr<CeriumFunction>> Program::make_program(
    uint32_t num_gpus,
    const std::map<std::string, std::map<std::string, std::string>> &config,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_wrapper) {

     auto cerium_functions =  make_cerium_functions(num_gpus, config, raw_inputs_wrapper);

  for (auto &[k, fns] : call_graph_by_depth) {
    for (auto &function_name : fns) {
      auto &cerium_function = cerium_functions.at(function_name);
      auto function_base_dir = directory_base + "/" + function_name;
      cerium_function->copy_inputs_to_device(function_base_dir + "/inputs",
                                         num_gpus);
      cerium_function->create_remapable_offsets(function_base_dir + "/inputs_remapable",
                                         num_gpus);
      auto function_config = config.find(function_name);
      if (function_config != config.end()) {
        const auto &function_config_map = function_config->second;
        const auto remapable_base =
            function_config_map.find("remapable_inputs_base");
        const auto remapable_keys = function_config_map.find("remapable_keys");
        const auto remapables_use_uvm =
            function_config_map.find("remapables_use_uvm");
        const bool use_uvm_for_remapables =
            remapables_use_uvm != function_config_map.end() &&
            remapables_use_uvm->second == "True";
        if (remapable_base != function_config_map.end() &&
            remapable_keys != function_config_map.end()) {
          std::vector<std::string> map_keys;
          std::vector<std::string> plaintexts_file_names;
          for (const auto &[map_key, plaintexts_file_name] :
               parse_remapable_keys(remapable_keys->second)) {
            map_keys.emplace_back(map_key);
            plaintexts_file_names.emplace_back(remapable_base->second + "/" +
                                               plaintexts_file_name);
          }
          cerium_function->generate_remapable_inputs_and_copy_to_device_batch(
              function_base_dir + "/remapable_inputs", plaintexts_file_names,
              map_keys, function_base_dir + "/inputs_remapable", num_gpus,
              use_uvm_for_remapables);
        }
      }
    }
  }

  std::unordered_map<std::string, bool> checked;
  for (auto &[function_name, calls] : call_graph) {
    use_cudagraphs[function_name] = false;
    checked[function_name] = false;
    auto default_config = config.find("default");
    if (default_config != config.end()) {
      auto &default_config_map = default_config->second;
      auto it = default_config_map.find("use_cudagraph");
      if (it != default_config_map.end()) {
        if (it->second == "True") {
          use_cudagraphs[function_name] = true;
        }
      }
    }
    auto function_config = config.find(function_name);
    if (function_config != config.end()) {
      auto &function_config_map = function_config->second;
      auto it = function_config_map.find("use_cudagraph");
      if (it != function_config_map.end()) {
        if (it->second != "False") {
          use_cudagraphs[function_name] = true;
        } else if (it->second == "False") {
          use_cudagraphs[function_name] = false;
        }
      }
    }
  }
  for (auto &[function_name, calls] : call_graph) {
    if (use_cudagraphs[function_name] == false) {
      continue;
    }
    if (checked[function_name] == true) {
      continue;
    }
    checked[function_name] = true;

    // If a function uses cudagraph, then all functions it calls must also use cudagraphs
    std::vector<std::string> stack;
    stack = calls;
    while (!stack.empty()) {
      auto fn = stack.back();
      stack.pop_back();
      use_cudagraphs[fn] = true;
      checked[fn] = true;
      for (auto &c : call_graph[fn]) {
        stack.push_back(c);
      }
    }
  }

  // XXX: If a parent function uses cudagraphs, all of its children must be compiled with cudagraphs
  for (auto &[k, fns] : call_graph_by_depth) {
    for (auto &function_name : fns) {
      auto &cerium_function = cerium_functions.at(function_name);
      auto use_cudagraph = use_cudagraphs.at(function_name);
      LOG(logger, INFO) << function_name << ": Create Cudagraph = "
                        << (use_cudagraph ? "True" : "False") << "\n"
                        << std::flush;
      if (use_cudagraph) {
        cerium_function->create_cuda_graph();
      } else {
        cerium_function->create_program_multithread();
      }
      // cerium_functions.at(f)->add_function_cerium_function(std::move(std::string(c)),cerium_functions.at(c));
    }
  }
  LOG(logger, INFO) << "Returning CeriumFunctions\n" << std::flush;

  return cerium_functions;
}

void Program::run_program() {
  if (use_cudagraphs[top_level_function]) {
    cerium_functions.at(top_level_function)->run_program_multithread_cugraph();
  } else {
    cerium_functions.at(top_level_function)->run_program_multithread();
  }
}

} // namespace Runtime
} // namespace Cerium