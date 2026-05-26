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

#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/execution/device_utils.h"
#include "cerium/runtime/execution/function.h"
#include "cerium/runtime/io/generator.h"
#include "cerium/runtime/utils/function_helpers.h"
#include "cerium/runtime/utils/logger.h"

#include <fstream>
#include <iomanip>

#include <algorithm>
#include <cctype>
#include <cstring>
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

namespace {

struct RuntimeInputRow {
  std::string term;
  std::string descriptor;
  std::vector<uint32_t> rns_base_ids;
};

std::string trim_copy(const std::string &value) {
  const auto begin = value.find_first_not_of(" \t\n\r\f\v");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = value.find_last_not_of(" \t\n\r\f\v");
  return value.substr(begin, end - begin + 1);
}

std::vector<uint32_t> parse_uint32_list(const std::string &list,
                                        const std::string &context) {
  auto values = trim_copy(list);
  if (values.size() < 2 || values.front() != '[' || values.back() != ']') {
    throw std::runtime_error("Invalid bracketed list in " + context + ": " +
                             list);
  }

  values = values.substr(1, values.size() - 2);
  std::vector<uint32_t> result;
  size_t start = 0;
  while (start <= values.size()) {
    const auto end = values.find(',', start);
    auto token = trim_copy(values.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (!token.empty()) {
      const auto value = std::stoul(token);
      if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("RNS base id exceeds uint32_t in " + context +
                                ": " + token);
      }
      result.push_back(static_cast<uint32_t>(value));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

struct RuntimeIOEntry {
  std::string term;
  std::string var;
  std::string type;
  std::vector<uint32_t> rns_base_ids;
};

RuntimeInputRow parse_runtime_input_row(const std::string &line) {
  const auto first_separator = line.find('|');
  if (first_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input row: " + line);
  }

  const auto second_separator = line.find('|', first_separator + 1);
  if (second_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input row: " + line);
  }

  RuntimeInputRow row;
  row.term = trim_copy(line.substr(0, first_separator));
  row.descriptor = trim_copy(
      line.substr(first_separator + 1, second_separator - first_separator - 1));
  row.rns_base_ids = parse_uint32_list(line.substr(second_separator + 1), line);
  return row;
}

RuntimeIOEntry parse_runtime_io(const std::string &line) {
  auto row = parse_runtime_input_row(line);
  const auto descriptor_separator = row.descriptor.find(':');
  if (descriptor_separator == std::string::npos) {
    throw std::runtime_error("Invalid generated input descriptor: " +
                             row.descriptor);
  }

  RuntimeIOEntry entry;
  entry.term = std::move(row.term);
  entry.var = row.descriptor.substr(0, descriptor_separator);
  entry.type = row.descriptor.substr(descriptor_separator + 1);
  entry.rns_base_ids = std::move(row.rns_base_ids);
  return entry;
}

} // namespace

CeriumFunction::EvalkeyInfo::EvalkeyInfo(std::string key_info) {

  std::vector<std::string> split;
  size_t pos = 0;
  while ((pos = key_info.find(":")) != std::string::npos) {
    split.push_back(key_info.substr(0, pos));
    key_info.erase(0, pos + 1);
  }
  split.push_back(key_info);
  if (split.size() != 7 || split[0] != "K") {
    throw std::runtime_error("Invalid Evalkey Entry" + key_info);
  }

  auto evk_type = std::stoi(split[1]);
  switch (evk_type) {
  case 0:
    key_type = KeyType::Mul;
    break;
  case 1:
    key_type = KeyType::Rot;
    break;
  case 2:
    key_type = KeyType::Con;
    break;
  case 3:
    key_type = KeyType::Boot;
    break;
  case 4:
    key_type = KeyType::Ephemeral;
    break;
  case 5:
    key_type = KeyType::Boot2;
    break;
  case 6:
    key_type = KeyType::RotInv;
    break;
  default:
    throw std::runtime_error("Invalid Evalkey type: " + split[1]);
  }

  id = split[1] + ":" + split[2] + ":" + split[3] + ":" + split[4] + ":" +
       split[6];

  level = std::stoi(split[2]);
  extension_size = std::stoi(split[3]);
  rotation_amount = std::stoi(split[4]);

  split[5][0] = '0';
  ct_number = std::stoi(split[5]);

  auto &digit_partition_str = split[6];
  digit_partition_str[0] = ' ';

  pos = 0;
  while ((pos = digit_partition_str.find(",")) != std::string::npos) {
    digit_partition.push_back(std::stoul(digit_partition_str.substr(0, pos)));
    digit_partition_str.erase(0, pos + 1);
  }
  pos = digit_partition_str.length();
  digit_partition.push_back(std::stoul(digit_partition_str.substr(0, pos - 1)));
}

CeriumFunction::EvalKeyEntry CeriumFunction::parse_evalkey(std::string &line) {
  auto row = parse_runtime_input_row(line);
  return EvalKeyEntry{std::move(row.term), EvalkeyInfo(row.descriptor),
                      std::move(row.rns_base_ids)};
};

void CeriumFunction::handle_evalkey_stream(std::ifstream &input_file,
                                        const std::string &evalkeys_file_name) {

  auto create_memory_pool = [](size_t size) -> MemoryManagerHandle {
    return std::make_shared<MemoryManagerCUDA>(
        size, MemoryManagerCUDA::MemoryType::Host);
  };
  auto evalkeys = IOGenerator::deserialize_evalkeys(
      context_, evalkeys_file_name, create_memory_pool);

  std::string line;
  std::vector<std::string> evk_ids;
  std::unordered_map<std::string, std::vector<EvalKeyEntry>> evalkey_entries;
  bool complete = false;
  while (std::getline(input_file, line)) {
    if (line == ";") {
      complete = true;
      break;
    }
    auto parse = parse_evalkey(line);
    auto &evk_info = parse.info;
    evalkey_entries[evk_info.id].push_back(parse);
  }

  for (auto &[k, v] : evalkey_entries) {
    evk_ids.push_back(k);
  }

  if (!complete) {
    throw std::logic_error("Unexpected EOF");
  }

  std::reverse(evk_ids.begin(), evk_ids.end());

  for (auto i = 0; i < evk_ids.size(); i++) {

    const auto &evk_entry = evalkey_entries.at(evk_ids.at(i));

    auto &parse = evk_entry.at(0);
    auto &evk_info = parse.info;
    std::pair<RnsPolynomialPtr, RnsPolynomialPtr> evk;

    try {
      evk = evalkeys.at(evk_info.id);
    } catch (const std::out_of_range &err) {
      throw std::runtime_error("Evalkey not found: " + evk_info.id);
    }

    for (auto &parse_ : evk_entry) {
      auto &term = parse_.term;
      auto &evk_info = parse_.info;
      auto &rns_base_ids = parse_.rns_base_ids;
      RnsPolynomialPtr poly;
      if (evk_info.ct_number == 0) {
        poly = std::get<0>(evk);
      } else if (evk_info.ct_number == 1) {
        poly = std::get<1>(evk);
      } else {
        throw std::runtime_error("Invalid CT Number");
      }

      for (auto &rns_base : rns_base_ids) {
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        const auto &limb = poly->at(rns_base);
        program_memory_.insert({term_name, limb});
      }
    }
  }
  CUDA::deviceSynchronize();
};

void CeriumFunction::generate_inputs(
    const std::string &inputs_file_name, const std::string &evalkeys_file_name,
    const std::string &plaintexts_file_name,
    const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_ptr) {
  auto &raw_inputs = raw_inputs_ptr->get_raw_inputs();

  LOG(logger, INFO) << "Generating Inputs" << "\n" << std::flush;
  auto input_file = open_input_file(inputs_file_name);
  std::string line;

  int stream_count = 0;

  CKKSEncoder encoder(context_);

  std::unordered_map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
      encrypted_inputs;

  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto handle_scalar_stream = [&]() {
    std::string line;
    size_t lpos = 0;
    size_t rpos = 0;

    std::vector<std::string> scalars_str;
    bool complete = false;
    while (std::getline(input_file, line)) {

      if (line == ";") {
        complete = true;
        break;
      }
      auto parse = parse_io(line);
      auto &var = parse.var;
      if (raw_inputs.find(var) == raw_inputs.end()) {
        std::cerr << "ERROR: Raw Input key error: " << var << "\n"
                  << std::flush;
        throw std::runtime_error("Raw Inputs key error: " + var);
      }
      scalars_str.push_back(line);
    }

    if (!complete) {
      throw std::logic_error("Unexpected EOF");
    }

    std::reverse(scalars_str.begin(), scalars_str.end());

    auto process_scalar = [&]() {
      for (int i = 0; i < scalars_str.size(); i++) {
        auto line = scalars_str.at(i);
        auto parse = parse_io(line);
        auto &term = parse.term;
        auto &var = parse.var;
        auto &type = parse.type;
        auto &rns_base_ids = parse.rns_base_ids;
        Cerium::Runtime::Utils::RawInputType raw_input;
        try {
          raw_input = raw_inputs.at(var);
        } catch (const std::out_of_range &err) {
          std::cerr << "ERROR: Raw Input key error: " << var << "\n"
                    << std::flush;
          throw std::runtime_error("Raw Inputs key error: " + var);
        }

        auto &message = std::get<0>(raw_input);
        auto &scale = std::get<1>(raw_input);
        if (type == "s") {
          std::map<uint32_t, Limb::Element_t> sc;
          std::visit(
              Cerium::Runtime::Utils::overloaded{
                  [](auto arg) { throw std::invalid_argument("Message"); },
                  [&](const double &arg) {
                    try {
                      sc = encoder.encode_scalar(arg, scale, rns_base_ids);
                    } catch (const std::exception &err) {
                      std::cerr << "ERROR: Failed to encode scalar: " << term
                                << "with error: " << err.what() << "\n"
                                << std::flush;
                      throw;
                    }
                  }},
              message);
          for (auto &rns_base : rns_base_ids) {
            auto term_name = term + "(" + std::to_string(rns_base) + ")";
            program_memory_scalar_.insert({term_name, sc.at(rns_base)});
          }
        } else {
          std::cerr << "ERROR: Invalid Type For Scalars: " << type << "\n"
                    << std::flush;
          throw std::runtime_error("Invalid Type: " + type);
        }
      }
    };
    process_scalar();
  };

  auto handle_plaintext_stream = [&]() {
    auto create_memory_pool = [](size_t size) -> MemoryManagerHandle {
      return std::make_shared<MemoryManagerCUDA>(
          size, MemoryManagerCUDA::MemoryType::Host);
    };
    auto plaintexts = IOGenerator::deserialize_plaintexts(
        context_, plaintexts_file_name, create_memory_pool);
    std::string line;
    size_t lpos = 0;
    size_t rpos = 0;

    std::vector<std::string> plaintexts_str;
    bool complete = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        complete = true;
        break;
      }
      auto parse = parse_io(line);
      auto &var = parse.var;
      if (plaintexts.find(var) == plaintexts.end()) {
        std::cerr << "ERROR: Plaintexts key error: " << var << "\n"
                  << std::flush;
        throw std::runtime_error("Plaintexts key error: " + var);
      }
      plaintexts_str.push_back(line);
    }

    if (!complete) {
      throw std::logic_error("Unexpected EOF");
    }

    std::reverse(plaintexts_str.begin(), plaintexts_str.end());

    int NUM_THREADS = 16;
    std::vector<decltype(program_memory_)> thread_local_program_memory_(
        NUM_THREADS);
    std::vector<decltype(program_memory_scalar_)>
        thread_local_program_memory_scalar_(NUM_THREADS);
    auto process_plaintext = [&](int tid) {
      auto &local_program_memory_ = thread_local_program_memory_.at(tid);
      auto &local_program_memory_scalar_ =
          thread_local_program_memory_scalar_.at(tid);
      for (int i = tid; i < plaintexts_str.size(); i += NUM_THREADS) {
        auto line = plaintexts_str.at(i);
        auto parse = parse_io(line);
        auto &term = parse.term;
        auto &var = parse.var;
        auto &type_ = parse.type;
        auto &rns_base_ids = parse.rns_base_ids;

        auto pos = type_.find(":");
        auto type = type_.substr(0, pos);
        auto elt_size = std::stoi(type_.substr(pos + 1));

        if (type == "p") {
          auto pt = plaintexts.at(var);
          for (auto &rns_base : rns_base_ids) {
            auto term_name = term + "(" + std::to_string(rns_base) + ")";
            const auto &limb = pt->at(rns_base);
            local_program_memory_.insert({term_name, limb});
          }
        } else {
          std::cerr << "ERROR: Invalid Type: " << type << "\n" << std::flush;
          throw std::runtime_error("Invalid Type: " + type);
        }
      }
    };

    std::vector<std::thread> threads;
    std::exception_ptr thread_exception;
    std::mutex thread_exception_mutex;
    for (size_t tid = 0; tid < NUM_THREADS; tid++) {
      threads.emplace_back([&, tid]() {
        capture_thread_exception([&]() { process_plaintext(tid); },
                                 thread_exception, thread_exception_mutex);
      });
    }

    for (size_t tid = 0; tid < NUM_THREADS; tid++) {
      threads[tid].join();
    }
    rethrow_thread_exception(thread_exception);

    for (int tid = 0; tid < NUM_THREADS; tid++) {
      const auto &local_program_memory_ = thread_local_program_memory_.at(tid);
      program_memory_.insert(local_program_memory_.begin(),
                             local_program_memory_.end());
    }
    CUDA::deviceSynchronize();
    thread_local_program_memory_.clear();
    thread_local_program_memory_scalar_.clear();
    plaintexts.clear();
  };

  auto handle_output_stream = [&]() {
    std::string line;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        return;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;

      auto &output = program_outputs_[var];
      if (type == "c0") {
        output.c0_term = term;
        output.rns_base_ids = rns_base_ids;
      } else if (type == "c1") {
        output.c1_term = term;
      } else {
        throw std::runtime_error("Invalid Type: " + type);
      }
    }

    throw std::logic_error("Unexpected EOF");
  };

  while (std::getline(input_file, line)) {
    size_t pos;
    if (stream_count == 0) {
      // Ciphertext Stream
      if ((pos = line.find("Ciphertext Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Header: " + line);
      } else {
        stream_count++;
        // Nothing to do
        while (std::getline(input_file, line) && line != ";")
          ;
      }
    } else if (stream_count == 1) {
      // Plaintext Stream
      if ((pos = line.find("Plaintext Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        handle_plaintext_stream();
        // // TODO: make a stream for scalars....
        // stream_count++;
      }
    } else if (stream_count == 2) {
      // Scalar Stream
      if ((pos = line.find("Scalar Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        handle_scalar_stream();
      }
    } else if (stream_count == 3) {
      // Output Stream
      if ((pos = line.find("Output Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        handle_output_stream();
      }
    } else if (stream_count == 4) {
      // Evalkey Stream
      if ((pos = line.find("Evalkey Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        // handle_evalkey_stream(input_file,encryptor);
        handle_evalkey_stream(input_file, evalkeys_file_name);
      }
    } else if (stream_count == 5) {
      // Evalkey Stream
      if ((pos = line.find("Function Inputs Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        // Nothing to do
        while (std::getline(input_file, line) && line != ";")
          ;
      }
    } else if (stream_count == 6) {
      // Evalkey Stream
      if ((pos = line.find("Function Outputs Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Heading: " + line);
      } else {
        stream_count++;
        // Nothing to do
        while (std::getline(input_file, line) && line != ";")
          ;
      }
    }
  }
}

void CeriumFunction::generate_remapable_inputs(
    const std::string &remapables_file_name,
    const std::string &plaintexts_file_name, const std::string &map_key,
    bool remapables_use_uvm) {

  bool SKIP_COPY = runtime_options_.skip_copy_remapables;
  if (SKIP_COPY) {
    LOG(logger, INFO) << "Skipping Copy for Remapable Inputs with key:"
                      << map_key << " from file:" << remapables_file_name
                      << "\n"
                      << std::flush;
    const auto memory_type = remapables_use_uvm
                                 ? MemoryManagerCUDA::MemoryType::HostPageable
                                 : MemoryManagerCUDA::MemoryType::Host;
    auto create_memory_pool = [memory_type](size_t size) -> MemoryManagerHandle {
      return std::make_shared<MemoryManagerCUDA>(size, memory_type);
    };
    auto plaintexts = IOGenerator::deserialize_plaintexts_no_copy(
        context_, plaintexts_file_name, create_memory_pool);
    remapable_program_memory_.insert({map_key, plaintexts});
    return;
  }

  LOG(logger, INFO) << "Generating Remapable Inputs with key:" << map_key
                    << " from inputs file:" << remapables_file_name
                    << " copying values from plaintexts file: " << plaintexts_file_name <<  "\n"
                    << std::flush;
  auto input_file = open_input_file(remapables_file_name);
  std::string line;

  int stream_count = 0;

  CKKSEncoder encoder(context_);

  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto handle_plaintext_stream = [&]() {
    const auto memory_type = remapables_use_uvm
                                 ? MemoryManagerCUDA::MemoryType::HostPageable
                                 : MemoryManagerCUDA::MemoryType::Host;
    auto create_memory_pool = [memory_type](size_t size) -> MemoryManagerHandle {
      return std::make_shared<MemoryManagerCUDA>(size, memory_type);
    };
    auto plaintexts = IOGenerator::deserialize_plaintexts(
        context_, plaintexts_file_name, create_memory_pool);

    {
      std::lock_guard<std::mutex> lock(sync_lock);
      remapable_program_memory_.insert({map_key, plaintexts});
    }

    bool NO_CHECK_REMAPABLES = runtime_options_.no_check_remapables;
    if (NO_CHECK_REMAPABLES) {
      return;
    }

    std::string line;
    size_t lpos = 0;
    size_t rpos = 0;

    std::vector<std::string> plaintexts_str;
    bool complete = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        complete = true;
        break;
      }
      auto parse = parse_io(line);
      auto &var = parse.var;
      if (plaintexts.find(var) == plaintexts.end()) {
        std::cerr << "ERROR: Plaintexts key error: " << var << "\n"
                  << std::flush;
        throw std::runtime_error("Plaintexts key error: " + var);
      }
      plaintexts_str.push_back(line);
    }

    if (!complete) {
      throw std::logic_error("Unexpected EOF");
    }
    return;
  };

  while (std::getline(input_file, line)) {
    size_t pos;
    if (stream_count == 0) {
      // Ciphertext Stream
      if ((pos = line.find("Remapable Stream")) == std::string::npos) {
        throw std::logic_error("Invalid Stream Header: " + line);
      } else {
        stream_count++;
        handle_plaintext_stream();
      }
    }
  }
}

void CeriumFunction::generate_remapable_inputs_and_copy_to_device_batch(
    const std::string &remapables_file_name,
    const std::vector<std::string> &plaintexts_file_names,
    const std::vector<std::string> &map_keys,
    const std::string &local_remapable_file_base, size_t num_partitions,
    bool remapables_use_uvm) {
  if (plaintexts_file_names.size() != map_keys.size()) {
    throw std::runtime_error("Mismatched sizes for remapable inputs batch");
  }
  if (plaintexts_file_names.empty()) {
    return;
  }

  if (num_partitions == 0) {
    throw std::runtime_error("Number of partitions must be greater than zero");
  }
  create_remapable_offsets(local_remapable_file_base, num_partitions);

  int NUM_THREADS = std::max(2, static_cast<int>(8 / num_partitions));
  if (runtime_options_.num_remapable_input_threads) {
    NUM_THREADS = *runtime_options_.num_remapable_input_threads;
  }
  if (NUM_THREADS <= 0) {
    throw std::runtime_error(
        "Number of remapable input threads must be greater than zero");
  }
  NUM_THREADS = std::min(NUM_THREADS, (int)plaintexts_file_names.size());
  auto thread_fn = [&](const int tid) {
    for (int i = tid; i < plaintexts_file_names.size(); i += NUM_THREADS) {
      generate_remapable_inputs(remapables_file_name, plaintexts_file_names[i],
                                map_keys[i], remapables_use_uvm);
      copy_remapable_inputs_to_device(local_remapable_file_base, num_partitions,
                                      map_keys[i], remapables_use_uvm);
    }
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back([&, i]() {
      capture_thread_exception([&]() { thread_fn(i); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t i = 0; i < NUM_THREADS; i++) {
    threads[i].join();
  }
  rethrow_thread_exception(thread_exception);
}

void CeriumFunction::copy_inputs_to_device(
    const std::string &local_inputs_file_base, size_t num_partitions) {
  LOG(logger, INFO) << "Copying Inputs to Device\n" << std::flush;
  program_memory_local_.resize(num_partitions);
  program_outputs_local_.resize(num_partitions);
  function_args_name_map_.resize(num_partitions);
  function_args_local_.resize(num_partitions);
  function_args_input_local_.resize(num_partitions);
  function_args_output_local_.resize(num_partitions);
  remapable_base_.resize(num_partitions);
  remapable_program_memory_local_.resize(num_partitions);
  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto calculate_input_size = [&](std::ifstream &input_file, const size_t tid) {
    std::size_t memory_size = 0;
    std::string line;
    bool completed = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      if (term[0] == 's') {
        continue;
      }
      size_t elt_size = context_.n();
      if (term[0] == 'p') {
        elt_size = 2 * std::stoul(type.substr(type.find(":") + 1));
        elt_size = std::max(32UL, elt_size);
      }
      memory_size += (rns_base_ids.size() * elt_size * sizeof(Limb::Element_t));
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
    return memory_size;
  };

  auto handle_input_stream = [&](std::ifstream &input_file,
                                 size_t inputs_size_bytes, const size_t tid) {
    std::string line;
    bool completed = false;
    auto stream = CUDA::createStream();

    MemoryManagerCUDAHandle memory_pool = nullptr;
    if (use_uvm_) {
      memory_pool = std::make_shared<MemoryManagerCUDA>(
          inputs_size_bytes, MemoryManagerCUDA::MemoryType::Managed);
    } else {
      memory_pool = std::make_shared<MemoryManagerCUDA>(
          inputs_size_bytes, MemoryManagerCUDA::MemoryType::Device);
    }

    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      if (type == "s") {
        continue;
      }
      for (auto &rns_base : rns_base_ids) {
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        LimbPtr limb = nullptr;
        if(type == "c0" || type == "c1") {
          // Allocate an unitialized placeholder limb for ciphertexts 
          limb = std::make_shared<Limb>(allocate_limb_elts(coeff_count),
                                            coeff_count, rns_base, true);
        } else { 
          limb = program_memory_.at(term_name);
        }
        auto limb_size = std::max(32UL, limb->size());
        auto device_ptr =
            DevicePointer<Limb::Element_t>(memory_pool, limb_size);
        CUDA::memcpyHostToDeviceAsync(device_ptr.get(), limb->data(),
                                      sizeof(Limb::Element_t) * limb->size(),
                                      stream);
        if (type == "p") {
          assert(!limb->is_ntt_form() &&
                 "Limb should not be in NTT form when copying to device");
        }
        auto device_limb = std::make_shared<LimbT>(
            std::move(device_ptr), limb->size(), limb->rns_base_id(), true);
        program_memory_local_[tid].insert({term_name, device_limb});
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
    CUDA::streamSynchronize(stream);
  };

  auto handle_output_stream = [&](std::ifstream &input_file, const size_t tid) {
    std::string line;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        return;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;

      for (auto &rns_base : rns_base_ids) {
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        program_outputs_local_[tid].insert(term_name);
      }
    }

    throw std::logic_error("Unexpected EOF");
  };

  auto handle_function_stream = [&](std::ifstream &input_file, const size_t tid,
                                    const bool is_input) {
    std::string line;
    bool completed = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      for (auto &rns_base : rns_base_ids) {
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        auto var_name = var + ":" + type + "(" + std::to_string(rns_base) + ")";
        auto device_ptr = DevicePointer<Limb::Element_t>(coeff_count, use_uvm_);
        auto device_limb = std::make_shared<LimbT>(std::move(device_ptr),
                                                   coeff_count, rns_base, true);
        if (is_input) {
          function_args_input_local_[tid].insert({var_name, device_limb});
        } else {
          function_args_output_local_[tid].insert({var_name, device_limb});
        }
        function_args_name_map_[tid].insert({var_name, term_name});
        function_args_local_[tid].insert({term_name, device_limb});
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
  };

  auto thread_fn = [&](const size_t tid) {
    LOG(logger, INFO) << "Launching Threadfn " << tid << "\n" << std::flush;
    CUDA::setDevice(tid);
    const auto input_file_name =
        local_inputs_file_base + std::to_string(tid % num_gpus_compiled);
    auto input_file = open_input_file(input_file_name);
    std::string line;

    int stream_count = 0;

    while (std::getline(input_file, line)) {
      size_t pos;
      if (stream_count == 0) {
        // Inputs Stream
        if ((pos = line.find("Input Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Header: " + line);
        } else {
          stream_count++;
          auto saved_pos = input_file.tellg();
          auto mem_size = calculate_input_size(input_file, tid);
          input_file.clear();
          input_file.seekg(saved_pos);
          handle_input_stream(input_file, mem_size, tid);
        }
      } else if (stream_count == 1) {
        // Output Stream
        if ((pos = line.find("Output Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          handle_output_stream(input_file, tid);
          // // TODO: make a stream for scalars....
          // stream_count++;
        }
      } else if (stream_count == 2) {
        // Output Stream
        if ((pos = line.find("Function Input Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          handle_function_stream(input_file, tid, true /*input*/);
        }
      } else if (stream_count == 3) {
        // Output Stream
        if ((pos = line.find("Function Output Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          handle_function_stream(input_file, tid, false /*output*/);
        }
      }
    }
    remapable_base_[tid] = DevicePointer<Limb::Element_t *>(1);
    CUDA::deviceSynchronize();
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { thread_fn(tid); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
}

void CeriumFunction::copy_ciphertext_inputs_to_gpu(
    const std::string &local_inputs_file_base, size_t num_partitions,
    const std::unordered_map<std::string, std::pair<RnsPolynomialPtr,RnsPolynomialPtr>> & encrypted_inputs) {
  LOG(logger, INFO) << "Copying Ciphertext Inputs To GPU\n" << std::flush;
  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto handle_input_stream = [&](std::ifstream &input_file, const size_t tid) {
    std::string line;
    bool completed = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      RnsPolynomialPtr poly = nullptr;
      if(type == "c0") {
        poly = encrypted_inputs.at(var).first;
      } else if(type == "c1") {
        poly = encrypted_inputs.at(var).second;
      } else {
        continue;
      }
      std::vector<Limb::Element_t *> limb_ptrs;
      size_t requires_ntt_count = 0;
      for (auto &rns_base : rns_base_ids) {
        const auto &limb = poly->at(rns_base);
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        auto device_limb = program_memory_local_[tid].at(term_name);
        CUDA::memcpyHostToDevice(device_limb->data(), limb->data(),
                                 device_limb->size() * sizeof(Limb::Element_t));
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
  };

  auto nop_stream = [&](std::ifstream &input_file) {
    std::string line;
    bool completed = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
  };

  auto thread_fn = [&](const size_t tid) {
    CUDA::setDevice(tid);
    const auto input_file_name =
        local_inputs_file_base + std::to_string(tid % num_gpus_compiled);
    auto input_file = open_input_file(input_file_name);
    std::string line;

    int stream_count = 0;

    while (std::getline(input_file, line)) {
      size_t pos;
      if (stream_count == 0) {
        // Inputs Stream
        if ((pos = line.find("Input Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Header: " + line);
        } else {
          stream_count++;
          handle_input_stream(input_file, tid);
        }
      } else if (stream_count == 1) {
        // Output Stream
        if ((pos = line.find("Output Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          nop_stream(input_file);
        }
      } else if (stream_count == 2) {
        // Output Stream
        if ((pos = line.find("Function Input Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          nop_stream(input_file);
        }
      } else if (stream_count == 3) {
        // Output Stream
        if ((pos = line.find("Function Output Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Heading: " + line);
        } else {
          stream_count++;
          nop_stream(input_file);
        }
      }
    }
    CUDA::deviceSynchronize();
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { thread_fn(tid); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
}

void CeriumFunction::create_remapable_offsets(
    const std::string &local_remapable_file_base, const size_t num_gpus) {
  std::lock_guard<std::mutex> lock(sync_lock);
  if (remapable_offsets_set_) {
    return;
  }
  remapable_program_memory_local_.resize(num_gpus);
  remapables_size_.resize(num_gpus, 0);

  for (size_t tid = 0; tid < num_gpus; tid++) {
    const auto input_file_name =
        local_remapable_file_base + std::to_string(tid % num_gpus_compiled);
    auto input_file = open_input_file(input_file_name);
    std::string header;
    if (!std::getline(input_file, header) ||
        header.find("Remapable Stream") == std::string::npos) {
      throw std::logic_error("Invalid Stream Header: " + header);
    }
    LOG(logger, INFO) << "Calculating Remapable Size for tid: " << tid << "\n"
                      << std::flush;
    std::size_t offset = 0;
    std::size_t memory_size = 0;
    std::string line;
    bool completed = false;
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_runtime_io(line);
      auto &term = parse.term;
      auto &type_ = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      auto pos = type_.find(":");
      auto type = type_.substr(0, pos);
      auto elt_size = std::stoi(type_.substr(pos + 1));
      if (type != "p") {
        throw std::runtime_error("Invalid type: " + type);
      }
      memory_size += (2 * elt_size) * rns_base_ids.size();
      for (auto &rns_base : rns_base_ids) {
        auto term_name = term + "(" + std::to_string(rns_base) + ")";
        remapable_program_memory_local_[tid].insert({term_name, offset});
        offset += 2 * elt_size;
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
    if (offset != memory_size) {
      throw std::runtime_error("Offset size doesn't match memory_size");
    }
    remapables_size_[tid] = memory_size * sizeof(Limb::Element_t);
    LOG(logger, INFO) << "Calculated Remapable Size for tid: " << tid
                      << " size: " << remapables_size_[tid] << "\n"
                      << std::flush;
  }
  remapable_offsets_set_ = true;
}

void CeriumFunction::copy_remapable_inputs_to_device(
    const std::string &local_remapable_file_base, size_t num_partitions,
    const std::string &map_key, bool remapables_use_uvm) {
  {
    std::lock_guard<std::mutex> lock(sync_lock);
    if (!remapable_offsets_set_) {
      throw std::logic_error(
          "Remapable offsets must be created before copying remapable inputs");
    }
  }
  bool SKIP_COPY = runtime_options_.skip_copy_remapables;

  bool remapables_use_explicit_copy =
      runtime_options_.remapables_use_explicit_copy;
  if (remapables_use_explicit_copy) {
    LOG(logger, INFO) << "Remapables Use Explicit Copy: "
                      << remapables_use_explicit_copy << "\n"
                      << std::flush;
    if (remapable_pointers_explicit_copy_.size() != num_partitions) {
      remapable_pointers_explicit_copy_.resize(num_partitions);
    }
  }
  if (remapables_use_explicit_copy) {
    remapables_type_ = RemapablesType::ExplicitCopy;
  } else if (remapables_use_uvm) {
    remapables_type_ = RemapablesType::UVM;
  } else {
    remapables_type_ = RemapablesType::Device;
  }

  if (SKIP_COPY) {
    LOG(logger, INFO) << "Skipping Copying Remapable Inputs to Device with key:"
                      << map_key << " from file:" << local_remapable_file_base
                      << " num_gpus: " << num_partitions
                      << " use_uvm: " << (uint32_t)remapables_use_uvm
                      << "SKIP_COPY: " << SKIP_COPY << "\n"
                      << std::flush;
  } else {
    LOG(logger, INFO) << "Copying Remapable Inputs to Device with key:"
                      << map_key << " from file:" << local_remapable_file_base
                      << " num_gpus: " << num_partitions
                      << " use_uvm: " << (uint32_t)remapables_use_uvm << "\n"
                      << std::flush;
  }
  decltype(remapable_program_memory_)::mapped_type memory;
  {
    std::lock_guard<std::mutex> lock(sync_lock);
    remapable_pointers_[map_key].resize(num_partitions);
    if (remapable_program_memory_local_.size() != num_partitions) {
      remapable_program_memory_local_.resize(num_partitions);
    }
    if (remapables_size_.size() != num_partitions) {
      remapables_size_.resize(num_partitions, 0);
    }
    try {
      memory = remapable_program_memory_.at(map_key);
    } catch (const std::exception &e) {
      std::cerr << "Unexpected error: " << e.what()
                << "In Copy Remapable Inputs to Device for map_key: "
                << map_key;
      throw;
    }
  }
  auto parse_io = [](const std::string &line) {
    return parse_runtime_io(line);
  };

  auto handle_remapable_stream = [&](std::ifstream &input_file,
                                     const std::size_t memory_size,
                                     const size_t tid) {
    if (memory_size == 0) {
      return;
    }
    std::string line;
    LimbDataType *device_ptr = nullptr;
    if (remapables_type_ == RemapablesType::ExplicitCopy) {
      auto memory_pool = std::make_shared<MemoryManagerCUDA>(
          memory_size * sizeof(Limb::Element_t),
          MemoryManagerCUDA::MemoryType::Host);
      auto mem_ptr = DevicePointer<Limb::Element_t>(memory_pool, memory_size);
      {
        std::lock_guard<std::mutex> lock(sync_lock);
        remapable_pointers_[map_key][tid] = std::move(mem_ptr);
        if (remapable_pointers_explicit_copy_[tid].get() == nullptr) {
          remapable_pointers_explicit_copy_[tid] =
              DevicePointer<Limb::Element_t>(memory_size);
        }
        device_ptr = remapable_pointers_[map_key][tid].get();
      }
    } else {
      auto mem_ptr =
          DevicePointer<Limb::Element_t>(memory_size, remapables_use_uvm);
      {
        std::lock_guard<std::mutex> lock(sync_lock);
        remapable_pointers_[map_key][tid] = std::move(mem_ptr);
        device_ptr = remapable_pointers_[map_key][tid].get();
      }
    }
    if (((uint64_t)device_ptr & 0x3UL) != 0) {
      throw std::runtime_error("Remapable Base Pointer is not 32 bit aligned");
    }
    std::size_t offset = 0;
    bool completed = false;
    std::vector<CUDA::StreamPtr> streams;
    if (remapables_type_ != RemapablesType::UVM) {
      for (auto i = 0; i < context_.num_rns_bases(); i++) {
        streams.emplace_back(CUDA::createStream());
      }
    }
    if (SKIP_COPY) {
      return;
    }
    if (remapables_type_ == RemapablesType::UVM) {
      const auto bytes = memory_size * sizeof(Limb::Element_t);
      auto host_prefetch_stream = CUDA::createStream();
      CUDA::memPrefetchDeviceToHostAsync(device_ptr, bytes,
                                         host_prefetch_stream);
      CUDA::streamSynchronize(host_prefetch_stream);
    }
    while (std::getline(input_file, line)) {
      if (line == ";") {
        completed = true;
        break;
      }
      auto parse = parse_io(line);
      auto &term = parse.term;
      auto &var = parse.var;
      auto &type_ = parse.type;
      auto &rns_base_ids = parse.rns_base_ids;
      auto pos = type_.find(":");
      auto type = type_.substr(0, pos);
      auto elt_size = std::stoi(type_.substr(pos + 1));
      if (type != "p") {
        throw std::runtime_error("Invalid type: " + type);
      }
      size_t requires_ntt_count = 0;
      auto &rns_poly = memory.at(var);
      for (auto &rns_base : rns_base_ids) {
        const auto &limb = rns_poly->at(rns_base);
        if (type == "p") {
          assert(limb->is_ntt_form() &&
                 "Limb should be in NTT form when copying to device");
        }
        LimbDataType *ptr = device_ptr + offset;
        if (remapables_type_ == RemapablesType::UVM) {
          std::memcpy(ptr, limb->data(),
                      limb->size() * sizeof(Limb::Element_t));
        } else if (remapables_type_ == RemapablesType::ExplicitCopy) {
          CUDA::memcpyHostToHostAsync(ptr, limb->data(),
                                      limb->size() * sizeof(Limb::Element_t),
                                      streams[rns_base % streams.size()]);
        } else {
          CUDA::memcpyHostToDeviceAsync(ptr, limb->data(),
                                        limb->size() * sizeof(Limb::Element_t),
                                        streams[rns_base % streams.size()]);
        }
        offset += limb->size();
      }
    }
    if (!completed) {
      throw std::logic_error("Unexpected EOF");
    }
    if (offset != memory_size) {
      throw std::runtime_error("Offset size doesn't match memory_size");
    }
    if (remapables_type_ == RemapablesType::UVM) {
      const auto bytes = memory_size * sizeof(Limb::Element_t);
      CUDA::memAdviseSetMostlyReadOnly(device_ptr, bytes);
      CUDA::memAdviseSetPreferredLocationDevice(device_ptr, bytes, tid);
      // The runtime prefetch stream moves this buffer to the device before use.
    }
  };

  auto thread_fn = [&](const size_t tid) {
    CUDA::setDevice(tid);
    const auto input_file_name =
        local_remapable_file_base + std::to_string(tid % num_gpus_compiled);
    auto input_file = open_input_file(input_file_name);
    std::string line;

    int stream_count = 0;

    while (std::getline(input_file, line)) {
      size_t pos;
      if (stream_count == 0) {
        // Inputs Stream
        if ((pos = line.find("Remapable Stream")) == std::string::npos) {
          throw std::logic_error("Invalid Stream Header: " + line);
        } else {
          stream_count++;
          const size_t mem_size =
              remapables_size_[tid] / sizeof(Limb::Element_t);
          handle_remapable_stream(input_file, mem_size, tid);
          break;
        }
      }
    }
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { thread_fn(tid); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < num_partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
  {
    std::lock_guard<std::mutex> lock(sync_lock);
    remapable_program_memory_.erase(map_key);
  }
}

std::map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>>
CeriumFunction::get_program_outputs() {
  decltype(program_memory_) outputs_memory_;
  for (size_t i = 0; i < program_outputs_local_.size(); i++) {
    CUDA::setDevice(i);
    for (const auto &term : program_outputs_local_[i]) {
      try {
        const auto &device_limb = program_memory_local_[i].at(term);
        auto host_limb = device_limb->move_to_host_ptr();
        outputs_memory_.insert({term, host_limb});
      } catch (const std::out_of_range &e) {
        std::cerr << "Error: " << e.what() << " for term: " << term
                  << " in device " << i << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "Unexpected error: " << e.what() << " for term: " << term
                  << " in device " << i << std::endl;
        throw;
      }
    }
    CUDA::deviceSynchronize();
  }
  std::map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>> outputs;
  for (auto &[key, value] : program_outputs_) {
    auto c0_term = value.c0_term;
    auto c1_term = value.c1_term;
    auto rns_base_ids = value.rns_base_ids;
    auto ct0 = std::make_shared<RnsPolynomial>();
    auto ct1 = std::make_shared<RnsPolynomial>();
    bool term_exists = true;
    for (auto &rns_base_id : rns_base_ids) {
      auto c0_addr = c0_term + '(' + std::to_string(rns_base_id) + ')';
      auto c1_addr = c1_term + '(' + std::to_string(rns_base_id) + ')';
      try {
        auto ct0_host = outputs_memory_.at(c0_addr)->move_to_host_ptr();
        ct0->write_limb(ct0_host, rns_base_id);
      } catch (const std::out_of_range &e) {
        std::cerr << "Error: " << e.what() << " for addr: " << c0_addr
                  << std::endl;
        term_exists = false;
      } catch (const std::exception &e) {
        std::cerr << "Unexpected error: " << e.what()
                  << " for addr: " << c0_addr << std::endl;
        term_exists = false;
        throw;
      }
      try {
        auto ct1_host = outputs_memory_.at(c1_addr)->move_to_host_ptr();
        ct1->write_limb(ct1_host, rns_base_id);
      } catch (const std::out_of_range &e) {
        LOG(logger, WARN) << "Error: " << e.what() << " for addr: " << c1_addr
                          << std::endl;
        term_exists = false;
      } catch (const std::exception &e) {
        LOG(logger, FATAL) << "Unexpected error: " << e.what()
                           << " for addr: " << c1_addr << std::endl;
        term_exists = false;
        throw;
      }
    }
    if (!term_exists) {
      LOG(logger, WARN) << "Term does not exist in outputs_memory_ for key: "
                        << key << std::endl;
      continue;
    }
    outputs[key] = std::make_pair(ct0, ct1);
  }

  print_runtime_stats();

  return std::move(outputs);
}

bool CeriumFunction::get_boolean_env_variable(const std::string &var) {
  if (const char *env_p = std::getenv(var.c_str())) {
    auto env_str = std::string(env_p);
    if (env_str == "1") {
      return true;
    } else {
      return false;
    }
  }
  return false;
}

} // namespace Runtime
} // namespace Cerium
