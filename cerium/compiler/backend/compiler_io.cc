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

#include "cerium/compiler/backend/compiler.h"
#include "cerium/compiler/util/mkdir.h"

#include <fstream>

namespace Cerium {
namespace Backend {

namespace {

template <typename KernelGroupContainer>
void print_graph_for_kernel_groups(const KernelGroupContainer &kernel_groups,
                                   const std::string &graph_path) {
  std::ofstream graph_file(graph_path);
  if (!graph_file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + graph_path);
  }

  graph_file << "Digraph G {\n";
  std::unordered_set<KernelGroup::IndexType> visited;

  for (size_t i = 0; i < kernel_groups.size(); ++i) {
    if (!kernel_groups[i]) {
      continue;
    }

    auto idx = kernel_groups[i]->index();
    if (visited.find(idx) != visited.end()) {
      continue;
    }
    visited.insert(idx);

    auto &kg = kernel_groups[idx];
    graph_file << "\tsubgraph cluster_KG" << idx << "{\n ";
    graph_file << "\t\tb" << idx << "[label=\"BN" << idx << ":("
               << static_cast<uint32_t>(kg->partition_size()) << ":"
               << static_cast<uint32_t>(kg->partition_id()) << ")\"]\n";

    uint64_t instructions = 0;
    for (auto &instr : kg->instructions()) {
      ++instructions;
      graph_file << "\t\tt" << instructions << "[label=\"" << instr->ppOp()
                 << "\"]\n";
    }

    graph_file << "\t}\n";
  }

  graph_file << "}";
  graph_file << std::flush;
}

template <typename LimbRange>
std::string format_io_term_entry(const std::shared_ptr<Term> &term,
                                 const LimbRange &limbs) {
  std::stringstream entry;
  entry << term->name() << " | " << term->symbol() << " | [";

  bool first = true;
  for (const auto &limb : limbs) {
    if (!first) {
      entry << ",";
    }
    entry << limb;
    first = false;
  }

  entry << "]\n";
  return entry.str();
}

void write_labeled_stream(std::ofstream &file, const char *label,
                          const std::stringstream &stream) {
  file << label << "\n" << stream.str() << ";\n";
}

void ensure_ofstream_open(const std::ofstream &file, const std::string &path) {
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + path);
  }
}

void ensure_directory_created(const std::string &path) {
  if (!Cerium::Util::mkdir_p(path)) {
    throw std::runtime_error("Failed to create output directory: " + path);
  }
}

} // namespace

void CeriumCompiler::print_graph_simple2(const std::string &name) {
  if (Cerium::Util::getEnvVarUint32("CERIUM_PRINT_GRAPHS", 0) == 0) {
    return;
  }

  const auto graph_path = function_name + "_cerium_graph_" + name + ".dot";
  print_graph_for_kernel_groups(kernel_groups, graph_path);
}

void CeriumCompiler::print_graph_split() {
  if (Cerium::Util::getEnvVarUint32("CERIUM_PRINT_GRAPHS", 0) == 0) {
    return;
  }

  for (size_t j = 0; j < mod_partitions; j++) {
    const auto graph_path =
        function_name + "_cerium_graph_split.dot" + std::to_string(j);
    print_graph_for_kernel_groups(kernel_groups_split[j], graph_path);
  }
}

void CeriumCompiler::print_limb_instructions() {

  for (auto j = 0; j < mod_partitions; j++) {
    auto kernel_factory_ = Backend::KernelFusionContext(function_name);
    auto kgs = kernel_groups_split[j];
    for (auto i = 0; i < kgs.size(); i++) {
      if (!kgs[i]) {
        continue;
      }

      auto idx = i;
      auto &kg = kgs[idx];
      // TODO:: Fix this by splitting across gpus
      auto kernel = kg->get_fused_kernel(kernel_factory_);
    }
    std::string output_directory =
        output_prefix + "/" + function_name + "/partition_" + std::to_string(j);
    ensure_directory_created(output_directory);
    do_next_use_analysis(kernel_factory_, j);
    kernel_factory_.write_code(output_directory);
    kernel_factory_.allocate_registers_cerium(num_vregs, num_bcus);
    kernel_factory_.write_function_map(output_directory);
    kernel_factory_.write_function_map_cugraph(output_directory);
    kernel_factory_.write_instructions(output_directory);
  }
}

void CeriumCompiler::do_next_use_analysis(
    Backend::KernelFusionContext &kernel_factory, uint32_t partition_id) {

  auto &kernels = kernel_factory.kernels();
  Backend::LimbMap<uint64_t> next_use;
  std::map<Backend::TermIndexType, uint64_t> next_use_base_conv;
  uint64_t kernel_count = kernels.size();

  auto register_io_limb = [&](const auto &limb) {
    program_io_local[limb.term()].insert(limb.limb_idx());
  };

  auto update_next_use = [&](auto &limb) {
    auto next_use_it = next_use.find(limb);
    if (next_use_it == next_use.end()) {
      limb.set_as_dead();
    }
    next_use[limb] = kernel_count;
    limb.set_next_use(kernel_count);
  };

  auto update_next_use_base_conv = [&](auto &limb) {
    auto next_use_it = next_use_base_conv.find(limb.term_idx());
    if (next_use_it == next_use_base_conv.end()) {
      limb.set_as_dead();
      CL_LOG3("Setting {} as dead", limb);
    }
    next_use_base_conv[limb.term_idx()] = kernel_count;
  };

  auto visit_nested_limb_groups = [&](auto &groups, auto &&handler) {
    for (auto group_it = groups.rbegin(); group_it != groups.rend();
         ++group_it) {
      for (auto limb_it = group_it->rbegin(); limb_it != group_it->rend();
           ++limb_it) {
        handler(std::get<Backend::Limb>(*limb_it));
      }
    }
  };

  for (auto it = kernels.rbegin(); it != kernels.rend(); it++) {
    auto &kernel = *it;
    auto &outputs = kernel->outputs();
    auto &inputs = kernel->inputs();
    auto &inputs_scalar = kernel->inputs_scalar();
    auto &inputs_remapable = kernel->inputs_remapable();

    visit_nested_limb_groups(outputs, [&](auto &o) {
      if (o.is_output()) {
        register_io_limb(o);
        return;
      }
      if (o.is_bcor()) {
        return;
      }
      update_next_use(o);
    });

    visit_nested_limb_groups(inputs, [&](auto &i) {
      if (i.is_input()) {
        register_io_limb(i);
      } else if (i.is_bcor()) {
        update_next_use_base_conv(i);
        return;
      } else {
        update_next_use(i);
        return;
      }
    });

    visit_nested_limb_groups(inputs_scalar, [&](auto &i) {
      if (i.is_input()) {
        register_io_limb(i);
      }
      update_next_use(i);
    });

    visit_nested_limb_groups(inputs_remapable, [&](auto &i) {
      if (i.is_input()) {
        register_io_limb(i);
      }
      update_next_use(i);
    });

    kernel_count--;
  }
  for (auto &[term, limbs] : program_io_local) {
    for (auto limb : limbs) {
      program_io[term].insert(limb);
    }
  }
  auto dirname = output_prefix + "/" + function_name + "/partition_" +
                 std::to_string(partition_id);
  ensure_directory_created(dirname);
  write_istream_inputs(dirname + "/inputs", partition_id);
  program_io_local.clear();
}

void CeriumCompiler::write_program_inputs() {
  std::stringstream ciphertext_stream, plaintext_stream, scalar_stream,
      output_stream, evalkey_stream;
  std::stringstream function_input_stream, function_output_stream;
  std::stringstream remapable_plaintext_stream;

  auto append_entry = [](std::stringstream &stream, const std::string &entry) {
    stream << entry;
  };

  auto route_input = [&](const auto &term, const std::string &entry) {
    if (term->is_eval_key()) {
      append_entry(evalkey_stream, entry);
      return;
    }

    if (term->is_plaintext()) {
      if (term->is_plaintext_remapable()) {
        append_entry(remapable_plaintext_stream, entry);
      } else if (term->is_scalar()) {
        append_entry(scalar_stream, entry);
      } else {
        append_entry(plaintext_stream, entry);
      }
      return;
    }

    if (term->is_function_arg()) {
      append_entry(function_input_stream, entry);
      return;
    }

    append_entry(ciphertext_stream, entry);
  };

  auto route_output = [&](const auto &term, const std::string &entry) {
    if (term->is_function_arg()) {
      append_entry(function_output_stream, entry);
    } else {
      append_entry(output_stream, entry);
    }
  };

  // for(auto it = Backend::io_limbs_global.begin(); it !=
  // Backend::io_limbs_global.end(); it++){
  for (auto it = program_io.begin(); it != program_io.end(); it++) {
    auto term = it->first;
    const auto entry = format_io_term_entry(term, it->second);
    if (term->is_input()) {
      route_input(term, entry);
    } else if (term->is_output()) {
      route_output(term, entry);
    } else {
      throw std::runtime_error("Program IO term is neither input nor output");
    }
  }
  auto dirname = output_prefix + "/" + function_name;
  ensure_directory_created(dirname);
  const auto program_inputs_path = dirname + "/program_inputs";
  std::ofstream program_inputs_file(program_inputs_path);
  ensure_ofstream_open(program_inputs_file, program_inputs_path);
  const auto remapable_inputs_path = dirname + "/remapable_inputs";
  std::ofstream remapable_plaintexts_file(remapable_inputs_path);
  ensure_ofstream_open(remapable_plaintexts_file, remapable_inputs_path);
  write_labeled_stream(program_inputs_file,
                       "Ciphertext Stream:", ciphertext_stream);
  write_labeled_stream(program_inputs_file,
                       "Plaintext Stream:", plaintext_stream);
  write_labeled_stream(program_inputs_file, "Scalar Stream:", scalar_stream);
  write_labeled_stream(program_inputs_file, "Output Stream:", output_stream);
  write_labeled_stream(program_inputs_file, "Evalkey Stream:", evalkey_stream);
  write_labeled_stream(program_inputs_file,
                       "Function Inputs Stream:", function_input_stream);
  write_labeled_stream(program_inputs_file,
                       "Function Outputs Stream:", function_output_stream);
  write_labeled_stream(remapable_plaintexts_file,
                       "Remapable Stream:", remapable_plaintext_stream);
}

void CeriumCompiler::write_istream_inputs(const std::string &inputs_file_name,
                                            uint32_t partition_id) {

  std::stringstream input_stream, output_stream;
  std::stringstream function_input_stream, function_output_stream;
  std::stringstream remapable_plaintext_stream;

  auto append_entry = [](std::stringstream &stream, const std::string &entry) {
    stream << entry;
  };

  auto route_input = [&](const auto &term, const std::string &entry) {
    if (term->is_function_arg()) {
      append_entry(function_input_stream, entry);
      return;
    }

    if (term->is_plaintext_remapable()) {
      append_entry(remapable_plaintext_stream, entry);
      return;
    }

    append_entry(input_stream, entry);
  };

  auto route_output = [&](const auto &term, const std::string &entry) {
    if (term->is_function_arg()) {
      append_entry(function_output_stream, entry);
    } else {
      append_entry(output_stream, entry);
    }
  };

  for (auto it = program_io_local.begin(); it != program_io_local.end(); it++) {
    auto term = it->first;
    const auto entry = format_io_term_entry(term, it->second);
    if (term->is_input()) {
      route_input(term, entry);
    } else if (term->is_output()) {
      route_output(term, entry);
    } else {
      throw std::runtime_error("Partition IO term is neither input nor output");
    }
  }
  const auto program_inputs_path =
      inputs_file_name + std::to_string(partition_id);
  std::ofstream program_inputs_file(program_inputs_path);
  ensure_ofstream_open(program_inputs_file, program_inputs_path);
  const auto remapable_inputs_path =
      inputs_file_name + "_remapable" + std::to_string(partition_id);
  std::ofstream remapable_plaintexts_file(remapable_inputs_path);
  ensure_ofstream_open(remapable_plaintexts_file, remapable_inputs_path);

  write_labeled_stream(program_inputs_file, "Input Stream:", input_stream);
  write_labeled_stream(program_inputs_file, "Output Stream:", output_stream);
  write_labeled_stream(program_inputs_file,
                       "Function Input Stream:", function_input_stream);
  write_labeled_stream(program_inputs_file,
                       "Function Output Stream:", function_output_stream);
  write_labeled_stream(remapable_plaintexts_file,
                       "Remapable Stream:", remapable_plaintext_stream);
}

} // namespace Backend
} // namespace Cerium
