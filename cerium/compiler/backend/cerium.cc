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

#include "cerium/compiler/util/logging.h"
#include "cerium/compiler/util/program_traversal.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cerium/compiler/backend/cerium.h"
#include "cerium/compiler/backend/compiler.h"
#include "cerium/compiler/backend/keyswitch_pass.h"
#include "cerium/compiler/util/mkdir.h"

using namespace std;

namespace Cerium {
namespace Backend {

void ceriumCompile(Frontend::Program &program,
                     const uint8_t num_partitions, const uint64_t num_vregs,
                     const uint64_t num_bcus,
                     const std::string &output_prefix) {
  for (auto &[k, v] : program.getFunctions()) {
    Function &f = *v;
    ceriumCompileFunction(f, num_partitions, num_vregs, num_bcus,
                            output_prefix);
  }
}

void ceriumCompileFunction(Frontend::Function &function,
                             const uint8_t num_partitions,
                             const uint64_t num_vregs, const uint64_t num_bcus,
                             const std::string &output_prefix) {

  static const uint64_t skip_receive_reorder =
      Cerium::Util::getEnvVarUint32("CERIUM_SKIP_RECEIVE_REORDER", 0);
  if (skip_receive_reorder == 1) {
    FunctionTraversalReceiveReorder programTraverse(function);
    auto ceriumCompiler =
        CeriumCompiler(function, function.getPartitionSize(), num_vregs,
                        output_prefix, num_bcus);
    programTraverse.forwardPass(ceriumCompiler);
    ceriumCompiler.finish();
  } else {
    if (function.getPartitionSize() > 1) {
      FunctionTraversal FunctionTraversal(function);
      auto commonReceiveEliminatorPass =
          CeriumP::CommonReceiveEliminatorPass(function);
      FunctionTraversal.forwardPass(commonReceiveEliminatorPass);
      reductionRewriterPass(function);
    }
    FunctionTraversalReceiveReorder programTraverse(function);
    auto ceriumCompiler =
        CeriumCompiler(function, function.getPartitionSize(), num_vregs,
                        output_prefix, num_bcus);
    programTraverse.forwardPass(ceriumCompiler);
    ceriumCompiler.finish();
  }



  std::string output_directory = output_prefix + "/" + function.getName();
  Cerium::Util::mkdir_p(output_directory);

  const std::string calls_path = output_directory + "/calls";
  std::ofstream function_file(calls_path);
  if (!function_file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + calls_path);
  }
  function_file << "{\n";
  for (auto &fns : function.getCalls()) {
    function_file << "\t" << fns->getName() << "\n";
  }
  function_file << "}";
  function_file.close();

  const std::string compile_config_path = output_directory + "/compile_config";
  std::ofstream compile_config_file(compile_config_path);
  if (!compile_config_file.is_open()) {
    throw std::runtime_error("Failed to open output file: " +
                             compile_config_path);
  }
  compile_config_file << "{\n";
  compile_config_file << "\tfunction: " << function.getName() << "\n";
  compile_config_file << "\tvregs: " << num_vregs << "\n";
  compile_config_file << "\tbcus: " << num_bcus << "\n";
  compile_config_file << "\tgpus: "
                      << static_cast<uint32_t>(function.getPartitionSize())
                      << "\n";
  compile_config_file << "\tblock_size: " << BLOCK_SIZE << "\n";
  compile_config_file << "}";
  compile_config_file.close();
}

void keyswitchPass(Frontend::Program &program) {
}
void reductionRewriterPass(Frontend::Function &function) {
  static const uint64_t run_reduction_rewriter =
      Cerium::Util::getEnvVarUint32("CERIUM_REDUCTION_REWRITE", 1);
  if (run_reduction_rewriter == 0) {
    return;
  }
  std::cout << "Running Cerium Reduction Rewriter Pass\n" << std::flush;
  FunctionTraversal functionTraverse(function);
  functionTraverse.backwardPass(CeriumP::ReductionRewriterPass(function));
}
} // namespace Backend
} // namespace Cerium
