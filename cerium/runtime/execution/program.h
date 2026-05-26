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
#include "cerium/runtime/execution/function.h"
#include "cerium/runtime/utils/logger.h"

#include <fstream>

namespace Cerium {
namespace Runtime {

class Program {
public:
  using LimbT = DeviceLimb;
  using LimbPtrT = DeviceLimbPtr;

  Program(const Context &context, const std::string &directory_base,
          const std::string &top_level_function)
      : context_(context), directory_base(directory_base),
        top_level_function(top_level_function), logger("[Program]") {}

  std::map<std::string, std::shared_ptr<CeriumFunction>> make_cerium_functions(
      uint32_t num_gpus,
      const std::map<std::string, std::map<std::string, std::string>> &config,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_wrapper);

  std::map<std::string, std::shared_ptr<CeriumFunction>> make_program(
      uint32_t num_gpus,
      const std::map<std::string, std::map<std::string, std::string>> &config,
      const Cerium::Runtime::Utils::RawInputsWrapperPtr &raw_inputs_wrapper);

  void run_program();

private:
  std::vector<DevicePointer<LimbT::Element_t>> register_files_memory_;
  std::vector<std::vector<std::unique_ptr<Runtime::DeviceBaseConverter>>>
      base_conversion_units_memory_;
  std::vector<std::shared_ptr<Runtime::EvaluatorContext>>
      evaluator_contexts;
  const Context &context_;
  std::map<std::string, std::shared_ptr<CeriumFunction>> cerium_functions;
  std::string directory_base;
  std::string top_level_function;
  std::map<size_t, std::vector<std::string>, std::greater<size_t>>
      call_graph_by_depth;
  std::map<std::string, std::vector<std::string>> call_graph;
  std::unordered_map<std::string, bool> use_cudagraphs;

  std::vector<std::string> get_calls(const std::string &function_name,
                                     const std::string &function_dir);
  std::map<std::string, std::string>
  get_config(const std::string &function_name, const std::string &function_dir);
  Cerium::Runtime::Utils::Logger logger;
};

} // namespace Runtime
} // namespace Cerium