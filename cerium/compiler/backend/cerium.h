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
#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace Cerium {
namespace Backend {

void ceriumCompile(Frontend::Program &program,
                     const uint8_t mod_partitions, const uint64_t num_vregs,
                     const uint64_t num_bcus, const std::string &output_prefix);
void ceriumCompileFunction(Frontend::Function &function,
                             const uint8_t mod_partitions,
                             const uint64_t num_vregs, const uint64_t num_bcus,
                             const std::string &output_prefix);

void keyswitchPass(Frontend::Program &program);
void reductionRewriterPass(Frontend::Function &function);

} // namespace Backend
} // namespace Cerium
