// Copyright contributors to the EVA project
// Licensed under the MIT License.
// Original source: https://github.com/microsoft/EVA
//
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

#include "cerium/compiler/frontend/function.h"
#include "cerium/compiler/frontend/term_map.h"
#include "cerium/compiler/util/logging.h"
#include "cerium/compiler/util/program_traversal.h"
#include <stack>

namespace Cerium {
namespace Frontend {

std::string Program::toDOT() const {
  std::stringstream s;
  s << "digraph \"" << getName() << "\" {\n";
  for (auto &[k, v] : functions) {
    v->toDOT(s);
  }
  s << "}\n";

  return s.str();
}
} // namespace Frontend
} // namespace Cerium