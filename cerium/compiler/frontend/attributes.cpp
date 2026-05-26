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

#include "cerium/compiler/frontend/attributes.h"
#include <stdexcept>

using namespace std;

namespace Cerium {
namespace Frontend {

#define X(name, type) name::isValid(k, v) ||
bool isValidAttribute(AttributeKey k, const AttributeValue &v) {
  return CERIUM_ATTRIBUTES false;
}
#undef X

#define X(name, type)                                                          \
  case detail::name##Index:                                                    \
    return #name;
string getAttributeName(AttributeKey k) {
  switch (k) {
    CERIUM_ATTRIBUTES
  default:
    throw runtime_error("Unknown attribute key");
  }
}
#undef X

} // namespace Frontend
} // namespace Cerium
