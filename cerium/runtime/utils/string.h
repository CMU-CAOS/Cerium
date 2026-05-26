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
#include <cstdint>
#include <string>
#include <vector>

namespace Cerium {
namespace Runtime {
namespace Utils {
std::vector<std::string> split_string(const std::string &str,
                                      const std::string &delimiter);

uint64_t get_uint_env_variable(const std::string &var,
                               const uint64_t default_val);

} //namespace Utils
} // namespace Runtime
} // namespace Cerium