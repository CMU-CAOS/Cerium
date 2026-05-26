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

#include "cerium/runtime/utils/string.h"
#include <cassert>

namespace Cerium {
namespace Runtime {
namespace Utils {

std::vector<std::string> split_string(const std::string &str,
                                      const std::string &delimiter) {
  std::vector<std::string> strings;

  std::string::size_type pos = 0;
  std::string::size_type prev = 0;
  while ((pos = str.find(delimiter, prev)) != std::string::npos) {
    const auto &substr = str.substr(prev, pos - prev);
    if (substr != "") {
      strings.push_back(substr);
    }
    prev = pos + delimiter.size();
  }

  // To get the last substring (or only, if delimiter is not found)
  const auto &substr = str.substr(prev);
  if (substr != "") {
    strings.push_back(substr);
  }
  return strings;
}

uint64_t get_uint_env_variable(const std::string &var,
                               const uint64_t default_val) {
  if (const char *env_p = std::getenv(var.c_str())) {
    auto env_str = std::string(env_p);
    return std::stoul(env_str);
  }
  return default_val;
}

} // namespace Utils
} // namespace Runtime
} // namespace Cerium