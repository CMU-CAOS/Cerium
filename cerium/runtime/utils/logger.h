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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "cerium/runtime/utils/prefixbuf.h"

namespace Cerium {
namespace Runtime {
namespace Utils {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger : private virtual Prefixbuf, public std::ostream {
public:
  Logger(std::string const &prefix, std::ostream &out)
      : Prefixbuf(prefix, out.rdbuf()),
        std::ios(static_cast<std::streambuf *>(this)),
        std::ostream(static_cast<std::streambuf *>(this)) {}

  Logger(std::string const &prefix)
      : Prefixbuf(prefix, std::cout.rdbuf()),
        std::ios(static_cast<std::streambuf *>(this)),
        std::ostream(static_cast<std::streambuf *>(this)) {}

  Logger &update_line_specific_prefix(const std::string &line_spefic_prefix) {
    this->Prefixbuf::update_line_specific_prefix(line_spefic_prefix);
    return *this;
  }
};

#define LOG(logger, level)                                                     \
  logger.update_line_specific_prefix("[" #level "][" __FILE__ ":" +            \
                                     std::to_string(__LINE__) + "]")

} // namespace Utils
} // namespace Runtime
} // namespace Cerium