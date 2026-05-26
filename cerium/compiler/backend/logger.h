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
#include <fmt/core.h>

namespace Cerium {
namespace Backend {

template <typename... Args>
void log(const char *file, uint64_t line, const char *fmt, Args... args) {
  if (0) {
    fmt::print("LOG {}:{}: ", file, line);
    fmt::print(fmt, args...);
    fmt::print("\n");
    fflush(stdout);
  }
}

template <typename... Args>
void log2(const char *file, uint64_t line, const char *fmt, Args... args) {
  return;
  fmt::print("LOG {}:{}: ", file, line);
  fmt::print(fmt, args...);
  // printf("LOG %lu: ",line);
  // va_list args;
  // va_start(args, fmt);
  // vprintf(fmt, args);
  // va_end(args);
  // printf("\n");
  fmt::print("\n");
  fflush(stdout);
}
template <typename... Args> void log3(const char *fmt, Args... args) {
  return;
  fmt::print(fmt, args...);
  // printf("LOG %lu: ",line);
  // va_list args;
  // va_start(args, fmt);
  // vprintf(fmt, args);
  // va_end(args);
  // printf("\n");
  fmt::print("\n");
  fflush(stdout);
}

#define CL_LOG(format, args...)                                                \
  do {                                                                         \
    Backend::log(__FILE__, __LINE__, format, args);                            \
  } while (0);
#define CL_LOG2(format, args...)                                               \
  do {                                                                         \
    Backend::log2(__FILE__, __LINE__, format, args);                           \
  } while (0);
#define CL_LOG3(format, args...)                                               \
  do {                                                                         \
    Backend::log3(format, args);                                               \
  } while (0);

bool get_boolean_env_variable(const std::string &var);

} // namespace Backend
} // namespace Cerium