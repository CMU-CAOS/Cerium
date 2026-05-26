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

#include "cerium/compiler/util/logging.h"
#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdlib>

namespace Cerium {

int getUserVerbosity() {
  static int userVerbosity = 0;
  static bool parsed = false;
  if (!parsed) {
    if (const char *envP = std::getenv("CERIUM_VERBOSITY")) {
      auto envStr = std::string(envP);
      try {
        userVerbosity = std::stoi(envStr);
      } catch (std::invalid_argument e) {
        std::transform(envStr.begin(), envStr.end(), envStr.begin(), ::tolower);
        if (envStr == "silent") {
          userVerbosity = 0;
        } else if (envStr == "info") {
          userVerbosity = (int)Verbosity::Info;
        } else if (envStr == "debug") {
          userVerbosity = (int)Verbosity::Debug;
        } else if (envStr == "trace") {
          userVerbosity = (int)Verbosity::Trace;
        } else {
          std::cerr << "Invalid verbosity CERIUM_VERBOSITY=" << envStr
                    << " Defaulting to silent.\n";
          userVerbosity = 0;
        }
      }
    }
    parsed = true;
  }
  return userVerbosity;
}

void log(Verbosity verbosity, const char *fmt, ...) {
  if (getUserVerbosity() >= (int)verbosity) {
    printf("CERIUM: ");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
  }
}

bool verbosityAtLeast(Verbosity verbosity) {
  return getUserVerbosity() >= (int)verbosity;
}

void warn(const char *fmt, ...) {
  fprintf(stderr, "WARNING: ");
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  fflush(stderr);
}

} // namespace Cerium
