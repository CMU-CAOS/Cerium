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

#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace Cerium {
namespace Runtime {

inline std::ifstream open_input_file(const std::string &file_name) {
  std::ifstream file(file_name, std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + file_name);
  }
  return file;
}

template <typename Fn>
void capture_thread_exception(Fn &&fn, std::exception_ptr &exception,
                              std::mutex &mutex) {
  try {
    fn();
  } catch (const std::exception &) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!exception) {
      exception = std::current_exception();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!exception) {
      exception = std::make_exception_ptr(std::runtime_error(
          "Non-standard exception thrown from worker thread"));
    }
  }
}

inline void rethrow_thread_exception(const std::exception_ptr &exception) {
  if (exception) {
    std::rethrow_exception(exception);
  }
}

} // namespace Runtime
} // namespace Cerium
