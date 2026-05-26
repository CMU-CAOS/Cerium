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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <dlfcn.h>

namespace Cerium {
namespace Runtime {
namespace Utils {

class DLHandler {

public:
  DLHandler(const std::string &library_path) {
    handle_ = dlopen(library_path.c_str(), RTLD_LAZY);
    if (!handle_) {
      throw std::runtime_error("Failed to load library: " + library_path +
                               " with error: " + dlerror());
    }
  }

  ~DLHandler() {
    if (handle_) {
      dlclose(handle_);
    }
  }

  DLHandler(const DLHandler &) = delete;
  DLHandler &operator=(const DLHandler &) = delete;
  DLHandler(DLHandler &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  void *get() const {
    if (!handle_) {
      throw std::runtime_error("Library not loaded");
    }
    return handle_;
  }

private:
  void *handle_{nullptr};
};

} // namespace Utils
} // namespace Runtime
} // namespace Cerium