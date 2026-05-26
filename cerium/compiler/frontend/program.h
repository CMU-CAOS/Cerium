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

#pragma once

#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term.h"
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Cerium {
namespace Frontend {

template <typename> class TermMapOptional;
template <typename> class TermMap;
class TermMapBase;

class Program {

public:
  Program(const Program &copy) = delete;

  Program &operator=(const Program &assign) = delete;

  Program(std::string name, std::uint32_t rnsBitSize,
          std::uint8_t partitionSize)
      : name(name), rnsBitSize(rnsBitSize), currentPartitionSize(partitionSize),
        currentPartitionId(0) {
    if (partitionSize == 0) {
      throw std::runtime_error("Partition Size cannot be zero");
    }
    if (rnsBitSize != 28) {
      throw std::runtime_error(
          "Unsupported RNS Bit Size: " + std::to_string(rnsBitSize) +
          ". Only 28bit RNS primes are supported as of now.");
    }
  }

  void addFunction(std::string name, std::shared_ptr<Function> function) {
    if (functions.find(name) != functions.end()) {
      throw std::runtime_error("Function with name " + name +
                               " already exists");
    }
    functions[name] = function;
  }

  std::shared_ptr<Function> getFunction(std::string name) {
    auto iter = functions.find(name);
    if (iter == functions.end()) {
      throw std::runtime_error("Function with name " + name +
                               " does not exist");
    }
    return iter->second;
  }

  std::unordered_map<std::string, std::shared_ptr<Function>> getFunctions() {
    return functions;
  }

  std::string toDOT() const;

  std::string getName() const { return name; }

private:
  std::unordered_map<std::string, std::shared_ptr<Function>> functions;

  std::string name;

  std::uint32_t rnsBitSize;

  std::uint8_t currentPartitionSize;
  std::uint8_t currentPartitionId;

  std::uint32_t slotSize = 32 * 1024;

  friend class Function;
};

} // namespace Frontend
} // namespace Cerium
