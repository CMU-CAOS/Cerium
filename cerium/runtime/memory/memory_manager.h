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

#include <cstddef>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Cerium {
namespace Runtime {

// Arena Allocator for Memory Management. Derived classes must implement allocate_memory to specify how memory is allocated for the pool.
class MemoryManager {
private:
  struct Block {
    size_t size;
    bool used{false};
    uint8_t *start;
    Block(uint8_t *start, size_t size) : size(size), start(start){};
  };

  std::list<Block> blocks;
  size_t pool_size{0};
  std::unique_ptr<uint8_t, std::function<void(void *)>> pool{nullptr};
  size_t total_allocated{0};
  decltype(blocks)::iterator next_allocate;

  MemoryManager(const MemoryManager &) = delete;
  inline auto &operator=(MemoryManager &assign) = delete;

public:
  MemoryManager() = default;
  ~MemoryManager();

  void *allocate(size_t size);

private:
  bool pool_initialized{false};

protected:
  // derived classes must call initialize memory pool after allocating memory
  void initialize_memory_pool(size_t pool_size);
  virtual std::unique_ptr<uint8_t, std::function<void(void *)>>
  allocate_memory(size_t size) = 0;
};

using MemoryManagerHandle = std::shared_ptr<MemoryManager>;

} // namespace Runtime
} // namespace Cerium