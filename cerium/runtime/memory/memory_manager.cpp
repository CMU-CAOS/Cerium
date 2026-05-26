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

#include "cerium/runtime/memory/memory_manager.h"

#include <iostream>
#include <iterator>
#include <list>
#include <stdexcept>

namespace Cerium {
namespace Runtime {

void MemoryManager::initialize_memory_pool(size_t pool_size_) {
  pool_size = pool_size_;
  pool = allocate_memory(pool_size);
  blocks.emplace_back(MemoryManager::Block(pool.get(), pool_size));
  pool_initialized = true;
  next_allocate = blocks.begin();
}

MemoryManager::~MemoryManager() {
  blocks.clear();
  pool = nullptr;
  next_allocate = blocks.begin();
}

void *MemoryManager::allocate(size_t size) {
  //TODO: Make This Thread Safe
  if (!pool_initialized) {
    throw std::runtime_error("Memory Pool Not Initialized");
  }
  if (total_allocated + size > pool_size) {
    throw std::runtime_error("Memory Pool Insufficient for alloc");
  }

  // Fast path so that we don't always iterate list to the end
  auto it = next_allocate;
  for (; it != blocks.end(); it++) {
    auto &block = *it;
    if (block.used) {
      continue;
    }
    if (block.size < size) {
      continue;
    }
    size_t remaining_size = block.size - size;
    block.size = size;
    block.used = true;
    auto alloc = block.start;
    next_allocate = blocks.emplace(std::next(it),
                                   Block(block.start + size, remaining_size));
    total_allocated += size;
    return alloc;
  }
  auto it2 = blocks.begin();
  for (; it2 != blocks.end(); it2++) {
    auto &block = *it2;
    if (block.used) {
      continue;
    }
    if (block.size < size) {
      continue;
    }
    size_t remaining_size = block.size - size;
    block.size = size;
    block.used = true;
    auto alloc = block.start;
    next_allocate = blocks.emplace(std::next(it2),
                                   Block(block.start + size, remaining_size));
    total_allocated += size;
    return alloc;
  }
  throw std::bad_alloc();
}

} // namespace Runtime
} // namespace Cerium