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

#include "cerium/runtime/memory/memory_manager.h"
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Cerium {
namespace Runtime {

class MemoryManagerCUDA : public MemoryManager {
public:
  enum class MemoryType { Device, Host, HostPageable, Managed } memory_type;

  MemoryManagerCUDA(size_t pool_size, MemoryType memory_type)
      : memory_type(memory_type), MemoryManager() {
    // Base Class function to setup internal structures
    initialize_memory_pool(pool_size);
  };

private:
  std::unique_ptr<uint8_t, std::function<void(void *)>>
  allocate_memory(size_t size) override;
};

using MemoryManagerCUDAHandle = std::shared_ptr<MemoryManagerCUDA>;

} // namespace Runtime
} // namespace Cerium