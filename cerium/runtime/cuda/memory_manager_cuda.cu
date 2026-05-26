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

#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/memory_manager_cuda.h"

#include <cstdlib>
#include <limits>
#include <new>
#include <sys/mman.h>

#include <linux/mman.h>

#include <iterator>
#include <list>

namespace Cerium {
namespace Runtime {

std::unique_ptr<uint8_t, std::function<void(void *)>>
MemoryManagerCUDA::allocate_memory(size_t size) {
  uint8_t *d_alloc = nullptr;
  std::function<void(void *)> deleter;
  switch (memory_type) {
  case MemoryType::Device: {
    CUDA::allocate_memory_on_device((void **)&d_alloc, size);
    deleter = [](void *d_ptr) { CUDA::free(d_ptr); };
  }; break;
  case MemoryType::Host: {
    CUDA::mallocHost((void **)&d_alloc, size);
    deleter = [](void *d_ptr) { CUDA::freeHost(d_ptr); };
  }; break;
  case MemoryType::HostPageable: {
    constexpr size_t huge_page_size = 2 * 1024 * 1024;
    if (size > std::numeric_limits<size_t>::max() - (huge_page_size - 1)) {
      throw std::bad_alloc();
    }
    const size_t allocation_size =
        ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
    void *allocation = mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE |
                                MAP_HUGE_2MB,
                            -1, 0);
    if (allocation == MAP_FAILED) {
      d_alloc = static_cast<uint8_t *>(std::malloc(size));
      if (d_alloc == nullptr) {
        throw std::bad_alloc();
      }
      deleter = [](void *d_ptr) { std::free(d_ptr); };
    } else {
      d_alloc = static_cast<uint8_t *>(allocation);
      deleter = [allocation_size](void *d_ptr) {
        munmap(d_ptr, allocation_size);
      };
    }
  }; break;
  case MemoryType::Managed: {
    CUDA::mallocManaged((void **)&d_alloc, size);
    deleter = [](void *d_ptr) { CUDA::free(d_ptr); };
  }; break;
  default:
    throw std::runtime_error("Invalid Memory Type");
  }
  return std::unique_ptr<uint8_t, std::function<void(void *)>>(d_alloc,
                                                               deleter);
}
} // namespace Runtime
} // namespace Cerium