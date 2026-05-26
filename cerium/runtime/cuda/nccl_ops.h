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
#include "cerium/runtime/cuda/cuda_ops.h"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <vector>

namespace Cerium {
namespace Runtime {
namespace CUDA {
namespace NCCL {
// NCCL functions
void ncclInit(const size_t devices);
void ncclAllGather_uint32(const void *sendbuff, void *recvbuff,
                          size_t sendcount, size_t comm_size, size_t device,
                          const StreamPtr stream = nullptr);
void ncclAllGather_uint32_graph(GraphPtr &graph, NodePtr &node,
                                const void *sendbuff, void *recvbuff,
                                size_t sendcount, size_t comm_size,
                                size_t device,
                                const StreamPtr stream = nullptr);
void ncclReduceScatter_uint32(const void *sendbuff, void *recvbuff,
                              size_t sendcount, size_t comm_size, size_t device,
                              const StreamPtr stream = nullptr);
void ncclReduceScatter_uint32_graph(GraphPtr &graph, NodePtr &node,
                                    const void *sendbuff, void *recvbuff,
                                    size_t sendcount, size_t comm_size,
                                    size_t device,
                                    const StreamPtr stream = nullptr);
void ncclAllReduce_uint32(const void *sendbuff, void *recvbuff,
                          size_t sendcount, size_t comm_size, size_t device,
                          const StreamPtr stream = nullptr);
void ncclAllReduce_uint32_graph(GraphPtr &graph, NodePtr &node,
                                const void *sendbuff, void *recvbuff,
                                size_t sendcount, size_t comm_size,
                                size_t device,
                                const StreamPtr stream = nullptr);

void addEmptyNode(GraphPtr &graph, NodePtr &node);

struct Buffer {
public:
  Buffer(size_t size);
  ~Buffer();
  void *get() const { return buffer_; }
  size_t size() const { return bytes_; }

private:
  void *buffer_;
  size_t bytes_;
};

using BufferPtr = std::shared_ptr<Buffer>;
BufferPtr createNcclBuffer(size_t bytes);
void ncclBufferRegister(size_t device, BufferPtr &buffer);

} // namespace NCCL
} // namespace CUDA
} // namespace Runtime
} // namespace Cerium