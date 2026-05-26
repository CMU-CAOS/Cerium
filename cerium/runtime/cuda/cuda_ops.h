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
#include <cstdlib>
#include <memory>
#include <vector>

#define CHECK_CUDA_ERROR() Cerium::Runtime::CUDA::check_cuda(__FILE__, __LINE__)

#define CHECK_CUDA_ERROR_FUNCTION_ID(FUNCTION_ID)                              \
  Cerium::Runtime::CUDA::check_cuda(__FILE__, __LINE__, (FUNCTION_ID))

namespace Cerium {
namespace Runtime {
namespace CUDA {

class Stream;
using StreamPtr = std::shared_ptr<Stream>;

class Graph;
using GraphPtr = std::shared_ptr<Graph>;
class Node;
using NodePtr = std::shared_ptr<Node>;

class Event;
using EventPtr = std::shared_ptr<Event>;
// using NodePtr = Node *;

void check_cuda(const char *const file, const int line,
                const int function_id = -1);

void allocate_memory_on_device(void **d_alloc, size_t size);
void allocate_memory_on_device_async(void **d_alloc, size_t size,
                                     const StreamPtr &stream);
void memcpyDeviceToHost(void *dst, const void *src, size_t count);
void memcpyHostToDevice(void *dst, const void *src, size_t count);
void memcpyDeviceToDevice(void *dst, const void *src, size_t count);
void memcpyDeviceToHostAsync(void *dst, const void *src, size_t count,
                             const StreamPtr &stream);
void memcpyHostToDeviceAsync(void *dst, const void *src, size_t count,
                             const StreamPtr &stream);
void memcpyHostToDeviceGraph(GraphPtr &graph, NodePtr &node, void *dst,
                             const void *src, size_t count, size_t length,
                             const StreamPtr &stream);
void memcpyDeviceToDeviceAsync(void *dst, const void *src, size_t count,
                               const StreamPtr &stream);
void memcpyHostToHostAsync(void *dst, const void *src, size_t count,
                           const StreamPtr &stream);
void memcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice,
                size_t count);
void free(void *ptr);
void freeAsync(void *ptr, const StreamPtr &stream);
void deviceSynchronize();
void streamSynchronize(const StreamPtr &stream);

void mallocManaged(void **ptr, const size_t size);
void memcpyUVM(void *dst, const void *src, size_t count);
void memPrefetchDeviceToHostAsync(const void *ptr, size_t size,
                                  const StreamPtr &stream);
void memPrefetchHostToDeviceAsync(const void *ptr, size_t size, int deviceId,
                                  const StreamPtr &stream);
void memAdviseSetPreferredLocationHost(const void *ptr, size_t count);
void memAdviseSetPreferredLocationDevice(const void *ptr, size_t count,
                                         int device_id);
// Read-mostly advice is location-independent; CUDA ignores location for it.
void memAdviseSetMostlyReadOnly(const void *ptr, size_t count);

void mallocHost(void **ptr, size_t size);
void freeHost(void *ptr);

void setDevice(int device);

void graphBeginCapture(const StreamPtr &stream, size_t partition);
void graphEndCapture(const StreamPtr &stream, size_t partition);
// void graphLaunch(const StreamPtr & stream, size_t partition);
void graphDebugDotPrint(const size_t partition);
void graphDebugDotPrint(const std::string &prefix, const GraphPtr &graph_,
                        const size_t partition);
void explicitGraphInit(const GraphPtr &graph_, size_t partition);

void addEdge(GraphPtr &graph, const std::vector<NodePtr> &predecessors,
             const std::vector<NodePtr> &children);
void addSingleEdge(GraphPtr &graph, const NodePtr &pred, const NodePtr &child);

StreamPtr createStream();
StreamPtr createStreamNonBlocking();
GraphPtr createGraph();
NodePtr createNode();
NodePtr createEmptyNode(GraphPtr &graph);
NodePtr createMemcpyDeviceToDeviceNode(GraphPtr &graph, void *dst,
                                       const void *src, const size_t count);
NodePtr createSubgraphNode(GraphPtr &graph, GraphPtr &subGraph);

class GraphExec;
using GraphExecPtr = std::shared_ptr<GraphExec>;

GraphExecPtr createGraphExec(const GraphPtr &graph);
void graphLaunch(const GraphExecPtr &graphInstance, const StreamPtr &stream);
void streamBeginCapture(const StreamPtr &stream);
void streamEndCapture(const StreamPtr &stream, GraphPtr &graph);

EventPtr createEvent();
void eventRecord(const EventPtr &event, const StreamPtr &stream);
void streamWaitEvent(const StreamPtr &stream, const EventPtr &event);
void eventSynchronize(const EventPtr &event);

} // namespace CUDA
} // namespace Runtime
} // namespace Cerium
