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

#include "cerium/runtime/cuda/cuda_ops.cuh"
#include "cerium/runtime/cuda/cuda_ops.h"
#include <iostream>

#include <assert.h>
#include <stdexcept>
#include <sstream>

#include "nccl.h"

namespace Cerium {
namespace Runtime {
namespace CUDA {

void graphDebugDotPrint(const std::string &prefix, const GraphPtr &graph_,
                        const size_t partition) {
  std::string graph_name =
      prefix + "_cuGraph_" + std::to_string(partition) + ".dot";
  cudaGraphDebugDotPrint(graph_->graph_, graph_name.c_str(), 1 << 0);
  CHECK_CUDA_ERROR();
}

void addEdge(GraphPtr &graph, const std::vector<NodePtr> &predecessors,
             const std::vector<NodePtr> &children) {
  std::vector<cudaGraphNode_t> nodes;
  for (auto &node : predecessors) {
    nodes.push_back(node->node_);
  }
  std::vector<cudaGraphNode_t> child_nodes;
  for (auto &child : children) {
    child_nodes.push_back(child->node_);
  }
  cudaGraphAddDependencies(graph->graph_, nodes.data(), child_nodes.data(), nullptr,
                           nodes.size());
  CHECK_CUDA_ERROR();
}

void addSingleEdge(GraphPtr &graph, const NodePtr &pred, const NodePtr &child) {
  cudaGraphAddDependencies(graph->graph_, &(pred->node_), &(child->node_), nullptr, 1);
  CHECK_CUDA_ERROR();
}

void check_cuda(const char *const file, const int line, const int function_id) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA Runtime Error at: " << file << ":" << line
       << " function_id: " << function_id << std::endl;
    ss << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error(ss.str());
  }
}

void allocate_memory_on_device(void **d_alloc, size_t size) {
  cudaMalloc(d_alloc, size);
  CHECK_CUDA_ERROR();
}

void allocate_memory_on_device_async(void **d_alloc, size_t size,
                                     const StreamPtr &stream) {
  cudaMallocAsync(d_alloc, size, stream->stream_);
  CHECK_CUDA_ERROR();
}

void mallocManaged(void **d_alloc, size_t size) {
  cudaMallocManaged(d_alloc, size);
  CHECK_CUDA_ERROR();
}

void memcpyUVM(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyDefault);
  CHECK_CUDA_ERROR();
}

void memPrefetchDeviceToHostAsync(const void *ptr, size_t count,
                                  const StreamPtr &stream) {
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeHost;
  unsigned int flags = 0; // Flags are reserved for future use, must be 0.
  cudaMemPrefetchAsync(ptr, count, location, flags, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memPrefetchHostToDeviceAsync(const void *ptr, size_t count, int deviceID,
                                  const StreamPtr &stream) {
  cudaMemLocation location{};
  location.id = deviceID;
  location.type = cudaMemLocationTypeDevice;
  unsigned int flags = 0; // Flags are reserved for future use, must be 0.
  cudaMemPrefetchAsync(ptr, count, location, flags, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memAdviseSetPreferredLocationHost(const void *ptr, size_t count) {
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeHost;
  cudaMemoryAdvise advice;
  advice = cudaMemAdviseSetPreferredLocation;
  cudaMemAdvise(ptr, count, advice, location);
  CHECK_CUDA_ERROR();
}

void memAdviseSetPreferredLocationDevice(const void *ptr, size_t count,
                                         int device_id) {
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeDevice;
  location.id = device_id;
  cudaMemAdvise(ptr, count, cudaMemAdviseSetPreferredLocation, location);
  CHECK_CUDA_ERROR();
}

void memAdviseSetMostlyReadOnly(const void *ptr, size_t count) {
  // CUDA ignores location for read-mostly advice; pass a valid placeholder.
  cudaMemLocation location{};
  location.type = cudaMemLocationTypeHost;
  cudaMemoryAdvise advice;
  advice = cudaMemAdviseSetReadMostly;
  cudaMemAdvise(ptr, count, advice, location);
  CHECK_CUDA_ERROR();
}

void mallocHost(void **ptr, size_t size) {
  cudaMallocHost(ptr, size);
  CHECK_CUDA_ERROR();
}

void free(void *ptr) {
  cudaFree(ptr);
  CHECK_CUDA_ERROR();
}

void freeAsync(void *ptr, const StreamPtr &stream) {
  cudaFreeAsync(ptr, stream->stream_);
  CHECK_CUDA_ERROR();
}

void freeHost(void *ptr) {
  cudaFreeHost(ptr);
  CHECK_CUDA_ERROR();
}

void memcpyHostToDevice(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();
}

void memcpyDeviceToHost(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();
}

void memcpyDeviceToDevice(void *dst, const void *src, size_t count) {
  cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
  CHECK_CUDA_ERROR();
}

void memcpyDeviceToHostAsync(void *dst, const void *src, size_t count,
                             const StreamPtr &stream) {
  cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memcpyHostToDeviceAsync(void *dst, const void *src, size_t count,
                             const StreamPtr &stream) {
  cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memcpyHostToHostAsync(void *dst, const void *src, size_t count,
                           const StreamPtr &stream) {
  cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memcpyHostToDeviceGraph(GraphPtr &graph, NodePtr &node, void *dst,
                             const void *src, size_t count, size_t length,
                             const StreamPtr &stream) {
  cudaGraphNode_t new_node;
  cudaMemcpy3DParms memcpyParams = {0};
  memcpyParams.srcPtr = make_cudaPitchedPtr((void *)src, count, length, 1);
  memcpyParams.dstPtr = make_cudaPitchedPtr((void *)dst, count, length, 1);
  memcpyParams.extent = make_cudaExtent(count, 1, 1);
  memcpyParams.kind = cudaMemcpyHostToDevice;

  cudaGraphAddMemcpyNode(&new_node, graph->graph_, NULL, 0, &memcpyParams);
  node->node_ = new_node;
  CHECK_CUDA_ERROR();
}

void memcpyDeviceToDeviceAsync(void *dst, const void *src, size_t count,
                               const StreamPtr &stream) {
  cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream->stream_);
  CHECK_CUDA_ERROR();
}

void memcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice,
                size_t count) {
  cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
  CHECK_CUDA_ERROR();
}

void deviceSynchronize() {
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
}

void streamSynchronize(const StreamPtr &stream) {
  cudaStreamSynchronize(stream->stream_);
  CHECK_CUDA_ERROR();
}

void setDevice(int device) {
  cudaSetDevice(device);
  CHECK_CUDA_ERROR();
}

Stream::Stream() {
  cudaStreamCreate(&stream_);
  CHECK_CUDA_ERROR();
}

Stream::Stream(bool blocking) {
  if (!blocking) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  } else {
    cudaStreamCreate(&stream_);
  }
  CHECK_CUDA_ERROR();
}

Stream::~Stream() { cudaStreamDestroy(stream_); }

std::shared_ptr<Stream> createStream() { return std::make_shared<Stream>(); }
std::shared_ptr<Stream> createStreamNonBlocking() {
  return std::make_shared<Stream>(false);
}

Graph::Graph() {
  cudaGraphCreate(&graph_, 0);
  CHECK_CUDA_ERROR();
}

Graph::~Graph() { cudaGraphDestroy(graph_); }

std::shared_ptr<Graph> createGraph() { return std::make_shared<Graph>(); }

Node::Node() : node_(nullptr) {}

EventPtr createEvent() { return std::make_shared<Event>(); }

Event::Event() {
  cudaEventCreate(&event_);
  CHECK_CUDA_ERROR();
}

void eventRecord(const EventPtr &event, const StreamPtr &stream) {
  cudaEventRecord(event->event_, stream->stream_);
  CHECK_CUDA_ERROR();
}

void streamWaitEvent(const StreamPtr &stream, const EventPtr &event) {
  cudaStreamWaitEvent(stream->stream_, event->event_, 0);
  CHECK_CUDA_ERROR();
}

void eventSynchronize(const EventPtr &event) {
  cudaEventSynchronize(event->event_);
  CHECK_CUDA_ERROR();
}

Event::~Event() { cudaEventDestroy(event_); }

NodePtr createNode() { return std::make_shared<Node>(); }

NodePtr createEmptyNode(GraphPtr &graph) {
  auto node = createNode();
  cudaGraphAddEmptyNode(&node->node_, graph->graph_, NULL, 0);
  CHECK_CUDA_ERROR();
  return node;
}

NodePtr createMemcpyDeviceToDeviceNode(GraphPtr &graph, void *dst,
                                       const void *src, const size_t count) {
  auto node = createNode();
  cudaGraphAddMemcpyNode1D(&node->node_, graph->graph_, NULL, 0, dst, src,
                           count, cudaMemcpyDeviceToDevice);
  CHECK_CUDA_ERROR();
  return node;
}

NodePtr createSubgraphNode(GraphPtr &graph, GraphPtr &subGraph) {
  auto node = createNode();
  cudaGraphAddChildGraphNode(&node->node_, graph->graph_, NULL, 0,
                             subGraph->graph_);
  CHECK_CUDA_ERROR();
  return node;
}

GraphExec::GraphExec() : instance_graph_(nullptr){};
GraphExec::~GraphExec() {
  if (instance_graph_) {
    cudaGraphExecDestroy(instance_graph_);
  }
  instance_graph_ = nullptr;
}

GraphExecPtr createGraphExec(const GraphPtr &graph) {
  auto graph_exec = std::make_shared<GraphExec>();
  cudaGraphInstantiate(&graph_exec->instance_graph_, graph->graph_, 0);
  CHECK_CUDA_ERROR();
  return graph_exec;
}

void graphLaunch(const GraphExecPtr &graphInstance, const StreamPtr &stream) {
  cudaGraphLaunch(graphInstance->instance_graph_, stream->stream_);
  CHECK_CUDA_ERROR();
}

void streamBeginCapture(const StreamPtr &stream) {
  cudaStreamBeginCapture(stream->stream_, cudaStreamCaptureModeThreadLocal);
  CHECK_CUDA_ERROR();
}

void streamEndCapture(const StreamPtr &stream, GraphPtr &graph) {
  cudaStreamEndCapture(stream->stream_, &graph->graph_);
  CHECK_CUDA_ERROR();
}

} // namespace CUDA
} // namespace Runtime
} // namespace Cerium
