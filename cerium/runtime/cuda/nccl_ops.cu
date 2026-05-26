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

#include <iostream>
#include <memory>
#include <thread>

#include "cerium/runtime/cuda/cuda_ops.cuh"
#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/nccl_ops.cuh"
#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/utils/string.h"
#include "nccl.h"
#include <assert.h>
#include <map>
#include <mutex>

namespace Cerium {
namespace Runtime {
namespace CUDA {
namespace NCCL {

namespace {

void checkCudaResult(cudaError_t err, const char *file, const int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA Runtime Error at: ") + file +
                             ":" + std::to_string(line) + "\n" +
                             cudaGetErrorString(err));
  }
}

void checkNcclResult(ncclResult_t res, const char *file, const int line) {
  if (res != ncclSuccess) {
    throw std::runtime_error(std::string("NCCL error at ") + file + ":" +
                             std::to_string(line) + " '" +
                             ncclGetErrorString(res) + "'");
  }
}

void enablePeerAccess(const size_t devices) {
  for (size_t src = 0; src < devices; src++) {
    checkCudaResult(cudaSetDevice(src), __FILE__, __LINE__);
    for (size_t dst = 0; dst < devices; dst++) {
      if (src == dst) {
        continue;
      }

      int can_access_peer = 0;
      checkCudaResult(cudaDeviceCanAccessPeer(&can_access_peer, src, dst),
                      __FILE__, __LINE__);
      if (!can_access_peer) {
        continue;
      }

      auto status = cudaDeviceEnablePeerAccess(dst, 0);
      if (status == cudaErrorPeerAccessAlreadyEnabled) {
        continue;
      }
      checkCudaResult(status, __FILE__, __LINE__);
    }
  }
}

} // namespace

std::map<size_t, std::map<size_t, std::vector<std::unique_ptr<NCCL_COMM>>>>
    comms;
std::map<size_t, std::vector<NodePtr>> prev_ncc;
bool nccl_init = false;
bool NCCL_DESTROY_NODES = Cerium::Runtime::Utils::get_uint_env_variable(
    "CERIUM_RUNTIME_DESTROY_NCCL_NODES", 0);

size_t NUM_COMMS = 1;
std::vector<size_t> last_comm;

namespace {

void rethrowThreadException(const std::exception_ptr &exception) {
  if (exception) {
    std::rethrow_exception(exception);
  }
}

void warmupNcclCollectives(const size_t devices) {
  for (size_t n = 0; n < NUM_COMMS; n++) {
    for (size_t dev = devices; dev > 1; dev /= 2) {
      for (size_t group_start = 0; group_start < devices; group_start += dev) {
        std::exception_ptr thread_exception;
        std::mutex thread_exception_mutex;
        std::vector<std::thread> threads;

        for (size_t rank = 0; rank < dev; rank++) {
          const size_t device = group_start + rank;
          threads.emplace_back([&, device, dev, n]() {
            try {
              checkCudaResult(cudaSetDevice(device), __FILE__, __LINE__);

              cudaStream_t stream = nullptr;
              checkCudaResult(cudaStreamCreateWithFlags(&stream,
                                                        cudaStreamNonBlocking),
                              __FILE__, __LINE__);

              uint32_t *send_buffer = nullptr;
              uint32_t *recv_buffer = nullptr;
              checkCudaResult(cudaMalloc(&send_buffer, dev * sizeof(uint32_t)),
                              __FILE__, __LINE__);
              checkCudaResult(cudaMalloc(&recv_buffer, dev * sizeof(uint32_t)),
                              __FILE__, __LINE__);
              checkCudaResult(cudaMemsetAsync(send_buffer, 0,
                                              dev * sizeof(uint32_t), stream),
                              __FILE__, __LINE__);
              checkCudaResult(cudaMemsetAsync(recv_buffer, 0,
                                              dev * sizeof(uint32_t), stream),
                              __FILE__, __LINE__);

              auto comm = comms[n][dev][device]->get_comm();
              checkNcclResult(ncclAllGather(send_buffer, recv_buffer, 1,
                                            ncclUint32, comm, stream),
                              __FILE__, __LINE__);
              checkNcclResult(ncclReduceScatter(send_buffer, recv_buffer, 1,
                                                ncclUint32, ncclSum, comm,
                                                stream),
                              __FILE__, __LINE__);
              checkNcclResult(ncclAllReduce(send_buffer, recv_buffer, 1,
                                            ncclUint32, ncclSum, comm, stream),
                              __FILE__, __LINE__);

              checkCudaResult(cudaStreamSynchronize(stream), __FILE__,
                              __LINE__);
              checkCudaResult(cudaFree(send_buffer), __FILE__, __LINE__);
              checkCudaResult(cudaFree(recv_buffer), __FILE__, __LINE__);
              checkCudaResult(cudaStreamDestroy(stream), __FILE__, __LINE__);
            } catch (...) {
              std::lock_guard<std::mutex> lock(thread_exception_mutex);
              if (!thread_exception) {
                thread_exception = std::current_exception();
              }
            }
          });
        }

        for (auto &thread : threads) {
          thread.join();
        }
        rethrowThreadException(thread_exception);
      }
    }
  }

  for (size_t device = 0; device < devices; device++) {
    checkCudaResult(cudaSetDevice(device), __FILE__, __LINE__);
    checkCudaResult(cudaDeviceSynchronize(), __FILE__, __LINE__);
  }
}

} // namespace

void ncclInit(const size_t devices) {
  if (nccl_init) {
    return;
  }
  if (devices < 2) {
    return;
  }

  last_comm.resize(devices);

  std::vector<int> device_ids(devices);
  for (size_t i = 0; i < devices; i++) {
    device_ids[i] = i;
    last_comm[i] = 0;
  }

  enablePeerAccess(devices);

  ncclConfig_t nccl_config = NCCL_CONFIG_INITIALIZER;
  nccl_config.blocking = 0;
  nccl_config.collnetEnable = 1;
  // NCCL_CTA_POLICY_ZERO to enable zero-CTA optimization whenever possible
  // nccl_config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  // nccl_config.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
  nccl_config.minCTAs = 32;

  for (int n = 0; n < NUM_COMMS; n++) {

    prev_ncc[n].resize(devices);

    size_t dev = devices;
    if (dev != 1 && dev != 2 && dev != 4 && dev != 8) {
      throw std::runtime_error("Only 1,2,4 or 8 devices are supported, but " +
                               std::to_string(dev) + " were requested");
    }
    for (; dev > 1; dev /= 2) {
      comms[n][dev].resize(devices);
      std::vector<ncclComm_t> comms_init(devices);
      for (size_t j = 0; j < devices; j += dev) {
        auto thread_fn = [&](int n, int device, int nRanks, ncclUniqueId id,
                             int rank, ncclConfig_t *config) {
          CUDA::setDevice(device);
          comms[n][nRanks][device] =
              std::make_unique<NCCL_COMM>(nRanks, id, rank, config);
        };
        ncclUniqueId id;
        NCCLCHECK(ncclGetUniqueId(&id));
        std::vector<std::thread> threads;
        for (int k = 0; k < dev; k++) {
          threads.push_back(
              std::thread(thread_fn, n, j + k, dev, id, k, &nccl_config));
        }

        for (int k = 0; k < dev; k++) {
          threads[k].join();
        }
        // std::cout << "dev: " << dev << " j: " << j << "\n";
        // NCCLCHECK(ncclCommInitAll(&(comms[n][dev])[j], dev, &device_ids[j]));
      }
    }
  }

  warmupNcclCollectives(devices);

  nccl_init = true;

  //initializing NCCL
  // NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
}

void ncclAllGather_uint32(const void *sendbuff, void *recvbuff,
                          size_t sendcount, size_t comm_size, size_t device,
                          const StreamPtr stream) {
  auto n = last_comm[device]++ % NUM_COMMS;
  auto cmd = ncclAllGather(sendbuff, recvbuff, sendcount, ncclUint32,
                           comms[n][comm_size][device]->get_comm(),
                           stream ? stream->stream_ : nullptr);
  NCCLCHECK(cmd);
  // CUDA::streamSynchronize(stream);
  CHECK_CUDA_ERROR();
}

void // __attribute__((optimize("O0")))
ncclAllGather_uint32_graph(GraphPtr &graph, NodePtr &node, const void *sendbuff,
                           void *recvbuff, size_t sendcount, size_t comm_size,
                           size_t device, const StreamPtr stream) {

  auto n = last_comm[device]++ % NUM_COMMS;
  auto capture_status =
      cudaStreamBeginCapture(stream->stream_, cudaStreamCaptureModeThreadLocal);
  checkCudaResult(capture_status, __FILE__, __LINE__);
  auto cmd = ncclAllGather(sendbuff, recvbuff, sendcount, ncclUint32,
                           comms[n][comm_size][device]->get_comm(),
                           stream ? stream->stream_ : nullptr);
  cudaGraph_t graph_gather_ = nullptr;
  auto end_status = cudaStreamEndCapture(stream->stream_, &graph_gather_);
  if (cmd != ncclSuccess) {
    if (end_status == cudaSuccess && graph_gather_) {
      cudaGraphDestroy(graph_gather_);
    }
    checkNcclResult(cmd, __FILE__, __LINE__);
  }
  checkCudaResult(end_status, __FILE__, __LINE__);

  if (NCCL_DESTROY_NODES) {
    cudaGraphNode_t *nodes = nullptr;
    size_t numNodes = 3;
    nodes = (cudaGraphNode_t *)malloc(numNodes * sizeof(cudaGraphNode_t));
    cudaGraphGetNodes(graph_gather_, nodes, &numNodes);
    CHECK_CUDA_ERROR();
    for (size_t i = 0; i < numNodes; i++) {
      cudaGraphNodeType nodeType;
      cudaGraphNodeGetType(nodes[i], &nodeType);
      CHECK_CUDA_ERROR();
      if (nodeType != cudaGraphNodeTypeKernel) {
        // cudaGraphDestroyNode ( nodes[i]);
        CHECK_CUDA_ERROR();
      }
    }
  }
  // free((void *)nodes);

  cudaGraphAddChildGraphNode(&node->node_, graph->graph_, nullptr, 0,
                             graph_gather_);
  CHECK_CUDA_ERROR();
  if (prev_ncc[n][device]) {
    // addSingleEdge(graph,prev_ncc[n][device],node);
    CHECK_CUDA_ERROR();
  }
  prev_ncc[n][device] = node;
  CHECK_CUDA_ERROR();
  cudaGraphDestroy(graph_gather_);
  CHECK_CUDA_ERROR();
}

void ncclReduceScatter_uint32(const void *sendbuff, void *recvbuff,
                              size_t recvcount, size_t comm_size, size_t device,
                              const StreamPtr stream) {
  // auto cmd = ncclAllGather(sendbuff, recvbuff, sendcount, ncclUint32, comms[device], stream ? stream->stream_ : nullptr);
  assert(stream);
  auto n = last_comm[device]++ % NUM_COMMS;
  auto cmd = ncclReduceScatter(sendbuff, recvbuff, recvcount, ncclUint32,
                               ncclSum, comms[n][comm_size][device]->get_comm(),
                               stream ? stream->stream_ : nullptr);
  // auto cmd = ncclAllReduce(sendbuff, recvbuff, recvcount, ncclUint32, ncclSum, comms[device], stream ? stream->stream_ : nullptr);
  NCCLCHECK(cmd);
  // CUDA::streamSynchronize(stream);
  CHECK_CUDA_ERROR();
}

void ncclAllReduce_uint32(const void *sendbuff, void *recvbuff,
                          size_t recvcount, size_t comm_size, size_t device,
                          const StreamPtr stream) {
  auto n = last_comm[device]++ % NUM_COMMS;
  auto cmd = ncclAllReduce(sendbuff, recvbuff, recvcount, ncclUint32, ncclSum,
                           comms[n][comm_size][device]->get_comm(),
                           stream ? stream->stream_ : nullptr);
  NCCLCHECK(cmd);
  CHECK_CUDA_ERROR();
}

void // __attribute__((optimize("O0")))
ncclReduceScatter_uint32_graph(GraphPtr &graph, NodePtr &node,
                               const void *sendbuff, void *recvbuff,
                               size_t recvcount, size_t comm_size,
                               size_t device, const StreamPtr stream) {
  auto n = last_comm[device]++ % NUM_COMMS;
  auto capture_status =
      cudaStreamBeginCapture(stream->stream_, cudaStreamCaptureModeThreadLocal);
  checkCudaResult(capture_status, __FILE__, __LINE__);
  auto cmd = ncclReduceScatter(sendbuff, recvbuff, recvcount, ncclUint32,
                               ncclSum, comms[n][comm_size][device]->get_comm(),
                               stream ? stream->stream_ : nullptr);
  cudaGraph_t graph_gather_ = nullptr;
  auto end_status = cudaStreamEndCapture(stream->stream_, &graph_gather_);
  if (cmd != ncclSuccess) {
    if (end_status == cudaSuccess && graph_gather_) {
      cudaGraphDestroy(graph_gather_);
    }
    checkNcclResult(cmd, __FILE__, __LINE__);
  }
  checkCudaResult(end_status, __FILE__, __LINE__);

  if (NCCL_DESTROY_NODES) {
    cudaGraphNode_t *nodes = nullptr;
    size_t numNodes = 3;
    nodes = (cudaGraphNode_t *)malloc(numNodes * sizeof(cudaGraphNode_t));
    cudaGraphGetNodes(graph_gather_, nodes, &numNodes);
    CHECK_CUDA_ERROR();
    for (size_t i = 0; i < numNodes; i++) {
      cudaGraphNodeType nodeType;
      cudaGraphNodeGetType(nodes[i], &nodeType);
      CHECK_CUDA_ERROR();
      if (nodeType != cudaGraphNodeTypeKernel) {
        // cudaGraphDestroyNode ( nodes[i]);
        CHECK_CUDA_ERROR();
      }
    }
    // free((void *) nodes);
  }

  cudaGraphAddChildGraphNode(&node->node_, graph->graph_, nullptr, 0,
                             graph_gather_);
  CHECK_CUDA_ERROR();
  if (prev_ncc[n][device]) {
    // addSingleEdge(graph,prev_ncc[n][device],node);
    CHECK_CUDA_ERROR();
  }
  prev_ncc[n][device] = node;
  CHECK_CUDA_ERROR();
  cudaGraphDestroy(graph_gather_);
  CHECK_CUDA_ERROR();
}
void // __attribute__((optimize("O0")))
ncclAllReduce_uint32_graph(GraphPtr &graph, NodePtr &node, const void *sendbuff,
                           void *recvbuff, size_t recvcount, size_t comm_size,
                           size_t device, const StreamPtr stream) {
  if (stream == nullptr || stream->stream_ == NULL) {
    throw std::runtime_error(
        "Stream is null in ncclAllReduce_uint32_graph: tid " +
        std::to_string(device));
  }

  auto n = last_comm[device]++ % NUM_COMMS;
  auto capture_status =
      cudaStreamBeginCapture(stream->stream_, cudaStreamCaptureModeThreadLocal);
  checkCudaResult(capture_status, __FILE__, __LINE__);
  // auto cmd = ncclReduceScatter(sendbuff, recvbuff, recvcount, ncclUint32, ncclSum, comms[comm_size][device], stream ? stream->stream_ : nullptr);
  auto cmd = ncclAllReduce(sendbuff, recvbuff, recvcount, ncclUint32, ncclSum,
                           comms[n][comm_size][device]->get_comm(),
                           stream ? stream->stream_ : nullptr);
  cudaGraph_t graph_gather_ = nullptr;
  auto end_status = cudaStreamEndCapture(stream->stream_, &graph_gather_);
  if (cmd != ncclSuccess) {
    if (end_status == cudaSuccess && graph_gather_) {
      cudaGraphDestroy(graph_gather_);
    }
    checkNcclResult(cmd, __FILE__, __LINE__);
  }
  checkCudaResult(end_status, __FILE__, __LINE__);

  if (NCCL_DESTROY_NODES) {
    cudaGraphNode_t *nodes = nullptr;
    size_t numNodes = 3;
    nodes = (cudaGraphNode_t *)malloc(numNodes * sizeof(cudaGraphNode_t));
    cudaGraphGetNodes(graph_gather_, nodes, &numNodes);
    CHECK_CUDA_ERROR();
    for (size_t i = 0; i < numNodes; i++) {
      cudaGraphNodeType nodeType;
      cudaGraphNodeGetType(nodes[i], &nodeType);
      CHECK_CUDA_ERROR();
      if (nodeType != cudaGraphNodeTypeKernel) {
        cudaGraphDestroyNode(nodes[i]);
        CHECK_CUDA_ERROR();
      }
    }
  }

  cudaGraphAddChildGraphNode(&node->node_, graph->graph_, nullptr, 0,
                             graph_gather_);
  CHECK_CUDA_ERROR();
  if (prev_ncc[n][device]) {
    // addSingleEdge(graph,prev_ncc[n][device],node);
    CHECK_CUDA_ERROR();
  }
  prev_ncc[n][device] = node;
  CHECK_CUDA_ERROR();
  cudaGraphDestroy(graph_gather_);
  CHECK_CUDA_ERROR();
}

void addEmptyNode(GraphPtr &graph, NodePtr &node) {
  cudaGraphAddEmptyNode(&node->node_, graph->graph_, nullptr, 0);
  CHECK_CUDA_ERROR();
}

Buffer::Buffer(size_t bytes) : bytes_(bytes) {
  auto cmd = ncclMemAlloc(&buffer_, bytes);
  NCCLCHECK(cmd);
}

Buffer::~Buffer() {
  if (buffer_) {
    auto cmd = ncclMemFree(buffer_);
  }
  buffer_ = nullptr;
}

BufferPtr createNcclBuffer(size_t bytes) {
  return std::make_shared<Buffer>(bytes);
}

void ncclBufferRegister(size_t device, BufferPtr &buffer) {
  for (size_t n = 0; n < NUM_COMMS; n++) {
    for (auto &[d, comm_vec] : comms[n]) {
      // TODO: Keep this handle somewhere to unregister later
      void *sendRegHandle;
      std::cout << "Register nccl Buffer of size: " << buffer->size()
                << " on device: " << device << "with comm"
                << (void *)comm_vec.at(device)->get_comm() << "\n"
                << std::flush;
      auto cmd = ncclCommRegister(comm_vec[device]->get_comm(), buffer->get(),
                                  buffer->size(), &sendRegHandle);
      NCCLCHECK(cmd);
    }
  }
}

} // namespace NCCL
} // namespace CUDA
} // namespace Runtime
} // namespace Cerium
