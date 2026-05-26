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
#include <stdexcept>
#include <string>
#include <vector>

#include "nccl.h"

namespace Cerium {
namespace Runtime {
namespace CUDA {
namespace NCCL {
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = (cmd);                                                  \
    if (res != ncclSuccess) {                                                  \
      throw std::runtime_error(std::string("NCCL error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + " '" +         \
                               ncclGetErrorString(res) + "'");                 \
    }                                                                          \
  } while (0)

class NCCL_COMM {

public:
  ncclComm_t get_comm() { return comm_; }

  NCCL_COMM() : comm_(nullptr){};

  // Take Ownership of a created Communicator
  NCCL_COMM(int nRanks, ncclUniqueId &commId, int rank) {
    NCCLCHECK(ncclCommInitRankConfig(&comm_, nRanks, commId, rank, nullptr));
  }

  NCCL_COMM(int nRanks, ncclUniqueId &commId, int rank, ncclConfig_t *config) {
    NCCLCHECK(ncclCommInitRankConfig(&comm_, nRanks, commId, rank, nullptr));
  }

  ~NCCL_COMM() {
    if (comm_) {
      ncclCommDestroy(comm_);
    }
    comm_ = nullptr;
  }

private:
  ncclComm_t comm_{nullptr};

  NCCL_COMM(const NCCL_COMM &) = delete;
  NCCL_COMM &operator=(const NCCL_COMM &) = delete;
};

} // namespace NCCL
} // namespace CUDA
} // namespace Runtime
} // namespace Cerium