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

#include "context.h"
#include "seal/util/ntt.h"

namespace Cerium {
namespace Runtime {
Context::Context(const std::vector<std::uint64_t> &rns_bases)
    : slots_(SLOTS), num_rns_bases_(rns_bases.size()) {

  n_ = SLOTS << 1;

  for (size_t i = 0; i < rns_bases.size(); i++) {
    rns_bases_.push_back(seal::Modulus(rns_bases.at(i)));
  }
  pool_ = seal::MemoryManager::GetPool();
  Math::CreateNTTTables(log2(n_), rns_bases_, ntt_tables_, pool_);
  seal::util::CreateNTTTables(log2(n_), rns_bases_, seal_ntt_tables_, pool_);
  galois_tool_ = seal::util::allocate<Math::GaloisTool>(pool_, log2(n_), pool_);
}
} // namespace Runtime
} // namespace Cerium