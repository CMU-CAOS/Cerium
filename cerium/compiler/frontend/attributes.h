// Copyright contributors to the EVA project
// Licensed under the MIT License.
// Original source: https://github.com/microsoft/EVA
//
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

#include "cerium/compiler/frontend/attribute_list.h"
#include <cstdint>
#include <string>
#include <tuple>

namespace Cerium {
namespace Frontend {

using VectorPair = std::vector<std::pair<uint32_t, uint32_t>>;

#define CERIUM_ATTRIBUTES                                                    \
  X(RotationAttribute, std::int32_t)                                           \
  X(RescaleLevelsAttribute, std::uint32_t)                                     \
  X(TypeAttribute, Type)                                                       \
  X(ModRaiseLevelAttribute, std::uint32_t)                                     \
  X(IsScalarAttribute, bool)                                                   \
  X(PlaintextSlotSize, std::uint32_t)                                          \
  X(IsRemapableAttribute, bool)                                                \
  X(RotMulAccRotationAttribute, std::vector<int32_t>)                          \
  X(RotAccRotationAttribute, std::vector<int32_t>)                             \
  X(MultiRotationAttribute, std::vector<int32_t>)                              \
  X(NameAttribute, std::string)                                                \
  X(BsgsMulAccBabyStepAttribute, std::vector<int32_t>)                         \
  X(BsgsMulAccGiantStepAttribute, std::vector<int32_t>)                        \
  X(TermIdxInVec, std::uint32_t)                                               \
  X(PartitionSizeAttribute, std::uint32_t)                                     \
  X(PartitionIdAttribute, std::uint32_t)                                       \
  X(ReceiveDestinationsAttribute, VectorPair)                                  \
  X(ReduceSourcesAttribute, VectorPair)                                        \
  X(FunctionNameAttribute, std::string)                                        \
  X(FunctionArgumentNameAttribute, std::string)                                \
  X(FunctionRemapableBaseAttribute, std::string) //\

namespace detail {
enum AttributeIndex {
  RESERVE_EMPTY_ATTRIBUTE_KEY = 0,
#define X(name, type) name##Index,
  CERIUM_ATTRIBUTES
#undef X
};
} // namespace detail

#define X(name, type) using name = Attribute<detail::name##Index, type>;
CERIUM_ATTRIBUTES
#undef X

bool isValidAttribute(AttributeKey k, const AttributeValue &v);

std::string getAttributeName(AttributeKey k);

} // namespace Frontend
} // namespace Cerium
