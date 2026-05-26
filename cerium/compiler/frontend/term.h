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

#include "cerium/compiler/frontend/attributes.h"
#include "cerium/compiler/frontend/ops.h"
#include "cerium/compiler/frontend/types.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_set>
#include <vector>

namespace Cerium {
namespace Frontend {

class Function;

class Term : public AttributeList, public std::enable_shared_from_this<Term> {
public:
  using Ptr = std::shared_ptr<Term>;
  using Scale_t = std::uint64_t;
  using Level_t = std::uint16_t;

  enum Type { Cipher, Plain };

  // Term(Op opcode, function &function);
  // Term(Op opcode, function &function, Type type, Scale_t scale, Level_t
  // level,
  Term(Op opcode, Function &function, Type type, Scale_t scale, Level_t level,
       bool ephemeralKey, std::uint8_t partitionSize, std::uint8_t partitionId);
  ~Term();

  void addOperand(const Ptr &term);
  bool eraseOperand(const Ptr &term);
  bool replaceOperand(Ptr oldTerm, Ptr newTerm);
  void setOperands(std::vector<Ptr> o);
  std::size_t numOperands() const;
  Ptr operandAt(size_t i);
  const std::vector<Ptr> &getOperands() const;

  void replaceUsesWithIf(Ptr term, std::function<bool(const Ptr &)>);
  void replaceAllUsesWith(Ptr term);
  void replaceOtherUsesWith(Ptr term);
  void setNeedsRelinearization(bool val);
  void setExtractPossible(bool val);
  void setExtractSize(size_t val);
  void setEphemeralKey(bool val);
  void setHasReceiveUse(bool val);
  void setRepeatSize(size_t val);
  void setRemapable(bool val);

  Type getType() const;
  std::string getTypeString() const;
  Scale_t getScale() const;
  Level_t getLevel() const;
  bool getNeedsRelinearization() const;
  bool getExtractPossible() const;
  size_t getExtractSize() const;
  bool getEphemeralKey() const;
  uint8_t getPartitionSize() const;
  uint8_t getPartitionId() const;
  bool getHasReceiveUse() const;
  int32_t getRepeatSize() const;
  bool getRemapable() const;

  std::size_t numUses() const;
  std::vector<Ptr> getUses();

  bool isInternal() const;

  Function &function;

  Op getOp() const;

  void setOp(Op o);

  // Unique index for this Term in the owning function. Managed by function
  // and used to index into TermMap instances.
  std::uint64_t index;

  friend std::ostream &operator<<(std::ostream &s, const Term &term);

private:
  std::vector<Ptr> operands; // use->def chain (unmanaged pointers)
  std::vector<Term *> uses;  // def->use chain (managed pointers)

  Op op;
  Type type;
  Scale_t scale;
  Level_t level;
  bool needsRelinearization;
  bool extractPossible;
  bool ephemeralKey;
  size_t extractSize;

  std::uint8_t partitionSize;
  std::uint8_t partitionId;
  bool hasReceiveUse;

  int32_t repeatSize{-1}; // -1 is the default value
  bool remapable{false};

  void addUse(Term *term);
  bool eraseUse(Term *term);
};

} // namespace Frontend
} // namespace Cerium
