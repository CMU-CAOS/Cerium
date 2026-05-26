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

#include "cerium/compiler/frontend/term.h"
#include "cerium/compiler/frontend/function.h"
#include <algorithm>
#include <utility>

namespace Cerium {
namespace Frontend {

Term::Term(Op op, Function &function, Type type, Scale_t scale, Level_t level,
           bool ephemeralKey, std::uint8_t partitionSize, uint8_t partitionId)
    : op(op), function(function), index(function.allocateIndex()), type(type),
      level(level), scale(scale), needsRelinearization(false),
      extractPossible(false), extractSize(0), ephemeralKey(ephemeralKey),
      partitionSize(partitionSize), partitionId(partitionId),
      hasReceiveUse(false) {
  function.sources.insert(this);
  function.sinks.insert(this);
}

Term::~Term() {
  for (Ptr &operand : operands) {
    operand->eraseUse(this);
  }
  if (operands.empty()) {
    function.sources.erase(this);
  }
  assert(uses.empty());
  function.sinks.erase(this);
}

void Term::addOperand(const Term::Ptr &term) {
  if (operands.empty()) {
    function.sources.erase(this);
  }
  operands.emplace_back(term);
  term->addUse(this);
}

bool Term::eraseOperand(const Ptr &term) {
  auto iter = find(operands.begin(), operands.end(), term);
  if (iter != operands.end()) {
    term->eraseUse(this);
    operands.erase(iter);
    if (operands.empty()) {
      function.sources.insert(this);
    }
    return true;
  }
  return false;
}

bool Term::replaceOperand(Ptr oldTerm, Ptr newTerm) {
  bool replaced = false;
  for (Ptr &operand : operands) {
    if (operand == oldTerm) {
      operand = newTerm;
      oldTerm->eraseUse(this);
      newTerm->addUse(this);
      replaced = true;
    }
  }
  return replaced;
}

void Term::replaceUsesWithIf(Ptr term,
                             std::function<bool(const Ptr &)> predicate) {
  auto thisPtr = shared_from_this(); // TODO: avoid this and similar
                                     // unnecessary reference counting
  for (auto &use : getUses()) {
    if (predicate(use)) {
      use->replaceOperand(thisPtr, term);
    }
  }
}

void Term::replaceAllUsesWith(Ptr term) {
  replaceUsesWithIf(term, [](const Ptr &) { return true; });
}

void Term::replaceOtherUsesWith(Ptr term) {
  replaceUsesWithIf(term, [&](const Ptr &use) { return use != term; });
}

void Term::setNeedsRelinearization(bool val) { needsRelinearization = val; }

void Term::setExtractPossible(bool val) { extractPossible = val; }

void Term::setExtractSize(size_t val) {
  if (!extractPossible) {
    throw std::runtime_error(
        "Extract Possible must be true to set extract size");
  }
  extractSize = val;
}

void Term::setHasReceiveUse(bool val) { hasReceiveUse = val; }
Term::Type Term::getType() const { return type; }

std::string Term::getTypeString() const {
  switch (type) {
  case Type::Cipher:
    return std::string("Cipher");
  case Type::Plain:
    return std::string("Plain");
  }
  throw std::runtime_error("Invalid Type");
  return "";
}

Term::Scale_t Term::getScale() const { return scale; }
Term::Level_t Term::getLevel() const { return level; };

bool Term::getNeedsRelinearization() const { return needsRelinearization; };

bool Term::getExtractPossible() const { return extractPossible; };

size_t Term::getExtractSize() const { return extractSize; };

bool Term::getEphemeralKey() const { return ephemeralKey; };

uint8_t Term::getPartitionSize() const { return partitionSize; };

uint8_t Term::getPartitionId() const { return partitionId; };

bool Term::getHasReceiveUse() const { return hasReceiveUse; };

void Term::setRepeatSize(size_t val) {
  if (type != Term::Type::Plain) {
    throw std::runtime_error("Repeat size can only be set for plain terms");
  }
  repeatSize = val;
}

int32_t Term::getRepeatSize() const { return repeatSize; }

void Term::setRemapable(bool val) {
  if (type != Term::Type::Plain) {
    throw std::runtime_error("Remapable can only be set for plain terms");
  }
  remapable = val;
}
bool Term::getRemapable() const { return remapable; }

void Term::setOperands(std::vector<Term::Ptr> o) {
  if (operands.empty()) {
    function.sources.erase(this);
  }

  for (auto &operand : operands) {
    operand->eraseUse(this);
  }
  operands = move(o);
  for (auto &operand : operands) {
    operand->addUse(this);
  }

  if (operands.empty()) {
    function.sources.insert(this);
  }
}

Op Term::getOp() const { return op; }

void Term::setOp(Op o) { op = o; }

size_t Term::numOperands() const { return operands.size(); }

Term::Ptr Term::operandAt(size_t i) { return operands.at(i); }

const std::vector<Term::Ptr> &Term::getOperands() const { return operands; }

size_t Term::numUses() const { return uses.size(); }

std::vector<Term::Ptr> Term::getUses() {
  std::vector<Term::Ptr> u;
  for (Term *use : uses) {
    u.emplace_back(use->shared_from_this());
  }
  return u;
}

bool Term::isInternal() const {
  return ((operands.size() != 0) && (uses.size() != 0));
}

void Term::addUse(Term *term) {
  if (uses.empty()) {
    function.sinks.erase(this);
  }
  uses.emplace_back(term);
}

bool Term::eraseUse(Term *term) {
  auto iter = find(uses.begin(), uses.end(), term);
  assert(iter != uses.end());
  uses.erase(iter);
  if (uses.empty()) {
    function.sinks.insert(this);
    return true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &s, const Term &term) {
  s << term.index << ':' << getOpName(term.op) << '(';
  bool first = true;
  for (const auto &operand : term.getOperands()) {
    if (first) {
      first = false;
    } else {
      s << ',';
    }
    s << operand->index;
  }
  s << ')';
  return s;
}

} // namespace Frontend
} // namespace Cerium
