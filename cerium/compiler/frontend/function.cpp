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

#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term_map.h"
#include "cerium/compiler/util/logging.h"
#include "cerium/compiler/util/program_traversal.h"
#include <stack>
#include <unordered_set>

namespace Cerium {
namespace Frontend {

// TODO: maybe replace with smart iterator to avoid allocation
std::vector<Term::Ptr> toTermPtrs(const std::unordered_set<Term *> &terms) {
  std::vector<Term::Ptr> termPtrs;
  termPtrs.reserve(terms.size());
  for (auto &term : terms) {
    termPtrs.emplace_back(term->shared_from_this());
  }
  return termPtrs;
}

template <class Attr>
void appendIntListAttribute(std::stringstream &s, const Term *term,
                            const char *label = nullptr) {
  if (!term->has<Attr>()) {
    return;
  }

  const auto &values = term->get<Attr>();
  assert(values.size() > 1);

  s << "(";
  if (label != nullptr) {
    s << label << ": ";
  }
  s << values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    s << ", " << values[i];
  }
  s << ")";
}

template <class EmitFn>
void traverseTermsFromSinks(const std::vector<Term::Ptr> &sinks,
                            EmitFn &&emit) {
  std::stack<std::pair<bool, Term *>> work;
  std::unordered_set<Term *> visited;

  for (const auto &sink : sinks) {
    work.emplace(true, sink.get());
  }

  while (!work.empty()) {
    const bool visit = work.top().first;
    Term *term = work.top().second;
    work.pop();

    if (!visited.insert(term).second) {
      continue;
    }

    if (visit) {
      work.emplace(false, term);
      for (const auto &operand : term->getOperands()) {
        work.emplace(true, operand.get());
      }
    } else {
      emit(*term);
    }
  }
}

std::vector<Term::Ptr> Function::getSources() const {
  return toTermPtrs(this->sources);
}

std::vector<Term::Ptr> Function::getSinks() const {
  return toTermPtrs(this->sinks);
}

std::vector<Term::Ptr> Function::getNewSources() const {
  return this->newSources;
}

std::vector<Term::Ptr> Function::getNewSinks() const { return this->newSinks; }

void Function::emptyNewSources() { this->newSources.clear(); }

void Function::emptyNewSinks() { this->newSinks.clear(); }

uint64_t Function::allocateIndex() {
  // TODO: reuse released indices to save space in TermMap instances
  uint64_t index = nextTermIndex++;
  for (TermMapBase *termMap : termMaps) {
    termMap->resize(nextTermIndex);
  }
  return index;
}

void Function::initTermMap(TermMapBase &termMap) {
  termMap.resize(nextTermIndex);
}

void Function::registerTermMap(TermMapBase *termMap) {
  termMaps.emplace_back(termMap);
}

void Function::unregisterTermMap(TermMapBase *termMap) {
  auto iter = find(termMaps.begin(), termMaps.end(), termMap);
  if (iter == termMaps.end()) {
    throw std::runtime_error("TermMap to unregister not found");
  } else {
    termMaps.erase(iter);
  }
}

std::string Function::dump(TermMapOptional<std::uint32_t> &scales,
                           TermMap<Type> &types,
                           TermMap<std::uint32_t> &level) const {
  // TODO: switch to use a non-parallel generic traversal
  std::stringstream s;
  s << getName() << "(){\n";

  // Add all terms in topologically sorted order
  traverseTermsFromSinks(getSinks(), [&](Term &term) {
    s << "t" << term.index << " = " << getOpName(term.getOp());
    if (term.has<RotationAttribute>()) {
      s << "(" << term.get<RotationAttribute>() << ")";
    }
    if (term.has<TypeAttribute>()) {
      s << ":" << getTypeName(term.get<TypeAttribute>());
    }
    appendIntListAttribute<RotMulAccRotationAttribute>(s, &term);
    for (size_t i = 0; i < term.numOperands(); ++i) {
      s << " t" << term.operandAt(i)->index;
    }
    if (types[term] == Type::Cipher) {
      s << ", " << "s" << "=" << scales[term] << ", t=cipher ";
    } else {
      s << ", " << "s" << "=" << scales[term] << ", t=plain ";
    }
    s << "\n";
    // ConstantValue TODO: printing constant values for simple cases
  });

  s << "}\n";
  return s.str();
}

std::string Function::toDOT(std::stringstream &s) const {
  // TODO: switch to use a non-parallel generic traversal
  // stringstream s;

  // s << "subgraph_cluster_ \"" << getName() << "\" {\n";
  auto name = getName();
  s << "subgraph cluster_" << getName() << " {\n";
  s << "label=\"" << getName() << "\";\n";

  // Add all terms in topologically sorted order
  traverseTermsFromSinks(getSinks(), [&](Term &term) {
    // Operands are guaranteed to have been added
    s << name << "_t" << term.index << " [label=\"" << getOpName(term.getOp());
    if (term.has<RotationAttribute>()) {
      s << "(" << term.get<RotationAttribute>() << ")";
    }
    if (term.has<TypeAttribute>()) {
      s << " : " << getTypeName(term.get<TypeAttribute>());
    }
    appendIntListAttribute<RotMulAccRotationAttribute>(s, &term);
    appendIntListAttribute<BsgsMulAccBabyStepAttribute>(s, &term, "BabySteps");
    appendIntListAttribute<BsgsMulAccGiantStepAttribute>(s, &term,
                                                         "GiantSteps");
    s << "\""; // End label
    s << "];\n";
    for (size_t i = 0; i < term.numOperands(); ++i) {
      s << name << "_t" << term.operandAt(i)->index << " -> " << name << "_t"
        << term.index << " [label=\"" << i << "\"];\n";
    }
  });

  s << "}\n";

  return s.str();
}

std::uint64_t Function::numTerms() const { return nextTermIndex; }

} // namespace Frontend
} // namespace Cerium
