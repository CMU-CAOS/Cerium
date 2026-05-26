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

#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term_map.h"
#include "cerium/compiler/util/logging.h"
#include <stdexcept>
#include <vector>

namespace Cerium {
using namespace Frontend;

/*
Implements efficient forward and backward traversals of Program in the
presence of modifications during traversal.
The rewriter is called for each term in the Program exactly once.
Rewriters must not modify the Program in such a way that terms that are
not uses/operands (for forward/backward traversal, respectively) of the
current term are enabled. With such modifications the whole program is
not guaranteed to be traversed.
*/
class FunctionTraversal {
  Function &program;

  TermMap<bool> ready;
  TermMap<bool> processed;

  uint64_t total_terms = 0;
  uint64_t terms_processed = 0;
  uint64_t percent_complete = 0;

  template <bool isForward> bool arePredecessorsDone(const Term::Ptr &term) {
    for (auto &operand : isForward ? term->getOperands() : term->getUses()) {
      if (!processed[operand]) return false;
    }
    return true;
  }

  template <typename Rewriter, bool isForward>
  void traverse(Rewriter &&rewrite) {
    processed.clear();
    ready.clear();

    std::vector<Term::Ptr> readyNodes =
        isForward ? program.getSources() : program.getSinks();
    for (auto &term : readyNodes) {
      ready[term] = true;
    }
    // Used for remembering uses/operands before rewrite is called. Using a
    // vector here is fine because duplicates in the list are handled
    // gracefully.
    std::vector<Term::Ptr> checkList;

    while (readyNodes.size() != 0) {
      // Pop term to transform
      auto term = readyNodes.back();
      readyNodes.pop_back();

      // If this term is removed, we will lose uses/operands of this term.
      // Remember them here for checking readyness after the rewrite.
      checkList.clear();
      for (auto &succ : isForward ? term->getUses() : term->getOperands()) {
        checkList.push_back(succ);
      }

      log(Verbosity::Trace, "Processing term with index=%lu", term->index);
      rewrite(program, term);
      processed[term] = true;

      // If transform adds new sources/sinks add them to ready terms.
      // for (auto &leaf : isForward ? program.getSources() :
      // program.getSinks()) {
      for (auto &leaf :
           isForward ? program.getNewSources() : program.getNewSinks()) {
        if (!ready[leaf]) {
          readyNodes.push_back(leaf);
          ready[leaf] = true;
        }
      }

      // clear new sources and sinks
      isForward ? program.emptyNewSources() : program.emptyNewSinks();

      // Also check current uses/operands in case any new ones were added.
      for (auto &succ : isForward ? term->getUses() : term->getOperands()) {
        checkList.push_back(succ);
      }

      // Push and mark uses/operands that are ready to be processed.
      TermMap<bool> visited(program);
      for (auto &succ : checkList) {
        if (!ready[succ]) {
          if (arePredecessorsDone<isForward>(succ)) {
            readyNodes.push_back(succ);
            ready[succ] = true;
            continue;
            ;
          }
          // Perform DFS to find all ready predecessor nodes
          std::vector<Term::Ptr> workList;
          workList.push_back(succ);
          visited[succ] = true;
          while (!workList.empty()) {
            Term::Ptr top = workList.back();
            workList.pop_back();
            const std::vector<Term::Ptr> &preds =
                isForward ? top->getOperands() : top->getUses();
            for (auto &pred : preds) {
              if (visited[pred]) {
                continue;
              }
              visited[pred] = true;
              if (!ready[pred]) {
                if (arePredecessorsDone<isForward>(pred)) {
                  readyNodes.push_back(pred);
                  ready[pred] = true;
                } else {
                  workList.push_back(pred);
                }
              }
            }
          }
        }
      }
    }
  }

public:
  FunctionTraversal(Function &g) : program(g), processed(g), ready(g) {
    total_terms = program.numTerms();
  }

  template <typename Rewriter> void forwardPass(Rewriter &&rewrite) {
    traverse<Rewriter, true>(std::forward<Rewriter>(rewrite));
  }

  template <typename Rewriter> void backwardPass(Rewriter &&rewrite) {
    traverse<Rewriter, false>(std::forward<Rewriter>(rewrite));
  }
};

class ProgramTraversal {
  Program &program;

public:
  ProgramTraversal(Program &g) : program(g) {}

  template <typename Rewriter> void forwardPass(Rewriter &&rewrite) {
    for (auto &[k, v] : program.getFunctions()) {
      Function &f = *v;
      FunctionTraversal traversal(f);
      rewrite.init(f);
      traversal.forwardPass(rewrite);
    }
  }

  template <typename Rewriter> void backwardPass(Rewriter &&rewrite) {
    for (auto &[k, v] : program.getFunctions()) {
      Function &f = *v;
      FunctionTraversal traversal(f);
      rewrite.init(f);
      traversal.backwardPass(rewrite);
    }
  }
};

class FunctionTraversalReceiveReorder {
  Function &program;

  TermMap<bool> ready;
  TermMap<bool> processed;
  TermMap<bool> checked;

  uint64_t total_terms = 0;
  uint64_t terms_processed = 0;
  uint64_t percent_complete = 0;

  template <bool isForward> bool arePredecessorsDone(const Term::Ptr &term) {
    for (auto &operand : isForward ? term->getOperands() : term->getUses()) {
      if (!processed[operand]) return false;
    }
    return true;
  }

  template <typename Rewriter, bool isForward>
  void traverse(Rewriter &&rewrite) {
    processed.clear();
    ready.clear();
    checked.clear();

    auto terms = program.getTerms();
    std::vector<Term::Ptr> checklist;

    if (isForward) {
      for (auto it = terms.begin(); it != terms.end(); it++) {
        // auto term = *it;
        checklist.push_back(*it);
        checked[*it] = true;
        while (!checklist.empty()) {
          auto term = checklist.back();
          if (processed[term]) {
            checklist.pop_back();
            continue;
          }
          if (!arePredecessorsDone<isForward>(term)) {
            for (auto &operand :
                 isForward ? term->getOperands() : term->getUses()) {
              if (!checked[operand]) {
                checklist.push_back(operand);
                checked[operand] = true;
              };
            }
            continue;
          }
          log(Verbosity::Trace, "Processing term with index=%lu", term->index);
          rewrite(program, term);
          processed[term] = true;
          checklist.pop_back();
          if (term->getHasReceiveUse()) {
            for (auto &use : term->getUses()) {
              if (use->getOp() == Op::Receive &&
                  use->getPartitionSize() < term->getPartitionSize()) {
                checklist.push_back(use);
              }
            }
          }
        }
      }
    } else {
      throw std::runtime_error("Unimplemented");
      assert(0);
      for (auto it = terms.rbegin(); it != terms.rend(); it++) {
        auto term = *it;
        log(Verbosity::Trace, "Processing term with index=%lu", term->index);
        rewrite(program, term);
      }
    }
  }

public:
  FunctionTraversalReceiveReorder(Function &g)
      : program(g), processed(g), ready(g), checked(g) {
    total_terms = program.numTerms();
  }

  template <typename Rewriter> void forwardPass(Rewriter &&rewrite) {
    traverse<Rewriter, true>(std::forward<Rewriter>(rewrite));
  }

  template <typename Rewriter> void backwardPass(Rewriter &&rewrite) {
    traverse<Rewriter, false>(std::forward<Rewriter>(rewrite));
  }
};

class ProgramTraversalNew {
  Program &program;

public:
  ProgramTraversalNew(Program &g) : program(g) {}

  template <typename Rewriter> void forwardPass(Rewriter &&rewrite) {
    for (auto &[k, v] : program.getFunctions()) {
      Function &f = *v;
      FunctionTraversalReceiveReorder traversal(f);
      rewrite.init(f);
      traversal.forwardPass(rewrite);
    }
  }

  template <typename Rewriter> void backwardPass(Rewriter &&rewrite) {
    for (auto &[k, v] : program.getFunctions()) {
      Function &f = *v;
      FunctionTraversalReceiveReorder traversal(f);
      rewrite.init(f);
      traversal.backwardPass(rewrite);
    }
  }
};

} // namespace Cerium
