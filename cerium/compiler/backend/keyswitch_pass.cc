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

#include "cerium/compiler/backend/keyswitch_pass.h"
#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term_map.h"

#include <iostream>
#include <set>
#include <tuple>
#include <unordered_map>

namespace CeriumP {

FuseAggregationPass::FuseAggregationPass(Program &program) : program(program) {}

Term::Ptr
FuseAggregationPass::makeTermFrom(Function &function, const Term::Ptr &term,
                                  Op op,
                                  const std::vector<Term::Ptr> &operands = {}) {
  return function.makeTerm(op, term->getType(), term->getScale(),
                           term->getLevel(), term->getEphemeralKey(),
                           term->getPartitionSize(), term->getPartitionId(),
                           operands);
}

void FuseAggregationPass::addOperandToRotAccTerm(Term::Ptr &rotAccTerm,
                                                 Term::Ptr &rotTerm) {
  Term::Ptr op = rotTerm;
  int32_t rotIdx = 0;
  rotAccTerm->addOperand(op);
}

bool FuseAggregationPass::isSingleUseRotateOp(const Term::Ptr &term) {
  if (term->numUses() != 1) {
    return false;
  }
  return term->getOp() == Op::RotateLeftConst ||
         term->getOp() == Op::RotateRightConst;
}
bool FuseAggregationPass::isAggregationOp(const Term::Ptr &term) {
  return term->getOp() == Op::RotAcc;
}

void FuseAggregationPass::operator()(Function &function, Term::Ptr &term) {

  if (term->getOp() == Op::Add) {
    auto operands = term->getOperands();
    assert(operands.size() == 2);
    if (isSingleUseRotateOp(operands[0]) || isSingleUseRotateOp(operands[1])) {
      term->setOp(Op::RotAcc);
      // DO DFS and replace all terms
      std::vector<Term::Ptr> checklist;
      for (auto &op : term->getOperands()) {
        checklist.push_back(op);
      }
      std::set<uint64_t> visited;
      while (!checklist.empty()) {
        auto visit = checklist.back();
        checklist.pop_back();
        if (visited.find(visit->index) != visited.end()) {
          continue;
        }
        visited.insert(visit->index);
        if (isSingleUseRotateOp(visit)) {
          assert(visit->numUses() == 1);
          auto use = visit->getUses()[0];
          if (use->getOp() == Op::RotAcc) {
            addOperandToRotAccTerm(term, visit);
            use->eraseOperand(visit);
            assert(visit->numUses() == 1);
            continue;
          }
          assert(use->getOp() == Op::Add);
          addOperandToRotAccTerm(term, visit);
          use->eraseOperand(visit);
          assert(visit->numUses() == 1);
          assert(use->numOperands() == 1);
          auto useOp = use->operandAt(0);
          use->eraseOperand(useOp);
          use->replaceAllUsesWith(useOp);
          assert(use->numUses() == 0);
          use->setOp(Op::Nop);
        } else if (visit->getOp() == Op::Add) {
          assert(visit->numOperands() == 2);
          for (auto &op : visit->getOperands()) {
            checklist.push_back(op);
          }
        }
      }
      auto rotOperands = term->getOperands();
      std::vector<int32_t> rotIdxs(rotOperands.size());
      for (int i = 0; i < rotOperands.size(); i++) {
        int32_t rotIdx = 0;
        auto rotTerm = rotOperands[i];
        if (isSingleUseRotateOp(rotTerm)) {
          if (rotTerm->getOp() == Op::RotateLeftConst) {
            rotIdx = rotTerm->get<RotationAttribute>();
          } else if (rotTerm->getOp() == Op::RotateRightConst) {
            rotIdx = rotTerm->get<RotationAttribute>();
            rotIdx = -rotIdx;
          }
          assert(rotTerm->numOperands() == 1);
          auto op = rotTerm->operandAt(0);
          rotTerm->eraseOperand(op);
          term->eraseOperand(rotTerm);
          assert(rotTerm->numUses() == 0);
          assert(rotTerm->numOperands() == 0);
          rotTerm->setOp(Op::Nop);
          rotOperands[i] = op;
          rotIdxs[i] = rotIdx;
        }
      }
      term->set<MultiRotationAttribute>(rotIdxs);
      term->setOperands(rotOperands);
    }
  }
}

HoistInputBroadcastPass::HoistInputBroadcastPass(Program &program)
    : program(program) {}

Term::Ptr HoistInputBroadcastPass::makeTermFrom(
    Function &function, const Term::Ptr &term, Op op,
    const std::vector<Term::Ptr> &operands = {}) {
  return function.makeTerm(op, term->getType(), term->getScale(),
                           term->getLevel(), term->getEphemeralKey(),
                           term->getPartitionSize(), term->getPartitionId(),
                           operands);
}

bool HoistInputBroadcastPass::isRotateTerm(const Term::Ptr &term) {
  return term->getOp() == Op::RotateLeftConst ||
         term->getOp() == Op::RotateRightConst;
}

void HoistInputBroadcastPass::operator()(Function &function,
                                         const Term::Ptr &term) {

  auto uses = term->getUses();
  std::vector<Term::Ptr> rotationUses;
  for (auto &use : uses) {
    if (isRotateTerm(use)) {
      rotationUses.push_back(use);
    }
  }
  if (rotationUses.size() <= 1) {
    return;
  }

  auto rotManyTerm =
      makeTermFrom(function, term, Op::HoistInpBroadcast, {term});
  std::vector<int32_t> rotationIndices;
  rotManyTerm->setExtractPossible(true);
  rotManyTerm->setExtractSize(rotationUses.size());
  for (int i = 0; i < rotationUses.size(); i++) {
    // TODO: make extract Vec Here
    auto use = rotationUses.at(i);
    if (use->getOp() == Op::RotateLeftConst) {
      rotationIndices.push_back(use->get<RotationAttribute>());
    } else if (use->getOp() == Op::RotateRightConst) {
      int32_t idx = use->get<RotationAttribute>();
      rotationIndices.push_back(-idx);
    } else {
      assert(0);
    }
    use->addOperand(rotManyTerm);
    use->eraseOperand(term);
    use->setOp(Op::TermInCiphertextVec);
    use->set<TermIdxInVec>(i);
  }
  rotManyTerm->set<MultiRotationAttribute>(rotationIndices);
}

CommonReceiveEliminatorPass::CommonReceiveEliminatorPass(Function &function)
    : function(function) {}

bool CommonReceiveEliminatorPass::isReceiveTerm(const Term::Ptr &term) {
  return term->getOp() == Op::Receive;
}

void CommonReceiveEliminatorPass::operator()(Function &function,
                                             const Term::Ptr &term) {

  auto uses = term->getUses();
  std::vector<Term::Ptr> receiveUses;
  for (auto &use : uses) {
    if (isReceiveTerm(use)) {
      receiveUses.push_back(use);
    }
  }
  if (!receiveUses.empty()) {
    term->setHasReceiveUse(true);
  }
  if (receiveUses.size() <= 1) {
    return;
  }

  struct PairHash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t> &pair) const {
      return std::hash<uint32_t>()(std::get<0>(pair)) ^
             std::hash<uint32_t>()(std::get<1>(pair));
    }
  };

  struct PairCompare {
    bool operator()(const std::pair<uint32_t, uint32_t> &lhs,
                    const std::pair<uint32_t, uint32_t> &rhs) const {
      return std::get<0>(lhs) == std::get<0>(rhs) &&
             std::get<1>(lhs) == std::get<1>(rhs);
    }
  };
  std::unordered_map<std::pair<uint32_t, uint32_t>, Term::Ptr, PairHash,
                     PairCompare>
      receiveMap;

  for (auto &use : receiveUses) {
    auto key = std::pair<uint32_t, uint32_t>(use->getPartitionSize(),
                                             use->getPartitionId());
    auto it = receiveMap.find(key);
    if (it == receiveMap.end()) {
      receiveMap[key] = use;
      continue;
    }
    auto prev_use = it->second;
    use->replaceAllUsesWith(prev_use);
    use->setOp(Op::Nop);
  }

  bool allDestsHaveSamePartitionSize = true;
  uint32_t destPartitionSize = 0;
  std::vector<std::pair<uint32_t, uint32_t>> receiveDests;
  for (auto &use : receiveUses) {
    if (!isReceiveTerm(use)) {
      continue;
    }
    if (destPartitionSize == 0) {
      destPartitionSize = use->getPartitionSize();
    } else if (destPartitionSize != use->getPartitionSize()) {
      allDestsHaveSamePartitionSize = false;
      break;
    }
    receiveDests.emplace_back(
        std::pair(use->getPartitionSize(), use->getPartitionId()));
  }
  if (!allDestsHaveSamePartitionSize || destPartitionSize == 0) {
    return;
  }
  if (receiveDests.size() < 2) {
    return;
  }
  assert(allDestsHaveSamePartitionSize);
  Term::Ptr batchedReceiveTerm = nullptr;
  for (auto &use : receiveUses) {
    if (!isReceiveTerm(use)) {
      continue;
    }
    if (!batchedReceiveTerm) {
      batchedReceiveTerm = use;
      continue;
    }
    use->replaceAllUsesWith(batchedReceiveTerm);
    use->setOp(Op::Nop);
  }
  batchedReceiveTerm->set<ReceiveDestinationsAttribute>(receiveDests);
}

ReductionRewriterPass::ReductionRewriterPass(Function &function)
    : function(function) {}

Term::Ptr ReductionRewriterPass::makeTermFrom(
    Function &function, const Term::Ptr &term, Op op,
    const std::vector<Term::Ptr> &operands = {}) {
  return function.makeTerm(op, term->getType(), term->getScale(),
                           term->getLevel(), term->getEphemeralKey(),
                           term->getPartitionSize(), term->getPartitionId(),
                           operands);
}
uint8_t ReductionRewriterPass::receiveSrcPartitionID(const Term::Ptr &term) {
  assert(term->getOp() == Op::Receive);
  assert(term->numOperands() == 1);
  return term->operandAt(0)->getPartitionId();
}
uint8_t ReductionRewriterPass::receiveSrcPartitionSize(const Term::Ptr &term) {
  assert(term->getOp() == Op::Receive);
  assert(term->numOperands() == 1);
  return term->operandAt(0)->getPartitionSize();
}

void ReductionRewriterPass::addOperandToReduceTerm(Term::Ptr &reductionTerm,
                                                   Term::Ptr term) {
  Term::Ptr op = term;
  reductionTerm->addOperand(op);
}

bool ReductionRewriterPass::isSingleUseReceiveOp(const Term::Ptr &term) {
  if (term->numUses() != 1) {
    return false;
  }
  return term->getOp() == Op::Receive;
}
bool ReductionRewriterPass::isReductionOp(const Term::Ptr &term) {
  return term->getOp() == Op::Reduce;
}

void ReductionRewriterPass::operator()(Function &function, Term::Ptr &term) {

  if (term->getOp() != Op::Add) {
    return;
  }

  auto operands = term->getOperands();
  assert(operands.size() == 2);
  if (!isSingleUseReceiveOp(operands[0]) &&
      !isSingleUseReceiveOp(operands[1])) {
    return;
  }
  bool operand0IsReceive = isSingleUseReceiveOp(operands[0]);
  bool operand1IsReceive = isSingleUseReceiveOp(operands[1]);
  if (operand0IsReceive &&
      receiveSrcPartitionSize(operands[0]) >= term->getPartitionSize()) {
    return;
  }
  if (operand1IsReceive &&
      receiveSrcPartitionSize(operands[1]) >= term->getPartitionSize()) {
    return;
  }
  if (operand0IsReceive && operand1IsReceive &&
      receiveSrcPartitionID(operands[0]) ==
          receiveSrcPartitionID(operands[1])) {
    throw std::runtime_error(
        "Unimplemented Case where both are in the same partition. Need to "
        "ensure that the current partition size is not adjusted when this "
        "operation takes place");
    auto add_term =
        makeTermFrom(function, operands[0]->operandAt(0), Op::Add,
                     {operands[0]->operandAt(0), operands[1]->operandAt(0)});
    operands[0]->eraseOperand(operands[0]->operandAt(0));
    operands[1]->eraseOperand(operands[1]->operandAt(0));
    operands[0]->addOperand(add_term);
    assert(operands[0]->numOperands() == 1);
    operands[1]->setOp(Op::Nop);
    operands[1]->setHasReceiveUse(false);
    add_term->setHasReceiveUse(true);
    term->replaceAllUsesWith(operands[0]);
    term->setOp(Op::Nop);
    term->eraseOperand(operands[0]);
    term->eraseOperand(operands[1]);
    return;
  }
  std::unordered_map<uint8_t, Term::Ptr> srcMap;
  term->setOp(Op::Reduce);
  std::vector<Term::Ptr> checklist;
  for (auto &op : term->getOperands()) {
    checklist.push_back(op);
  }
  auto termSrcPartitionSize = -1;
  if (operand1IsReceive) {
    termSrcPartitionSize = receiveSrcPartitionSize(operands[1]);
  } else if (operand0IsReceive) {
    termSrcPartitionSize = receiveSrcPartitionSize(operands[0]);
  } else {
    throw std::runtime_error(
        "ReductionRewriterPass: neither operand is a receive");
  }

  std::set<uint64_t> visited;
  while (!checklist.empty()) {
    auto visit = checklist.back();
    checklist.pop_back();
    if (visited.find(visit->index) != visited.end()) {
      continue;
    }
    visited.insert(visit->index);
    if (visit->getOp() == Op::Add) {
      assert(visit->numOperands() == 2);
      for (auto &op : visit->getOperands()) {
        checklist.push_back(op);
      }
    }
    if (!isSingleUseReceiveOp(visit)) {
      continue;
    }

    if (receiveSrcPartitionSize(visit) != termSrcPartitionSize) {
      continue;
    }
    if (receiveSrcPartitionSize(visit) >= term->getPartitionSize()) {
      continue;
    }

    assert(visit->numUses() == 1);
    auto use = visit->getUses()[0];
    if (use->getOp() == Op::Reduce) {
      auto srcPartitionID = receiveSrcPartitionID(visit);
      auto it = srcMap.find(srcPartitionID);
      if (it == srcMap.end()) {
        srcMap[srcPartitionID] = visit->operandAt(0);
        addOperandToReduceTerm(term, visit->operandAt(0));
        visit->eraseOperand(visit->operandAt(0));
        use->eraseOperand(visit);
        assert(visit->numUses() == 0);
        visit->setOp(Op::Nop);
      } else {
        auto add_term =
            makeTermFrom(function, srcMap.at(srcPartitionID), Op::Add,
                         {srcMap.at(srcPartitionID), visit->operandAt(0)});
        term->eraseOperand(srcMap.at(srcPartitionID));
        visit->operandAt(0)->setHasReceiveUse(false);
        visit->eraseOperand(visit->operandAt(0));
        term->addOperand(add_term);
        visit->setOp(Op::Nop);
        add_term->setHasReceiveUse(true);
        srcMap[srcPartitionID] = add_term;
        use->eraseOperand(visit);
        assert(visit->numOperands() == 0);
        assert(visit->numUses() == 0);
      }
      continue;
    }
    assert(use->getOp() == Op::Add);
    auto srcPartitionID = receiveSrcPartitionID(visit);
    auto it = srcMap.find(srcPartitionID);
    if (it == srcMap.end()) {
      srcMap[srcPartitionID] = visit->operandAt(0);
      addOperandToReduceTerm(term, visit->operandAt(0));
      visit->eraseOperand(visit->operandAt(0));
      use->eraseOperand(visit);
      assert(visit->numUses() == 0);
      visit->setOp(Op::Mul);
    } else {
      auto add_term =
          makeTermFrom(function, srcMap.at(srcPartitionID), Op::Add,
                       {srcMap.at(srcPartitionID), visit->operandAt(0)});
      term->eraseOperand(srcMap.at(srcPartitionID));
      visit->operandAt(0)->setHasReceiveUse(false);
      visit->eraseOperand(visit->operandAt(0));
      term->addOperand(add_term);
      visit->setOp(Op::Mul);
      add_term->setHasReceiveUse(true);
      srcMap[srcPartitionID] = add_term;
      use->eraseOperand(visit);
      assert(visit->numOperands() == 0);
      assert(visit->numUses() == 0);
    }

    use->eraseOperand(visit);
    assert(visit->numUses() == 0);
    assert(use->numOperands() == 1);
    auto useOp = use->operandAt(0);
    use->eraseOperand(useOp);
    use->replaceAllUsesWith(useOp);
    assert(use->numUses() == 0);
    visit->setOp(Op::Nop);
    use->setOp(Op::Nop);
    continue;
  }

  std::vector<Term::Ptr> reductionOperands;
  std::vector<std::pair<uint32_t, uint32_t>> srcPartitions;
  for (auto &[i, op] : srcMap) {
    reductionOperands.push_back(op);
    term->eraseOperand(op);
    srcPartitions.emplace_back(
        std::pair<uint32_t, uint32_t>(termSrcPartitionSize, i));
  }
  std::vector<Term::Ptr> otherOperands = term->getOperands();
  term->setOperands(reductionOperands);
  term->set<ReduceSourcesAttribute>(srcPartitions);
  auto prev_add_term = term;
  for (auto &op : otherOperands) {
    auto add_term = makeTermFrom(function, op, Op::Add, {prev_add_term, op});
    prev_add_term = add_term;
  }

}

} // namespace CeriumP