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
#include "cerium/compiler/frontend/term_map.h"

using namespace Cerium;

namespace CeriumP {
using namespace Frontend;
class FuseAggregationPass {

public:
  FuseAggregationPass(Program &program);
  void operator()(Function &function, Term::Ptr &term);
  void init(Function &function) {}

private:
  Program &program;
  Term::Ptr makeTermFrom(Function &function, const Term::Ptr &term, Op op,
                         const std::vector<Term::Ptr> &operands);
  bool isSingleUseRotateOp(const Term::Ptr &term);
  bool isAggregationOp(const Term::Ptr &term);
  void addOperandToRotAccTerm(Term::Ptr &rotAccTerm, Term::Ptr &rotTerm);
};
} // namespace CeriumP

namespace CeriumP {
class HoistInputBroadcastPass {

public:
  HoistInputBroadcastPass(Program &program);
  void operator()(Function &function, const Term::Ptr &term);
  void init(Function &function) {}

private:
  Program &program;
  Term::Ptr makeTermFrom(Function &function, const Term::Ptr &term, Op op,
                         const std::vector<Term::Ptr> &operands);
  bool isRotateTerm(const Term::Ptr &term);
  bool isAggregationOp(const Term::Ptr &term);
  void addOperandToRotAccTerm(Term::Ptr &rotAccTerm, Term::Ptr &rotTerm);
};
} // namespace CeriumP

namespace CeriumP {
class CommonReceiveEliminatorPass {

public:
  CommonReceiveEliminatorPass(Function &function);
  void operator()(Function &function, const Term::Ptr &term);
  void init(Function &function) {}

private:
  Function &function;
  Term::Ptr makeTermFrom(Function &function, const Term::Ptr &term, Op op,
                         const std::vector<Term::Ptr> &operands);
  bool isReceiveTerm(const Term::Ptr &term);
  bool isAggregationOp(const Term::Ptr &term);
  void addOperandToRotAccTerm(Term::Ptr &rotAccTerm, Term::Ptr &rotTerm);
};

class ReductionRewriterPass {

public:
  ReductionRewriterPass(Function &function);
  void operator()(Function &function, Term::Ptr &term);
  void init(Function &function) {}

private:
  Function &function;
  Term::Ptr makeTermFrom(Function &function, const Term::Ptr &term, Op op,
                         const std::vector<Term::Ptr> &operands);
  bool isReceiveTerm(const Term::Ptr &term);
  bool isReductionOp(const Term::Ptr &term);
  void addOperandToReduceTerm(Term::Ptr &rotAccTerm, Term::Ptr rotTerm);
  uint8_t receiveSrcPartitionID(const Term::Ptr &term);
  uint8_t receiveSrcPartitionSize(const Term::Ptr &term);
  void addOperandToReductionTerm(Term::Ptr &reductionTerm, Term::Ptr &term);
  bool isSingleUseReceiveOp(const Term::Ptr &term);
};
} // namespace CeriumP
