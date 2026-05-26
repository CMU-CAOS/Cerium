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

#include <stdexcept>
#include <string>

namespace Cerium {
namespace Frontend {

#define CERIUM_OPS                                                           \
  X(Undef, 0)                                                                  \
  X(Input, 1)                                                                  \
  X(Output, 2)                                                                 \
  X(Constant, 3)                                                               \
  X(Receive, 4)                                                                \
  X(Send, 5)                                                                   \
  X(Reduce, 6)                                                                 \
  X(Negate, 10)                                                                \
  X(Add, 11)                                                                   \
  X(Sub, 12)                                                                   \
  X(Mul, 13)                                                                   \
  X(RotateLeftConst, 14)                                                       \
  X(RotateRightConst, 15)                                                      \
  X(Rotate2, 16)                                                               \
  X(Rotate3, 17)                                                               \
  X(Relinearize, 20)                                                           \
  X(ModSwitch, 21)                                                             \
  X(Rescale, 22)                                                               \
  X(DoubleRescale, 23)                                                         \
  X(BootstrapModRaise, 31)                                                     \
  X(Conjugate, 32)                                                             \
  X(RotMulAcc, 33)                                                             \
  X(MulRotAcc, 34)                                                             \
  X(RotAcc, 35)                                                                \
  X(BsgsMulAcc, 36)                                                            \
  X(BsgsMulAccVec, 37)                                                         \
  X(TermInCiphertextVec, 38)                                                   \
  X(ToEphemeral, 39)                                                           \
  X(Relinearize2, 50)                                                          \
  X(Relinearize3, 51)                                                          \
  X(Conjugate2, 52)                                                            \
  X(Partition, 40)                                                             \
  X(Nop, 49)                                                                   \
  X(RotAcc2, 59)                                                               \
  X(HoistInpBroadcast, 60)                                                     \
  X(MakeVector, 70)                                                            \
  X(MulVec, 71)                                                                \
  X(AddVec, 72)                                                                \
  X(RescaleVec, 73)                                                            \
  X(DoubleRescaleVec, 74)                                                      \
  X(FunctionInputArg, 80)                                                      \
  X(FunctionOutputArg, 81)                                                     \
  X(FunctionCall, 82)                                                          \
  X(FunctionInput, 83)                                                         \
  X(FunctionOutput, 84) //\

enum class Op {
#define X(op, code) op = code,
  CERIUM_OPS
#undef X
};

inline bool isValidOp(Op op) {
  switch (op) {
#define X(op, code) case Op::op:
    CERIUM_OPS
#undef X
    return true;
  default:
    return false;
  }
}

inline std::string getOpName(Op op) {
  switch (op) {
#define X(op, code)                                                            \
  case Op::op:                                                                 \
    return #op;
    CERIUM_OPS
#undef X
  default:
    throw std::runtime_error("Invalid op");
  }
}

} // namespace Frontend
} // namespace Cerium
