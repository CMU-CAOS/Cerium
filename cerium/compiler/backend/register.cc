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

#include "cerium/compiler/backend/register.h"

namespace Cerium {
namespace Backend {

std::ostream &operator<<(std::ostream &s, const Register &reg) {
  if (reg.is_bcor()) {
    s << "b" << reg.bcu_id_ << "(" << reg.id_ << ")";
  } else if (reg.reg_type() == Register::RegType::Scalar) {
    s << "s" << reg.id_;
  } else {
    s << "r" << reg.id_;
  }
  if (reg.is_dead_) {
    s << "[X]";
  }
  return s;
}

std::ostream &operator<<(std::ostream &s, const PolynomialRegisterGroup &prg) {
  s << "{";
  auto registers = prg.registers();
  auto it = registers.begin();
  s << *it;
  it++;
  for (; it != registers.end(); it++) {
    s << ", " << *it;
  }
  s << "}";
  return s;
}

std::ostream &operator<<(std::ostream &s, const BaseConversionUnit &bcu) {
  s << "B" << bcu.id_;
  return s;
}

} // namespace Backend
} // namespace Cerium
