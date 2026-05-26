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

#include "cerium/compiler/backend/terms.h"
namespace Cerium {
namespace Backend {

std::ostream &operator<<(std::ostream &s, const Backend::Polynomial &poly) {

  if (!poly.term_share() || !poly.term()) {
    s << "Z";
    return s;
  }

  auto &term = poly.term();

  std::stringstream ss;
  uint16_t prev = 0;
  uint16_t comb = 0;
  bool start = true;
  uint16_t count = 0;
  for (auto &i : poly.shares()) {
    count++;
    if (comb == 0) {
      if (start) {
        start = false;
        prev = i;
        ss << prev;
        comb = 1;
        continue;
      } else {
        ss << ",";
      }
      ss << prev;
      comb = 1;
    }

    if (prev + 1 == i) {
      prev = i;
      comb++;
      continue;
    } else if (comb == 1) {
      comb = 0;
      prev = i;
    } else {
      ss << ":";
      ss << prev + 1;
      comb = 0;
      prev = i;
    }
  }

  if (comb > 1) {
    ss << ":" << prev + 1;
  } else if (count > 1) {
    ss << "," << prev;
  }

  std::string reg_name = term->name();
  reg_name += "[" + ss.str() + "]";

  ss.str("");
  ss.clear();

  prev = 0;
  comb = 0;
  start = true;
  count = 0;

  for (auto &i : poly.limbs_) {
    count++;
    if (comb == false) {
      if (start) {
        start = false;
        prev = i;
        ss << prev;
        comb = 1;
        continue;
      } else {
        ss << ",";
      }
      ss << prev;
      comb = 1;
    }

    if (prev + 1 == i) {
      prev = i;
      comb++;
      continue;
    } else if (comb == 1) {
      comb = 0;
      prev = i;
    } else {
      ss << ":";
      ss << prev + 1;
      comb = 0;
      prev = i;
    }
  }

  if (comb > 1) {
    ss << ":" << prev + 1;
  } else if (count > 1) {
    ss << "," << prev;
  }

  reg_name += "(" + ss.str() + ")";
  s << reg_name;

  if (poly.dead_) {
    s << "[X]";
  }

  return s;
}

} // namespace Backend
} // namespace Cerium
