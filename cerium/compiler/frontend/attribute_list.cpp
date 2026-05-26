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

#include "cerium/compiler/frontend/attribute_list.h"
#include "cerium/compiler/frontend/attributes.h"
#include "cerium/compiler/util/overloaded.h"
#include <stdexcept>

namespace Cerium {
namespace Frontend {

bool AttributeList::has(AttributeKey k) const {
  const AttributeList *curr = this;
  while (true) {
    if (curr->key < k) {
      if (curr->tail) {
        curr = curr->tail.get();
      } else {
        return false;
      }
    } else {
      return curr->key == k;
    }
  }
}

const AttributeValue &AttributeList::get(AttributeKey k) const {
  const AttributeList *curr = this;
  while (true) {
    if (curr->key == k) {
      return curr->value;
    } else if (curr->key < k && curr->tail) {
      curr = curr->tail.get();
    } else {
      throw std::out_of_range("Attribute not in list: " + getAttributeName(k));
    }
  }
}

void AttributeList::set(AttributeKey k, AttributeValue v) {
  if (this->key == 0) {
    this->key = k;
    this->value = move(v);
  } else {
    AttributeList *curr = this;
    AttributeList *prev = nullptr;
    while (true) {
      if (curr->key < k) {
        if (curr->tail) {
          prev = curr;
          curr = curr->tail.get();
        } else { // Insert at end
          // AttributeList constructor is private
          curr->tail =
              std::unique_ptr<AttributeList>{new AttributeList(k, move(v))};
          return;
        }
      } else if (curr->key > k) {
        if (prev) { // Insert between
          // AttributeList constructor is private
          auto newList = std::unique_ptr<AttributeList>{
              new AttributeList(k, std::move(v))};
          newList->tail = std::move(prev->tail);
          prev->tail = std::move(newList);
        } else { // Insert at beginning
          // AttributeList constructor is private
          curr->tail = std::unique_ptr<AttributeList>{
              new AttributeList(std::move(*curr))};
          curr->key = k;
          curr->value = move(v);
        }
        return;
      } else {
        assert(curr->key == k);
        curr->value = move(v);
        return;
      }
    }
  }
}

void AttributeList::assignAttributesFrom(const AttributeList &other) {
  if (this->key != 0) {
    this->key = 0;
    this->value = std::monostate();
    this->tail = nullptr;
  }
  if (other.key == 0) {
    return;
  }
  AttributeList *lhs = this;
  const AttributeList *rhs = &other;
  while (true) {
    lhs->key = rhs->key;
    lhs->value = rhs->value;
    if (rhs->tail) {
      rhs = rhs->tail.get();
      lhs->tail = std::make_unique<AttributeList>();
      lhs = lhs->tail.get();
    } else {
      return;
    }
  }
}

} // namespace Frontend
} // namespace Cerium
