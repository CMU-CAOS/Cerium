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

#include "cerium/compiler/frontend/types.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <variant>

#include <vector>

namespace Cerium {
namespace Frontend {

using AttributeValue =
    std::variant<std::monostate, std::uint32_t, std::int32_t, Type,
                 std::vector<int32_t>, std::string, bool,
                 std::vector<std::pair<std::uint32_t, std::uint32_t>>>;

template <class T, class TypeList> struct IsInVariant;
template <class T, class... Ts>
struct IsInVariant<T, std::variant<Ts...>>
    : std::bool_constant<(... || std::is_same<T, Ts>{})> {};

using AttributeKey = std::uint8_t;

template <std::uint64_t Key, class T> struct Attribute {
  static_assert(IsInVariant<T, AttributeValue>::value,
                "Attribute type not in AttributeValue");
  static_assert(Key > 0, "Keys must be strictly positive");
  static_assert(Key <= std::numeric_limits<AttributeKey>::max(),
                "Key larger than current AttributeKey type");

  using Value = T;
  static constexpr AttributeKey key = Key;

  static bool isValid(AttributeKey k, const AttributeValue &v) {
    return k == Key && std::holds_alternative<T>(v);
  }
};

class AttributeList {
public:
  AttributeList() : key(0), tail(nullptr) {}

  template <class TAttr> bool has() const { return has(TAttr::key); }

  template <class TAttr> const typename TAttr::Value &get() const {
    return std::get<typename TAttr::Value>(get(TAttr::key));
  }

  template <class TAttr> void set(typename TAttr::Value value) {
    set(TAttr::key, std::move(value));
  }

  void assignAttributesFrom(const AttributeList &other);

private:
  AttributeKey key;
  AttributeValue value;
  std::unique_ptr<AttributeList> tail;

  AttributeList(AttributeKey k, AttributeValue v)
      : key(k), value(std::move(v)) {}

  bool has(AttributeKey k) const;
  const AttributeValue &get(AttributeKey k) const;
  void set(AttributeKey k, AttributeValue v);
};

} // namespace Frontend
} // namespace Cerium
