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

#include "cerium/compiler/frontend/function.h"
#include "cerium/compiler/frontend/program.h"
#include "cerium/compiler/frontend/term.h"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <variant>
#include <vector>

namespace Cerium {
namespace Frontend {

class TermMapBase {
public:
  // TODO: Maybe remove the default constructor
  //  TermMapBase() : function(nullptr) {  }
  TermMapBase(Function &p) : function(&p) { function->registerTermMap(this); }
  ~TermMapBase() {
    if (function) {
      function->unregisterTermMap(this);
    }
  }
  TermMapBase(const TermMapBase &other) : function(other.function) {
    function->registerTermMap(this);
  }
  TermMapBase &operator=(const TermMapBase &other) = default;

  friend class Function;

protected:
  void init() { function->initTermMap(*this); }

  std::uint64_t getIndex(const Term &term) const { return term.index; }

private:
  virtual void resize(std::size_t size) = 0;

  Function *function;
};

template <class TValue> class TermMap : TermMapBase {
public:
  TermMap(Function &p) : TermMapBase(p) { init(); }

  TValue &operator[](const Term &term) { return values.at(getIndex(term)); }

  const TValue &operator[](const Term &term) const {
    return values.at(getIndex(term));
  }

  TValue &operator[](const Term::Ptr &term) { return this->operator[](*term); }

  const TValue &operator[](const Term::Ptr &term) const {
    return this->operator[](*term);
  }

  void clear() { values.assign(values.size(), {}); }

private:
  void resize(std::size_t size) override { values.resize(size); }

  std::deque<TValue> values;
};

template <> class TermMap<bool> : TermMapBase {
public:
  TermMap(Function &p) : TermMapBase(p) { init(); }

  std::vector<bool>::reference operator[](const Term &term) {
    return values.at(getIndex(term));
  }

  bool operator[](const Term &term) const { return values.at(getIndex(term)); }

  std::vector<bool>::reference operator[](const Term::Ptr &term) {
    return this->operator[](*term);
  }

  bool operator[](const Term::Ptr &term) const {
    return this->operator[](*term);
  }

  void clear() { values.assign(values.size(), false); }

private:
  void resize(std::size_t size) override { values.resize(size); }

  std::vector<bool> values;
};

template <class TOptionalValue> class TermMapOptional : TermMapBase {
public:
  // TODO: Maybe remove the default constructor?
  // TermMapOptional() : TermMapBase() { }
  TermMapOptional(Function &p) : TermMapBase(p) { init(); }

  TOptionalValue &operator[](const Term &term) {
    auto &value = values.at(getIndex(term));
    if (!value.has_value()) {
      value.emplace();
    }
    return *value;
  }

  TOptionalValue &operator[](const Term::Ptr &term) {
    return this->operator[](*term);
  }

  TOptionalValue &at(const Term &term) {
    return values.at(getIndex(term)).value();
  }

  TOptionalValue &at(const Term::Ptr &term) { return this->at(*term); }

  bool has(const Term &term) const {
    return values.at(getIndex(term)).has_value();
  }

  bool has(const Term::Ptr &term) const { return has(*term); }

  void clear() { values.assign(values.size(), std::nullopt); }

private:
  void resize(std::size_t size) override { values.resize(size); }

  std::deque<std::optional<TOptionalValue>> values;
};

} // namespace Frontend
} // namespace Cerium
