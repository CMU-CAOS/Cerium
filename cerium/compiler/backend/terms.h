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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Cerium {
namespace Backend {

using TermIndexType = uint64_t;
using TermShareIndexType = uint64_t;
using LimbIndexType = uint16_t;
using IndexType = std::pair<TermIndexType, LimbIndexType>;

class SSAInstruction;

enum EvalKeyType {
  Relinearization,
  Rotation,
  Conjugation,
  Bootstrap,
  Ephemeral,
  Bootstrap2,
  RotationInv,
};

struct PartitionInfo {
  uint16_t partition_id;
  uint16_t partition_size;
};

class Term {

public:
  enum LimbType {
    Spl,    // Splittable
    Com,    // Complement
    Ext,    // Extension
    ExC,    // Extension Complement
    Par,    // Partition
    PaE,    // Partition Extension
    Usp,    // Unsplittable
    SplPar, // Split + Par

    Ext_Split,
    Ext_Split2,
    PaE_Split,
  };

private:
  TermIndexType index_;
  bool input;
  bool output;
  bool temp;
  bool bcor;
  bool eval_key;
  bool eval_key_poly_num_; // 1 if this is the random polynomial in the pair, 0
                           // otherwise
  bool eval_key_dont_split_shares_; // don't split this evalkey accross shares
  EvalKeyType eval_key_type_;
  bool join_reqd_;
  bool join_here_;
  bool unmergeable_downward_;
  bool unmergeable_upward_;
  bool plaintext_;
  bool scalar_;
  bool receive_;
  bool send_;
  bool function_arg_;
  bool zero_;
  uint32_t extension_size_;
  uint32_t plaintext_repeat_size_;
  bool plaintext_remapable_;
  std::string symbol_;

public:
  Term(){};
  Term(uint64_t index)
      : index_(index), input(false), output(false), temp(true), bcor(false),
        eval_key(false), eval_key_poly_num_(0),
        eval_key_dont_split_shares_(false), join_reqd_(false),
        join_here_(false), unmergeable_downward_(false),
        unmergeable_upward_(false), plaintext_(false), scalar_(false),
        receive_(false), send_(false), function_arg_(false), zero_(false),
        /*type_(LimbType::Spl),*/ extension_size_(0), plaintext_repeat_size_(0),
        plaintext_remapable_(false), symbol_(""){};
  Term(uint64_t index, const std::shared_ptr<Term> &other)
      : index_(index), input(other->input), output(other->output),
        temp(other->temp), bcor(other->bcor), eval_key(other->eval_key),
        eval_key_poly_num_(other->eval_key_poly_num_),
        eval_key_dont_split_shares_(other->eval_key_dont_split_shares_),
        eval_key_type_(other->eval_key_type_), join_reqd_(other->join_reqd_),
        join_here_(other->join_here_),
        unmergeable_downward_(other->unmergeable_downward_),
        unmergeable_upward_(other->unmergeable_upward_),
        plaintext_(other->plaintext_), scalar_(other->scalar_),
        receive_(other->receive_), send_(other->send_),
        function_arg_(other->function_arg_),
        zero_(other->zero_), /*type_(other->type_),*/
        extension_size_(other->extension_size_),
        plaintext_repeat_size_(other->plaintext_repeat_size_),
        plaintext_remapable_(other->plaintext_remapable_),
        symbol_(other->symbol_){};

  void set_as_input() { input = true; }

  bool is_input() const { return input; }

  void set_as_plaintext() {
    input = true;
    plaintext_ = true;
  }

  void set_as_scalar() {
    input = true;
    plaintext_ = true;
    scalar_ = true;
  }

  bool is_plaintext() const { return plaintext_; }

  uint32_t plaintext_repeat_size() const { return plaintext_repeat_size_; }

  void set_plaintext_repeat_size(uint32_t val) {
    assert(is_plaintext() && !is_scalar());
    plaintext_repeat_size_ = val;
  }

  // XXX: For now we don't support remapable scalars
  void set_plaintext_remapable(bool val) {
    assert(is_plaintext() && !is_scalar());
    plaintext_remapable_ = val;
  }

  bool is_plaintext_remapable() const { return plaintext_remapable_; }

  bool is_scalar() const { return scalar_; }

  void set_as_receive() { receive_ = true; }

  bool is_receive() const { return receive_; }

  void set_send(bool val) { send_ = val; }

  bool is_send() const { return send_; }

  void set_as_function_arg(bool val) { function_arg_ = val; }

  bool is_function_arg() const { return function_arg_; }

  void set_as_zero(bool val) { zero_ = val; }

  bool is_zero() const { return zero_; }

  void set_as_eval_key(EvalKeyType eval_key_type, bool eval_key_poly_num,
                       bool eval_key_dont_split_shares) {
    input = true;
    eval_key = true;
    eval_key_poly_num_ = eval_key_poly_num;
    eval_key_type_ = eval_key_type;
    eval_key_dont_split_shares_ = eval_key_dont_split_shares;
  }

  bool is_eval_key() const { return eval_key; }

  bool eval_key_poly_num() const { return eval_key_poly_num_; }

  bool eval_key_dont_split_shares() const {
    return eval_key_dont_split_shares_;
  }

  EvalKeyType eval_key_type() const { return eval_key_type_; }

  void set_output(bool val) { output = val; }

  bool is_output() const { return output; }

  void set_temp(bool val) { temp = val; }

  bool is_temp() const { return temp; }

  void set_as_bcor() { bcor = true; }

  bool is_bcor() const { return bcor; }

  uint64_t term_idx() const { return index_; }

  void set_join_reqd(bool val) { join_reqd_ = val; }

  bool join_reqd() const { return join_reqd_; }

  void set_join_here(bool val) { join_here_ = val; }

  bool join_here() const { return join_here_; }

  void set_unmergeable_downward(bool val) { unmergeable_downward_ = val; }

  bool unmergeable_downward() const { return unmergeable_downward_; }

  void set_unmergeable_upward(bool val) { unmergeable_upward_ = val; }

  bool unmergeable_upward() const { return unmergeable_upward_; }

  // void set_limbtype(LimbType type) { type_ = type; }

  // LimbType limbtype() const { return type_; }

  void set_extension_size(uint32_t extension_size) {
    extension_size_ = extension_size;
  }

  uint32_t extension_size() const { return extension_size_; }

  void set_symbol(const std::string &symbol) { symbol_ = symbol; }

  const std::string &symbol() const { return symbol_; }

  std::string name() const {
    std::string reg_name("");
    if (is_eval_key()) {
      reg_name += "k";
    } else if (is_zero()) {
      reg_name += "Z";
    } else if (is_function_arg()) {
      reg_name += "f";
    } else if (is_plaintext()) {
      if (is_scalar()) {
        reg_name += "s";
      } else {
        reg_name += "p";
      }
    } else if (is_input()) {
      reg_name += "i";
    } else if (is_bcor()) {
      reg_name += "b";
    } else if (is_temp()) {
      reg_name += "t";
    } else {
      reg_name += "v";
    }
    return reg_name + std::to_string(index_);
  }
};

struct TermHash {
  std::size_t operator()(const Term &poly) const {
    return std::hash<TermIndexType>()(poly.term_idx());
  }
};

struct TermCompare {
  bool operator()(const Term &lhs, const Term &rhs) const {
    return lhs.term_idx() == rhs.term_idx();
  }
};

template <typename T>
using TermMap = std::unordered_map<Term, T, TermHash, TermCompare>;

using TermSet = std::unordered_set<Term, TermHash, TermCompare>;

struct TermPtrHash {
  std::size_t operator()(const std::shared_ptr<Term> &term) const {
    return std::hash<TermIndexType>()(term->term_idx());
  }
};

struct TermPtrCompare {
  bool operator()(const std::shared_ptr<Term> &lhs,
                  const std::shared_ptr<Term> &rhs) const {
    return lhs->term_idx() == rhs->term_idx();
  }
};

template <typename T>
using TermPtrMap =
    std::unordered_map<std::shared_ptr<Term>, T, TermPtrHash, TermPtrCompare>;

using TermPtrSet =
    std::unordered_set<std::shared_ptr<Term>, TermPtrHash, TermPtrCompare>;

class BigNode;
class TermShare {
  TermShareIndexType index_;
  std::shared_ptr<Term> term_;
  std::set<uint16_t> shares_;
  uint64_t bignode_idx_;
  uint16_t level_;
  Term::LimbType type_ = Term::LimbType::Spl; // Default to Splittable

public:
  TermShare() {}
  TermShare(TermShareIndexType index, const std::shared_ptr<Term> &term,
            uint16_t level)
      : index_(index), term_(term), level_(level) {
    for (int i = 0; i < level; ++i) {
      shares_.insert(shares_.end(), i);
    }
  }
  const std::set<uint16_t> &shares() const { return shares_; }

  void set_shares(const std::set<uint16_t> &shares) { shares_ = shares; }

  uint32_t num_shares() const { return shares_.size(); }

  void set_bignode_idx(uint64_t idx) { bignode_idx_ = idx; }

  uint64_t bignode_index() const { return bignode_idx_; }

  TermShareIndexType term_share_idx() { return index_; }

  std::shared_ptr<Term> &term() { return term_; }

  const std::shared_ptr<Term> &term() const { return term_; }

  TermIndexType term_idx() { return term_->term_idx(); }

  bool is_bcor() const { return term_->is_bcor(); }

  bool is_input() const { return term_->is_input(); }

  bool is_plaintext() const { return term_->is_plaintext(); }

  uint32_t plaintext_repeat_size() const {
    return term_->plaintext_repeat_size();
  }

  void set_plaintext_repeat_size(uint32_t val) {
    term_->set_plaintext_repeat_size(val);
  }

  void set_plaintext_remapable(bool val) {
    term_->set_plaintext_remapable(val);
  }

  bool is_plaintext_remapable() const {
    return term_->is_plaintext_remapable();
  }

  bool is_scalar() const { return term_->is_scalar(); }

  bool is_receive() const { return term_->is_receive(); }

  void set_send(bool val) { term_->set_send(val); }

  bool is_send() const { return term_->is_send(); }

  void set_as_function_arg(bool val) { term_->set_as_function_arg(val); }

  bool is_function_arg() const { return term_->is_function_arg(); }

  void set_as_zero(bool val) { term_->set_as_zero(val); }

  bool is_zero() const { return term_->is_zero(); }

  bool is_eval_key() const { return term_->is_eval_key(); }

  bool eval_key_poly_num() const { return term_->eval_key_poly_num(); }

  bool eval_key_dont_split_shares() const {
    return term_->eval_key_dont_split_shares();
  }

  EvalKeyType eval_key_type() const { return term_->eval_key_type(); }

  void set_output(bool val) { term_->set_output(val); }

  bool is_output() const { return term_->is_output(); }

  void set_temp(bool val) { term_->set_temp(val); }

  bool is_temp() const { return term_->is_temp(); }

  void set_join_reqd(bool val) { term_->set_join_reqd(val); }

  bool join_reqd() const { return term_->join_reqd(); }

  void set_join_here(bool val) { term_->set_join_here(val); }

  bool join_here() const { return term_->join_here(); }

  void set_unmergeable_downward(bool val) {
    term_->set_unmergeable_downward(val);
  }

  bool unmergeable_downward() { return term_->unmergeable_downward(); }

  void set_unmergeable_upward(bool val) { term_->set_unmergeable_upward(val); }

  bool unmergeable_upward() { return term_->unmergeable_upward(); }

  void set_limbtype(Term::LimbType type) { type_ = type; }

  Term::LimbType limbtype() const { return type_; }

  void set_extension_size(uint32_t extension_size) {
    term_->set_extension_size(extension_size);
  }

  uint32_t extension_size() const { return term_->extension_size(); }

  void update_evalkey_term_symbol() {
    assert(is_eval_key());
    std::stringstream s;
    s << term_->symbol();
    s << ":[";
    uint16_t count = 0;
    if (eval_key_type() == EvalKeyType::Bootstrap ||
        eval_key_type() == EvalKeyType::Bootstrap2) {
      for (auto i = 0; i < level_; i++) {
        if (count != 0) {
          s << ",";
        }
        s << i;
        count++;
      }
      s << "]";

    } else {
      for (auto &share : shares()) {
        if (count != 0) {
          s << ",";
        }
        s << share;
        count++;
      }
      s << "]";
    }
    term_->set_symbol(s.str());
  }

  void set_symbol(const std::string &symbol) { term_->set_symbol(symbol); }

  const std::string &symbol() const { return term_->symbol(); }
};

class Polynomial {
  uint16_t level_;
  std::set<uint16_t> limbs_;
  std::shared_ptr<TermShare> term_share_;
  uint64_t next_use_;
  bool dead_;
  SSAInstruction *instruction_ = nullptr;

  // Hack to implement ext_split_from_shares_;
  bool interpret_ext_spl_from_shares_;
  std::int16_t adjustment_size_;
  std::int16_t ext_adjustment_size_;
  std::int16_t upper_ext_adjustment_size_;

public:
  Polynomial(){};
  Polynomial(std::shared_ptr<TermShare> &term_share, uint16_t level)
      : term_share_(term_share), level_(level), next_use_(-1), dead_(false),
        interpret_ext_spl_from_shares_(false), adjustment_size_(0),
        ext_adjustment_size_(0), upper_ext_adjustment_size_(0){};

  void set_instruction(SSAInstruction *const instruction) {
    instruction_ = instruction;
  }

  SSAInstruction *instruction() { return instruction_; }

  const std::set<uint16_t> &shares() const { return term_share_->shares(); }

  void set_shares(const std::set<uint16_t> &shares) {
    term_share_->set_shares(shares);
  }

  void set_limbs(const std::set<uint16_t> &limbs) { limbs_ = limbs; }

  void set_adjustment_size(std::int16_t adjustment_size) {
    adjustment_size_ = adjustment_size;
  }

  auto adjustment_size() const { return adjustment_size_; }

  void set_ext_adjustment_size(std::int16_t ext_adjustment_size) {
    ext_adjustment_size_ = ext_adjustment_size;
  }

  auto ext_adjustment_size() const { return ext_adjustment_size_; }

  void set_upper_ext_adjustment_size(std::int16_t upper_ext_adjustment_size) {
    upper_ext_adjustment_size_ = upper_ext_adjustment_size;
  }

  auto upper_ext_adjustment_size() const { return upper_ext_adjustment_size_; }

  const std::set<uint16_t> &limbs() const { return limbs_; }

  uint32_t num_limbs() const { return limbs_.size(); }

  uint32_t num_shares() const { return term_share_->num_shares(); }

  void set_level(const uint32_t level) { level_ = level; }

  uint32_t level() const { return level_; }

  std::shared_ptr<TermShare> term_share() { return term_share_; }

  const std::shared_ptr<TermShare> &term_share() const { return term_share_; }

  TermIndexType term_share_idx() { return term_share_->term_share_idx(); }

  std::shared_ptr<Term> term() { return term_share_->term(); }

  const std::shared_ptr<Term> &term() const { return term_share_->term(); }

  TermIndexType term_idx() { return term_share_->term_idx(); }

  const TermIndexType term_idx() const { return term_share_->term_idx(); }

  void set_bignode_idx(uint64_t idx) { term_share_->set_bignode_idx(idx); }

  uint64_t bignode_index() const { return term_share_->bignode_index(); }

  bool is_bcor() const { return term_share_->is_bcor(); }

  bool is_input() const { return term_share_->is_input(); }

  bool is_plaintext() const { return term_share_->is_plaintext(); }

  uint32_t plaintext_repeat_size() const {
    return term_share_->plaintext_repeat_size();
  }

  void set_plaintext_repeat_size(uint32_t val) {
    term_share_->set_plaintext_repeat_size(val);
  }

  void set_plaintext_remapable(bool val) {
    term_share_->set_plaintext_remapable(val);
  }

  bool is_plaintext_remapable() const {
    return term_share_->is_plaintext_remapable();
  }

  bool is_scalar() const { return term_share_->is_scalar(); }

  bool is_receive() const { return term_share_->is_receive(); }

  void set_send(bool val) { term_share_->set_send(val); }

  bool is_send() const { return term_share_->is_send(); }

  void set_as_function_arg(bool val) { term_share_->set_as_function_arg(val); }

  bool is_function_arg() const { return term_share_->is_function_arg(); }

  void set_as_zero(bool val) { term_share_->set_as_zero(val); }

  bool is_zero() const { return term_share_->is_zero(); }

  void set_output(bool val) { term_share_->set_output(val); }

  bool is_output() const { return term_share_->is_output(); }

  bool is_eval_key() const { return term_share_->is_eval_key(); }

  bool eval_key_poly_num() const { return term_share_->eval_key_poly_num(); }

  void set_temp(bool val) { term_share_->set_temp(val); }

  void set_join_reqd(bool val) { term_share_->set_join_reqd(val); }

  bool join_reqd() const { return term_share_->join_reqd(); }

  void set_join_here(bool val) { term_share_->set_join_here(val); }
  bool join_here() const { return term_share_->join_here(); }

  void set_unmergeable_downward(bool val) {
    term_share_->set_unmergeable_downward(val);
  }

  bool unmergeable_downward() { return term_share_->unmergeable_downward(); }

  void set_unmergeable_upward(bool val) {
    term_share_->set_unmergeable_upward(val);
  }

  bool unmergeable_upward() { return term_share_->unmergeable_upward(); }

  void set_limbtype(Term::LimbType type) { term_share_->set_limbtype(type); }

  Term::LimbType limbtype() const { return term_share_->limbtype(); }

  void set_extension_size(uint32_t extension_size) {
    term_share_->set_extension_size(extension_size);
  }

  uint32_t extension_size() const { return term_share_->extension_size(); }

  void set_interpret_extension_split_from_shares(bool val) {
    interpret_ext_spl_from_shares_ = true;
  }

  void set_limbs_from_termshare(size_t partition_id, size_t partition_size) {

    auto compl_max_level = 63;
    auto &shares_ = term_share_->shares();
    if (shares_.empty()) {
      limbs_.clear();
      return;
    }
    switch (term_share_->limbtype()) {
    case Term::Spl:
      limbs_ = shares_;
      for (auto i = 1; i <= adjustment_size(); i++) {
        limbs_.insert(level_ - i);
      }
      for (auto i = adjustment_size(); i < 0; i++) {
        limbs_.erase(level_ + i);
      }
      break;
    case Term::SplPar:
      limbs_.clear();
      for (auto &sh : shares_) {
        auto end = std::min(level_ + adjustment_size(), (int)level_);
        if (sh % partition_size == partition_id && sh < end) {
          limbs_.insert(sh);
        }
      }
      for (auto i = 1; i <= adjustment_size(); i++) {
        limbs_.insert(level_ - i);
      }
      break;
    case Term::Usp:
      limbs_.clear();
      for (auto i = 0; i < level_ + adjustment_size(); i++) {
        limbs_.insert(i);
      }
      break;
    case Term::ExC:
      limbs_.clear();
      for (auto i = 0; i < level_ + adjustment_size(); i++) {
        limbs_.insert(i);
      }
      assert(shares_.size() <= extension_size());
      for (auto i = 0; i < extension_size(); i++) {
        limbs_.insert(compl_max_level - i);
      }
      for (auto &i : shares_) {
        limbs_.erase(i);
      }
      break;
    case Term::Par: {
      limbs_.clear();
      auto end = std::min(level_ + adjustment_size(), (int)level_);
      for (auto i = (partition_id) % partition_size; i < end;
           i += partition_size) {
        limbs_.insert(i);
      }
      for (auto i = 1; i <= adjustment_size(); i++) {
        limbs_.insert(level_ - i);
      }
      for (int i = 0; i < ext_adjustment_size(); i++) {
        limbs_.insert(compl_max_level - i);
      }
    }; break;
    case Term::PaE:
      limbs_.clear();
      for (auto i = (partition_id) % partition_size; i < level_;
           i += partition_size) {
        limbs_.insert(i);
      }
      assert(shares_.size() <= extension_size());
      for (auto i = 0; i < extension_size(); i++) {
        limbs_.insert(compl_max_level - i);
      }
      for (auto &i : shares_) {
        limbs_.erase(i);
      }
      for (auto i = 1; i <= adjustment_size(); i++) {
        limbs_.insert(level_ - i);
      }
      for (int i = 0; i < ext_adjustment_size(); i++) {
        limbs_.insert(compl_max_level - i);
      }
      break;
    case Term::Com:
      limbs_.clear();
      for (auto i = 0; i < level_; i++) {
        limbs_.insert(i);
      }
      for (auto &i : shares_) {
        limbs_.erase(i);
      }
      break;
    case Term::Ext:
      limbs_.clear();
      assert(extension_size() != 0);
      for (int i = -ext_adjustment_size();
           i < extension_size() + upper_ext_adjustment_size(); i++) {
        limbs_.insert(compl_max_level - i);
      }
      for (auto i = 1; i <= adjustment_size(); i++) {
        limbs_.insert(level_ - i);
      }

      break;

    case Term::PaE_Split:
      limbs_.clear();
      partition_id = partition_id % partition_size;
      for (auto i = (partition_id) % partition_size; i < level_;
           i += partition_size) {
        limbs_.insert(i);
      }
      assert(shares_.size() <= extension_size());
      for (auto i = 0; i < extension_size(); i++) {
        auto l = compl_max_level - i;
        if ((l % partition_size) == partition_id) {
          limbs_.insert(l);
        }
      }
      for (auto &i : shares_) {
        limbs_.erase(i);
      }
      break;

    case Term::Ext_Split:
    case Term::Ext_Split2:
      limbs_.clear();
      assert(extension_size() != 0);
      if (interpret_ext_spl_from_shares_) {
        bool has_normal_ = false;
        bool has_complement_ = false;
        partition_id = partition_id % partition_size;
        for (auto &s : shares_) {
          if ((s % partition_size) != partition_id) {
            has_complement_ = true;
          }
          if ((s % partition_size) == partition_id) {
            has_normal_ = true;
          }
        }
        assert(has_normal_ || has_complement_);
        for (auto &s : shares_) {
          if (has_complement_ && !has_normal_) {
            assert((s % partition_size) != partition_id);
          }
          if (has_normal_ && !has_complement_) {
            assert((s % partition_size) == partition_id);
          }
        }

        for (auto i = 0; i < extension_size(); i++) {
          auto l = compl_max_level - i;
          if (has_normal_ && ((l % partition_size) == partition_id)) {
            limbs_.insert(l);
          }
          if (has_complement_ && ((l % partition_size) != partition_id)) {
            limbs_.insert(l);
          }
        }

      } else {
        for (auto i = 0; i < extension_size(); i++) {
          auto l = compl_max_level - i;
          if ((l % partition_size) == (partition_id % partition_size)) {
            limbs_.insert(l);
          }
        }
      }
      break;

    default:
      assert(0);
    }
  }

  void set_dead(bool val) { dead_ = val; }

  bool is_dead() const { return dead_; }

  void set_next_use(const uint64_t next_use) { next_use_ = next_use; }

  uint64_t next_use() const { return next_use_; }

  void set_symbol(const std::string &symbol) {
    term_share_->set_symbol(symbol);
  }

  const std::string &symbol() const { return term_share_->symbol(); }

  friend std::ostream &operator<<(std::ostream &s, const Polynomial &poly);
};

std::ostream &operator<<(std::ostream &s, const Backend::Polynomial &poly);

class Plaintext {
public:
  uint32_t level;
  Polynomial pt;
};

class Ciphertext {
public:
  uint32_t level;
  Polynomial ct0;
  Polynomial ct1;
  std::optional<Polynomial> ct2;
};

class Raw {};

class CiphertextVector {
public:
  std::vector<Ciphertext> vec;
};

class PlaintextVector {
public:
  std::vector<Plaintext> vec;
};

struct PolynomialHash {
  std::size_t operator()(const Polynomial &poly) const {
    return std::hash<TermIndexType>()(poly.term_idx());
  }
};

struct PolynomialCompare {
  bool operator()(const Polynomial &lhs, const Polynomial &rhs) const {
    return lhs.term_idx() == rhs.term_idx();
  }
};

template <typename T>
using PolynomialMap =
    std::unordered_map<Polynomial, T, PolynomialHash, PolynomialCompare>;

using PolynomialSet =
    std::unordered_set<Polynomial, PolynomialHash, PolynomialCompare>;

} // namespace Backend
} // namespace Cerium
