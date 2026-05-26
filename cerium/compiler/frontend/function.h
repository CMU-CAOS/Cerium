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
#include "cerium/compiler/frontend/term.h"
#include <cstdint>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Cerium {
namespace Frontend {

template <typename> class TermMapOptional;
template <typename> class TermMap;
class TermMapBase;
class Program;
class Function : public std::enable_shared_from_this<Function> {
public:
public:
  using Ptr = std::shared_ptr<Function>;
  Function(std::string name, Program &program, std::uint8_t partitionSize,
           std::uint8_t partitionId)
      : name(name), program(program), nextTermIndex(0),
        rnsBitSize(program.rnsBitSize), slotSize(program.slotSize),
        partitionSize(partitionSize), partitionId(partitionId),
        currentPartitionSize(partitionSize), currentPartitionId(partitionId) {
    if (currentPartitionSize == 0) {
      throw std::runtime_error("Partition Size cannot be zero");
    }
    if (rnsBitSize != 28) {
      throw std::runtime_error(
          "Unsupported RNS Bit Size: " + std::to_string(rnsBitSize) +
          ". Only 28bit RNS primes are supported as of now.");
    }
  }

  void addFunctionToProgram() { program.addFunction(name, shared_from_this()); }

  uint8_t getPartitionSize() const { return partitionSize; }
  Function(const Function &copy) = delete;

  Function &operator=(const Function &assign) = delete;

  bool receiveCheck(const Term::Ptr &term) {
    auto termPartitionSize = term->getPartitionSize();
    auto termPartitionId = term->getPartitionId();

    auto currentPartitionSize = getCurrentPartitionSize();
    auto currentPartitionId = getCurrentPartitionId();

    if (termPartitionSize < currentPartitionSize) {
      if (currentPartitionSize % termPartitionSize != 0) {
        throw std::runtime_error(
            "ERROR: Receive called accross neither subpartitions: size");
        return false;
      }
      if (termPartitionId / (currentPartitionSize / termPartitionSize) !=
          currentPartitionId) {
        throw std::runtime_error(
            "ERROR: Receive called accross neither subpartitions: id");
        return false;
      }

    } else if (termPartitionSize > currentPartitionSize) {
      if (termPartitionSize % currentPartitionSize != 0) {
        throw std::runtime_error(
            "ERROR: Receive called accross neither subpartitions: size");
        return false;
      }
      if (currentPartitionId / (termPartitionSize / currentPartitionSize) !=
          termPartitionId) {
        throw std::runtime_error(
            "ERROR: Receive called accross neither subpartitions: id");
        return false;
      }

    } else {
      throw std::runtime_error(
          "ERROR: Receive called accross neither subpartitions: equal");
      return false;
    }

    return true;
  }

  void checkAndSetVectorType(Term::Ptr &out, const Term::Ptr &op1) {
    if (op1->getExtractPossible()) {
      out->setExtractPossible(true);
      out->setExtractSize(op1->getExtractSize());
    }
  }

  void checkAndSetVectorType(Term::Ptr &out, const Term::Ptr &op1,
                             const Term::Ptr &op2) {
    if (op2->getExtractPossible() && !op1->getExtractPossible()) {
      checkAndSetVectorType(out, op2, op1);
    }
    if (op1->getExtractPossible()) {
      if (op2->getExtractPossible() &&
          op1->getExtractSize() != op2->getExtractSize()) {
        throw std::runtime_error("Both operands must be of same vector size");
      }
      out->setExtractPossible(true);
      out->setExtractSize(op1->getExtractSize());
    }
  }

  Term::Ptr makeTerm(Op op, const Term::Type type, const Term::Scale_t scale,
                     const Term::Level_t level, bool ephemeralKey,
                     const std::vector<Term::Ptr> &operands_ = {}) {
    std::vector<Term::Ptr> operands;
    for (auto &o : operands_) {
      if (op == Op::Receive) {
        operands.push_back(o);
        continue;
      }
      if (o->getPartitionSize() != currentPartitionSize ||
          o->getPartitionId() != currentPartitionId) {
        if (o->getType() != Term::Type::Cipher) {
          throw std::runtime_error(
              "ERROR: Receive called on value of Type: " + o->getTypeString() +
              ". Receive Only Valid for Type::Cipher");
        }
        if (o->getNeedsRelinearization()) {
          throw std::runtime_error(
              "ERROR: Receive called on a term that needs relinearization. "
              "Operand of Receive Needs to be relinearized");
        }
        if (!receiveCheck(o)) {
          throw std::runtime_error("ERROR: Receive Check Failed");
        }
        auto receiveTerm = std::make_shared<Term>(
            Op::Receive, *this, type, scale, level, ephemeralKey,
            currentPartitionSize, currentPartitionId);
        terms.push_back(receiveTerm);
        receiveTerm->setOperands({o});
        operands.push_back(receiveTerm);
      } else {
        operands.push_back(o);
      }
    }
    auto term =
        std::make_shared<Term>(op, *this, type, scale, level, ephemeralKey,
                               currentPartitionSize, currentPartitionId);
    if (operands.size() > 0) {
      term->setOperands(operands);
    }
    terms.push_back(term);
    return term;
  }

  Term::Ptr makeTerm(Op op, const Term::Type type, const Term::Scale_t scale,
                     const Term::Level_t level, bool ephemeralKey,
                     std::uint8_t partitionSize, std::uint8_t partitionId,
                     const std::vector<Term::Ptr> &operands = {}) {
    auto term =
        std::make_shared<Term>(op, *this, type, scale, level, ephemeralKey,
                               partitionSize, partitionId);
    if (operands.size() > 0) {
      term->setOperands(operands);
    }
    terms.push_back(term);
    return term;
  }

  Term::Ptr makeCiphertextInput(const std::string &name, Term::Scale_t scale,
                                Term::Level_t level) {

    auto term = makeTerm(Op::Input, Term::Type::Cipher, scale, level, false);
    term->set<NameAttribute>(name);

    term->set<TypeAttribute>(Type::Cipher);

    inputs.emplace(name, term);

    // add new inputs to sources and sinks
    this->newSources.push_back(term);
    this->newSinks.push_back(term);
    return term;
  }

  Term::Ptr makePlaintextInput(const std::string &name, std::uint32_t scale,
                               std::uint32_t level, bool is_scalar = false,
                               bool is_remapable = false) {
    auto encode = makeTerm(Op::Input, Term::Type::Plain, scale, level, false);
    encode->set<NameAttribute>(name);
    encode->set<IsScalarAttribute>(is_scalar);
    encode->set<IsRemapableAttribute>(is_remapable);
    encode->setRemapable(is_remapable);
    if (!is_scalar) {
      encode->set<PlaintextSlotSize>(slotSize);
    } else {
      encode->setRepeatSize(0);
    }
    encode->set<TypeAttribute>(Type::Plain);
    inputs.emplace(name, encode);

    // add new inputs to sources and sinks
    this->newSources.push_back(encode);
    this->newSinks.push_back(encode);
    return encode;
  }

  Term::Ptr makeRepeatedPlaintextInput(const std::string &name,
                                       std::uint32_t scale, std::uint32_t level,
                                       std::uint32_t elts_size,
                                       bool is_remapable = false) {

    if (elts_size & (elts_size - 1) != 0) {
      throw std::runtime_error("ERROR: MakeRepeatedPlaintextInput called with "
                               "elts_size that is not a power of 2");
    }
    if (elts_size == 0) {
      throw std::runtime_error("ERROR: MakeRepeatedPlaintextInput called with "
                               "elts_size that is zero");
    }
    if (elts_size > slotSize) {
      throw std::runtime_error("ERROR: MakeRepeatedPlaintextInput called with "
                               "elts_size that is greater than slot size");
    }
    auto encode = makeTerm(Op::Input, Term::Type::Plain, scale, level, false);
    encode->set<NameAttribute>(name);
    encode->set<IsScalarAttribute>(false);
    encode->set<PlaintextSlotSize>(elts_size);
    encode->set<TypeAttribute>(Type::Plain);
    encode->set<IsRemapableAttribute>(is_remapable);
    encode->setRemapable(is_remapable);
    encode->setRepeatSize(elts_size);
    inputs.emplace(name, encode);

    // add new inputs to sources and sinks
    this->newSources.push_back(encode);
    this->newSinks.push_back(encode);
    return encode;
  }

  Term::Ptr makeVector(const std::vector<Term::Ptr> &terms) {
    if (terms.empty()) {
      throw std::runtime_error("ERROR: MakeVector called with empty vector");
    }
    for (size_t i = 0; i < terms.size(); i++) {
      // if(terms[i]->getType() != Term::Type::Cipher || terms[i]->getType() !=
      // Term::Type::Plain) {
      //   throw std::runtime_error("ERROR: MakeVector called with invalid
      //   type");
      // }
      if (terms[i]->getExtractPossible()) {
        throw std::runtime_error("ERROR: MakeVector called with invalid type");
      }
      if (terms[i]->getType() != terms[0]->getType()) {
        throw std::runtime_error(
            "ERROR: MakeVector called with terms of different types");
      }
      if (terms[i]->getScale() != terms[0]->getScale()) {
        throw std::runtime_error(
            "ERROR: MakeVector called with terms of different scales");
      }
      if (terms[i]->getLevel() != terms[0]->getLevel()) {
        throw std::runtime_error(
            "ERROR: MakeVector called with terms of different levels");
      }
      if (terms[i]->getNeedsRelinearization() !=
          terms[0]->getNeedsRelinearization()) {
        throw std::runtime_error("ERROR: MakeVector called with terms of "
                                 "different relinearzation states");
      }
      if (terms[i]->getEphemeralKey() != terms[0]->getEphemeralKey()) {
        throw std::runtime_error("ERROR: MakeVector called with terms of "
                                 "different ephemeral key states");
      }
      if (terms[i]->getRepeatSize() != terms[0]->getRepeatSize()) {
        throw std::runtime_error(
            "ERROR: MakeVector called with terms of different dimensions");
      }
      if (terms[i]->getRemapable() != terms[0]->getRemapable()) {
        throw std::runtime_error(
            "ERROR: MakeVector called with terms of different remapable types");
      }
    }
    // auto type = terms[0]->type();
    auto term =
        makeTerm(Op::MakeVector, terms[0]->getType(), terms[0]->getScale(),
                 terms[0]->getLevel(), terms[0]->getEphemeralKey(), terms);
    term->setNeedsRelinearization(terms[0]->getNeedsRelinearization());
    term->setExtractPossible(true);
    term->setExtractSize(terms.size());
    if (term->getType() == Term::Type::Plain) {
      term->setRepeatSize(terms[0]->getRepeatSize());
      term->setRemapable(terms[0]->getRemapable());
    }

    return term;
  }

  std::vector<Term::Ptr> extractVector(const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Extract called on operand not of type Cipher");
    }
    // if(term->getNeedsRelinearization()){
    //   throw std::runtime_error("ERROR: Extract called on term that needs
    //   relinearization");
    // }
    if (!term->getExtractPossible()) {
      throw std::runtime_error(
          "ERROR: Extract called on term that is not extractable");
    }
    std::vector<Term::Ptr> ret;
    for (size_t i = 0; i < term->getExtractSize(); i++) {
      auto extract =
          makeTerm(Op::TermInCiphertextVec, term->getType(), term->getScale(),
                   term->getLevel(), term->getEphemeralKey(), {term});
      extract->set<TermIdxInVec>(i);
      extract->setNeedsRelinearization(term->getNeedsRelinearization());
      ret.push_back(extract);
    }
    return std::move(ret);
  }

  Term::Ptr makeReceive(const Term::Ptr &term) {
    if (term->getPartitionSize() == getCurrentPartitionSize() &&
        term->getPartitionId() == getCurrentPartitionId()) {
      return term;
    }
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Receive called on value of Type: " + term->getTypeString() +
          ". Receive Only Valid for Type::Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Receive called on a term that needs relinearization. Operand "
          "of Receive Needs to be relinearized");
    }
    if (!receiveCheck(term)) {
      throw std::runtime_error("ERROR: Receive check Failed");
    }
    auto receive = makeTerm(Op::Receive, term->getType(), term->getScale(),
                            term->getLevel(), term->getEphemeralKey(), {term});
    return receive;
  }

  void makePartition(uint8_t partitionSize, uint8_t partitionID) {

    auto term = makeTerm(Op::Partition, Term::Type::Cipher, 0, 0, false);
    term->set<PartitionSizeAttribute>(partitionSize);
    term->set<PartitionIdAttribute>(partitionID);
    this->setCurrentPartitionSize(partitionSize);
    this->setCurrentPartitionId(partitionID);
  }

  Term::Ptr makeOutput(std::string name, const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Output called on value of Type:" + term->getTypeString() +
          "Output Only Valid for Type::Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Output called on a term that needs relinearization");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: Output called on a term that is "
                               "encrypted with the ephemeral key.");
    }
    auto output = makeTerm(Op::Output, term->getType(), term->getScale(),
                           term->getLevel(), false, {term});
    outputs.emplace(name, output);
    output->set<NameAttribute>(name);
    return output;
  }

  Term::Ptr makeAdd(const Term::Ptr &op1, const Term::Ptr &op2) {
    if (op1->getScale() != op2->getScale()) {
      std::stringstream s;
      s << "Addition Between values of unequal scale: " << op1->getScale()
        << ", " << op2->getScale();
      throw std::runtime_error(s.str());
    }
    if (op1->getLevel() != op2->getLevel()) {
      std::stringstream s;
      s << "Addition Between values of unequal level : " << op1->getLevel()
        << ", " << op2->getLevel();
      throw std::runtime_error(s.str());
    }
    if (op1->getType() == Term::Type::Cipher &&
        op2->getType() == Term::Type::Cipher &&
        (op1->getEphemeralKey() != op2->getEphemeralKey())) {
      std::stringstream s;
      s << "Ciphertext Ciphertext addition Between values of invalid ephemeral "
           "key states : "
        << op1->getEphemeralKey() << ", " << op2->getEphemeralKey();
      throw std::runtime_error(s.str());
    }
    bool ephemeralKeyState = op1->getEphemeralKey() || op2->getEphemeralKey();

    Term::Type type = Term::Type::Plain;
    if (op1->getType() == Term::Type::Cipher ||
        op2->getType() == Term::Type::Cipher) {
      type = Term::Type::Cipher;
    }

    auto add = makeTerm(Op::Add, type, op1->getScale(), op1->getLevel(),
                        ephemeralKeyState, {op1, op2});
    bool needsRelinearization =
        op1->getNeedsRelinearization() || op2->getNeedsRelinearization();
    add->setNeedsRelinearization(needsRelinearization);
    checkAndSetVectorType(add, op1, op2);
    return add;
  }

  Term::Ptr makeSubtract(const Term::Ptr &op1, const Term::Ptr &op2) {
    if (op1->getScale() != op2->getScale()) {
      std::stringstream s;
      s << "Subtraction Between values of unequal scale: " << op1->getScale()
        << ", " << op2->getScale();
      throw std::runtime_error(s.str());
    }
    if (op1->getLevel() != op2->getLevel()) {
      std::stringstream s;
      s << "Subtraction Between values of unequal level : " << op1->getLevel()
        << ", " << op2->getLevel();
      throw std::runtime_error(s.str());
    }

    if (op1->getType() == Term::Type::Cipher &&
        op2->getType() == Term::Type::Cipher &&
        (op1->getEphemeralKey() != op2->getEphemeralKey())) {
      std::stringstream s;
      s << "Ciphertext Ciphertext subtraction Between values of invalid "
           "ephemeral states : "
        << op1->getEphemeralKey() << ", " << op2->getEphemeralKey();
      throw std::runtime_error(s.str());
    }
    bool ephemeralKeyState = op1->getEphemeralKey() || op2->getEphemeralKey();

    Term::Type type = Term::Type::Plain;
    if (op1->getType() == Term::Type::Cipher ||
        op2->getType() == Term::Type::Cipher) {
      type = Term::Type::Cipher;
    }

    auto subtract = makeTerm(Op::Sub, type, op1->getScale(), op1->getLevel(),
                             ephemeralKeyState, {op1, op2});
    bool needsRelinearization =
        op1->getNeedsRelinearization() || op2->getNeedsRelinearization();
    subtract->setNeedsRelinearization(needsRelinearization);
    checkAndSetVectorType(subtract, op1, op2);
    return subtract;
  }

  Term::Ptr makeMultiply(const Term::Ptr &op1, const Term::Ptr &op2) {
    if (op1->getLevel() != op2->getLevel()) {
      std::stringstream s;
      s << "Multiplication Between values of unequal level : "
        << op1->getLevel() << ", " << op2->getLevel();
      throw std::runtime_error(s.str());
    }

    auto newScale = op1->getScale() + op2->getScale();

    bool needsRelinearization =
        op1->getNeedsRelinearization() || op2->getNeedsRelinearization();
    Term::Type type = Term::Type::Plain;
    if (op1->getType() == Term::Type::Cipher ||
        op2->getType() == Term::Type::Cipher) {
      type = Term::Type::Cipher;
    }

    if (op1->getType() == Term::Type::Cipher &&
        op2->getType() == Term::Type::Cipher) {
      if (op1->getNeedsRelinearization() || op2->getNeedsRelinearization()) {
        throw std::runtime_error(
            "ERROR: Multiply called on a term that needs relinearization. Both "
            "operands of a Ciphertext Ciphertext Multiply need to be "
            "relinearized");
      }
      needsRelinearization = true;
    }

    if (op1->getType() == Term::Type::Cipher &&
        op2->getType() == Term::Type::Cipher &&
        (op1->getEphemeralKey() != op2->getEphemeralKey())) {
      std::stringstream s;
      s << "Ciphertext Ciphertext multiplication Between values of invalid "
           "ephemeral key states : "
        << op1->getEphemeralKey() << ", " << op2->getEphemeralKey();
      throw std::runtime_error(s.str());
    }
    bool ephemeralKeyState = op1->getEphemeralKey() || op2->getEphemeralKey();

    auto multiply = makeTerm(Op::Mul, type, newScale, op1->getLevel(),
                             ephemeralKeyState, {op1, op2});
    multiply->setNeedsRelinearization(needsRelinearization);

    checkAndSetVectorType(multiply, op1, op2);
    return multiply;
  }

  Term::Ptr makeNegate(const Term::Ptr &term) {
    auto negate = makeTerm(Op::Negate, term->getType(), term->getScale(),
                           term->getLevel(), term->getEphemeralKey(), {term});
    negate->setNeedsRelinearization(term->getNeedsRelinearization());
    checkAndSetVectorType(negate, term);
    return negate;
  }

  Term::Ptr makeConjugate(const Term::Ptr &term) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Conjugate called on a term that needs relinearization. "
          "Operand of Conjugate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Conjugate called on a term that is encrypted with the "
          "ephemeral key. Operand of Conjugate Needs to be encrypted with the "
          "standard key");
    }
    auto conjugate = makeTerm(Op::Conjugate, term->getType(), term->getScale(),
                              term->getLevel(), false, {term});

    checkAndSetVectorType(conjugate, term);
    return conjugate;
  }

  Term::Ptr makeConjugate2(const Term::Ptr &term) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Conjugate called on a term that needs relinearization. "
          "Operand of Conjugate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Conjugate called on a term that is encrypted with the "
          "ephemeral key. Operand of Conjugate Needs to be encrypted with the "
          "standard key");
    }
    auto conjugate = makeTerm(Op::Conjugate2, term->getType(), term->getScale(),
                              term->getLevel(), false, {term});

    checkAndSetVectorType(conjugate, term);
    return conjugate;
  }

  Term::Ptr makeLeftRotation(const Term::Ptr &term, std::int32_t slots) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that needs relinearization. Operand "
          "of Rotate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that is encrypted with the ephemeral "
          "key. Operand of Rotate Needs to be encrypted with the standard key");
    }
    auto rotation = makeTerm(Op::RotateLeftConst, term->getType(),
                             term->getScale(), term->getLevel(), false, {term});
    rotation->set<RotationAttribute>(slots);
    checkAndSetVectorType(rotation, term);
    return rotation;
  }

  Term::Ptr makeRightRotation(const Term::Ptr &term, std::int32_t slots) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that needs relinearization. Operand "
          "of Rotate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that is encrypted with the ephemeral "
          "key. Operand of Rotate Needs to be encrypted with the standard key");
    }
    auto rotation = makeTerm(Op::RotateRightConst, term->getType(),
                             term->getScale(), term->getLevel(), false, {term});
    rotation->set<RotationAttribute>(slots);
    checkAndSetVectorType(rotation, term);
    return rotation;
  }

  Term::Ptr makeRotate2(const Term::Ptr &term, std::int32_t slots) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that needs relinearization. Operand "
          "of Rotate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that is encrypted with the ephemeral "
          "key. Operand of Rotate Needs to be encrypted with the standard key");
    }
    auto rotation = makeTerm(Op::Rotate2, term->getType(), term->getScale(),
                             term->getLevel(), false, {term});
    rotation->set<RotationAttribute>(slots);
    checkAndSetVectorType(rotation, term);
    return rotation;
  }

  Term::Ptr makeRotate3(const Term::Ptr &term, std::int32_t slots) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that needs relinearization. Operand "
          "of Rotate Needs to be relinearized");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate called on a term that is encrypted with the ephemeral "
          "key. Operand of Rotate Needs to be encrypted with the standard key");
    }
    auto rotation = makeTerm(Op::Rotate3, term->getType(), term->getScale(),
                             term->getLevel(), false, {term});
    rotation->set<RotationAttribute>(slots);
    checkAndSetVectorType(rotation, term);
    return rotation;
  }

  Term::Ptr
  makeRotateMultiplyAccumulate(const Term::Ptr &CipherTextOperand,
                               const std::vector<Term::Ptr> &PlaintextOperands,
                               std::vector<std::int32_t> RotationIndices) {
    if (CipherTextOperand->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rotate Multiply Accumulate called on a term that needs "
          "relinearization. Operand of Rotate Needs to be relinearized");
    }
    if (CipherTextOperand->getType() != Term::Type::Cipher) {
      throw std::runtime_error("ERROR: Rotate Multiply Accumulate CipherText "
                               "Operand of Invalid Type");
    }
    if (CipherTextOperand->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate Multiply Accumulate called on a term that is "
          "encrypted with the ephemeral key. Operand of Rotate Needs to be "
          "encrypted with the standard key");
    }
    if (RotationIndices.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Rotation Indices for RotateMultiplyAcc");
    }
    if (PlaintextOperands.size() != RotationIndices.size()) {
      throw std::runtime_error(
          "Must have one plaintext operand for every rotation index");
    }
    auto plaintextScale = PlaintextOperands.at(0)->getScale();
    auto ciphertextLevel = CipherTextOperand->getLevel();
    for (auto &plaintext : PlaintextOperands) {
      if (plaintextScale != plaintext->getScale()) {
        throw std::runtime_error(
            "All plaintext operands must have the same scale");
      }
      if (ciphertextLevel != plaintext->getLevel()) {
        throw std::runtime_error(
            "Ciphertext Level and plaintext Level must be equal");
      }
      if (plaintext->getType() != Term::Type::Plain) {
        throw std::runtime_error("ERROR: Rotate Multiply Accumulate Plaintext "
                                 "Operand of Invalid Type");
      }
    }

    std::vector<Term::Ptr> operands;
    operands.push_back(CipherTextOperand);
    operands.insert(operands.end(), PlaintextOperands.begin(),
                    PlaintextOperands.end());
    auto rotMulAcc = makeTerm(Op::RotMulAcc, Term::Type::Cipher,
                              CipherTextOperand->getScale() + plaintextScale,
                              CipherTextOperand->getLevel(), false, operands);
    rotMulAcc->set<RotMulAccRotationAttribute>(RotationIndices);
    return rotMulAcc;
  }

  Term::Ptr
  makeMultiplyRotateAccumulate(const Term::Ptr &CipherTextOperand,
                               const std::vector<Term::Ptr> &PlaintextOperands,
                               std::vector<std::int32_t> RotationIndices) {
    if (CipherTextOperand->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Multiply Rotate Accumulate called on a term that needs "
          "relinearization. Operand of Rotate Needs to be relinearized");
    }
    if (CipherTextOperand->getType() != Term::Type::Cipher) {
      throw std::runtime_error("ERROR: Multiply Rotate Accumulate CipherText "
                               "Operand of Invalid Type");
    }
    if (CipherTextOperand->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Multiply Rotate Accumulate called on a term that is "
          "encrypted with the ephemeral key. Operand of Rotate Needs to be "
          "encrypted with the standard key");
    }
    if (RotationIndices.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Rotation Indices for RotateMultiplyAcc");
    }
    if (PlaintextOperands.size() != RotationIndices.size()) {
      throw std::runtime_error(
          "Must have one plaintext operand for every rotation index");
    }
    auto plaintextScale = PlaintextOperands.at(0)->getScale();
    auto ciphertextLevel = CipherTextOperand->getLevel();
    for (auto &plaintext : PlaintextOperands) {
      if (plaintextScale != plaintext->getScale()) {
        throw std::runtime_error(
            "All plaintext operands must have the same scale");
      }
      if (ciphertextLevel != plaintext->getLevel()) {
        throw std::runtime_error(
            "Ciphertext Level and plaintext Level must be equal");
      }
      if (plaintext->getType() != Term::Type::Plain) {
        throw std::runtime_error("ERROR: Multiply Rotate Accumulate Plaintext "
                                 "Operand of Invalid Type");
      }
    }

    std::vector<Term::Ptr> operands;
    operands.push_back(CipherTextOperand);
    operands.insert(operands.end(), PlaintextOperands.begin(),
                    PlaintextOperands.end());
    auto mulRotAcc = makeTerm(Op::MulRotAcc, Term::Type::Cipher,
                              CipherTextOperand->getScale() + plaintextScale,
                              CipherTextOperand->getLevel(), false, operands);
    mulRotAcc->set<RotMulAccRotationAttribute>(RotationIndices);
    return mulRotAcc;
  }

  Term::Ptr makeRotateAccumulate(const Term::Ptr &CipherTextOperand,
                                 std::vector<std::int32_t> RotationIndices) {
    if (CipherTextOperand->getNeedsRelinearization()) {
      throw std::runtime_error("ERROR: Rotate Accumulate called on a term that "
                               "needs relinearization. Operand of Rotate "
                               "Multiply Needs to be relinearized");
    }
    if (CipherTextOperand->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Rotate Accumulate CipherText Operand of Invalid Type");
    }
    if (CipherTextOperand->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Rotate Accumulate called on a term that is encrypted with "
          "the ephemeral key. Operand of Rotate Needs to be encrypted with the "
          "standard key");
    }
    if (RotationIndices.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Rotation Indices for RotateAccumulate");
    }
    auto ciphertextLevel = CipherTextOperand->getLevel();

    std::vector<Term::Ptr> operands;
    operands.push_back(CipherTextOperand);
    auto rotAcc =
        makeTerm(Op::RotAcc2, Term::Type::Cipher, CipherTextOperand->getScale(),
                 CipherTextOperand->getLevel(), false, operands);
    rotAcc->set<RotAccRotationAttribute>(RotationIndices);
    return rotAcc;
  }

  Term::Ptr
  makeRotateAccumulateMany(const std::vector<Term::Ptr> &CipherTextOperands,
                           std::vector<std::int32_t> RotationIndices) {

    if (CipherTextOperands.empty()) {
      throw std::runtime_error(
          "ERROR: Operands to Rotate Accumulate Many Can't be empty");
    }

    if (CipherTextOperands.size() != RotationIndices.size()) {
      throw std::runtime_error(
          "ERROR: Rotation Indices size doesn't match number of operands");
    }

    if (RotationIndices.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Rotation Indices for RotateAccumulate");
    }

    auto ciphertextLevel = CipherTextOperands[0]->getLevel();
    auto ciphertextScale = CipherTextOperands[0]->getScale();
    for (auto &ct : CipherTextOperands) {
      if (ct->getNeedsRelinearization()) {
        throw std::runtime_error(
            "ERROR: Rotate Accumulate called on a term that "
            "needs relinearization. Operand of Rotate "
            "Needs to be relinearized");
      }
      if (ct->getType() != Term::Type::Cipher) {
        throw std::runtime_error(
            "ERROR: Rotate Accumulate CipherText Operand of Invalid Type");
      }
      if (ct->getEphemeralKey()) {
        throw std::runtime_error(
            "ERROR: Rotate Accumulate called on a term that is encrypted with "
            "the ephemeral key. Operand of Rotate Needs to be encrypted with "
            "the "
            "standard key");
      }
      if (ct->getLevel() != ciphertextLevel) {
        throw std::runtime_error("All operands must have the same level");
      }

      if (ct->getScale() != ciphertextScale) {
        throw std::runtime_error("All operands must have the same scale");
      }
    }

    auto rotAcc = makeTerm(Op::RotAcc, Term::Type::Cipher, ciphertextScale,
                           ciphertextLevel, false, CipherTextOperands);
    // rotAcc->set<RotAccRotationAttribute>(RotationIndices);
    rotAcc->set<MultiRotationAttribute>(RotationIndices);
    return rotAcc;
  }

  std::vector<Term::Ptr>
  makeHoistedRotate(const Term::Ptr &CipherTextOperand,
                    std::vector<std::int32_t> RotationIndices) {

    if (RotationIndices.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Rotation Indices for Hoisted Rotate");
    }

    if (CipherTextOperand->getNeedsRelinearization()) {
      throw std::runtime_error("ERROR: Hoisted Rotate called on a term that "
                               "needs relinearization. Operand of Rotate "
                               "Needs to be relinearized");
    }
    if (CipherTextOperand->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Hoisted Rotate CipherText Operand of Invalid Type");
    }
    if (CipherTextOperand->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Hoisted Rotate called on a term that is encrypted with "
          "the ephemeral key. Operand of Rotate Needs to be encrypted with the "
          "standard key");
    }

    auto hoistedRotate =
        makeTerm(Op::HoistInpBroadcast, Term::Type::Cipher,
                 CipherTextOperand->getScale(), CipherTextOperand->getLevel(),
                 false, {CipherTextOperand});
    hoistedRotate->setExtractPossible(true);
    hoistedRotate->setExtractSize(RotationIndices.size());
    hoistedRotate->set<MultiRotationAttribute>(RotationIndices);
    return extractVector(hoistedRotate);
  }

  Term::Ptr makeBsgsMultiplyAccumulate(
      const Term::Ptr &CipherTextOperand,
      const std::vector<Term::Ptr> &PlaintextOperands,
      const std::vector<std::int32_t> &BabyStepRotationIndicies,
      const std::vector<std::int32_t> &GiantStepRotationIndices,
      const uint32_t rescaleLevels) {
    if (CipherTextOperand->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: BSGS Multiply Accumulate CipherText Operand of Invalid Type");
    }
    if (CipherTextOperand->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Bsgs Multiply Accumulate called on a term that needs "
          "relinearization. Operand of Rotate Needs to be relinearized");
    }
    if (CipherTextOperand->getEphemeralKey()) {
      throw std::runtime_error(
          "ERROR: Bsgs Multiply Accumulate called on a term that is encrypted "
          "with the ephemeral key. Operand of Rotate Needs to be encrypted "
          "with the standard key");
    }
    if (BabyStepRotationIndicies.size() <= 1) {
      throw std::runtime_error(
          "Must have atleast 2 Baby Step Rotation Indices for RotMulAcc");
    }
    if (PlaintextOperands.size() !=
        GiantStepRotationIndices.size() * BabyStepRotationIndicies.size()) {
      throw std::runtime_error("Incorrect Number of Plaintext Operands for "
                               "Specified Baby Step operation Giant Step");
    }

    auto plaintextScale = PlaintextOperands.at(0)->getScale();
    auto ciphertextLevel = CipherTextOperand->getLevel();
    for (auto &plaintext : PlaintextOperands) {
      if (plaintextScale != plaintext->getScale()) {
        throw std::runtime_error(
            "All plaintext operands must have the same scale");
      }
      if (ciphertextLevel != plaintext->getLevel()) {
        throw std::runtime_error(
            "Ciphertext Level and plaintext Level must be equal");
      }
      if (plaintext->getType() != Term::Type::Plain) {
        throw std::runtime_error("ERROR: BSGS Multiply Accumulate Plaintext "
                                 "Operand of Invalid Type");
      }
    }

    std::vector<Term::Ptr> operands;
    operands.push_back(CipherTextOperand);
    operands.insert(operands.end(), PlaintextOperands.begin(),
                    PlaintextOperands.end());
    const auto fixedRescale = 28;
    if (CipherTextOperand->getScale() + plaintextScale <=
        fixedRescale * rescaleLevels) {
      throw std::runtime_error("Rescale results in a term with 0 scale");
    }
    auto bsgsScale = CipherTextOperand->getScale() + plaintextScale -
                     rescaleLevels * fixedRescale;
    auto bsgsLevel = CipherTextOperand->getLevel() - rescaleLevels;
    auto bsgsMulAcc = makeTerm(Op::BsgsMulAcc, Term::Type::Cipher, bsgsScale,
                               bsgsLevel, false, operands);
    bsgsMulAcc->set<BsgsMulAccBabyStepAttribute>(BabyStepRotationIndicies);
    bsgsMulAcc->set<BsgsMulAccGiantStepAttribute>(GiantStepRotationIndices);
    bsgsMulAcc->set<RescaleLevelsAttribute>(rescaleLevels);
    checkAndSetVectorType(bsgsMulAcc, CipherTextOperand);
    return bsgsMulAcc;
  }

  Term::Ptr makeRescale(const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Rescale called on operand not of type Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Rescale called on term that needs relinearization");
    }
    auto fixedRescale = rnsBitSize;
    if (term->getScale() <= fixedRescale) {
      throw std::runtime_error("Rescale results in a term with 0 scale");
    }
    auto newScale = term->getScale() - fixedRescale;
    if (term->getLevel() <= 1) {
      throw std::runtime_error("Rescale results in a term with level 0");
    }
    // auto newLevel = term->getLevel() + 1;
    auto newLevel = term->getLevel() - 1;
    auto rescale = makeTerm(Op::Rescale, term->getType(), newScale, newLevel,
                            term->getEphemeralKey(), {term});

    checkAndSetVectorType(rescale, term);
    return rescale;
  }

  Term::Ptr makeDoubleRescale(const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Double Rescale called on operand not of type Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Double Rescale called on term that needs relinearization");
    }
    auto fixedRescale = rnsBitSize * 2;
    if (term->getScale() <= fixedRescale) {
      throw std::runtime_error("Double Rescale results in a term with 0 scale");
    }
    auto newScale = term->getScale() - fixedRescale;
    if (term->getLevel() <= 2) {
      throw std::runtime_error("Double Rescale results in a term with 0 level");
    }
    auto newLevel = term->getLevel() - 2;
    auto rescale = makeTerm(Op::DoubleRescale, term->getType(), newScale,
                            newLevel, term->getEphemeralKey(), {term});
    // rescale->set<RescaleDivisorAttribute>(fixedRescale);
    checkAndSetVectorType(rescale, term);
    return rescale;
  }
  Term::Ptr makeRelinearize(const Term::Ptr &term,
                            const uint32_t rescaleLevels) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Relinearize called on operand not of type Cipher");
    }
    if (!term->getNeedsRelinearization()) {
      throw std::runtime_error("ERROR: Relinearization called on term that "
                               "does not need Relinearization");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: Relinearization called on term that is "
                               "encrypted with the ephemeral key");
    }
    const uint32_t fixedRescale = rnsBitSize;
    if (term->getScale() <= rescaleLevels * fixedRescale) {
      throw std::runtime_error("Rescale results in a term with 0 scale");
    }
    auto newScale = term->getScale() - rescaleLevels * fixedRescale;
    auto newLevel = term->getLevel() - rescaleLevels;
    auto relin = makeTerm(Op::Relinearize, Term::Type::Cipher, newScale,
                          newLevel, false, {term});
    relin->setNeedsRelinearization(false);
    relin->set<RescaleLevelsAttribute>(rescaleLevels);
    checkAndSetVectorType(relin, term);
    return relin;
  }

  Term::Ptr makeRelinearize2(const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Relinearize called on operand not of type Cipher");
    }
    if (!term->getNeedsRelinearization()) {
      throw std::runtime_error("ERROR: Relinearization called on term that "
                               "does not need Relinearization");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: Relinearization called on term that is "
                               "encrypted with the ephemeral key");
    }
    auto relin = makeTerm(Op::Relinearize2, Term::Type::Cipher,
                          term->getScale(), term->getLevel(), false, {term});
    relin->setNeedsRelinearization(false);
    return relin;
  }

  Term::Ptr makeRelinearize3(const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Relinearize called on operand not of type Cipher");
    }
    if (!term->getNeedsRelinearization()) {
      throw std::runtime_error("ERROR: Relinearization called on term that "
                               "does not need Relinearization");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: Relinearization called on term that is "
                               "encrypted with the ephemeral key");
    }
    auto relin = makeTerm(Op::Relinearize3, Term::Type::Cipher,
                          term->getScale(), term->getLevel(), false, {term});
    relin->setNeedsRelinearization(false);
    return relin;
  }

  Term::Ptr makeToEphemeral(const Term::Ptr &term) {
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Ephemeral called on a term that needs relinearization. "
          "Operand of needs to be relineraised");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: To Ephemeral called on term that is "
                               "encrypted with the ephemeral key");
    }
    auto ephemeral = makeTerm(Op::ToEphemeral, term->getType(),
                              term->getScale(), term->getLevel(), true, {term});

    checkAndSetVectorType(ephemeral, term);
    return ephemeral;
  }

  Term::Ptr makeModSwitch(const Term::Ptr &term) {
    // auto newLevel = term->getLevel() + 1;
    if (term->getLevel() <= 1) {
      throw std::runtime_error(
          "ERROR: ModSwitch will result in a term with level 0");
    }
    auto newLevel = term->getLevel() - 1;
    auto modSwitch = makeTerm(Op::ModSwitch, term->getType(), term->getScale(),
                              newLevel, term->getEphemeralKey(), {term});
    modSwitch->setNeedsRelinearization(term->getNeedsRelinearization());
    checkAndSetVectorType(modSwitch, term);
    return modSwitch;
  }

  Term::Ptr makeBootstrapModRaise(const Term::Ptr &term,
                                  const Term::Level_t newLevel) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: BootstrapModRaise called on operand not of type Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: BootstrapModRaise called on term needs to be reliniearized");
    }
    if (!term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: To BootstrapModRaise called on term "
                               "that is not encrypted with the ephemeral key");
    }
    auto modRaise = makeTerm(Op::BootstrapModRaise, Term::Type::Cipher,
                             term->getScale(), newLevel, false, {term});
    modRaise->set<ModRaiseLevelAttribute>(newLevel);
    checkAndSetVectorType(modRaise, term);
    return modRaise;
  }

  Term::Ptr makeCiphertextArgument(const std::string &name, Term::Scale_t scale,
                                   Term::Level_t level) {

    auto term =
        makeTerm(Op::FunctionInput, Term::Type::Cipher, scale, level, false);
    term->set<NameAttribute>(name);

    term->set<TypeAttribute>(Type::Cipher);
    term->set<FunctionArgumentNameAttribute>(name);

    inputs.emplace(name, term);
    function_input_args.emplace(name, term);

    // add new inputs to sources and sinks
    // this->newSources.push_back(term);
    // this->newSinks.push_back(term);
    return term;
  }

  Term::Ptr makeFunctionOutput(std::string name, const Term::Ptr &term) {
    if (term->getType() != Term::Type::Cipher) {
      throw std::runtime_error(
          "ERROR: Output called on value of Type:" + term->getTypeString() +
          "Output Only Valid for Type::Cipher");
    }
    if (term->getNeedsRelinearization()) {
      throw std::runtime_error(
          "ERROR: Output called on a term that needs relinearization");
    }
    if (term->getEphemeralKey()) {
      throw std::runtime_error("ERROR: Output called on a term that is "
                               "encrypted with the ephemeral key.");
    }
    auto output = makeTerm(Op::FunctionOutput, term->getType(),
                           term->getScale(), term->getLevel(), false, {term});
    outputs.emplace(name, output);
    function_output_args.emplace(name, output);
    output->set<FunctionArgumentNameAttribute>(name);
    return output;
  }

  std::unordered_map<std::string, Term::Ptr>
  makeFunctionCall(Function &func,
                   const std::unordered_map<std::string, Term::Ptr> &arguments,
                   const std::string remapableBase = "") {

    if (getCurrentPartitionSize() != func.getPartitionSize()) {
      throw std::runtime_error("Function \"" + func.name +
                               "\" defined with a stream size of: " +
                               std::to_string(func.getPartitionSize()) +
                               " cannot be called from a stream of size: " +
                               std::to_string(getCurrentPartitionSize()));
    }
    // auto & func_args = func.inputs;
    auto &func_args = func.function_input_args;
    std::vector<Term::Ptr> input_arguments;
    for (auto &[k, v] : func_args) {
      if (arguments.find(k) == arguments.end()) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k;
        throw std::runtime_error(s.str());
      }
      auto arg = arguments.at(k);
      if (arg->getType() != v->getType()) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k
          << " of type " << v->getTypeString() << ", but got "
          << arg->getTypeString();
        throw std::runtime_error(s.str());
      }
      if (arg->getScale() != v->getScale()) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k
          << " of scale " << v->getScale() << ", but got " << arg->getScale();
        throw std::runtime_error(s.str());
      }

      if (arg->getLevel() != v->getLevel()) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k
          << " of level " << v->getLevel() << ", but got " << arg->getLevel();
        throw std::runtime_error(s.str());
      }

      if (arg->getPartitionSize() != v->getPartitionSize()) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k
          << " of partition size"
          << static_cast<uint32_t>(v->getPartitionSize()) << ", but got "
          << static_cast<uint32_t>(arg->getPartitionSize());
        throw std::runtime_error(s.str());
      }

      if ((arg->getPartitionSize() * arg->getPartitionId()) %
              (func.getPartitionSize()) !=
          (v->getPartitionId())) {
        std::stringstream s;
        s << "Function " << func.name << " requires argument " << k
          << " of partition size and id" << v->getPartitionSize() << ":"
          << v->getPartitionId() << ", but got " << arg->getPartitionSize()
          << ":" << arg->getPartitionId();
        throw std::runtime_error(s.str());
      }

      // if (arg->getPartitionSize() != 1) {
      //   std::stringstream s;
      //   s << "Function " << func.name << " requires argument " << k
      //     << " of partition size" << 1 << ", but got "
      //     << arg->getPartitionSize();
      //   throw std::runtime_error(s.str());
      // }

      auto farg = makeTerm(Op::FunctionInputArg, v->getType(), v->getScale(),
                           v->getLevel(), false, arg->getPartitionSize(),
                           arg->getPartitionId(), {arg});
      checkAndSetVectorType(farg, arg);
      farg->set<FunctionArgumentNameAttribute>(k);
      farg->set<FunctionNameAttribute>(func.name);
      input_arguments.push_back(farg);
    }

    auto functionCallTerm = makeTerm(Op::FunctionCall, Term::Type::Cipher, 0, 0,
                                     false, getCurrentPartitionSize(),
                                     getCurrentPartitionId(), input_arguments);
    functionCallTerm->set<FunctionNameAttribute>(func.name);
    if (!remapableBase.empty()) {
      functionCallTerm->set<FunctionRemapableBaseAttribute>(remapableBase);
    }

    auto &func_outputs = func.function_output_args;
    std::unordered_map<std::string, Term::Ptr> outputs;
    for (auto &[k, v] : func_outputs) {
      auto partitionId = getCurrentPartitionId() * getCurrentPartitionSize() +
                         v->getPartitionId();
      auto term = makeTerm(Op::FunctionOutputArg, v->getType(), v->getScale(),
                           v->getLevel(), false, v->getPartitionSize(),
                           partitionId, {functionCallTerm});
      checkAndSetVectorType(term, v);
      term->set<FunctionArgumentNameAttribute>(k);
      term->set<FunctionNameAttribute>(func.name);
      outputs.emplace(k, term);
      // term->set<NameAttribute>(k);

      auto func_ptr = program.getFunction(func.name);
      calls.insert(func_ptr);
      func_ptr->called_by.insert(program.getFunction(name));

      // calls.insert(func.shared_ptr());
      // calls.insert(func.shared_ptr());

      // calls.insert(func.shared_ptr());
    }

    return outputs;
    // return 1;
  }

  Term::Ptr getInput(std::string name) const {
    if (inputs.find(name) == inputs.end()) {
      std::stringstream s;
      s << "No input named " << name;
      throw std::out_of_range(s.str());
    }
    return inputs.at(name);
  }

  const auto &getInputs() const { return inputs; }

  const auto &getOutputs() const { return outputs; }

  const auto &getTerms() const { return terms; }

  std::string getName() const { return name; }
  void setName(std::string newName) { name = newName; }

  std::uint8_t getCurrentPartitionSize() const { return currentPartitionSize; }

  std::uint8_t getCurrentPartitionId() const { return currentPartitionId; }

  void setCurrentPartitionSize(std::uint8_t size) {
    currentPartitionSize = size;
  }

  void setCurrentPartitionId(std::uint8_t id) { currentPartitionId = id; }

  std::vector<Term::Ptr> getSources() const;

  std::vector<Term::Ptr> getSinks() const;

  std::vector<Term::Ptr> getNewSources() const;

  std::vector<Term::Ptr> getNewSinks() const;

  void emptyNewSources();

  void emptyNewSinks();

  std::string toDOT(std::stringstream &s) const;
  std::string dump(TermMapOptional<std::uint32_t> &scales, TermMap<Type> &types,
                   TermMap<std::uint32_t> &level) const;

  std::uint64_t numTerms() const;

  const auto &getCalls() { return calls; }

private:
  Program &program;

  std::uint64_t allocateIndex();
  void initTermMap(TermMapBase &termMap);
  void registerTermMap(TermMapBase *annotation);
  void unregisterTermMap(TermMapBase *annotation);

  std::string name;

  std::uint32_t rnsBitSize;

  std::uint32_t slotSize;

  std::uint8_t partitionSize;
  std::uint8_t partitionId;
  std::uint8_t currentPartitionSize;
  std::uint8_t currentPartitionId;

  // These are managed automatically by Term
  std::unordered_set<Term *> sources;
  std::unordered_set<Term *> sinks;

  // hold constant terms created during transformations
  std::vector<Term::Ptr> newSources;
  std::vector<Term::Ptr> newSinks;

  std::uint64_t nextTermIndex;
  std::vector<TermMapBase *> termMaps;

  std::vector<Term::Ptr> terms;

  // These members must currently be last, because their destruction triggers
  // associated Terms to be destructed, which still use the sources and sinks
  // structures above.
  // TODO: move away from shared ownership for Terms and have Program own them
  // uniquely. It is an error to hold onto a Term longer than a Program, but
  // the shared_ptr is misleading on this regard.
  std::unordered_map<std::string, Term::Ptr> outputs;
  std::unordered_map<std::string, Term::Ptr> inputs;
  std::unordered_map<std::string, Term::Ptr> function_input_args;
  std::unordered_map<std::string, Term::Ptr> function_output_args;

  std::set<std::shared_ptr<Function>> calls;
  std::set<std::shared_ptr<Function>> called_by;

  friend class Term;
  friend class TermMapBase;
  friend class Program;
};

} // namespace Frontend
} // namespace Cerium
