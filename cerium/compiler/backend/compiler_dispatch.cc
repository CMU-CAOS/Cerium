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

#include "cerium/compiler/backend/compiler.h"

namespace Cerium {
namespace Backend {

namespace {

void set_cipher_input_symbols(Backend::Ciphertext &output,
                              const std::string &name) {
  output.ct0.set_symbol(name + ":c0");
  output.ct1.set_symbol(name + ":c1");
}

} // namespace

void CeriumCompiler::initialize_cipher_input(
    const Frontend::Term::Ptr &term, uint16_t limbs, const std::string &name,
    Backend::Polynomial (CeriumCompiler::*make_poly)(uint16_t)) {
  auto &output = initValue<Backend::Ciphertext>(term);
  output.ct0 = (this->*make_poly)(limbs);
  output.ct1 = (this->*make_poly)(limbs);
  output.level = limbs;
  set_cipher_input_symbols(output, name);

  auto kg_input = create_kernel_group();
  auto inp_instr = std::make_shared<InputInstruction>();
  inp_instr->add_operands(output.ct0);
  inp_instr->add_operands(output.ct1);
  kg_input->push_instruction(inp_instr);
}

void CeriumCompiler::process_input(const Frontend::Term::Ptr &term) {
  using namespace Frontend;
  switch (term->get<TypeAttribute>()) {
  case Type::Cipher: {
    uint16_t limbs = term->getLevel(); // change to max levels
    auto &name = term->get<NameAttribute>();
    initialize_cipher_input(term, limbs, name, &CeriumCompiler::make_input);

  } break;
  case Type::Plain: {
    auto &output = initValue<Backend::Plaintext>(term);
    uint16_t level = term->getLevel(); // change to max levels
    output.level = level;
    auto &name = term->get<NameAttribute>();
    if (term->get<IsScalarAttribute>() == true) {
      output.pt = make_scalar(level);
      output.pt.set_symbol(name + ":s");
    } else {
      auto pt_slot_size = term->get<PlaintextSlotSize>();
      if (pt_slot_size == 0 || pt_slot_size > SLOTS) {
        throw std::runtime_error("Plaintext slot size must be between 1 and " +
                                 std::to_string(SLOTS));
      }
      if ((pt_slot_size & (pt_slot_size - 1)) != 0) {
        throw std::runtime_error("Plaintext slot size must be a power of 2");
      }
      output.pt = make_plaintext(level);
      output.pt.set_plaintext_repeat_size(pt_slot_size);
      output.pt.set_symbol(name + ":p" + ":" + std::to_string(pt_slot_size));
    }
    if (term->get<IsRemapableAttribute>()) {
      output.pt.set_plaintext_remapable(true);
    }
    auto kg_input = create_kernel_group();
    auto inp_instr = std::make_shared<InputInstruction>();
    inp_instr->add_operands(output.pt);
    kg_input->push_instruction(inp_instr);

  } break;
  default:
    throw std::runtime_error("Invalid Type");
  }
  return;
}

void CeriumCompiler::process_function_call(const Frontend::Term::Ptr &term) {

  using namespace Frontend;
  auto function_name = term->get<FunctionNameAttribute>();
  std::string remapable_base;
  if (term->has<FunctionRemapableBaseAttribute>()) {
    remapable_base = term->get<FunctionRemapableBaseAttribute>();
  }
  std::unordered_map<std::string, Backend::Polynomial> function_input_args_map;

  std::unordered_map<std::string, Backend::Polynomial> function_output_args_map;

  std::unordered_map<std::string, Backend::PartitionInfo>
      function_partition_info;
  auto iargs = term->getOperands();
  auto oargs = term->getUses();
  for (auto &arg : iargs) {
    auto arg_name = arg->get<FunctionArgumentNameAttribute>();
    auto function_name_arg = arg->get<FunctionNameAttribute>();
    if (arg->getOp() != Op::FunctionInputArg) {
      std::stringstream s;
      s << "Function " << function_name << " got argument " << arg_name
        << " of type " << getOpName(arg->getOp());
      throw std::runtime_error(s.str());
    }
    if (function_name != function_name_arg) {
      std::stringstream s;
      s << "Function " << function_name << " got argument " << arg_name
        << " from function " << function_name_arg;
      throw std::runtime_error(s.str());
    }

    assert(arg->numOperands() == 1);

    // TODO: Check for all types
    Backend::Ciphertext &arg_value =
        std::get<Backend::Ciphertext>(Objects.at(arg->getOperands()[0]));
    Backend::PartitionInfo part_info{.partition_id = arg->getPartitionId(),
                                     .partition_size = arg->getPartitionSize()};

    function_input_args_map[arg_name + ":c0"] = arg_value.ct0;
    function_input_args_map[arg_name + ":c1"] = arg_value.ct1;
    function_partition_info[arg_name + ":c0"] = part_info;
    function_partition_info[arg_name + ":c1"] = part_info;
  }

  for (auto &arg : oargs) {
    auto arg_name = arg->get<FunctionArgumentNameAttribute>();
    auto function_name_arg = arg->get<FunctionNameAttribute>();
    if (arg->getOp() != Op::FunctionOutputArg) {
      std::stringstream s;
      s << "Function " << function_name << " got argument " << arg_name
        << " of type " << getOpName(arg->getOp());
      throw std::runtime_error(s.str());
    }
    if (function_name != function_name_arg) {
      std::stringstream s;
      s << "Function " << function_name << " got argument " << arg_name
        << " from function " << function_name_arg;
      throw std::runtime_error(s.str());
    }

    assert(arg->numOperands() == 1);

    // TODO: Check for all types
    // auto arg_value = Objects.at(arg->getOperands()[0]);
    Backend::Ciphertext &arg_value = initValue<Backend::Ciphertext>(arg);
    arg_value.level = arg->getLevel();
    arg_value.ct0 = make_treg(arg_value.level);
    arg_value.ct1 = make_treg(arg_value.level);

    Backend::PartitionInfo part_info{.partition_id = arg->getPartitionId(),
                                     .partition_size = arg->getPartitionSize()};
    function_output_args_map[arg_name + ":c0"] = arg_value.ct0;
    function_output_args_map[arg_name + ":c1"] = arg_value.ct1;
    function_partition_info[arg_name + ":c0"] = part_info;
    function_partition_info[arg_name + ":c1"] = part_info;
  }

  auto kg = create_kernel_group_call();
  auto function_call = std::make_shared<Backend::CallInstruction>(
      function_name, function_output_args_map, function_input_args_map,
      function_partition_info, remapable_base);
  kg->push_instruction(function_call);
}

void CeriumCompiler::process_function_input(const Frontend::Term::Ptr &term) {

  using namespace Frontend;
  switch (term->get<TypeAttribute>()) {
  case Type::Cipher: {
    uint16_t limbs = term->getLevel();
    auto &name = term->get<FunctionArgumentNameAttribute>();
    initialize_cipher_input(term, limbs, name,
                            &CeriumCompiler::make_function_input);

  } break;
  case Type::Plain: {
    throw std::runtime_error("Invalid Type");
  } break;
  default:
    throw std::runtime_error("Invalid Type");
  }
  return;
}

void CeriumCompiler::process_function_output(
    const Frontend::Term::Ptr &term) {

  using namespace Frontend;
  auto &args = term->getOperands();
  assert(args.size() == 1);
  assert(isCipher(args[0]));

  auto &name = term->get<FunctionArgumentNameAttribute>();
  Backend::Ciphertext &input1 =
      std::get<Backend::Ciphertext>(Objects.at(args[0]));
  assert(!input1.ct2.has_value());
  input1.ct0.set_output(true);
  input1.ct1.set_output(true);

  input1.ct0.set_as_function_arg(true);
  input1.ct1.set_as_function_arg(true);

  input1.ct0.set_symbol(name + ":c0");
  input1.ct1.set_symbol(name + ":c1");
}

void CeriumCompiler::operator()(Frontend::Function &f,
                                  const Frontend::Term::Ptr &term) {
  using namespace Frontend;
  // if (verbosityAtLeast(Verbosity::Debug)) {
  if (false) {
    printf("EVA: Execute t%lu = %s(", term->index,
           getOpName(term->getOp()).c_str());
    bool first = true;
    for (auto &operand : term->getOperands()) {
      if (first) {
        first = false;
        printf("t%lu", operand->index);
      } else {
        printf(",t%lu", operand->index);
      }
    }
    printf(")\n");
    fflush(stdout);
  }

  auto args = term->getOperands();
  switch (term->getOp()) {

  case Op::Nop:
    // Do nothing
    break;
  case Op::Input:
    process_input(term);
    break;
  case Op::FunctionInputArg:
    // Nothing to Do. This is internally handeled by Op::FunctionCall
    break;
  case Op::FunctionOutputArg:
    // Nothing to Do. This is internally handeled by Op::FunctionCall
    break;
  case Op::FunctionCall:
    process_function_call(term);
    break;
  case Op::FunctionInput:
    process_function_input(term);
    break;
  case Op::FunctionOutput:
    process_function_output(term);
    break;
  // case Op::Negate:
  //   assert(args.size() == 1);
  //   if (isRaw(args[0])) {
  //     // Unimplimented for now
  //     throw std::runtime_error("Unhandled op " + getOpName(term->getOp()));
  //     assert(0);
  //   } else { // works on cipher, no plaintext support
  //     assert(isCipher(args[0]));
  //     auto &output = initValue<Backend::Ciphertext>(term);
  //     negate(output, args[0]);
  //     // printf("New Negate Ciphertext at level %u\n", output.level);
  //   }
  //   break;
  case Op::MakeVector: {
    assert(args.size() > 0);
    switch (term->getType()) {
    case Frontend::Term::Type::Cipher: {
      auto &output = initValue<Backend::CiphertextVector>(term);
      std::vector<Backend::Ciphertext> cts;
      for (auto &arg : args) {
        assert(isCipher(arg));
        cts.push_back(std::get<Backend::Ciphertext>(Objects.at(arg)));
      }
      output.vec = cts;
    } break;
    case Frontend::Term::Type::Plain: {
      auto &output = initValue<Backend::PlaintextVector>(term);
      std::vector<Backend::Plaintext> pts;
      for (auto &arg : args) {
        assert(isPlain(arg));
        pts.push_back(std::get<Backend::Plaintext>(Objects.at(arg)));
      }
      output.vec = pts;
      assert(isVector(term));
    } break;
    }

  } break;
  case Op::Add: {
    assert(args.size() == 2);
    add(term, args[0], args[1]);
  } break;
  case Op::Sub: {
    assert(args.size() == 2);
    sub(term, args[0], args[1]);
  } break;
  case Op::Mul: {
    assert(args.size() == 2);
    mul(term, args[0], args[1]);
  } break;
  case Op::ToEphemeral: { // works on cipher
    assert(isCipher(args[0]));
    auto &output = initValue<Backend::Ciphertext>(term);
    to_ephemeral(output, args[0]);
  } break;
  case Op::BootstrapModRaise:
    assert(args.size() == 1);
    { 
      assert(isCipher(args[0]));
      auto &output = initValue<Backend::Ciphertext>(term);
      uint16_t raise_to_level =
          term->get<ModRaiseLevelAttribute>(); 
      bootstrap_mod_raise(output, args[0], raise_to_level, true);
    }
    break;
  case Op::Relinearize: {
    assert(args.size() == 1);
    relinearize(term, args[0]);
  } break;
  case Op::Relinearize2: {
    assert(args.size() == 1);
    relinearize2(term, args[0]);
  } break;
  case Op::Rescale: {
    assert(args.size() == 1);
    rescale(term, args[0]);
  } break;
  case Op::DoubleRescale:
    assert(args.size() == 1);
    double_rescale(term, args[0]);
    break;
  case Op::BsgsMulAcc: {
    auto &babystep_rotation_indices = term->get<BsgsMulAccBabyStepAttribute>();
    auto &giantstep_rotation_indices =
        term->get<BsgsMulAccGiantStepAttribute>();
    assert(args.size() == babystep_rotation_indices.size() *
                                  giantstep_rotation_indices.size() +
                              1);
    uint32_t rescale_levels = 0;
    if (term->has<RescaleLevelsAttribute>()) {
      rescale_levels = term->get<RescaleLevelsAttribute>();
    }
    bsgs_multiply_accumulate(term, args, babystep_rotation_indices,
                                 giantstep_rotation_indices, rescale_levels);
  } break;

  case Op::HoistInpBroadcast: {
    auto rotataion_indices = term->get<MultiRotationAttribute>();
    hoisted_input_broadcast(term, rotataion_indices, 0);
  }; break;
  case Op::RotAcc: {
    auto rotataion_indices = term->get<MultiRotationAttribute>();
    rotate_accumulate(term, rotataion_indices, 0);
  }; break;
  case Op::RotAcc2: {
    auto rotataion_indices = term->get<RotAccRotationAttribute>();
    rotate_accumulate(term, rotataion_indices, 0);
  }; break;
  case Op::RotMulAcc: {
    auto &rotation_indices = term->get<RotMulAccRotationAttribute>();
    assert(args.size() == rotation_indices.size() + 1);
    uint32_t rescale_levels = 0;
    if (term->has<RescaleLevelsAttribute>()) {
      rescale_levels = term->get<RescaleLevelsAttribute>();
    }
    rotate_multiply_accumulate(term, args, rotation_indices, rescale_levels);
  } break;

  case Op::MulRotAcc: {
    auto &rotation_indices = term->get<RotMulAccRotationAttribute>();
    assert(args.size() == rotation_indices.size() + 1);
    uint32_t rescale_levels = 0;
    if (term->has<RescaleLevelsAttribute>()) {
      rescale_levels = term->get<RescaleLevelsAttribute>();
    }
    multiply_rotate_accumulate(term, args, rotation_indices, rescale_levels);
  } break;

  case Op::RotateLeftConst:
    assert(args.size() == 1);
    rotate(term, args[0], term->get<RotationAttribute>());
    break;
  case Op::RotateRightConst:
    assert(args.size() == 1);
    rotate(term, args[0], -term->get<RotationAttribute>());
    break;
  case Op::Conjugate:
    assert(args.size() == 1);
    conjugate(term, args[0]);
    break;
  case Op::Conjugate2:
    assert(args.size() == 1);
    conjugate(term, args[0], true);
    break;

  case Op::Rotate2:
    assert(args.size() == 1);
    { 
      assert(isCipher(args[0]));
      auto &output = initValue<Backend::Ciphertext>(term);
      rotate(output, args[0], term->get<RotationAttribute>(),
             true /*use keyswitch2*/);
    }
    break;
  case Op::Rotate3:
    assert(args.size() == 1);
    { 
      assert(isCipher(args[0]));
      auto &output = initValue<Backend::Ciphertext>(term);
      throw std::runtime_error("Rotate3 is currently disabled");
    }
    break;
  case Op::ModSwitch:
    assert(args.size() == 1);
    mod_switch(term, args[0]);
    break;
  case Op::TermInCiphertextVec:
    assert(args.size() == 1);
    { 
      auto &extract_idx = term->get<TermIdxInVec>();
      assert(isCiphertextVector(args[0]));
      auto &output = initValue<Backend::Ciphertext>(term);
      Backend::CiphertextVector &ct_vec =
          std::get<Backend::CiphertextVector>(Objects.at(args[0]));
      output = ct_vec.vec.at(extract_idx);
    }
    break;
  case Op::Receive: {
    if (args.size() != 1) {
      throw std::runtime_error("Receive op expects exactly one operand");
    }
    if (term->has<Frontend::ReceiveDestinationsAttribute>()) {
      receive2(term, args[0]);
    } else {
      receive(term, args[0]);
    }
  } break;
  case Op::Reduce: {
    reduce(term, args);
  } break;
  case Op::Partition: {
    Backend::PartitionInfo partition_info{
        static_cast<uint16_t>(term->get<PartitionIdAttribute>()),
        static_cast<uint16_t>(term->get<PartitionSizeAttribute>())};
    partition(partition_info);
  } break;
  case Op::Output: {
    if (args.size() != 1) {
      throw std::runtime_error("Output op expects exactly one operand");
    }
    if (!isCipher(args[0])) {
      throw std::runtime_error("Output op expects a ciphertext operand");
    }
    set_output(args[0], term->get<NameAttribute>());
  } break;
  default:
    throw std::runtime_error("Unhandled op : " + getOpName(term->getOp()));
  }
}

} // namespace Backend
} // namespace Cerium
