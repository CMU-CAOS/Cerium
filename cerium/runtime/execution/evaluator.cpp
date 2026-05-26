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

#include <assert.h>
#include <cstdint>
#include <dlfcn.h>
#include <seal/seal.h>
#include <seal/util/mempool.h>
#include <seal/util/polyarithsmallmod.h>
#include <seal/util/rns.h>
#include <seal/util/uintarithmod.h>
#include <stdio.h>
#include <thread>
#include <tuple>

#include "cerium/runtime/utils/logger.h"
#include "cerium/runtime/utils/string.h"

#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/execution/device_utils.h"
#include "cerium/runtime/execution/evaluator.h"
#include "cerium/runtime/math/util.h"

namespace Cerium {
namespace Runtime {
void Evaluator::init() {

  auto num_rns_bases = context_.num_rns_bases();
  const auto &rns_bases_ = context_.rns_bases();

  // nccl_buffer_ = allocate_device<uint32_t>(context_.n()*NCCL_BUFFER_MAX_LIMBS);

  // stream_ = CUDA::createStream();
  stream_ = CUDA::createStreamNonBlocking();

  if (evaluator_context_->num_devices > 1) {
    NCCL_BUFFER_MAX_LIMBS = 64;
    auto NUM_DEVICES = evaluator_context_->num_devices;
    nccl_buffer___ =
        CUDA::NCCL::createNcclBuffer(context_.n() * NCCL_BUFFER_MAX_LIMBS *
                                     NUM_DEVICES * sizeof(Limb::Element_t));
    nccl_buffer_ = static_cast<LimbT::Element_t *>(nccl_buffer___->get());
  }
}

void EvaluatorContext::init_params() {

  auto num_rns_bases = context_.num_rns_bases();
  barrett_k_host_.resize(context_.num_rns_bases());
  barrett_ratios_host_.resize(context_.num_rns_bases());
  modulii_host_.resize(context_.num_rns_bases());

  const auto &rns_bases_ = context_.rns_bases();

  for (size_t i = 0; i < context_.num_rns_bases(); i++) {
    barrett_k_host_[i] = 58;
    modulii_host_[i] = rns_bases_[i].value();
    barrett_ratios_host_[i] = Math::safe_cast<LimbT::Element_t>(
        (1ULL << barrett_k_host_[i]) / modulii_host_[i]);
  }

  barrett_ratios_ = allocate_uint_device(num_rns_bases);
  barrett_ratios_.copy(barrett_ratios_host_.data(), num_rns_bases);

  modulii_ = allocate_uint_device(num_rns_bases);
  modulii_.copy(modulii_host_.data(), num_rns_bases);

  barrett_k_ = allocate_device<uint32_t>(num_rns_bases);
  barrett_k_.copy(barrett_k_host_.data(), num_rns_bases);

  rotate_permutations_device_.resize(context_.n());

  NCCL_BUFFER_MAX_LIMBS = 64 * 8;
  // nccl_buffer_ = allocate_device<uint32_t>(context_.n()*NCCL_BUFFER_MAX_LIMBS);
  // nccl_buffer___ = CUDA::NCCL::createNcclBuffer(context_.n()*NCCL_BUFFER_MAX_LIMBS*sizeof(Limb::Element_t));
  // CUDA::NCCL::ncclBufferRegister(device_id, nccl_buffer___);

  // nccl_buffer_ = static_cast<LimbT::Element_t *>(nccl_buffer___->get());
}

void EvaluatorContext::permute_root_powers(
    std::vector<LimbT::Element_t> &root_powers_host, const uint32_t m_start,
    const uint32_t m_end, const size_t coeff_count, const size_t per_thread,
    const size_t num_rns_bases) {

  std::vector<LimbT::Element_t> root_powers_host_(root_powers_host);
  for (size_t r = 0; r < num_rns_bases; r++) {
    for (size_t mm = m_start; mm < m_end; mm *= per_thread) {
      // std::cout << "mm: " << mm << " base: " << r << "\n";
      for (size_t tt = 1; tt < per_thread; tt *= 2) {
        for (size_t k = 0; k < tt; k++) {
          for (size_t i = 0; i < mm; i++) {
            root_powers_host_[r * coeff_count + mm * (tt + k) + i] =
                root_powers_host[r * coeff_count + (mm + i) * tt + k];
          }
        }
      }
    }
  }
  root_powers_host = root_powers_host_;
}

void EvaluatorContext::init_ntt_params() {
  auto coeff_count = context_.n();
  auto coeff_count_power = log2(coeff_count);
  auto num_rns_bases = context_.num_rns_bases();

  std::vector<LimbT::Element_t> root_powers_host(coeff_count * num_rns_bases);
  std::vector<LimbT::Element_t> root_powers_scaled_host(coeff_count *
                                                        num_rns_bases);
  std::vector<LimbT::Element_t> inv_root_powers_host(coeff_count *
                                                     num_rns_bases);
  std::vector<LimbT::Element_t> inv_root_powers_div_two_host(coeff_count *
                                                             num_rns_bases);
  std::vector<LimbT::Element_t> inv_root_powers_div_two_scaled_host(
      coeff_count * num_rns_bases);
  std::vector<LimbT::Element_t2> root_powers_host_uint2(coeff_count *
                                                        num_rns_bases);
  std::vector<LimbT::Element_t2> inv_root_powers_div_two_host_uint2(
      coeff_count * num_rns_bases);
  for (size_t i = 0; i < num_rns_bases; i++) {
    const auto &modulus = context_.get_rns_modulus(i);
    auto modulus_word = Math::safe_cast<LimbT::Element_t>(modulus.value());

    // We defer parameter checking to try_minimal_primitive_root(...)
    uint64_t root = 0;
    uint64_t inv_root = 0;
    if (!seal::util::try_minimal_primitive_root(2 * coeff_count, modulus,
                                                root)) {
      throw std::invalid_argument("invalid modulus");
    }
    if (!seal::util::try_invert_uint_mod(root, modulus, inv_root)) {
      throw std::invalid_argument("invalid modulus");
    }

    // Populate tables with powers of root in specific orders.
    uint64_t power = root;
    for (size_t j = 1; j < coeff_count; j++) {
      root_powers_host[coeff_count * i +
                       seal::util::reverse_bits(j, coeff_count_power)] =
          Math::safe_cast<LimbT::Element_t>(power);
      //TODO: Check this for type safety
      auto shoup =
          Math::Shoup(Math::safe_cast<LimbT::Element_t>(power), modulus_word);
      root_powers_scaled_host[coeff_count * i +
                              seal::util::reverse_bits(j, coeff_count_power)] =
          Math::safe_cast<LimbT::Element_t>(shoup);

      power = seal::util::multiply_uint_mod(power, root, modulus);
    }
    root_powers_host[coeff_count * i] = static_cast<LimbT::Element_t>(1);
    root_powers_scaled_host[coeff_count * i] =
        Math::Shoup(Math::safe_cast<LimbT::Element_t>(1), modulus_word);

    uint64_t two_inv = 0;
    if (!seal::util::try_invert_uint_mod(2, modulus, two_inv)) {
      throw std::invalid_argument("invalid modulus");
    }

    power = inv_root;
    for (size_t j = 1; j < coeff_count; j++) {
      inv_root_powers_host[coeff_count * i +
                           seal::util::reverse_bits(j - 1, coeff_count_power) +
                           1] = Math::safe_cast<LimbT::Element_t>(power);

      auto power_div_two =
          seal::util::multiply_uint_mod(power, two_inv, modulus);
      inv_root_powers_div_two_host
          [coeff_count * i + seal::util::reverse_bits(j, coeff_count_power)] =
              Math::safe_cast<LimbT::Element_t>(power_div_two);

      auto shoup = Math::Shoup(Math::safe_cast<LimbT::Element_t>(power_div_two),
                               modulus_word);
      inv_root_powers_div_two_scaled_host
          [coeff_count * i + seal::util::reverse_bits(j, coeff_count_power)] =
              Math::safe_cast<LimbT::Element_t>(shoup);

      power = seal::util::multiply_uint_mod(power, inv_root, modulus);
    }
    inv_root_powers_host[coeff_count * i] = static_cast<LimbT::Element_t>(1);
    inv_root_powers_div_two_host[coeff_count * i] =
        static_cast<LimbT::Element_t>(two_inv);
    inv_root_powers_div_two_scaled_host[coeff_count * i] =
        Math::Shoup(Math::safe_cast<LimbT::Element_t>(two_inv), modulus_word);
  }

  permute_root_powers(root_powers_host, 1, 256, coeff_count, 4, num_rns_bases);
  permute_root_powers(root_powers_scaled_host, 1, 256, coeff_count, 4,
                      num_rns_bases);
  permute_root_powers(root_powers_host, 256, coeff_count, coeff_count, 4,
                      num_rns_bases);
  permute_root_powers(root_powers_scaled_host, 256, coeff_count, coeff_count, 4,
                      num_rns_bases);

  permute_root_powers(inv_root_powers_div_two_host, 128, coeff_count,
                      coeff_count, 8, num_rns_bases);
  permute_root_powers(inv_root_powers_div_two_scaled_host, 128, coeff_count,
                      coeff_count, 8, num_rns_bases);

  for (size_t i = 0; i < num_rns_bases; i++) {
    for (size_t j = 0; j < coeff_count; j++) {
      root_powers_host_uint2[coeff_count * i + j] = {
          root_powers_host[coeff_count * i + j],
          root_powers_scaled_host[coeff_count * i + j]};
      inv_root_powers_div_two_host_uint2[coeff_count * i + j] = {
          inv_root_powers_div_two_host[coeff_count * i + j],
          inv_root_powers_div_two_scaled_host[coeff_count * i + j]};
    }
  }

  root_powers_uint2_ = allocate_device<uint2>(coeff_count * num_rns_bases);
  inv_root_powers_div_two_uint2_ =
      allocate_device<uint2>(coeff_count * num_rns_bases);

  root_powers_uint2_.copy(root_powers_host_uint2.data(),
                          coeff_count * num_rns_bases);
  inv_root_powers_div_two_uint2_.copy(inv_root_powers_div_two_host_uint2.data(),
                                      coeff_count * num_rns_bases);
}

const uint32_t *EvaluatorContext::get_rotation_map_ptr(const int32_t amount) {
  auto galois_tool = context_.galois_tool();
  auto galois_elt = galois_tool->get_elt_from_step(amount);
  auto elt_id = (galois_elt - 1) >> 1;

  // TODO: rotate does memcpy on table every time, can we optimize this?
  if (!rotate_permutations_device_[elt_id]) {

    bool use_rot_write = Cerium::Runtime::Utils::get_uint_env_variable(
        "CERIUM_RUNTIME_USE_ROT_WRITE", 1);

    if (use_rot_write) {
      auto table = galois_tool->get_galois_table_inverse(galois_elt);
      auto table_d = allocate_device<uint32_t>(context_.n());
      CUDA::memcpyHostToDevice(table_d.get(), table,
                               context_.n() * sizeof(uint32_t));
      rotate_permutations_device_[elt_id] = std::move(table_d);
    } else {
      auto table = galois_tool->get_galois_table(galois_elt);
      auto table_d = allocate_device<uint32_t>(context_.n());
      CUDA::memcpyHostToDevice(table_d.get(), table,
                               context_.n() * sizeof(uint32_t));
      rotate_permutations_device_[elt_id] = std::move(table_d);
    }
  }
  const auto &table_d = rotate_permutations_device_[elt_id];
  return table_d.get();
}

EvaluatorContext::EvaluatorContext(const Context &context, uint32_t device_id,
                                   uint32_t num_devices)
    : context_(context), coeff_count_(context.n()),
      coeff_count_power_(log2(coeff_count_)), device_id(device_id),
      num_devices(num_devices) {
  init_params();
  init_ntt_params();
}

Evaluator::Evaluator(const Context &context,
                     std::shared_ptr<EvaluatorContext> evaluator_context)
    : context_(context), coeff_count_(context.n()),
      coeff_count_power_(log2(coeff_count_)),
      evaluator_context_(evaluator_context) {
  init();
}

void Evaluator::fn_ewi(void *function, const CUDA::StreamPtr &stream,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &scalars,
                       const LimbDataType **remapable_base,
                       const std::vector<std::size_t> &remapables,
                       const std::vector<uint32_t> &bases,
                       std::vector<int32_t> rotations) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto scalars_device = allocate_device<LimbT::Element_t>(scalars.size());
  scalars_device.copy(scalars.data(), scalars.size());
  auto remapables_device = allocate_device<size_t>(remapables.size());
  remapables_device.copy(remapables.data(), remapables.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  DevicePointer<const uint32_t *> rotations_map_device;
  if (rotations.size() > 0) {
    std::vector<const uint32_t *> rotations_map_host;
    for (auto &amt : rotations) {
      rotations_map_host.push_back(
          evaluator_context_->get_rotation_map_ptr(amt));
    }
    rotations_map_device =
        allocate_device<const uint32_t *>(rotations_map_host.size());
    rotations_map_device.copy(rotations_map_host.data(),
                              rotations_map_host.size());
  }

  execute_ewi(function, stream, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              scalars_device.get(), remapable_base, remapables_device.get(),
              const_cast<const uint32_t *>(bases_device.get()),
              evaluator_context_->modulii_.get(),
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get(), rotations_map_device.get());

}

void Evaluator::fn_ewi(void *function, CUDA::GraphPtr &graph,
                       CUDA::NodePtr &node0,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &scalars,
                       const LimbDataType **remapable_base,
                       const std::vector<std::size_t> &remapables,
                       const std::vector<uint32_t> &bases,
                       std::vector<int32_t> rotations) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto scalars_device = allocate_device<LimbT::Element_t>(scalars.size());
  scalars_device.copy(scalars.data(), scalars.size());
  auto remapables_device = allocate_device<size_t>(remapables.size());
  remapables_device.copy(remapables.data(), remapables.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  DevicePointer<const uint32_t *> rotations_map_device;
  if (rotations.size() > 0) {
    std::vector<const uint32_t *> rotations_map_host;
    for (auto &amt : rotations) {
      rotations_map_host.push_back(
          evaluator_context_->get_rotation_map_ptr(amt));
    }
    rotations_map_device =
        allocate_device<const uint32_t *>(rotations_map_host.size());
    rotations_map_device.copy(rotations_map_host.data(),
                              rotations_map_host.size());
  }

  execute_ewi(function, graph, node0, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              scalars_device.get(), remapable_base, remapables_device.get(),
              const_cast<const uint32_t *>(bases_device.get()),
              evaluator_context_->modulii_.get(),
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get(), rotations_map_device.get());

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(scalars_device));
  args_sca.push_back(std::move(bases_device));
  args_rot.push_back(std::move(rotations_map_device));
  args_size.push_back(std::move(remapables_device));

}
void Evaluator::print_memory_location(const uint32_t **ptr, size_t num_inputs) {
  auto host_ptr = allocate<const uint32_t *>(num_inputs);
  CUDA::memcpyDeviceToHost(host_ptr.get(), ptr,
                           num_inputs * sizeof(uint32_t *));
  for (int i = 0; i < num_inputs; i++) {
    std::cout << "i: [" << i << "] ptr: [" << host_ptr[i] << "]: ";
    if (host_ptr[i] == nullptr) {
      std::cout << "\n";
      continue;
    }
    auto host_ptr_inner = allocate_limb_elts(context_.n());
    CUDA::memcpyDeviceToHost(host_ptr_inner.get(), host_ptr[i], context_.n());
    for (size_t j = 0; j < 10; j++) {
      std::cout << host_ptr_inner[j] << ",";
    }
    std::cout << "\n";
  }
}

std::vector<void *>
Evaluator::fn_ewi_args(void *function, std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &scalars,
                       const LimbDataType **remapable_base,
                       const std::vector<std::size_t> &remapables,
                       const std::vector<uint32_t> &bases,
                       std::vector<int32_t> rotations) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto scalars_device = allocate_device<LimbT::Element_t>(scalars.size());
  scalars_device.copy(scalars.data(), scalars.size());
  auto remapables_device = allocate_device<size_t>(remapables.size());
  remapables_device.copy(remapables.data(), remapables.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  DevicePointer<const uint32_t *> rotations_map_device;
  if (rotations.size() > 0) {
    std::vector<const uint32_t *> rotations_map_host;
    for (auto &amt : rotations) {
      rotations_map_host.push_back(
          evaluator_context_->get_rotation_map_ptr(amt));
    }
    rotations_map_device =
        allocate_device<const uint32_t *>(rotations_map_host.size());
    rotations_map_device.copy(rotations_map_host.data(),
                              rotations_map_host.size());
  }

  std::vector<void *> function_args(10);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)inputs_device.get();
  function_args[3] = (void *)scalars_device.get();
  function_args[4] = (void *)remapable_base;
  function_args[5] = (void *)remapables_device.get();
  function_args[6] = (void *)bases_device.get();
  function_args[7] = (void *)rotations_map_device.get();
  function_args[8] = (void *)inputs.size();
  function_args[9] = (void *)outputs.size();

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(scalars_device));
  args_sca.push_back(std::move(bases_device));
  args_rot.push_back(std::move(rotations_map_device));
  args_size.push_back(std::move(remapables_device));

  return function_args;
}

void Evaluator::fn_ewi(const CUDA::StreamPtr &stream,
                       std::vector<void *> &args) {
  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto inputs_device = (const LimbDataType **)args[2];
  auto scalars_device = (const LimbDataType *)args[3];
  auto remapable_base = (const LimbDataType **)args[4];
  auto remapables_device = (const size_t *)args[5];
  auto bases_device = (const uint32_t *)args[6];
  auto rotations_map_device = (const uint32_t **)args[7];
  auto num_inputs = (size_t)args[8];
  auto num_outputs = (size_t)args[9];

  execute_ewi(function, stream, outputs_device, inputs_device, scalars_device,
              remapable_base, remapables_device, bases_device,
              evaluator_context_->modulii_.get(),
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get(), rotations_map_device);

}

void Evaluator::fn_int(void *function, const CUDA::StreamPtr &stream,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  execute_int(function, stream, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->inv_root_powers_div_two_uint2_.get());
  CUDA::deviceSynchronize();
  CHECK_CUDA_ERROR();

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
}

void Evaluator::fn_int(void *function, CUDA::GraphPtr &graph,
                       CUDA::NodePtr &node0, CUDA::NodePtr &node1,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  execute_int(function, graph, node0, node1, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->inv_root_powers_div_two_uint2_.get());

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));
}

std::vector<void *>
Evaluator::fn_int_args(void *function, std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  std::vector<void *> function_args(5);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)inputs_device.get();
  function_args[3] = (void *)bases_device.get();
  function_args[4] = (void *)bases.size();

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));

  return function_args;
}

void Evaluator::fn_int(const CUDA::StreamPtr &stream,
                       std::vector<void *> &args) {
  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto inputs_device = (const LimbDataType **)args[2];
  auto bases_device = (const uint32_t *)args[3];
  auto num_bases = (size_t)args[4];

  execute_int(function, stream, outputs_device, inputs_device, bases_device,
              num_bases, evaluator_context_->modulii_.get(),
              evaluator_context_->inv_root_powers_div_two_uint2_.get());
}

void Evaluator::fn_ntt(void *function, const CUDA::StreamPtr &stream,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  execute_ntt(function, stream, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get());

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
}

void Evaluator::fn_ntt(void *function, CUDA::GraphPtr &graph,
                       CUDA::NodePtr &node0, CUDA::NodePtr &node1,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {
  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  execute_ntt(function, graph, node0, node1, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get());

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));
}

std::vector<void *>
Evaluator::fn_ntt_args(void *function, std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases) {
  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  std::vector<void *> function_args(5);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)inputs_device.get();
  function_args[3] = (void *)bases_device.get();
  function_args[4] = (void *)bases.size();

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));

  return function_args;
}

void Evaluator::fn_ntt(const CUDA::StreamPtr &stream,
                       std::vector<void *> &args) {
  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto inputs_device = (const LimbDataType **)args[2];
  auto bases_device = (const uint32_t *)args[3];
  auto num_bases = (size_t)args[4];

  execute_ntt(function, stream, outputs_device, inputs_device, bases_device,
              num_bases, evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get());
}

void Evaluator::fn_sud(void *function, const CUDA::StreamPtr &stream,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases,
                       const std::vector<LimbDataType> &sud_factors) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  auto sud_factors_device =
      allocate_device<LimbT::Element_t>(sud_factors.size());
  sud_factors_device.copy(sud_factors.data(), sud_factors.size());

  execute_sud(function, stream, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get(),
              sud_factors_device.get(),
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get());
}

void Evaluator::fn_sud(void *function, CUDA::GraphPtr &graph,
                       CUDA::NodePtr &node0, CUDA::NodePtr &node1,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases,
                       const std::vector<LimbDataType> &sud_factors) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  auto sud_factors_device =
      allocate_device<LimbT::Element_t>(sud_factors.size());
  sud_factors_device.copy(sud_factors.data(), sud_factors.size());

  execute_sud(function, graph, node0, node1, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              const_cast<const uint32_t *>(bases_device.get()), bases.size(),
              evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get(),
              sud_factors_device.get(),
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get());

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));
  args_sca.push_back(std::move(sud_factors_device));
}

std::vector<void *>
Evaluator::fn_sud_args(void *function, std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<uint32_t> &bases,
                       const std::vector<LimbDataType> &sud_factors) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());

  auto sud_factors_device =
      allocate_device<LimbT::Element_t>(sud_factors.size());
  sud_factors_device.copy(sud_factors.data(), sud_factors.size());

  std::vector<void *> function_args(6);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)inputs_device.get();
  function_args[3] = (void *)bases_device.get();
  function_args[4] = (void *)bases.size();
  function_args[5] = (void *)sud_factors_device.get();

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(bases_device));
  args_sca.push_back(std::move(sud_factors_device));

  return function_args;
}

void Evaluator::fn_sud(const CUDA::StreamPtr &stream,
                       std::vector<void *> &args) {
  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto inputs_device = (const LimbDataType **)args[2];
  auto bases_device = (const uint32_t *)args[3];
  auto num_bases = (size_t)args[4];
  auto sud_factors_device = (const LimbDataType *)args[5];

  execute_sud(function, stream, outputs_device, inputs_device, bases_device,
              num_bases, evaluator_context_->modulii_.get(),
              evaluator_context_->root_powers_uint2_.get(), sud_factors_device,
              evaluator_context_->barrett_ratios_.get(),
              evaluator_context_->barrett_k_.get());
}

void Evaluator::fn_rsv(void *function, const CUDA::StreamPtr &stream,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &factors,
                       const std::vector<LimbDataType> &base,
                       const uint64_t barrett_ratio, const uint32_t barrett_k) {

  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());

  auto rsv_factors_device = allocate_device<LimbT::Element_t>(factors.size());
  rsv_factors_device.copy(factors.data(), factors.size());

  auto rsv_base_device = allocate_device<LimbT::Element_t>(base.size());
  rsv_base_device.copy(base.data(), base.size());

  auto rsv_barrett_ratio_device = allocate_uint_device(base.size());
  rsv_barrett_ratio_device.copy((LimbDataType *)&barrett_ratio, base.size());

  execute_rsv(function, stream, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              rsv_factors_device.get(), rsv_base_device.get(),
              rsv_barrett_ratio_device.get(), barrett_k);
}

void Evaluator::fn_rsv(void *function, CUDA::GraphPtr &graph,
                       CUDA::NodePtr &node0,
                       std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &factors,
                       const std::vector<LimbDataType> &base,
                       const uint64_t barrett_ratio, const uint32_t barrett_k) {
  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());

  auto rsv_factors_device = allocate_device<LimbT::Element_t>(factors.size());
  rsv_factors_device.copy(factors.data(), factors.size());

  auto rsv_base_device = allocate_device<LimbT::Element_t>(base.size());
  rsv_base_device.copy(base.data(), base.size());

  auto rsv_barrett_ratio_device = allocate_uint_device(base.size());
  rsv_barrett_ratio_device.copy((LimbDataType *)&barrett_ratio, base.size());

  execute_rsv(function, graph, node0, outputs_device.get(),
              const_cast<const LimbDataType **>(inputs_device.get()),
              rsv_factors_device.get(), rsv_base_device.get(),
              rsv_barrett_ratio_device.get(), barrett_k);

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(rsv_factors_device));
  args_sca.push_back(std::move(rsv_base_device));
  args_sca.push_back(std::move(rsv_barrett_ratio_device));
}

std::vector<void *>
Evaluator::fn_rsv_args(void *function, std::vector<LimbDataType *> &outputs,
                       const std::vector<LimbDataType *> &inputs,
                       const std::vector<LimbDataType> &factors,
                       const std::vector<LimbDataType> &base,
                       const uint64_t barrett_ratio, const uint32_t barrett_k) {
  auto outputs_device = allocate_device<LimbT::Element_t *>(outputs.size());
  outputs_device.copy(outputs.data(), outputs.size());
  auto inputs_device = allocate_device<LimbT::Element_t *>(inputs.size());
  inputs_device.copy(inputs.data(), inputs.size());

  auto rsv_factors_device = allocate_device<LimbT::Element_t>(factors.size());
  rsv_factors_device.copy(factors.data(), factors.size());

  auto rsv_base_device = allocate_device<LimbT::Element_t>(base.size());
  rsv_base_device.copy(base.data(), base.size());

  auto rsv_barrett_ratio_device = allocate_uint_device(base.size());
  rsv_barrett_ratio_device.copy((LimbDataType *)&barrett_ratio, base.size());

  std::vector<void *> function_args(7);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)inputs_device.get();
  function_args[3] = (void *)rsv_factors_device.get();
  function_args[4] = (void *)rsv_base_device.get();
  function_args[5] = (void *)rsv_barrett_ratio_device.get();
  function_args[6] = (void *)barrett_k;

  args_vec.push_back(std::move(outputs_device));
  args_vec.push_back(std::move(inputs_device));
  args_sca.push_back(std::move(rsv_factors_device));
  args_sca.push_back(std::move(rsv_base_device));
  args_sca.push_back(std::move(rsv_barrett_ratio_device));

  return function_args;
}

void Evaluator::fn_rsv(const CUDA::StreamPtr &stream,
                       std::vector<void *> &args) {
  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto inputs_device = (const LimbDataType **)args[2];
  auto rsv_factors_device = (const LimbDataType *)args[3];
  auto rsv_base_device = (const uint32_t *)args[4];
  auto rsv_barrett_ratio_device = (const LimbDataType *)args[5];
  auto barrett_k = static_cast<uint32_t>((uint64_t)(args[6]));

  execute_rsv(function, stream, outputs_device, inputs_device,
              rsv_factors_device, rsv_base_device, rsv_barrett_ratio_device,
              barrett_k);
}

void Evaluator::fn_bco(
    void *function, const CUDA::StreamPtr &stream,
    std::vector<DeviceBaseConverter *> &
        base_converters) { 

  auto count = base_converters.size();

  auto outputs = std::vector<LimbT::Element_t *>(count);
  auto num_output_bases = std::vector<size_t>(count);
  auto inputs = std::vector<LimbT::Element_t *>(count);
  auto num_input_bases = std::vector<size_t>(count);
  auto base_conversion_factors = std::vector<LimbT::Element_t *>(count);
  auto output_modulii = std::vector<LimbT::Element_t *>(count);
  auto barrett_ratios = std::vector<LimbT::Element_t *>(count);
  auto barrett_k = std::vector<uint32_t *>(count);

  size_t shared_mem_size = 0;

  for (size_t i = 0; i < count; i++) {
    outputs[i] = base_converters[i]->output_limbs_ptr_.get();
    num_output_bases[i] = base_converters[i]->output_rns_base_ids_.size();
    inputs[i] = base_converters[i]->input_limbs_ptr_.get();
    num_input_bases[i] = base_converters[i]->input_rns_base_ids_.size();
    base_conversion_factors[i] =
        base_converters[i]->base_change_factors_vec_device_.back().get();
    output_modulii[i] =
        base_converters[i]->output_modulus_vec_device_.back().get();
    barrett_ratios[i] =
        base_converters[i]->barrett_ratios_vec_device_.back().get();
    barrett_k[i] = base_converters[i]->barrett_k_vec_device_.back().get();
    auto mem_size = num_output_bases[i] * num_input_bases[i];
    mem_size = (mem_size / 32 + 1) * 32;
    mem_size += num_input_bases[i] * 256;
    if (shared_mem_size <= mem_size) {
      shared_mem_size = mem_size;
    }
  }
  shared_mem_size *= sizeof(LimbT::Element_t);

  auto outputs_device = allocate_device<LimbT::Element_t *>(count);
  outputs_device.copy(outputs.data(), outputs.size());

  auto num_output_bases_device = allocate_device<size_t>(count);
  num_output_bases_device.copy(num_output_bases.data(),
                               num_output_bases.size());

  auto inputs_device = allocate_device<LimbT::Element_t *>(count);
  inputs_device.copy(inputs.data(), inputs.size());

  auto num_input_bases_device = allocate_device<size_t>(count);
  num_input_bases_device.copy(num_input_bases.data(), num_input_bases.size());

  auto base_conversion_factors_device =
      allocate_device<LimbT::Element_t *>(count);
  base_conversion_factors_device.copy(base_conversion_factors.data(),
                                      base_conversion_factors.size());

  auto output_modulii_device = allocate_device<LimbT::Element_t *>(count);
  output_modulii_device.copy(output_modulii.data(), output_modulii.size());

  auto barrett_ratios_device = allocate_device<LimbT::Element_t *>(count);
  barrett_ratios_device.copy(barrett_ratios.data(), barrett_ratios.size());

  auto barrett_k_device = allocate_device<uint32_t *>(count);
  barrett_k_device.copy(barrett_k.data(), barrett_k.size());

  execute_bco(
      function, stream, outputs_device.get(),
      const_cast<const size_t *>(num_output_bases_device.get()),
      const_cast<const LimbDataType **>(inputs_device.get()),
      const_cast<const size_t *>(num_input_bases_device.get()),
      const_cast<const LimbDataType **>(base_conversion_factors_device.get()),
      const_cast<const LimbDataType **>(output_modulii_device.get()),
      const_cast<const LimbDataType **>(barrett_ratios_device.get()),
      const_cast<const LimbDataType **>(barrett_k_device.get()),
      shared_mem_size, count);

  for (size_t i = 0; i < count; i++) {
    base_converters[i]->set_input_bases({});
  }
}

void Evaluator::fn_bco(
    void *function, CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
    std::vector<DeviceBaseConverter *> &
        base_converters) { 

  auto count = base_converters.size();

  auto outputs = std::vector<LimbT::Element_t *>(count);
  auto num_output_bases = std::vector<size_t>(count);
  auto inputs = std::vector<LimbT::Element_t *>(count);
  auto num_input_bases = std::vector<size_t>(count);
  auto base_conversion_factors = std::vector<LimbT::Element_t *>(count);
  auto output_modulii = std::vector<LimbT::Element_t *>(count);
  auto barrett_ratios = std::vector<LimbT::Element_t *>(count);
  auto barrett_k = std::vector<uint32_t *>(count);

  size_t shared_mem_size = 0;

  for (size_t i = 0; i < count; i++) {
    outputs[i] = base_converters[i]->output_limbs_ptr_.get();
    num_output_bases[i] = base_converters[i]->output_rns_base_ids_.size();
    inputs[i] = base_converters[i]->input_limbs_ptr_.get();
    num_input_bases[i] = base_converters[i]->input_rns_base_ids_.size();
    base_conversion_factors[i] =
        base_converters[i]->base_change_factors_vec_device_.back().get();
    output_modulii[i] =
        base_converters[i]->output_modulus_vec_device_.back().get();
    barrett_ratios[i] =
        base_converters[i]->barrett_ratios_vec_device_.back().get();
    barrett_k[i] = base_converters[i]->barrett_k_vec_device_.back().get();
    auto mem_size = num_output_bases[i] * num_input_bases[i];
    mem_size = (mem_size / 32 + 1) * 32;
    mem_size += num_input_bases[i] * 256;
    if (shared_mem_size <= mem_size) {
      shared_mem_size = mem_size;
    }
  }
  shared_mem_size *= sizeof(LimbT::Element_t);

  auto outputs_device = allocate_device<LimbT::Element_t *>(count);
  outputs_device.copy(outputs.data(), outputs.size());

  auto num_output_bases_device = allocate_device<size_t>(count);
  num_output_bases_device.copy(num_output_bases.data(),
                               num_output_bases.size());

  auto inputs_device = allocate_device<LimbT::Element_t *>(count);
  inputs_device.copy(inputs.data(), inputs.size());

  auto num_input_bases_device = allocate_device<size_t>(count);
  num_input_bases_device.copy(num_input_bases.data(), num_input_bases.size());

  auto base_conversion_factors_device =
      allocate_device<LimbT::Element_t *>(count);
  base_conversion_factors_device.copy(base_conversion_factors.data(),
                                      base_conversion_factors.size());

  auto output_modulii_device = allocate_device<LimbT::Element_t *>(count);
  output_modulii_device.copy(output_modulii.data(), output_modulii.size());

  auto barrett_ratios_device = allocate_device<LimbT::Element_t *>(count);
  barrett_ratios_device.copy(barrett_ratios.data(), barrett_ratios.size());

  auto barrett_k_device = allocate_device<uint32_t *>(count);
  barrett_k_device.copy(barrett_k.data(), barrett_k.size());

  execute_bco(
      function, graph, node0, outputs_device.get(),
      const_cast<const size_t *>(num_output_bases_device.get()),
      const_cast<const LimbDataType **>(inputs_device.get()),
      const_cast<const size_t *>(num_input_bases_device.get()),
      const_cast<const LimbDataType **>(base_conversion_factors_device.get()),
      const_cast<const LimbDataType **>(output_modulii_device.get()),
      const_cast<const LimbDataType **>(barrett_ratios_device.get()),
      const_cast<const LimbDataType **>(barrett_k_device.get()),
      shared_mem_size, count);

  args_vec.push_back(std::move(outputs_device));
  args_size.push_back(std::move(num_output_bases_device));
  args_vec.push_back(std::move(inputs_device));
  args_size.push_back(std::move(num_input_bases_device));
  args_vec.push_back(std::move(base_conversion_factors_device));
  args_vec.push_back(std::move(output_modulii_device));
  args_vec.push_back(std::move(barrett_ratios_device));
  args_vec.push_back(std::move(barrett_k_device));

  for (size_t i = 0; i < count; i++) {
    base_converters[i]->set_input_bases({});
  }
}

std::vector<void *> Evaluator::fn_bco_args(
    void *function,
    std::vector<DeviceBaseConverter *> &
        base_converters) { 

  auto count = base_converters.size();

  auto outputs = std::vector<LimbT::Element_t *>(count);
  auto num_output_bases = std::vector<size_t>(count);
  auto inputs = std::vector<LimbT::Element_t *>(count);
  auto num_input_bases = std::vector<size_t>(count);
  auto base_conversion_factors = std::vector<LimbT::Element_t *>(count);
  auto output_modulii = std::vector<LimbT::Element_t *>(count);
  auto barrett_ratios = std::vector<LimbT::Element_t *>(count);
  auto barrett_k = std::vector<uint32_t *>(count);

  size_t shared_mem_size = 0;

  for (size_t i = 0; i < count; i++) {
    outputs[i] = base_converters[i]->output_limbs_ptr_.get();
    num_output_bases[i] = base_converters[i]->output_rns_base_ids_.size();
    inputs[i] = base_converters[i]->input_limbs_ptr_.get();
    num_input_bases[i] = base_converters[i]->input_rns_base_ids_.size();
    base_conversion_factors[i] =
        base_converters[i]->base_change_factors_vec_device_.back().get();
    output_modulii[i] =
        base_converters[i]->output_modulus_vec_device_.back().get();
    barrett_ratios[i] =
        base_converters[i]->barrett_ratios_vec_device_.back().get();
    barrett_k[i] = base_converters[i]->barrett_k_vec_device_.back().get();
    auto mem_size = num_output_bases[i] * num_input_bases[i];
    mem_size = (mem_size / 32 + 1) * 32;
    mem_size += num_input_bases[i] * 256;
    if (shared_mem_size <= mem_size) {
      shared_mem_size = mem_size;
    }
  }
  shared_mem_size *= sizeof(LimbT::Element_t);

  auto outputs_device = allocate_device<LimbT::Element_t *>(count);
  outputs_device.copy(outputs.data(), outputs.size());

  auto num_output_bases_device = allocate_device<size_t>(count);
  num_output_bases_device.copy(num_output_bases.data(),
                               num_output_bases.size());

  auto inputs_device = allocate_device<LimbT::Element_t *>(count);
  inputs_device.copy(inputs.data(), inputs.size());

  auto num_input_bases_device = allocate_device<size_t>(count);
  num_input_bases_device.copy(num_input_bases.data(), num_input_bases.size());

  auto base_conversion_factors_device =
      allocate_device<LimbT::Element_t *>(count);
  base_conversion_factors_device.copy(base_conversion_factors.data(),
                                      base_conversion_factors.size());

  auto output_modulii_device = allocate_device<LimbT::Element_t *>(count);
  output_modulii_device.copy(output_modulii.data(), output_modulii.size());

  auto barrett_ratios_device = allocate_device<LimbT::Element_t *>(count);
  barrett_ratios_device.copy(barrett_ratios.data(), barrett_ratios.size());

  auto barrett_k_device = allocate_device<uint32_t *>(count);
  barrett_k_device.copy(barrett_k.data(), barrett_k.size());

  std::vector<void *> function_args(11);

  function_args[0] = function;
  function_args[1] = (void *)outputs_device.get();
  function_args[2] = (void *)num_output_bases_device.get();
  function_args[3] = (void *)inputs_device.get();
  function_args[4] = (void *)num_input_bases_device.get();
  function_args[5] = (void *)base_conversion_factors_device.get();
  function_args[6] = (void *)output_modulii_device.get();
  function_args[7] = (void *)barrett_ratios_device.get();
  function_args[8] = (void *)barrett_k_device.get();
  function_args[9] = (void *)shared_mem_size;
  function_args[10] = (void *)count;

  args_vec.push_back(std::move(outputs_device));
  args_size.push_back(std::move(num_output_bases_device));
  args_vec.push_back(std::move(inputs_device));
  args_size.push_back(std::move(num_input_bases_device));
  args_vec.push_back(std::move(base_conversion_factors_device));
  args_vec.push_back(std::move(output_modulii_device));
  args_vec.push_back(std::move(barrett_ratios_device));
  args_vec.push_back(std::move(barrett_k_device));

  for (size_t i = 0; i < count; i++) {
    base_converters[i]->set_input_bases({});
  }

  return function_args;
}

void Evaluator::fn_bco(
    const CUDA::StreamPtr &stream,
    std::vector<void *> &
        args) { 

  auto function = (void *)args[0];
  auto outputs_device = (LimbDataType **)args[1];
  auto num_output_bases_device = (const size_t *)args[2];
  auto inputs_device = (const LimbDataType **)args[3];
  auto num_input_bases_device = (const size_t *)args[4];
  auto base_conversion_factors_device = (const LimbDataType **)args[5];
  auto output_modulii_device = (const LimbDataType **)args[6];
  auto barrett_ratios_device = (const LimbDataType **)args[7];
  auto barrett_k_device = (const uint32_t **)args[8];
  auto shared_mem_size = (const size_t)args[9];
  auto num_conversions = (const size_t)args[10];

  execute_bco(function, stream, outputs_device, num_output_bases_device,
              inputs_device, num_input_bases_device,
              base_conversion_factors_device, output_modulii_device,
              barrett_ratios_device, barrett_k_device, shared_mem_size,
              num_conversions);
}

void Evaluator::drm(const CUDA::StreamPtr &stream,
                    std::vector<LimbDataType *> &gather,
                    const std::vector<LimbDataType *> scatter,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto n = context_.n();
  auto nccl_buffer_ptr_ = nccl_buffer_ + n * tid * num_comm_limbs;
  copy_to_nccl_buffer(stream, nccl_buffer_ptr_,
                      const_cast<const LimbDataType **>(scatter_device.get()),
                      scatter.size(), n);
  CUDA::NCCL::ncclAllGather_uint32(nccl_buffer_ptr_, nccl_buffer_,
                                   n * num_comm_limbs, comm_size, tid, stream);
  copy_from_nccl_buffer(stream, gather_device.get(), nccl_buffer_,
                        gather.size(), n);
}

void Evaluator::drm(CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                    CUDA::NodePtr &node1, std::vector<LimbDataType *> &gather,
                    const std::vector<LimbDataType *> scatter,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid) {
  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto n = context_.n();
  auto nccl_buffer_ptr_ = nccl_buffer_ + n * tid * num_comm_limbs;
  assert(gather.size() < NCCL_BUFFER_MAX_LIMBS);
  copy_to_nccl_buffer(graph, node0, nccl_buffer_ptr_,
                      const_cast<const LimbDataType **>(scatter_device.get()),
                      scatter.size(), n);
  auto node_gather = CUDA::createNode();
  CUDA::NCCL::ncclAllGather_uint32_graph(graph, node_gather, nccl_buffer_ptr_,
                                         nccl_buffer_, n * num_comm_limbs,
                                         comm_size, tid, stream_);
  copy_from_nccl_buffer(graph, node1, gather_device.get(), nccl_buffer_,
                        gather.size(), n);
  CUDA::addSingleEdge(graph, node0, node_gather);
  CUDA::addSingleEdge(graph, node_gather, node1);
  args_vec.push_back(std::move(gather_device));
  args_vec.push_back(std::move(scatter_device));
}

std::vector<void *>
Evaluator::drm_args(std::vector<LimbDataType *> &gather,
                    const std::vector<LimbDataType *> scatter,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid) {
  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto n = context_.n();
  auto nccl_buffer_ptr_ = nccl_buffer_ + n * tid * num_comm_limbs;
  assert(gather.size() < NCCL_BUFFER_MAX_LIMBS);
  std::vector<void *> function_args(10);

  function_args[0] = nullptr;
  function_args[1] = (void *)gather_device.get();
  function_args[2] = (void *)gather.size();
  function_args[3] = (void *)scatter_device.get();
  function_args[4] = (void *)scatter.size();
  function_args[5] = (void *)nullptr /*Reserved*/;
  function_args[6] = (void *)nccl_buffer_ptr_;
  function_args[7] = (void *)num_comm_limbs;
  function_args[8] = (void *)comm_size;
  function_args[9] = (void *)tid;

  args_vec.push_back(std::move(gather_device));
  args_vec.push_back(std::move(scatter_device));
  return function_args;
}

void Evaluator::drm(const CUDA::StreamPtr &stream, std::vector<void *> &args) {

  auto gather_device = (LimbDataType **)args[1];
  auto gather_size = (size_t)args[2];
  auto scatter_device = (const LimbDataType **)args[3];
  auto scatter_size = (size_t)args[4];
  auto bases_device = (uint32_t *)args[5];
  auto nccl_buffer_ptr_ = (LimbDataType *)args[6];
  auto num_comm_limbs = (size_t)args[7];
  auto comm_size = (size_t)args[8];
  auto tid = (size_t)args[9];

  auto n = context_.n();

  copy_to_nccl_buffer(stream, nccl_buffer_ptr_, scatter_device, scatter_size,
                      n);
  CUDA::NCCL::ncclAllGather_uint32(nccl_buffer_ptr_, nccl_buffer_,
                                   n * num_comm_limbs, comm_size, tid, stream);
  copy_from_nccl_buffer(stream, gather_device, nccl_buffer_, gather_size, n);
}

void Evaluator::ags(const CUDA::StreamPtr &stream,
                    std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();
  auto nccl_buffer_ptr_ = nccl_buffer_ + tid * n * num_comm_limbs;
  copy_to_nccl_buffer(stream, nccl_buffer_,
                      const_cast<const LimbDataType **>(gather_device.get()),
                      gather.size(), n);
  CUDA::NCCL::ncclReduceScatter_uint32(nccl_buffer_, nccl_buffer_ptr_,
                                       n * num_comm_limbs, comm_size, tid,
                                       stream);
  copy_from_nccl_buffer_with_mod(stream, scatter_device.get(), nccl_buffer_ptr_,
                                 scatter.size(), n, bases_device.get(),
                                 evaluator_context_->modulii_.get());
}

void Evaluator::ags(CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                    CUDA::NodePtr &node1, std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();

  auto nccl_buffer_ptr_ = nccl_buffer_ + tid * n * num_comm_limbs;
  assert(scatter.size() < NCCL_BUFFER_MAX_LIMBS);
  copy_to_nccl_buffer(graph, node0, nccl_buffer_,
                      const_cast<const LimbDataType **>(gather_device.get()),
                      gather.size(), n);
  auto node_gather = CUDA::createNode();
  CUDA::NCCL::ncclReduceScatter_uint32_graph(
      graph, node_gather, nccl_buffer_, nccl_buffer_ptr_, n * num_comm_limbs,
      comm_size, tid, stream_);
  copy_from_nccl_buffer_with_mod(
      graph, node1, scatter_device.get(), nccl_buffer_ptr_, scatter.size(), n,
      bases_device.get(), evaluator_context_->modulii_.get());
  CUDA::addSingleEdge(graph, node0, node_gather);
  CUDA::addSingleEdge(graph, node_gather, node1);
  args_vec.emplace_back(std::move(gather_device));
  args_vec.emplace_back(std::move(scatter_device));
  args_sca.emplace_back(std::move(bases_device));
}

std::vector<void *>
Evaluator::ags_args(std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();

  auto nccl_buffer_ptr_ = nccl_buffer_ + tid * n * num_comm_limbs;

  std::vector<void *> function_args(10);

  function_args[0] = nullptr;
  function_args[1] = (void *)gather_device.get();
  function_args[2] = (void *)gather.size();
  function_args[3] = (void *)scatter_device.get();
  function_args[4] = (void *)scatter.size();
  function_args[5] = (void *)bases_device.get();
  function_args[6] = (void *)nccl_buffer_ptr_;
  function_args[7] = (void *)num_comm_limbs;
  function_args[8] = (void *)comm_size;
  function_args[9] = (void *)tid;

  args_vec.emplace_back(std::move(gather_device));
  args_vec.emplace_back(std::move(scatter_device));
  args_sca.emplace_back(std::move(bases_device));
  return function_args;
}

void Evaluator::ags(const CUDA::StreamPtr &stream, std::vector<void *> &args) {

  auto gather_device = (const LimbDataType **)args[1];
  auto gather_size = (size_t)args[2];
  auto scatter_device = (LimbDataType **)args[3];
  auto scatter_size = (size_t)args[4];
  auto bases_device = (uint32_t *)args[5];
  auto nccl_buffer_ptr_ = (LimbDataType *)args[6];
  auto num_comm_limbs = (size_t)args[7];
  auto comm_size = (size_t)args[8];
  auto tid = (size_t)args[9];

  auto n = context_.n();

  copy_to_nccl_buffer(stream, nccl_buffer_, gather_device, gather_size, n);
  CUDA::NCCL::ncclReduceScatter_uint32(nccl_buffer_, nccl_buffer_ptr_,
                                       n * num_comm_limbs, comm_size, tid,
                                       stream);
  copy_from_nccl_buffer_with_mod(stream, scatter_device, nccl_buffer_ptr_,
                                 scatter_size, n, bases_device,
                                 evaluator_context_->modulii_.get());
}

void Evaluator::ard(const CUDA::StreamPtr &stream,
                    std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();
  auto nccl_buffer_ptr_ = nccl_buffer_;
  copy_to_nccl_buffer(stream, nccl_buffer_,
                      const_cast<const LimbDataType **>(gather_device.get()),
                      gather.size(), n);
  CUDA::NCCL::ncclAllReduce_uint32(nccl_buffer_, nccl_buffer_ptr_,
                                   n * num_comm_limbs, comm_size, tid, stream);
  copy_from_nccl_buffer_with_mod(stream, scatter_device.get(), nccl_buffer_ptr_,
                                 scatter.size(), n, bases_device.get(),
                                 evaluator_context_->modulii_.get());
}

void Evaluator::ard(CUDA::GraphPtr &graph, CUDA::NodePtr &node0,
                    CUDA::NodePtr &node1, std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();

  auto nccl_buffer_ptr_ = nccl_buffer_;
  assert(scatter.size() < NCCL_BUFFER_MAX_LIMBS);
  copy_to_nccl_buffer(graph, node0, nccl_buffer_,
                      const_cast<const LimbDataType **>(gather_device.get()),
                      gather.size(), n);
  auto node_gather = CUDA::createNode();
  CUDA::NCCL::ncclAllReduce_uint32_graph(graph, node_gather, nccl_buffer_,
                                         nccl_buffer_ptr_, n * num_comm_limbs,
                                         comm_size, tid, stream_);
  copy_from_nccl_buffer_with_mod(
      graph, node1, scatter_device.get(), nccl_buffer_ptr_, scatter.size(), n,
      bases_device.get(), evaluator_context_->modulii_.get());
  CUDA::addSingleEdge(graph, node0, node_gather);
  CUDA::addSingleEdge(graph, node_gather, node1);
  args_vec.emplace_back(std::move(gather_device));
  args_vec.emplace_back(std::move(scatter_device));
  args_sca.emplace_back(std::move(bases_device));
}

std::vector<void *>
Evaluator::ard_args(std::vector<LimbDataType *> &scatter,
                    const std::vector<LimbDataType *> gather,
                    const size_t comm_size, const size_t num_comm_limbs,
                    const size_t tid, const std::vector<uint32_t> &bases) {

  auto gather_device = allocate_device<LimbT::Element_t *>(gather.size());
  gather_device.copy(gather.data(), gather.size());
  auto scatter_device = allocate_device<LimbT::Element_t *>(scatter.size());
  scatter_device.copy(scatter.data(), scatter.size());
  auto bases_device = allocate_device<uint32_t>(bases.size());
  bases_device.copy(bases.data(), bases.size());
  auto n = context_.n();

  auto nccl_buffer_ptr_ = nccl_buffer_;

  std::vector<void *> function_args(10);

  function_args[0] = nullptr;
  function_args[1] = (void *)gather_device.get();
  function_args[2] = (void *)gather.size();
  function_args[3] = (void *)scatter_device.get();
  function_args[4] = (void *)scatter.size();
  function_args[5] = (void *)bases_device.get();
  function_args[6] = (void *)nccl_buffer_ptr_;
  function_args[7] = (void *)num_comm_limbs;
  function_args[8] = (void *)comm_size;
  function_args[9] = (void *)tid;

  args_vec.emplace_back(std::move(gather_device));
  args_vec.emplace_back(std::move(scatter_device));
  args_sca.emplace_back(std::move(bases_device));
  return function_args;
}

void Evaluator::ard(const CUDA::StreamPtr &stream, std::vector<void *> &args) {

  auto gather_device = (const LimbDataType **)args[1];
  auto gather_size = (size_t)args[2];
  auto scatter_device = (LimbDataType **)args[3];
  auto scatter_size = (size_t)args[4];
  auto bases_device = (uint32_t *)args[5];
  auto nccl_buffer_ptr_ = (LimbDataType *)args[6];
  auto num_comm_limbs = (size_t)args[7];
  auto comm_size = (size_t)args[8];
  auto tid = (size_t)args[9];

  auto n = context_.n();

  copy_to_nccl_buffer(stream, nccl_buffer_, gather_device, gather_size, n);
  CUDA::NCCL::ncclAllReduce_uint32(nccl_buffer_, nccl_buffer_ptr_,
                                   n * num_comm_limbs, comm_size, tid, stream);
  copy_from_nccl_buffer_with_mod(stream, scatter_device, nccl_buffer_ptr_,
                                 scatter_size, n, bases_device,
                                 evaluator_context_->modulii_.get());
}

// DeviceBaseConverter // -----------------------------------------------------------------

DeviceBaseConverter::LimbT::Element_t *
DeviceBaseConverter::get_pointer(std::uint32_t rns_base_id) {
  for (auto i = 0; i < input_rns_base_ids_.size(); i++) {
    if (input_rns_base_ids_[i] == rns_base_id) {
      return input_limbs_ptr_.get() + i * coeff_count_;
    }
  }
  for (auto i = 0; i < output_rns_base_ids_.size(); i++) {
    if (output_rns_base_ids_[i] == rns_base_id) {
      return output_limbs_ptr_.get() + i * coeff_count_;
    }
  }
  throw std::runtime_error("Invalid RNS base");

  return nullptr;
}

void DeviceBaseConverter::set_input_bases(
    const std::vector<std::uint64_t> &input_rns_base_ids) {
  input_rns_base_ids_ = input_rns_base_ids;
  converted_ = false;
}

void DeviceBaseConverter::append_input_bases(
    const std::vector<std::uint64_t> &input_rns_base_ids) {
  input_rns_base_ids_.insert(input_rns_base_ids_.end(),
                             input_rns_base_ids.begin(),
                             input_rns_base_ids.end());
  converted_ = false;
}

void DeviceBaseConverter::set_output_bases(
    const std::vector<std::uint64_t> &output_rns_base_ids) {
  output_rns_base_ids_ = output_rns_base_ids;
  converted_ = false;

  std::vector<seal::Modulus> input_rns_modulii;
  for (auto &id : input_rns_base_ids_) {
    input_rns_modulii.push_back(context_.get_rns_modulus(id));
  }

  input_rns_bases_ = seal::util::allocate<seal::util::RNSBase>(
      pool_, input_rns_modulii, pool_);
  seal::util::StrideIter<const uint64_t *> ibase_punctured_prod_array(
      input_rns_bases_->punctured_prod_array(), input_rns_bases_->size());

  // barrettK = 58;
  std::vector<LimbT::Element_t> barrett_k_;
  std::vector<LimbT::Element_t> barrett_ratios_;
  std::vector<LimbT::Element_t> output_modulii_;
  barrett_k_.resize(output_rns_base_ids_.size());
  barrett_ratios_.resize(output_rns_base_ids_.size());
  output_modulii_.resize(output_rns_base_ids_.size());

  std::vector<std::uint64_t> inv_punctured_product(input_rns_base_ids_.size());

  SEAL_ITERATE(
      seal::util::iter(inv_punctured_product, ibase_punctured_prod_array,
                       input_rns_bases_->inv_punctured_prod_mod_base_array(),
                       input_rns_bases_->base()),
      input_rns_base_ids_.size(), [&](const auto &I) {
        // multiply by 1 to convert from MutliplyOperand to std::uint64_t
        std::get<0>(I) =
            seal::util::multiply_uint_mod(1, std::get<2>(I), std::get<3>(I));
      });

  SEAL_ITERATE(
      seal::util::iter(std::size_t(0), output_rns_base_ids_, barrett_ratios_,
                       barrett_k_, output_modulii_),
      output_rns_base_ids_.size(), [&](const auto &I) {
        auto output_base_idx = std::get<0>(I);
        const seal::Modulus &output_modulus =
            context_.get_rns_modulus(std::get<1>(I));

        uint32_t barrettK = 58;
        std::get<2>(I) = Math::safe_cast<LimbT::Element_t>(
            (1ULL << barrettK) / output_modulus.value());
        std::get<3>(I) = barrettK;
        std::get<4>(I) =
            Math::safe_cast<LimbT::Element_t>(output_modulus.value());

        SEAL_ITERATE(
            seal::util::iter(std::size_t(0), inv_punctured_product,
                             ibase_punctured_prod_array),
            input_rns_base_ids_.size(), [&](const auto &J) {
              const std::uint64_t x = seal::util::modulo_uint(
                  std::get<2>(J), input_rns_base_ids_.size(), output_modulus);
              auto y = seal::util::multiply_uint_mod(x, std::get<1>(J),
                                                     output_modulus);
              auto idx = std::get<0>(J) +
                         (output_base_idx * input_rns_base_ids_.size());
              base_change_factors_[idx] = Math::safe_cast<LimbT::Element_t>(y);
            });
      });

  size_t matrix_size_ = output_rns_base_ids.size() * input_rns_base_ids_.size();
  // base_change_matrix_.copy_async(base_change_factors_.get(),matrix_size_);

  // auto base_change_factors_vec_entry_ = allocate<LimbT::Element_t>(matrix_size_);
  std::vector<LimbDataType> base_change_factors_vec_entry_(matrix_size_);
  for (size_t i = 0; i < matrix_size_; i++) {
    base_change_factors_vec_entry_[i] = base_change_factors_[i];
  }
  base_change_matrix_ = allocate_uint_device(matrix_size_);
  base_change_matrix_.copy(base_change_factors_vec_entry_, matrix_size_);
  // base_change_facors_vec_.push_back(std::move(base_change_factors_vec_entry_));
  base_change_factors_vec_.push_back(base_change_factors_vec_entry_);
  base_change_factors_vec_device_.push_back(std::move(base_change_matrix_));

  output_modulus_device_ = allocate_uint_device(output_rns_base_ids.size());
  barrett_k_device_ = allocate_uint_device(output_rns_base_ids.size());
  barrett_ratios_device_ = allocate_uint_device(output_rns_base_ids.size());
  output_modulus_device_.copy(&output_modulii_[0], output_rns_base_ids.size());
  barrett_k_device_.copy(&barrett_k_[0], output_rns_base_ids.size());
  barrett_ratios_device_.copy(&barrett_ratios_[0], output_rns_base_ids.size());

  output_modulus_vec_.push_back(std::move(output_modulii_));
  barrett_k_vec_.push_back(std::move(barrett_k_));
  barrett_ratios_vec_.push_back(std::move(barrett_ratios_));

  output_modulus_vec_device_.push_back(std::move(output_modulus_device_));
  barrett_k_vec_device_.push_back(std::move(barrett_k_device_));
  barrett_ratios_vec_device_.push_back(std::move(barrett_ratios_device_));
}

} // namespace Runtime

} // namespace Cerium
