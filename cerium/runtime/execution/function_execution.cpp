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

#include "cerium/runtime/cuda/cuda_ops.h"
#include "cerium/runtime/cuda/nccl_ops.h"
#include "cerium/runtime/execution/device_utils.h"
#include "cerium/runtime/execution/function.h"
#include "cerium/runtime/io/generator.h"
#include "cerium/runtime/utils/function_helpers.h"
#include "cerium/runtime/utils/logger.h"

#include <fstream>
#include <iomanip>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <locale>
#include <regex>
#include <stdexcept>
#include <string>

#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <execution>
#include <optional>
#include <thread>

#include <dlfcn.h>

#include "cerium/runtime/cuda/memory_manager_cuda.h"
#include "cerium/runtime/utils/overloaded.h"
#include "cerium/runtime/utils/string.h"

namespace Cerium {
namespace Runtime {

auto CeriumFunction::compute_rsv_factors(
    const std::vector<uint32_t> &dest_rns_base_ids) {
  if (dest_rns_base_ids.empty()) {
    throw std::invalid_argument("RSV bases must not be empty");
  }
  std::vector<seal::Modulus> modulii;
  for (uint64_t i = 0; i < dest_rns_base_ids.size(); i++) {
    modulii.push_back(context_.get_rns_modulus(dest_rns_base_ids[i]));
  }
  seal::util::RNSBase rns_base(modulii, pool_glo);
  seal::util::StrideIter<const uint64_t *> ibase_punctured_prod_array(
      rns_base.punctured_prod_array(), rns_base.size());
  auto rns_base_prod = rns_base.base_prod();

  std::vector<LimbDataType> factors(dest_rns_base_ids.size() *
                                    dest_rns_base_ids.size());
  uint64_t *temp_data_64 = (uint64_t *)factors.data();

  for (size_t i = 0; i < dest_rns_base_ids.size(); i++) {

    auto punctured_product = ibase_punctured_prod_array[i];
    auto inv_punctured_product_modop =
        rns_base.inv_punctured_prod_mod_base_array()[i];
    // multiply by 1 to convert from MutliplyOperand to std::uint64_t
    auto temp_prod = seal::util::multiply_uint_mod(
        static_cast<uint64_t>(1), inv_punctured_product_modop, modulii.at(i));
    seal::util::set_zero_uint(
        rns_base.size(),
        (uint64_t *)(factors.data() + i * dest_rns_base_ids.size()));
    seal::util::multiply_uint(
        punctured_product, rns_base.size(), temp_prod, rns_base.size(),
        (uint64_t *)(factors.data() + i * dest_rns_base_ids.size()));
  }
  typedef unsigned __int128 uint128_t;
  uint32_t rsv_barrett_k_ = floor(log2(rns_base_prod[0])) + 63;
  uint128_t temp = ((uint64_t)1 << (rsv_barrett_k_ - 64));
  temp <<= 64;
  auto rsv_barrett_ratio_host_ = (uint64_t)(temp / rns_base_prod[0]);

  std::vector<LimbDataType> rns_base_prod_vec(dest_rns_base_ids.size());
  for (size_t i = 0; i < dest_rns_base_ids.size(); i++) {
    rns_base_prod_vec[i] = ((LimbDataType *)rns_base_prod)[i];
  }

  return std::tuple{factors, rns_base_prod_vec, rsv_barrett_ratio_host_,
                    rsv_barrett_k_};
}

void CeriumFunction::create_program_multithread() {

  if (partitions > 1) {
    CUDA::NCCL::ncclInit(partitions);
  }

  LOG(logger, INFO) << "Creating Program \n" << std::flush;

  init_streams(partitions);
  init_prefetch_streams(partitions);
  init_register_file(partitions, registers);
  init_base_conversion_units(partitions, num_bcus);
  init_evaluators(partitions);
  init_comm_records(partitions, registers);
  load_function_map(code_file_base, partitions);
  kernel_args_.resize(partitions);
  prefetch_args_.resize(partitions);
  prefetch_last_use_events_.resize(partitions);
  prefetch_queued_functions_.resize(partitions);

  std::atomic<size_t> num_threads_ready = 0;
  std::atomic<size_t> num_threads_complete = 0;

  auto thread_fn = [&](const size_t tid) {
    CUDA::setDevice(tid);
    auto &fmap = function_maps.at(tid);

    auto instruction_file_name =
        instruction_file_base + std::to_string(tid % num_gpus_compiled);
    auto ifile = open_input_file(instruction_file_name);
    std::string instruction;

    while (true) {
      if (!std::getline(ifile, instruction, '}')) {
        break;
      }
      KernelArgs kernel_args;
      auto fn = FunctionMetaData(instruction, this);
      if (!fn.valid) {
        continue;
      }
      size_t kernel_idx = kernel_args_[tid].size();
      bool is_comm = (fn.kernel_type == KernelType::Drm ||
                      fn.kernel_type == KernelType::Ags ||
                      fn.kernel_type == KernelType::Ard);
      auto bcu_ids =
          parse_term_base_conv_vec(fn.base_conv_string, fn.kernel_type, tid);
      kernel_args.kernel_type = fn.kernel_type;
      kernel_args.function_id = fn.function_id;
      std::vector<LimbT::Element_t *> inputs_ptrs_;
      std::vector<LimbT::Element_t *> outputs_ptrs_;
      for (auto &input : fn.inputs) {
        inputs_ptrs_.push_back(get_pointer_from_term(input, false, tid));
      }
      for (auto &output : fn.outputs) {
        outputs_ptrs_.push_back(get_pointer_from_term(output, true, tid));
      }

      std::vector<size_t> true_deps;
      std::vector<size_t> false_deps;

      if (fn.kernel_type == KernelType::Generic) {
        if (fn.outputs.empty()) {
          continue;
        }
        std::vector<LimbT::Element_t> scalar_vals_;
        for (auto &scalar : fn.scalars) {
          scalar_vals_.push_back(get_scalar_from_term(scalar));
        }
        std::vector<size_t> remapable_offsets_;
        for (auto &term : fn.remapables) {
          remapable_offsets_.push_back(
              get_remapable_offset_from_term(term, tid));
        }
        kernel_args.args = evaluators_[tid]->fn_ewi_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, scalar_vals_,
            const_cast<const LimbDataType **>(remapable_base_[tid].get()),
            remapable_offsets_, fn.bases, fn.rotations);
      } else if (fn.kernel_type == KernelType::Pmu) {
        if (fn.outputs.empty()) {
          continue;
        }
        std::vector<LimbT::Element_t> pmu_factors_ =
            compute_pmu_factors(fn.pmu_bases);
        kernel_args.args = evaluators_[tid]->fn_ewi_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, pmu_factors_,
            const_cast<const LimbDataType **>(remapable_base_[tid].get()), {},
            fn.bases, fn.rotations);
      } else if (fn.kernel_type == KernelType::Int) {
        if (fn.outputs.empty()) {
          continue;
        }
        kernel_args.args = evaluators_[tid]->fn_int_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, fn.bases);
      } else if (fn.kernel_type == KernelType::Sud) {
        if (fn.outputs.empty()) {
          continue;
        }
        auto bcu_id = std::optional<uint32_t>();
        if (!bcu_ids.empty()) {
          bcu_id = bcu_ids[0];
        }
        auto sud_factors = compute_sud_factors(tid, bcu_id, fn.divide_bases);
        kernel_args.args = evaluators_[tid]->fn_sud_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, fn.bases,
            sud_factors);
      } else if (fn.kernel_type == KernelType::Ntt) {
        if (fn.outputs.empty()) {
          continue;
        }
        kernel_args.args = evaluators_[tid]->fn_ntt_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, fn.bases);
      } else if (fn.kernel_type == KernelType::Rsv) {
        const auto &[factors, base, barrett_ratio, barrett_k] =
            compute_rsv_factors(fn.rsv_bases);
        kernel_args.args = evaluators_[tid]->fn_rsv_args(
            fmap.at(fn.function_id), outputs_ptrs_, inputs_ptrs_, factors, base,
            barrett_ratio, barrett_k);
      } else if (fn.kernel_type == KernelType::Bco) {
        if (bcu_ids.empty()) {
          continue;
        }
        std::vector<DeviceBaseConverter *> bcus;
        for (auto &bcu_id : bcu_ids) {
          bcus.push_back(base_conversion_units[tid].at(bcu_id));
        }
        kernel_args.args =
            evaluators_[tid]->fn_bco_args(fmap.at(fn.function_id), bcus);
      } else if (fn.kernel_type == KernelType::Drm) {
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, false, tid));
        }
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, true, tid));
        }

        kernel_args.args = evaluators_[tid]->drm_args(
            gather_ptrs_, scatter_ptrs_, fn.comm_size, fn.num_comm_limbs, tid);
      } else if (fn.kernel_type == KernelType::Ags) {
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, false, tid));
        }
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, true, tid));
        }
        kernel_args.args = evaluators_[tid]->ags_args(
            scatter_ptrs_, gather_ptrs_, fn.comm_size, fn.num_comm_limbs, tid,
            fn.bases);
      } else if (fn.kernel_type == KernelType::Ard) {
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, false, tid));
        }
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, true, tid));
        }
        kernel_args.args = evaluators_[tid]->ard_args(
            scatter_ptrs_, gather_ptrs_, fn.comm_size, fn.num_comm_limbs, tid,
            fn.bases);
      } else if (fn.kernel_type == KernelType::Call) {
        LOG(logger, INFO) << "Function Name: " << fn.function_name << "\n"
                          << std::flush;

        std::string remapable_base = "";
        std::string fname = fn.function_name;
        auto lpos = fname.find("(");
        auto rpos = fname.find(")");
        if (lpos != std::string::npos) {
          assert(rpos != std::string::npos);
          remapable_base = fn.function_name.substr(lpos + 1, rpos - lpos - 1);
          fname = fn.function_name.erase(lpos);

          LOG(logger, INFO) << "Function Name: " << fname << "\n" << std::flush;
          LOG(logger, INFO) << "Remapable Base: " << remapable_base << "\n"
                            << std::flush;
        }

        if (called_cerium_functions_.find(fname) ==
            called_cerium_functions_.end()) {
          throw std::runtime_error("Called CeriumFunction " + fname +
                                   " not found");
        }
        auto called_cerium_function = called_cerium_functions_.at(fname);
        kernel_args.args.emplace_back(called_cerium_function.get());

        kernel_args.args.emplace_back((void *)-1);
        if (remapable_base == "") {
          kernel_args.args.emplace_back(nullptr);
        } else {
          kernel_args.args.emplace_back(
              called_cerium_function->remapable_pointers_.at(remapable_base)[tid]
                  .get());
          PrefetchArgs prefetch_args;
          prefetch_args.cerium_function = called_cerium_function.get();
          prefetch_args.pointer =
              called_cerium_function->remapable_pointers_.at(remapable_base)[tid].get();
          prefetch_args.size = called_cerium_function->remapables_size_[tid];

          prefetch_args.event = CUDA::createEvent();
          kernel_args.wait_for_event = prefetch_args.event;

          kernel_args.event_end = CUDA::createEvent();
          prefetch_args.wait_for_event = kernel_args.event_end;

          prefetch_args_[tid].emplace_back(std::move(prefetch_args));
        }
        kernel_args.args.emplace_back((void *)-1);

        for (auto &[i, l] : called_cerium_function->function_args_input_local_[tid]) {
          auto name_in_func = i;
          auto term_here = fn.function_args_map.at(i);
          kernel_args.args.emplace_back(l->data());
          kernel_args.args.emplace_back(
              get_pointer_from_term(term_here, false, tid));
        }

        kernel_args.args.emplace_back((void *)-1);

        for (auto &[o, l] : called_cerium_function->function_args_output_local_[tid]) {
          auto name_in_func = o;
          auto it = fn.function_args_map.find(o);
          if (it == fn.function_args_map.end()) {
            continue;
          }
          auto &term_here = it->second;
          kernel_args.args.emplace_back(
              get_pointer_from_term(term_here, true, tid));
          kernel_args.args.emplace_back(l->data());
        }

        // We use -1 to denote the end of the seqeunce
        kernel_args.args.emplace_back((void *)-1);
      } else if (fn.kernel_type == KernelType::Mov) {
        continue;
      }
      CHECK_CUDA_ERROR();
      kernel_args_[tid].emplace_back(std::move(kernel_args));
    }
    CUDA::deviceSynchronize();
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t tid = 0; tid < partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { thread_fn(tid); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
}

void CeriumFunction::prefetch_threadfn(const std::size_t tid,
                                    const CUDA::StreamPtr &stream,
                                    const CUDA::StreamPtr &prefetch_stream,
                                    size_t prefetch_idx,
                                    const CUDA::EventPtr &wait_for_event) {

  bool perform_prefetching = runtime_options_.perform_uvm_prefetching;
  if (!perform_prefetching) {
    LOG(logger, INFO) << "Perform Prefetching: "
                      << (uint32_t)perform_prefetching << "\n"
                      << std::flush;
  }

  if (prefetch_idx >= prefetch_args_.at(tid).size()) {
    return;
  }

  auto &prefetch_args = prefetch_args_[tid][prefetch_idx];
  auto called_cerium_function = prefetch_args.cerium_function;
  assert(called_cerium_function != nullptr);
  if (called_cerium_function->remapables_type_ == RemapablesType::UVM &&
      perform_prefetching && prefetch_args.pointer != nullptr &&
      prefetch_args.size > 0) {
    // Ensure that prefetch stream is empty before starting prefetching to avoid deferring prefetch
    // CUDA::streamSynchronize(prefetch_stream);
    LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid << "]:"
                      << "Prefetching Remapable Inputs to Device Memory\n"
                      << std::flush;
    CUDA::memPrefetchHostToDeviceAsync(prefetch_args.pointer,
                                       called_cerium_function->remapables_size_[tid],
                                       tid, prefetch_stream);
    LOG(logger, INFO)
        << called_cerium_function->function_name() << "[" << tid << "]:"
        << "Recording Event and Making Main Stream Wait for Prefetching\n"
        << std::flush;
    CUDA::eventRecord(prefetch_args.event, prefetch_stream);
    // LOG(logger,INFO) << called_cerium_function->function_name() << "["<< tid << "]:" << "Event Recorded, Prefetch Stream Waiting\n" << std::flush;
    // CUDA::streamWaitEvent(prefetch_stream,prefetch_args.wait_for_event);
  }
  if (called_cerium_function->remapables_type_ == RemapablesType::ExplicitCopy &&
      perform_prefetching && prefetch_args.pointer != nullptr &&
      prefetch_args.size > 0) {
    auto copy_wait_event = wait_for_event;
    if (runtime_options_.prefetch_wait_per_function) {
      const auto last_use = prefetch_last_use_events_.at(tid).find(
          called_cerium_function);
      copy_wait_event = last_use == prefetch_last_use_events_.at(tid).end()
                            ? nullptr
                            : last_use->second;
    }
    if (copy_wait_event != nullptr) {
      LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid << "]:"
                        << "Prefetch Thread Waiting for Event Before Copying\n"
                        << std::flush;
      CUDA::streamWaitEvent(prefetch_stream, copy_wait_event);
    }
    LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid << "]:"
                      << "Explicity Copying Remapable Inputs to Device Memory\n"
                      << std::flush;
    CUDA::memcpyHostToDeviceAsync(
        called_cerium_function->remapable_pointers_explicit_copy_[tid].get(),
        prefetch_args.pointer, called_cerium_function->remapables_size_[tid],
        prefetch_stream);
    LOG(logger, INFO)
        << called_cerium_function->function_name() << "[" << tid << "]:"
        << "Recording Event and Making Main Stream Wait for Prefetching\n"
        << std::flush;
    CUDA::eventRecord(prefetch_args.event, prefetch_stream);
    // LOG(logger,INFO) << called_cerium_function->function_name() << "["<< tid << "]:" << "Event Recorded, Prefetch Stream Waiting\n" << std::flush;
    // CUDA::streamWaitEvent(prefetch_stream,prefetch_args.wait_for_event);
    // CUDA::memcpyHostToDeviceAsync(called_cerium_function->remapable_pointers_explicit_copy_[tid].get(), remapable_base_ptr, called_cerium_function->remapables_size_[tid], stream);
  }
}

void CeriumFunction::run_program_multithread_threadfn(
    const std::size_t tid, const CUDA::StreamPtr &stream,
    const CUDA::StreamPtr &prefetch_stream) {

  bool perform_prefetching = runtime_options_.perform_uvm_prefetching;
  bool perform_prefetching_back_to_host =
      runtime_options_.perform_uvm_prefetching_back_to_host;
  if (!perform_prefetching) {
    LOG(logger, INFO) << "Perform Prefetching: "
                      << (uint32_t)perform_prefetching << "\n"
                      << std::flush;
  }
  if (perform_prefetching_back_to_host) {
    LOG(logger, INFO) << "Perform Prefetching Back To Host: "
                      << (uint32_t)perform_prefetching_back_to_host << "\n"
                      << std::flush;
  }
  size_t prefetches_issued = 0;

  auto issue_prefetches = [&](const CUDA::EventPtr &wait_for_event) {
    const auto issue_one = [&]() -> std::optional<size_t> {
      if (prefetches_issued >= prefetch_args_[tid].size()) {
        return std::nullopt;
      }
      const auto &prefetch_args = prefetch_args_[tid][prefetches_issued];
      // Do not overwrite a function's sole explicit-copy staging buffer before
      // its already-queued invocation has consumed it.
      if (prefetch_queued_functions_[tid].find(
              prefetch_args.cerium_function) !=
          prefetch_queued_functions_[tid].end()) {
        return std::nullopt;
      }
      prefetch_threadfn(tid, stream, prefetch_stream, prefetches_issued,
                        wait_for_event);
      prefetch_queued_functions_[tid].insert(prefetch_args.cerium_function);
      return prefetches_issued++;
    };

    const auto first_prefetch = issue_one();
    if (first_prefetch &&
        prefetch_args_[tid][*first_prefetch].size <
            runtime_options_.small_prefetch_lookahead_bytes) {
      issue_one();
    }
  };

  if (perform_prefetching && !prefetch_args_[tid].empty()) {
    issue_prefetches(nullptr);
  }

  for (auto &kernel_args : kernel_args_.at(tid)) {
    if (kernel_args.kernel_type == KernelType::Generic) {
      evaluators_[tid]->fn_ewi(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Pmu) {
      evaluators_[tid]->fn_ewi(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Int) {
      evaluators_[tid]->fn_int(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Sud) {
      evaluators_[tid]->fn_sud(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Ntt) {
      evaluators_[tid]->fn_ntt(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Rsv) {
      evaluators_[tid]->fn_rsv(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Bco) {
      evaluators_[tid]->fn_bco(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Drm) {
      evaluators_[tid]->drm(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Ags) {
      evaluators_[tid]->ags(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Ard) {
      evaluators_[tid]->ard(stream, kernel_args.args);
    } else if (kernel_args.kernel_type == KernelType::Call) {

      auto &args = kernel_args.args;
      assert(args.size() > 4);
      auto called_cerium_function = (CeriumFunction *)kernel_args.args[0];
      if (perform_prefetching) {
        prefetch_queued_functions_[tid].erase(called_cerium_function);
      }
      assert((int64_t)args[1] == -1);
      auto remapable_base_ptr = (LimbDataType *)kernel_args.args[2];
      if (called_cerium_function->remapables_type_ != RemapablesType::ExplicitCopy &&
          remapable_base_ptr != nullptr) {
        set_pointer(called_cerium_function->remapable_base_[tid].get(),
                    remapable_base_ptr, stream);
      } else if (called_cerium_function->remapables_type_ ==
                     RemapablesType::ExplicitCopy &&
                 remapable_base_ptr != nullptr) {
        set_pointer(
            called_cerium_function->remapable_base_[tid].get(),
            called_cerium_function->remapable_pointers_explicit_copy_[tid].get(),
            stream);
      }
      CUDA::EventPtr event = nullptr;
      if (called_cerium_function->remapables_type_ == RemapablesType::UVM &&
          perform_prefetching && remapable_base_ptr &&
          called_cerium_function->remapables_size_[tid] > 0) {
        LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid
                          << "]:"
                          << "Prefetching Remapable Inputs to Device Memory\n"
                          << std::flush;
        event = kernel_args.wait_for_event;
      }
      if (called_cerium_function->remapables_type_ == RemapablesType::ExplicitCopy &&
          remapable_base_ptr && called_cerium_function->remapables_size_[tid] > 0) {
        LOG(logger, INFO)
            << called_cerium_function->function_name() << "[" << tid
            << "]:" << "Explicity Copying Remapable Inputs to Device Memory\n"
            << std::flush;
        event = kernel_args.wait_for_event;
      }
      assert((int64_t)args[3] == -1);
      int count = 4;
      while (count < args.size() && (int64_t)args[count] != -1) {
        auto dest = (LimbDataType *)args[count];
        auto src = (const LimbDataType *)args[count + 1];
        CUDA::memcpyDeviceToDeviceAsync(
            dest, src, context_.n() * sizeof(LimbT::Element_t), stream);
        count += 2;
      }
      assert((int64_t)args[count] == -1);
      count++;
      if (event) {
        LOG(logger, INFO)
            << called_cerium_function->function_name() << "[" << tid
            << "]:" << "Waiting for Prefetch Event before launching kernel\n"
            << std::flush;
        CUDA::streamWaitEvent(stream, event);
      }
      if (!called_cerium_function->graphs_init_.empty() &&
          called_cerium_function->graphs_init_[tid]) {
        CUDA::graphLaunch(called_cerium_function->graph_exec_[tid], stream);
      } else {
        called_cerium_function->run_program_multithread_threadfn(tid, stream,
                                                         prefetch_stream);
      }
      while (count < args.size() && (int64_t)args[count] != -1) {
        auto dest = (LimbDataType *)args[count];
        auto src = (const LimbDataType *)args[count + 1];
        CUDA::memcpyDeviceToDeviceAsync(
            dest, src, context_.n() * sizeof(LimbT::Element_t), stream);
        count += 2;
      }

      if (kernel_args.event_end) {
        LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid
                          << "]:" << "Recording Event after launching kernels\n"
                          << std::flush;
        CUDA::eventRecord(kernel_args.event_end, stream);
        if (runtime_options_.prefetch_wait_per_function) {
          prefetch_last_use_events_.at(tid)[called_cerium_function] =
              kernel_args.event_end;
        }
        issue_prefetches(kernel_args.event_end);
      }
      assert((int64_t)args[count] == -1);
      count++;
      assert(count == args.size());

      if (perform_prefetching_back_to_host &&
          called_cerium_function->remapables_type_ == RemapablesType::UVM &&
          remapable_base_ptr && called_cerium_function->remapables_size_[tid] > 0) {
        LOG(logger, INFO) << called_cerium_function->function_name() << "[" << tid
                          << "]:"
                          << "Prefetching Remapable Inputs to Host Memory\n"
                          << std::flush;
        CUDA::memPrefetchDeviceToHostAsync(
            remapable_base_ptr, called_cerium_function->remapables_size_[tid], stream);
      }
    } else if (kernel_args.kernel_type == KernelType::Mov) {
      continue;
    }
    CHECK_CUDA_ERROR_FUNCTION_ID(kernel_args.function_id);
  }
}

void CeriumFunction::run_program_multithread() {

  std::atomic<size_t> num_threads_ready = 0;
  std::atomic<size_t> num_threads_complete = 0;

  auto thread_fn1 = [&](const std::size_t tid) {
    CUDA::setDevice(tid);
    CUDA::deviceSynchronize();
  };

  NUM_ITERS = runtime_options_.num_iters;

  runtime_stats_.start = std::chrono::steady_clock::now();

  auto thread_fn = [&](const std::size_t tid, const CUDA::StreamPtr &stream,
                       const CUDA::StreamPtr &prefetch_stream) {
    CUDA::setDevice(tid);
    for (int i = 0; i < NUM_ITERS; i++) {
      run_program_multithread_threadfn(tid, stream, prefetch_stream);
    }
    CUDA::deviceSynchronize();
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  bool USE_PREFETCH_STREAM = runtime_options_.use_prefetch_stream;
  LOG(logger, INFO) << "Using Prefetch Stream: "
                    << (uint32_t)USE_PREFETCH_STREAM << "\n"
                    << std::flush;
  if (USE_PREFETCH_STREAM) {
    for (size_t tid = 0; tid < partitions; tid++) {
      threads.emplace_back([&, tid]() {
        capture_thread_exception(
            [&]() { thread_fn(tid, streams_[tid], prefetch_streams_[tid]); },
            thread_exception, thread_exception_mutex);
      });
    }
  } else {
    for (size_t tid = 0; tid < partitions; tid++) {
      threads.emplace_back([&, tid]() {
        capture_thread_exception(
            [&]() { thread_fn(tid, streams_[tid], streams_[tid]); },
            thread_exception, thread_exception_mutex);
      });
    }
  }

  for (size_t tid = 0; tid < partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);

  runtime_stats_.end = std::chrono::steady_clock::now();
  update_runtime_stats();
}

void CeriumFunction::create_cuda_graph() {

  if (partitions > 1) {
    CUDA::NCCL::ncclInit(partitions);
  }

  CUDA::deviceSynchronize();
  use_cugraph = true;
  init_register_file(partitions, registers);
  LOG(logger, INFO) << "Created Register File:\n" << std::flush;
  init_base_conversion_units(partitions, num_bcus);
  LOG(logger, INFO) << "Created BCU Units:\n" << std::flush;
  init_evaluators(partitions);
  CUDA::deviceSynchronize();
  LOG(logger, INFO) << "Creating Graphs:\n" << std::flush;
  init_graphs(partitions, registers);
  graphs_have_collectives_.assign(partitions, false);
  LOG(logger, INFO) << "Created Graphs:\n" << std::flush;
  load_function_map(code_file_base, partitions);
  LOG(logger, INFO) << "Loaded Function Map:\n" << std::flush;

  bool all_ops_in_same_stream = runtime_options_.all_ops_in_same_stream;
  if (all_ops_in_same_stream) {
    LOG(logger, INFO) << "All Kernels in 1 Stream: "
                      << (uint32_t)all_ops_in_same_stream << "\n"
                      << std::flush;
  }

  auto thread_fn = [&](const size_t tid) {
    LOG(logger, INFO) << "Creating Cuda Graphs tid:" + std::to_string(tid)
                      << "\n";

    CUDA::setDevice(tid);
    CUDA::NodePtr prev_nccl = nullptr;
    auto fmap = function_maps.at(tid);

    CUDA::NodePtr prev_node = nullptr;

    auto instruction_file_name =
        instruction_file_base + std::to_string(tid % num_gpus_compiled);
    auto ifile = open_input_file(instruction_file_name);
    std::string instruction;

    bool no_cross_chip_comm = runtime_options_.no_cross_chip_comm;
    bool no_cross_chip_drm = runtime_options_.no_cross_chip_drm;
    bool no_cross_chip_ags = runtime_options_.no_cross_chip_ags;
    bool no_cross_chip_ard = runtime_options_.no_cross_chip_ard;
    bool no_comm_false_deps = runtime_options_.no_comm_false_deps;

    while (true) {
      if (!std::getline(ifile, instruction, '}')) {
        break;
      }

      auto fn = FunctionMetaData(instruction, this);
      if (!fn.valid) {
        continue;
      }
      auto bcu_ids =
          parse_term_base_conv_vec(fn.base_conv_string, fn.kernel_type, tid);

      std::vector<LimbT::Element_t *> inputs_ptrs_;
      std::vector<LimbT::Element_t *> outputs_ptrs_;
      for (auto &input : fn.inputs) {
        inputs_ptrs_.push_back(get_pointer_from_term(input, false, tid));
      }
      for (auto &output : fn.outputs) {
        outputs_ptrs_.push_back(get_pointer_from_term(output, true, tid));
      }

      std::vector<CUDA::NodePtr> true_deps;
      std::vector<CUDA::NodePtr> false_deps;

      CUDA::NodePtr in;
      CUDA::NodePtr out;

      for (auto &input : fn.inputs) {
        auto term_true_deps = get_true_deps_from_term(input, tid);
        true_deps.insert(true_deps.end(), term_true_deps.begin(),
                         term_true_deps.end());
      }

      for (auto &output : fn.outputs) {
        auto term_false_deps = get_false_deps_from_term(output, tid);
        false_deps.insert(false_deps.end(), term_false_deps.begin(),
                          term_false_deps.end());
      }

      if (fn.kernel_type == KernelType::Generic) {
        if (fn.outputs.empty() && fn.kernel_type != KernelType::Call) {
          continue;
        }
        std::vector<LimbT::Element_t> scalar_vals_;
        for (auto &scalar : fn.scalars) {
          scalar_vals_.push_back(get_scalar_from_term(scalar));
        }
        std::vector<size_t> remapable_offsets_;
        for (auto &term : fn.remapables) {
          remapable_offsets_.push_back(
              get_remapable_offset_from_term(term, tid));
        }
        auto node0 = CUDA::createNode();
        evaluators_[tid]->fn_ewi(
            fmap.at(fn.function_id), graphs_[tid], node0, outputs_ptrs_,
            inputs_ptrs_, scalar_vals_,
            const_cast<const LimbDataType **>(remapable_base_[tid].get()),
            remapable_offsets_, fn.bases, fn.rotations);
        in = node0;
        out = node0;
      } else if (fn.kernel_type == KernelType::Pmu) {
        if (fn.outputs.empty()) {
          continue;
        }
        std::vector<LimbT::Element_t> pmu_factors_ =
            compute_pmu_factors(fn.pmu_bases);
        auto node0 = CUDA::createNode();
        evaluators_[tid]->fn_ewi(
            fmap.at(fn.function_id), graphs_[tid], node0, outputs_ptrs_,
            inputs_ptrs_, pmu_factors_,
            const_cast<const LimbDataType **>(remapable_base_[tid].get()), {},
            fn.bases, fn.rotations);
        in = node0;
        out = node0;
      } else if (fn.kernel_type == KernelType::Int) {
        if (fn.outputs.empty()) {
          continue;
        }
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        evaluators_[tid]->fn_int(fmap.at(fn.function_id), graphs_[tid], node0,
                                 node1, outputs_ptrs_, inputs_ptrs_, fn.bases);
        in = node0;
        out = node1;
      } else if (fn.kernel_type == KernelType::Sud) {
        if (fn.outputs.empty()) {
          continue;
        }
        auto bcu_id = std::optional<uint32_t>();
        if (!bcu_ids.empty()) {
          bcu_id = bcu_ids[0];
        }
        auto sud_factors = compute_sud_factors(tid, bcu_id, fn.divide_bases);
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        evaluators_[tid]->fn_sud(fmap.at(fn.function_id), graphs_[tid], node0,
                                 node1, outputs_ptrs_, inputs_ptrs_, fn.bases,
                                 sud_factors);
        in = node0;
        out = node1;
      } else if (fn.kernel_type == KernelType::Ntt) {
        if (fn.outputs.empty()) {
          continue;
        }
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        evaluators_[tid]->fn_ntt(fmap.at(fn.function_id), graphs_[tid], node0,
                                 node1, outputs_ptrs_, inputs_ptrs_, fn.bases);
        in = node0;
        out = node1;
      } else if (fn.kernel_type == KernelType::Rsv) {
        if (fn.outputs.empty()) {
          continue;
        }

        auto node0 = CUDA::createNode();
        const auto &[factors, base, barrett_ratio, barrett_k] =
            compute_rsv_factors(fn.rsv_bases);
        evaluators_[tid]->fn_rsv(fmap.at(fn.function_id), graphs_[tid], node0,
                                 outputs_ptrs_, inputs_ptrs_, factors, base,
                                 barrett_ratio, barrett_k);

        in = node0;
        out = node0;
      } else if (fn.kernel_type == KernelType::Bco) {
        if (bcu_ids.empty()) {
          continue;
        }
        std::vector<DeviceBaseConverter *> bcus;
        for (auto &bcu_id : bcu_ids) {
          bcus.push_back(base_conversion_units[tid].at(bcu_id));
        }
        auto node0 = CUDA::createNode();
        evaluators_[tid]->fn_bco(fmap.at(fn.function_id), graphs_[tid], node0,
                                 bcus);
        DependencySet bcu_dependencies;
        for (auto &bcu_id : bcu_ids) {
          auto &dep_w = bcu_last_write[tid][bcu_id];
          bcu_dependencies.insert({dep_w, node0});
          bcu_last_write[tid][bcu_id] = node0;
          auto &dep_r = bcu_last_read[tid][bcu_id];
          for (auto &d : dep_r) {
            bcu_dependencies.insert({d, node0});
          }
        }
        if (all_ops_in_same_stream) {
          bcu_dependencies.insert({prev_node, node0});
        }
        for (auto &d : bcu_dependencies) {
          CUDA::addSingleEdge(graphs_[tid], d.first, d.second);
        }
        prev_node = node0;
        continue; // XXX
      } else if (fn.kernel_type == KernelType::Drm) {
        if (no_cross_chip_comm || no_cross_chip_drm) {
          continue;
        }
        graphs_have_collectives_[tid] = true;
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, false, tid));
        }
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, true, tid));
        }
        evaluators_[tid]->drm(graphs_[tid], node0, node1, gather_ptrs_,
                              scatter_ptrs_, fn.comm_size, fn.num_comm_limbs,
                              tid);
        in = node0;
        out = node1;
        if (prev_nccl && no_comm_false_deps == 0) {
          false_deps.push_back(prev_nccl);
        }
        prev_nccl = out;
      } else if (fn.kernel_type == KernelType::Ags) {
        if (no_cross_chip_comm || no_cross_chip_ags) {
          continue;
        }
        graphs_have_collectives_[tid] = true;
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, false, tid));
        }
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, true, tid));
        }
        evaluators_[tid]->ags(graphs_[tid], node0, node1, scatter_ptrs_,
                              gather_ptrs_, fn.comm_size, fn.num_comm_limbs,
                              tid, fn.bases);
        in = node0;
        out = node1;
        if (prev_nccl && no_comm_false_deps == 0) {
          false_deps.push_back(prev_nccl);
        }
        prev_nccl = out;
      } else if (fn.kernel_type == KernelType::Ard) {
        if (no_cross_chip_comm || no_cross_chip_ard) {
          continue;
        }
        graphs_have_collectives_[tid] = true;
        auto node0 = CUDA::createNode();
        auto node1 = CUDA::createNode();
        std::vector<LimbT::Element_t *> scatter_ptrs_;
        std::vector<LimbT::Element_t *> gather_ptrs_;
        for (auto &g : fn.gather) {
          gather_ptrs_.push_back(get_pointer_from_term(g, false, tid));
        }
        for (auto &s : fn.scatter) {
          scatter_ptrs_.push_back(get_pointer_from_term(s, true, tid));
        }
        evaluators_[tid]->ard(graphs_[tid], node0, node1, scatter_ptrs_,
                              gather_ptrs_, fn.comm_size, fn.num_comm_limbs,
                              tid, fn.bases);
        in = node0;
        out = node1;
        if (prev_nccl && no_comm_false_deps == 0) {
          false_deps.push_back(prev_nccl);
        }
        prev_nccl = out;
      } else if (fn.kernel_type == KernelType::Call) {

        LOG(logger, INFO) << "[" << tid << "]: Function Name: " << fn.function_name << "\n";

        std::string remapable_base = "";
        std::string fname = fn.function_name;
        auto lpos = fname.find("(");
        auto rpos = fname.find(")");
        if (lpos != std::string::npos) {
          assert(rpos != std::string::npos);
          remapable_base = fn.function_name.substr(lpos + 1, rpos - lpos - 1);
          fname = fn.function_name.erase(lpos);

          LOG(logger, INFO) << "Function Name: " << fname << "\n";
          LOG(logger, INFO) << "Remapable Base: " << remapable_base << "\n";
          throw std::string("Unimplemented Call with remapable_base cudagraph");
        }

        auto node0 = CUDA::createEmptyNode(graphs_[tid]);
        auto node1 = CUDA::createEmptyNode(graphs_[tid]);

        auto called_cerium_function = called_cerium_functions_.at(fn.function_name);

        auto call_subgraph_node = CUDA::createSubgraphNode(
            graphs_[tid], called_cerium_function->graphs_[tid]);

        const bool child_has_collectives =
            called_cerium_function->graphs_have_collectives_.at(tid);
        graphs_have_collectives_[tid] |= child_has_collectives;
        if (prev_nccl && no_comm_false_deps == 0 && child_has_collectives) {
          CUDA::addSingleEdge(graphs_[tid], prev_nccl, call_subgraph_node);
        }

        // Keep the wrapper nodes connected even when this call has no arguments.
        CUDA::addSingleEdge(graphs_[tid], node0, call_subgraph_node);
        CUDA::addSingleEdge(graphs_[tid], call_subgraph_node, node1);
        // Ordinary execution enqueues argument copies on one stream. Preserve
        // that order in the graph too: sibling copies may otherwise execute
        // concurrently, which is unsafe if mapped argument storage aliases.
        auto previous_input_copy = node0;
        for (auto &[i, l] : called_cerium_function->function_args_input_local_[tid]) {
          auto name_in_func = i;
          auto term_here = fn.function_args_map.at(i);
          auto memcpyNode = CUDA::createMemcpyDeviceToDeviceNode(
              graphs_[tid], l->data(),
              get_pointer_from_term(term_here, false, tid),
              context_.n() * sizeof(LimbT::Element_t));
          CUDA::addSingleEdge(graphs_[tid], previous_input_copy, memcpyNode);
          CUDA::addSingleEdge(graphs_[tid], memcpyNode, call_subgraph_node);
          previous_input_copy = memcpyNode;
        }

        auto previous_output_copy = call_subgraph_node;
        for (auto &[o, l] : called_cerium_function->function_args_output_local_[tid]) {
          auto name_in_func = o;
          // A caller may intentionally omit an output it does not consume.
          // Match the non-graph Call path by not creating a copy in that case.
          auto it = fn.function_args_map.find(o);
          if (it == fn.function_args_map.end()) {
            continue;
          }
          auto &term_here = it->second;

          auto memcpyNode = CUDA::createMemcpyDeviceToDeviceNode(
              graphs_[tid], get_pointer_from_term(term_here, true, tid),
              l->data(), context_.n() * sizeof(LimbT::Element_t));
          CUDA::addSingleEdge(graphs_[tid], previous_output_copy, memcpyNode);
          CUDA::addSingleEdge(graphs_[tid], memcpyNode, node1);
          previous_output_copy = memcpyNode;
        }

        // auto last_use = fn_last_use[tid][function_name];
        // XXX: Need to make sure functions with internal false dependencies are handled correctly
        // XXX: For now, treat all functions as dependent on each other
        auto last_use = fn_last_use[tid][""];
        if (last_use) {
          false_deps.push_back(last_use);
        }

        // fn_last_use[tid][function_name] = node1;
        // XXX: Need to make sure functions with internal false dependencies are handled correctly
        // XXX: For now, treat all functions as dependent on each other
        fn_last_use[tid][""] = node1;
        // A collective-bearing child must also order the next direct collective
        // in this graph. node1 is downstream of the child graph (and of any
        // output copies), so the existing direct-collective dependency path
        // makes the next NCCL operation wait for the call to finish.
        if (child_has_collectives) {
          prev_nccl = node1;
        }
        in = node0;
        out = node1;

      } else if (fn.kernel_type == KernelType::Mov) {
        continue;
      }
      CHECK_CUDA_ERROR();

      DependencySet dependencies;
      for (auto &d : true_deps) {
        if (d != nullptr && in != nullptr) {
          dependencies.insert({d, in});
        }
      }

      if (all_ops_in_same_stream && prev_node != nullptr) {
        false_deps.push_back(prev_node);
      }
      for (auto &d : false_deps) {
        if (d != nullptr && in != nullptr) {
          dependencies.insert({d, in});
        }
      }

      for (auto &edge : dependencies) {
        CUDA::addSingleEdge(graphs_[tid], edge.first, edge.second);
      }

      for (auto &input : fn.inputs) {
        update_term_reads(input, tid, in);
        update_term_reads(input, tid, out);
      }

      for (auto &output : fn.outputs) {
        update_term_writes(output, tid, out);
      }
      prev_node = out;
    }
    graph_exec_[tid] = CUDA::createGraphExec(graphs_[tid]);
    graphs_init_[tid] = true;
    CUDA::deviceSynchronize();
    bool PRINT_GRAPH = runtime_options_.print_cuda_graph;
    if (PRINT_GRAPH) {
      CUDA::graphDebugDotPrint(function_name_, graphs_[tid], tid);
    }
  };

  std::vector<std::thread> threads;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  for (size_t tid = 0; tid < partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { thread_fn(tid); }, thread_exception,
                               thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
  LOG(logger, INFO) << "Completed CUDA Graph Creation:" << "\n" << std::flush;
}

void CeriumFunction::run_program_multithread_cugraph() {
  NUM_ITERS = runtime_options_.num_iters;

  std::vector<CUDA::StreamPtr> stream;
  for (size_t tid = 0; tid < partitions; tid++) {
    CUDA::setDevice(tid);
    stream.push_back(CUDA::createStream());
  }

  auto graph_launch_thread_fn = [&](size_t tid) {
    CUDA::setDevice(tid);
    CUDA::graphLaunch(graph_exec_[tid], stream[tid]);
    for (int i = 1; i < NUM_ITERS; i++) {
      CUDA::graphLaunch(graph_exec_[tid], stream[tid]);
    }
    CUDA::deviceSynchronize();
  };

  std::vector<std::thread> threads;
  threads.clear();
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;
  LOG(logger, INFO) << "Launching Cuda Graphs\n";
  runtime_stats_.start = std::chrono::steady_clock::now();
  for (size_t tid = 0; tid < partitions; tid++) {
    threads.emplace_back([&, tid]() {
      capture_thread_exception([&]() { graph_launch_thread_fn(tid); },
                               thread_exception, thread_exception_mutex);
    });
  }

  for (size_t tid = 0; tid < partitions; tid++) {
    threads[tid].join();
  }
  rethrow_thread_exception(thread_exception);
  runtime_stats_.end = std::chrono::steady_clock::now();
  update_runtime_stats();
  for (size_t tid = 0; tid < partitions; tid++) {
    CUDA::setDevice(tid);
    CUDA::deviceSynchronize();
  }
}

} // namespace Runtime
} // namespace Cerium
