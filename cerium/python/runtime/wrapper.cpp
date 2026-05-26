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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <iostream>
#include <variant>
#include <vector>

#include "cerium/runtime/context.h"
#include "cerium/runtime/execution/function.h"
#include "cerium/runtime/execution/program.h"
#include "cerium/runtime/io/ckks-encoder.h"
#include "cerium/runtime/io/generator.h"
#include "cerium/runtime/math/ntt.h"
#include "cerium/runtime/polynomial/rns_polynomial.h"
#include "cerium/runtime/utils/map_wrapper.h"
#include "cerium/runtime/utils/overloaded.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using Ciphertext =
    std::pair<Cerium::Runtime::RnsPolynomialPtr, Cerium::Runtime::RnsPolynomialPtr>;
PYBIND11_MAKE_OPAQUE(Ciphertext);
std::unordered_map<uint32_t, std::vector<Cerium::Runtime::Limb::Element_t>>
rns_polynomial_repr(const std::shared_ptr<Cerium::Runtime::RnsPolynomial> &rns_poly) {
  std::unordered_map<uint32_t, std::vector<Cerium::Runtime::Limb::Element_t>> repr;
  for (const auto &[id, limb] : rns_poly->limb_index_map()) {
    repr[id] = std::vector<Cerium::Runtime::Limb::Element_t>(
        limb->data(), limb->data() + limb->size());
  }
  return repr;
}

PYBIND11_MODULE(_cerium_runtime, m) {
  m.attr("__name__") = "cerium.runtime._cerium_runtime";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<Cerium::Runtime::Context>(m, "Context", "Cerium Runtime Context")
      .def(py::init([](const size_t slots,
                       const std::vector<std::uint64_t> &rns_modulii) {
        return std::unique_ptr<Cerium::Runtime::Context>(
            new Cerium::Runtime::Context(rns_modulii));
      }))
      .def_property_readonly("coeff_count", &Cerium::Runtime::Context::n);

  py::class_<Ciphertext>(m, "Ciphertext");

  py::class_<Cerium::Runtime::CKKSEncryptor>(m, "CKKSEncryptor",
                                      "Cerium Runtime CKKS Encryptor")
      .def(
          py::init([](const Cerium::Runtime::CKKSEncryptorContext &encryptor_context) {
            return std::unique_ptr<Cerium::Runtime::CKKSEncryptor>(
                new Cerium::Runtime::CKKSEncryptor(encryptor_context));
          }))
      .def(py::init([](const Cerium::Runtime::CKKSEncryptorContext &encryptor_context,
                       const seal::prng_seed_type &prng_seed) {
        return std::unique_ptr<Cerium::Runtime::CKKSEncryptor>(
            new Cerium::Runtime::CKKSEncryptor(encryptor_context, prng_seed));
      }))
      .def("encode_and_encrypt",
           [](Cerium::Runtime::CKKSEncryptor &encryptor,
              const Cerium::Runtime::Utils::MessageType &message, const double scale,
              const std::vector<uint32_t> rns_base_ids) {
             return std::visit(
                 Cerium::Runtime::Utils::overloaded{
                     [](auto arg) { throw std::invalid_argument("Message"); },
                     [&](const double &arg) {
                       return encryptor.encode_and_encrypt(arg, scale,
                                                           rns_base_ids);
                     },
                     [&](const std::vector<double> &arg) {
                       return encryptor.encode_and_encrypt(arg, scale,
                                                           rns_base_ids);
                     },
                     [&](const std::vector<std::complex<double>> &arg) {
                       return encryptor.encode_and_encrypt(arg, scale,
                                                           rns_base_ids);
                     }},
                 message);
           })
      .def("decrypt_and_decode", [](Cerium::Runtime::CKKSEncryptor &encryptor,
                                    const Ciphertext &input,
                                    const double scale) {
        return encryptor.decrypt_and_decode<std::complex<double>>(input, scale);
      });

  py::class_<Cerium::Runtime::CKKSEncryptorContext>(
      m, "CKKSEncryptorContext", "Cerium Runtime CKKS Encryptor Context")
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::vector<std::int64_t> &secret_key) {
        return std::unique_ptr<Cerium::Runtime::CKKSEncryptorContext>(
            new Cerium::Runtime::CKKSEncryptorContext(context, secret_key,
                                               secret_key));
      }))
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::vector<std::int64_t> &secret_key,
                       const seal::prng_seed_type &prng_seed) {
        return std::unique_ptr<Cerium::Runtime::CKKSEncryptorContext>(
            new Cerium::Runtime::CKKSEncryptorContext(context, secret_key, secret_key,
                                               prng_seed));
      }))
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::vector<std::int64_t> &secret_key,
                       const std::vector<std::int64_t> &ephemeral_key) {
        return std::unique_ptr<Cerium::Runtime::CKKSEncryptorContext>(
            new Cerium::Runtime::CKKSEncryptorContext(context, secret_key,
                                               ephemeral_key));
      }))
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::vector<std::int64_t> &secret_key,
                       const std::vector<std::int64_t> &ephemeral_key,
                       const seal::prng_seed_type &prng_seed) {
        return std::unique_ptr<Cerium::Runtime::CKKSEncryptorContext>(
            new Cerium::Runtime::CKKSEncryptorContext(context, secret_key,
                                               ephemeral_key, prng_seed));
      }));

  py::class_<Cerium::Runtime::IOGenerator>(m, "IOGenerator",
                                                  "IO Generator")
      .def(py::init([](const Cerium::Runtime::Context &context) {
        return std::make_unique<Cerium::Runtime::IOGenerator>(context);
      }))
      .def("generate_and_serialize_plaintexts",
           &Cerium::Runtime::IOGenerator::
               generate_and_serialize_plaintexts,
           py::arg("outputs_file_name"), py::arg("inputs_file_name"),
           py::arg("raw_inputs_wrapper"),
           py::arg("stream_name") = "Plaintext Stream")
      .def("generate_and_serialize_remapables",
           &Cerium::Runtime::IOGenerator::
               generate_and_serialize_plaintexts,
           py::arg("outputs_file_name"), py::arg("inputs_file_name"),
           py::arg("raw_inputs_wrapper"),
           py::arg("stream_name") = "Remapable Stream")
      .def("generate_and_serialize_evalkeys",
           &Cerium::Runtime::IOGenerator::
               generate_and_serialize_evalkeys)
      .def("generate_and_serialize_ciphertexts",
           &Cerium::Runtime::IOGenerator::
               generate_and_serialize_ciphertexts,
           py::arg("output_file_name"), py::arg("inputs_file_name"),
           py::arg("raw_inputs_wrapper"), py::arg("encryptor_context"))
      .def("generate_ciphertext_inputs",
           &Cerium::Runtime::IOGenerator::
               generate_ciphertext_inputs)
      .def_static(
          "deserialize_ciphertexts",
          [](const Cerium::Runtime::Context &context,
             const std::string &ciphertexts_file_name) {
            return Cerium::Runtime::IOGenerator::deserialize_ciphertexts(
                context, ciphertexts_file_name);
          },
          py::arg("context"), py::arg("ciphertexts_file_name"))
        ;

  py::class_<Cerium::Runtime::CeriumFunction,
             std::shared_ptr<Cerium::Runtime::CeriumFunction>>(
      m, "CeriumFunctionExecutor", "Cerium Function Executor")
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::string name = "",
                       const bool use_uvm = false) {
        return std::shared_ptr<Cerium::Runtime::CeriumFunction>(
            new Cerium::Runtime::CeriumFunction(context, name, use_uvm));
      }))
      .def_property_readonly(
          "function_name", &Cerium::Runtime::CeriumFunction::function_name)
      .def("generate_inputs",
           &Cerium::Runtime::CeriumFunction::generate_inputs)
      .def("generate_remapable_inputs",
           &Cerium::Runtime::CeriumFunction::generate_remapable_inputs,
           py::arg("remapables_file_name"),
           py::arg("plaintexts_file_name"), py::arg("map_key"),
           py::arg("remapables_use_uvm") = false)
      .def("copy_inputs_to_device",
           &Cerium::Runtime::CeriumFunction::copy_inputs_to_device)
      .def(
          "copy_remapable_inputs_to_device",
          &Cerium::Runtime::CeriumFunction::copy_remapable_inputs_to_device)
      .def("create_remapable_offsets",
           &Cerium::Runtime::CeriumFunction::create_remapable_offsets)
      .def("get_program_outputs",
           &Cerium::Runtime::CeriumFunction::get_program_outputs)
      .def("copy_ciphertext_inputs_to_gpu",
           &Cerium::Runtime::CeriumFunction::copy_ciphertext_inputs_to_gpu)

      ;

  py::class_<Cerium::Runtime::Utils::RawInputsWrapper,
             std::shared_ptr<Cerium::Runtime::Utils::RawInputsWrapper>>(
      m, "RawInputsWrapper",
      "Cerium Runtime Raw Inputs Wrapper. This is used to avoid having to "
      "copy inputs multiple times")
      .def(py::init(
          [](const std::unordered_map<
              std::string, Cerium::Runtime::Utils::RawInputType> &raw_inputs) {
            return std::shared_ptr<Cerium::Runtime::Utils::RawInputsWrapper>(
                new Cerium::Runtime::Utils::RawInputsWrapper(raw_inputs));
          }));

  py::class_<Cerium::Runtime::Program,
             std::shared_ptr<Cerium::Runtime::Program>>(
      m, "FusedKernelProgram", "Cerium CKKS Program")
      .def(py::init([](const Cerium::Runtime::Context &context,
                       const std::string &directory_base,
                       const std::string &top_level_function) {
        return std::make_shared<Cerium::Runtime::Program>(
            context, directory_base, top_level_function);
      }))
      .def("make_program", &Cerium::Runtime::Program::make_program,
           py::keep_alive<1, 0>())
      .def("run", &Cerium::Runtime::Program::run_program);
}
