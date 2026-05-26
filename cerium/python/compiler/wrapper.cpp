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

#include "cerium/compiler/cerium.h"
#include <cstdint>
#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

const char *const SAVE_DOC_STRING =
    R"DELIMITER(Serialize and save an Cerium object to a file.

Parameters
----------
path : str
    Path of the file to save to
)DELIMITER";

// clang-format off
PYBIND11_MODULE(_cerium_compiler, m) {
  m.doc() = "Python wrapper for Cerium";
  m.attr("__name__") = "cerium.compiler._cerium_compiler";

  py::enum_<Cerium::Frontend::Op>(m, "Op")
#define X(op,code) .value(#op, Cerium::Frontend::Op::op)
CERIUM_OPS
#undef X
  ;
  py::enum_<Cerium::Frontend::Type>(m, "Type")
#define X(type,code) .value(#type, Cerium::Frontend::Type::type)
CERIUM_TYPES
#undef X
  ;
  py::class_<Cerium::Frontend::Term, shared_ptr<Cerium::Frontend::Term>>(m, "Term", "Cerium's Term Class")
    .def_property_readonly("op", &Cerium::Frontend::Term::getOp, "The operation performed by this term")
    .def_property_readonly("scale", &Cerium::Frontend::Term::getScale, "The Scale of this term")
    .def_property_readonly("level", &Cerium::Frontend::Term::getLevel, "The Level of this term");


  py::class_<Cerium::Frontend::Function, shared_ptr<Cerium::Frontend::Function>>(m, "Function", "Cerium's Function class")
    .def(py::init<string,Cerium::Frontend::Program &, std::uint8_t, std::uint8_t>(), py::arg("name"), py::arg("program"), py::arg("partitionSize"), py::arg("partitionId"))
    .def_property("name", &Cerium::Frontend::Function::getName, &Cerium::Frontend::Function::setName, "The name of this program")
    .def("_currentPartitionSize", &Cerium::Frontend::Function::getCurrentPartitionSize, "" )
    .def("_currentPartitionID", &Cerium::Frontend::Function::getCurrentPartitionId, "")
    .def("to_DOT", &Cerium::Frontend::Function::toDOT, R"DELIMITER(Produce a graph representation of the program in the DOT format.

Returns
-------
str
    The graph in DOT format)DELIMITER")
    .def("_make_add", &Cerium::Frontend::Function::makeAdd, py::keep_alive<0,1>())
    .def("_make_subtract", &Cerium::Frontend::Function::makeSubtract, py::keep_alive<0,1>())
    .def("_make_multiply", &Cerium::Frontend::Function::makeMultiply, py::keep_alive<0,1>())
    .def("_make_negate", &Cerium::Frontend::Function::makeNegate, py::keep_alive<0,1>())
    .def("_make_left_rotation", &Cerium::Frontend::Function::makeLeftRotation, py::keep_alive<0,1>())
    .def("_make_right_rotation", &Cerium::Frontend::Function::makeRightRotation, py::keep_alive<0,1>())
    .def("_make_conjugate", &Cerium::Frontend::Function::makeConjugate, py::keep_alive<0,1>())
    .def("_make_conjugate2", &Cerium::Frontend::Function::makeConjugate, py::keep_alive<0,1>())
    .def("_make_rotate2", &Cerium::Frontend::Function::makeRotate2, py::keep_alive<0,1>())
    // .def("_make_rotate3", &Cerium::Frontend::Function::makeRotate3, py::keep_alive<0,1>())
    .def("_make_rotate_multiply_accumulate", &Cerium::Frontend::Function::makeRotateMultiplyAccumulate, py::keep_alive<0,1>())
    .def("_make_multiply_rotate_accumulate", &Cerium::Frontend::Function::makeMultiplyRotateAccumulate, py::keep_alive<0,1>())
    .def("_make_rotate_accumulate", &Cerium::Frontend::Function::makeRotateAccumulate, py::keep_alive<0,1>())
    .def("_make_rotate_accumulate_many", &Cerium::Frontend::Function::makeRotateAccumulateMany, py::keep_alive<0,1>())
    .def("_make_hoisted_rotate", &Cerium::Frontend::Function::makeHoistedRotate)
    .def("_make_bsgs_multiply_accumulate", &Cerium::Frontend::Function::makeBsgsMultiplyAccumulate, py::keep_alive<0,1>())
    .def("_make_plaintext_input", &Cerium::Frontend::Function::makePlaintextInput, py::keep_alive<0,1>())
    .def("_make_repeated_plaintext_input", &Cerium::Frontend::Function::makeRepeatedPlaintextInput, py::keep_alive<0,1>())
    .def("_make_ciphertext_input", &Cerium::Frontend::Function::makeCiphertextInput, py::keep_alive<0,1>())
    .def("_make_receive", &Cerium::Frontend::Function::makeReceive, py::keep_alive<0,1>())
    .def("_make_rescale", &Cerium::Frontend::Function::makeRescale, py::keep_alive<0,1>())
    .def("_make_double_rescale", &Cerium::Frontend::Function::makeDoubleRescale, py::keep_alive<0,1>())
    .def("_make_ephemeral", &Cerium::Frontend::Function::makeToEphemeral, py::keep_alive<0,1>())
    .def("_make_relinearize", &Cerium::Frontend::Function::makeRelinearize, py::keep_alive<0,1>())
    .def("_make_relinearize2", &Cerium::Frontend::Function::makeRelinearize2, py::keep_alive<0,1>())
    // .def("_make_relinearize3", &Cerium::Frontend::Function::makeRelinearize3, py::keep_alive<0,1>())
    .def("_make_modswitch", &Cerium::Frontend::Function::makeModSwitch, py::keep_alive<0,1>())
    .def("_make_bootstrap_modraise", &Cerium::Frontend::Function::makeBootstrapModRaise, py::keep_alive<0,1>())
    .def("_make_output", &Cerium::Frontend::Function::makeOutput, py::keep_alive<0,1>())
    .def("_make_partition", &Cerium::Frontend::Function::makePartition, py::keep_alive<0,1>())
    .def("_make_vector", &Cerium::Frontend::Function::makeVector, py::keep_alive<0,1>())
    .def("_make_extract_vector", &Cerium::Frontend::Function::extractVector)
    .def("_make_ciphertext_argument", &Cerium::Frontend::Function::makeCiphertextArgument, py::keep_alive<0,1>())
    .def("_make_function_output", &Cerium::Frontend::Function::makeFunctionOutput, py::keep_alive<0,1>())
    .def("_make_function_call", &Cerium::Frontend::Function::makeFunctionCall)
    .def("_add_function_to_program", &Cerium::Frontend::Function::addFunctionToProgram, py::keep_alive<0,1>())
    ;

  py::class_<Cerium::Frontend::Program>(m, "Program", "Cerium's Program class")
    .def(py::init<string,uint32_t,uint8_t>(), py::arg("name"), py::arg("rns_bit_size")=28, py::arg("num_gpus")=1)
    .def("toDOT", &Cerium::Frontend::Program::toDOT)
    .def("get_functions",&Cerium::Frontend::Program::getFunctions)
  ;


  py::module m_passes = m.def_submodule("_passes", "Python wrapper for Cerium Passes");
  m_passes.def("keyswitch_pass", &Cerium::Backend::keyswitchPass, "", py::arg("program"))
              ;

  py::module m_compiler = m.def_submodule("_compiler", "Python wrapper for Cerium Compiler");
  m_compiler
  .def("cerium_compile", &Cerium::Backend::ceriumCompile, "", py::arg("program"), py::arg("num_gpus"), py::arg("num_vregs"), py::arg("num_bcus"), py::arg("output_prefix"))
  .def("cerium_compile_function", &Cerium::Backend::ceriumCompileFunction, "", py::arg("function"), py::arg("num_gpus"), py::arg("num_vregs"), py::arg("num_bcus"), py::arg("output_prefix"))
              ;

}
// clang-format on
