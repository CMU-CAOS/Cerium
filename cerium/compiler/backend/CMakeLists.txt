# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



add_library(CeriumCompilerBackend OBJECT
    keyswitch_pass.cc
    cuda_kernel.cc
    terms.cc
    limb.cc
    logger.cc
    register.cc
    cerium.cc
    compiler.cc
    compiler_dispatch.cc
    compiler_partition.cc
    compiler_keyswitch.cc
    compiler_rescale.cc
    compiler_io.cc
    compiler_fusion.cc
    keyswitch_digits.cc
)

target_include_directories(CeriumCompilerBackend
    PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
        ${FMT_INCLUDE_DIR}
)

target_compile_options(CeriumCompilerBackend PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${CERIUM_DEBUG_COMPILER}>>:-g;-O0;-UNDEBUG>"
)

target_link_libraries(CeriumCompilerBackend PUBLIC fmt::fmt)
target_link_libraries(CeriumCompiler PUBLIC CeriumCompilerBackend)
