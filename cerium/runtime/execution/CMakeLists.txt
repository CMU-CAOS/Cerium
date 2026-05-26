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

add_library(CeriumRuntimeExec OBJECT
    evaluator.cpp
    function.cpp
    function_execution.cpp
    function_io.cpp
    program.cpp
    device_utils.cu
)

target_link_libraries(CeriumRuntimeExec
    PUBLIC
        SEAL::seal
    PRIVATE
        CUDA::cudart
)


target_include_directories(CeriumRuntimeExec
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

target_compile_features(CeriumRuntimeExec PUBLIC cxx_std_17 cuda_std_17)

target_compile_options(CeriumRuntimeExec PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${CERIUM_DEBUG_EXEC}>>:-g;-O0>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${CERIUM_DEBUG_EXEC}>>:-G;-lineinfo;-Xcompiler=-g,-O0>"
)

target_sources(CeriumRuntime PRIVATE
    $<TARGET_OBJECTS:CeriumRuntimeExec>
)
