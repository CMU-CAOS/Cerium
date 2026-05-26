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

add_library(CeriumRuntimeCuda OBJECT
    cuda_ops.cu
    nccl_ops.cu
    memory_manager_cuda.cu
)

target_include_directories(CeriumRuntimeCuda
    PRIVATE
        ${NCCL_INCLUDE_DIR}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

target_compile_definitions(CeriumRuntimeCuda PRIVATE USE_NCCL)

target_compile_features(CeriumRuntimeCuda PUBLIC cxx_std_17 cuda_std_17)

target_compile_options(CeriumRuntimeCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${CERIUM_DEBUG_EXEC}>>:-G;-lineinfo;-Xcompiler=-g,-O0>"
)

target_sources(CeriumRuntime PRIVATE
    $<TARGET_OBJECTS:CeriumRuntimeCuda>
)