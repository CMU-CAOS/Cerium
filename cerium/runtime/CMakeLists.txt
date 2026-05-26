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

enable_language(CUDA)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
find_package(NCCL REQUIRED)

find_package(SEAL 4.1 REQUIRED)

add_library(CeriumRuntime SHARED
    context.cpp
)

target_link_libraries(CeriumRuntime
    PUBLIC
        SEAL::seal
        CUDA::cudart
    PRIVATE
        ${NCCL_LIBRARY}
)

target_include_directories(CeriumRuntime
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
)

add_subdirectory(utils)
add_subdirectory(math)
add_subdirectory(memory)
add_subdirectory(polynomial)
add_subdirectory(io)
add_subdirectory(cuda)
add_subdirectory(execution)