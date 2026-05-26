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


add_library(CeriumCompiler STATIC
    cerium.cpp
    version.cpp
)

target_include_directories(CeriumCompiler
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(CeriumCompiler PRIVATE CERIUM_VERSION_STR="${PROJECT_VERSION}")

target_compile_options(CeriumCompiler PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${CERIUM_DEBUG_COMPILER}>>:-g;-O0;-UNDEBUG>"
)

include(FetchContent)
FetchContent_Declare(
    fmtlib
    PREFIX third_party/fmtlib
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 9.1.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS ON
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/third_party/fmtlib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
FetchContent_MakeAvailable(fmtlib)

target_link_libraries(CeriumCompiler PUBLIC fmt::fmt)


add_subdirectory(util)
add_subdirectory(frontend)
add_subdirectory(backend)
