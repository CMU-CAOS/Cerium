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

cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(Cerium
    VERSION 1.0.0
    LANGUAGES CXX
)

option(CERIUM_DEBUG_EXEC "Enable debug flags for CUDA and runtime object libraries" OFF)
option(CERIUM_DEBUG_COMPILER "Enable debug flags for compiler targets only" OFF)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(NOT DEFINED Python3_EXECUTABLE_DIR)
    set(Python3_EXECUTABLE_DIR "/usr/bin/")
endif()
if(NOT DEFINED Python3_LIBRARY_DIR_BASE)
    set(Python3_LIBRARY_DIR_BASE "/usr/lib/")
endif()

find_package(Python3 COMPONENTS Interpreter Development)
if(NOT Python3_VERSION_MAJOR EQUAL 3)
    message(FATAL_ERROR "Cerium requires Python 3. Please ensure you have it
        installed in a location searched by CMake.")
endif()

add_subdirectory(third_party/pybind11 "pybind11_${Python3_VERSION_MINOR}")

add_subdirectory(cerium)
