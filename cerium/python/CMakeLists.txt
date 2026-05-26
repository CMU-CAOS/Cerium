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


configure_file(setup.py.in setup.py.after_configure)
file(GENERATE OUTPUT setup.py INPUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py.after_configure)

add_subdirectory(compiler cerium/compiler)
add_subdirectory(runtime cerium/runtime)

add_custom_target("python_${Python3_VERSION_MINOR}" ALL DEPENDS "cerium_compiler_py_${Python3_VERSION_MINOR}" "cerium_runtime_py_${Python3_VERSION_MINOR}")
add_custom_command(TARGET "python_${Python3_VERSION_MINOR}"
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/cerium
            ${CMAKE_CURRENT_BINARY_DIR}/cerium
)

add_custom_command(TARGET "python_${Python3_VERSION_MINOR}"
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/compiler
            ${CMAKE_CURRENT_BINARY_DIR}/cerium/compiler
)

add_custom_command(TARGET "python_${Python3_VERSION_MINOR}"
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime
            ${CMAKE_CURRENT_BINARY_DIR}/cerium/runtime
)
