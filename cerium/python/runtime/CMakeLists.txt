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

pybind11_add_module("cerium_runtime_py_${Python3_VERSION_MINOR}"
    wrapper.cpp
)

target_link_options("cerium_runtime_py_${Python3_VERSION_MINOR}"
    PRIVATE
        "LINKER:--no-as-needed"
)


target_compile_features("cerium_runtime_py_${Python3_VERSION_MINOR}" PUBLIC cxx_std_17)
target_link_libraries("cerium_runtime_py_${Python3_VERSION_MINOR}" PRIVATE CeriumRuntime)
set_target_properties("cerium_runtime_py_${Python3_VERSION_MINOR}" PROPERTIES OUTPUT_NAME _cerium_runtime)