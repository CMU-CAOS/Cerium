#!/bin/bash

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

BASE_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$BASE_DIR/../
shopt -s globstar
clang-format -i $PROJECT_ROOT_DIR/cerium/**/*.h
clang-format -i $PROJECT_ROOT_DIR/cerium/**/*.cpp
clang-format -i $PROJECT_ROOT_DIR/cerium/**/*.cu
clang-format -i $PROJECT_ROOT_DIR/cerium/**/*.cuh
