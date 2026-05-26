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

#! /bin/bash

compile_all () {
    FN=$1
    CERIUM_DIR=$2
    COMPILED_DIR=$3
    echo $FN $GPUS $CERIUM_DIR $COMPILED_DIR
    mkdir -p ${COMPILED_DIR}/$FN/
    make all -j FNAME=$FN DIRNAME=$CERIUM_DIR && cp -r ${CERIUM_DIR}/$FN/output/* ${COMPILED_DIR}/$FN/
    wait
}

compile_nocode () {
    FN=$1
    CERIUM_DIR=$2
    COMPILED_DIR=$3
    echo $FN $GPUS $CERIUM_DIR $COMPILED_DIR
    mkdir -p ${COMPILED_DIR}/$FN/
    make no_code -j FNAME=$FN DIRNAME=$CERIUM_DIR && cp -r ${CERIUM_DIR}/$FN/output/* ${COMPILED_DIR}/$FN/
    wait
}

compile_clean () {
    FN=$1
    CERIUM_DIR=$2
    COMPILED_DIR=$3
    echo $FN $CERIUM_DIR $COMPILED_DIR
    make clean -j FNAME=$FN DIRNAME=$CERIUM_DIR
    wait
 
}

compile_clean_cu () {
    FN=$1
    CERIUM_DIR=$2
    COMPILED_DIR=$3
    echo $FN $GPUS $CERIUM_DIR $COMPILED_DIR
    make clean_cu -j FNAME=$FN DIRNAME=$CERIUM_DIR
    wait
}


CERIUM_DIR="outputs"

compile_all "bootstrap_32Kslots_33levels_1gpus" $CERIUM_DIR $CERIUM_DIR/compiled

compile_all "bootstrap_32Kslots_33levels_2gpus" $CERIUM_DIR $CERIUM_DIR/compiled

compile_all "bootstrap_32Kslots_33levels_4gpus" $CERIUM_DIR $CERIUM_DIR/compiled

compile_all "bootstrap_32Kslots_33levels_8gpus" $CERIUM_DIR $CERIUM_DIR/compiled