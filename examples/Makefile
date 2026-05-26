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

NVCC=/usr/local/cuda-13.3/bin/nvcc

DIR=$(abspath ./${DIRNAME}/${FNAME})
CERIUM_KERNELS_PATH=$(abspath ../cerium/compiler/backend/kernels/)

HEADER_DIR=$(CERIUM_KERNELS_PATH)
BUILD_DIR=$(abspath $(DIR)/build)
BUILD_DIR_PARTITIONS=$(addprefix $(BUILD_DIR)/partition_,$(PARTITIONS))
OUTPUT_DIR=$(abspath $(DIR)/output)
LIB=$(abspath $(DIR)/build/function_map.so)
INCLUDE_DIR=$(HEADER_DIR)


GPUS := $(shell grep "gpus:" $(DIR)/compile_config | cut -f2 -d ":")
LAST := $(shell echo $$(( ${GPUS} - 1 )))
PARTITIONS := $(shell seq 0 ${LAST})

CUDA_ARCHITECTURES ?= 80 90 100
CUDA_GENCODE_FLAG = $(foreach arch,$(CUDA_ARCHITECTURES),-gencode=arch=compute_$(arch),code=sm_$(arch))


.PHONY: no_code
no_code: build_dir output_dir ll_function inputs program_inputs inputs_remapable remapable_inputs calls compile_config

.PHONY: all
all: build_dir output_dir test_lib no_code

build_dir:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR_PARTITIONS)

output_dir:
	mkdir -p $(OUTPUT_DIR)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)

.PHONY: clean_cu
clean_cu: clean
	$(foreach p, $(PARTITIONS), rm -rf $(DIR)/partition_$p/*.cu)
	@echo "Cleaned up build and output directories."

.PHONY: ll_function
ll_function: $(foreach p, $(PARTITIONS), $(OUTPUT_DIR)/functions$p)

.PHONY: inputs 
inputs : $(foreach p, $(PARTITIONS), $(OUTPUT_DIR)/inputs$p)

.PHONY: inputs_remapable 
inputs_remapable : $(foreach p, $(PARTITIONS), $(OUTPUT_DIR)/inputs_remapable$p)

.PHONY: program_inputs 
program_inputs : $(OUTPUT_DIR)/program_inputs

.PHONY: calls 
calls : $(OUTPUT_DIR)/calls

.PHONY: compile_config 
compile_config : $(OUTPUT_DIR)/compile_config

.PHONY: remapable_inputs 
remapable_inputs : $(OUTPUT_DIR)/remapable_inputs

BUILD_DIR_PARTITIONS=$(addprefix $(BUILD_DIR)/partition_,$(PARTITIONS))

$(BUILD_DIR)/common_function.o : $(INCLUDE_DIR)/function.cu
	@echo "$@"
	$(NVCC) -D__FNAME__="${FNAME}" -ccbin ${CXX} ${CUDA_GENCODE_FLAG} -I${INCLUDE_DIR}  -Xcompiler -fPIC -pg -g -c -O3  -Xcudafe --diag_suppress=177 $< -o $@

.PHONY: test_lib

test_lib: $(foreach p, $(PARTITIONS), $(OUTPUT_DIR)/function_map_$p.so)

define PARTITION_RULE
$(BUILD_DIR)/partition_$1/%.o: $(DIR)/partition_$1/%.cu
	@echo "$@"
	$(NVCC) -D__FNAME__="${FNAME}"  -ccbin ${CXX} ${CUDA_GENCODE_FLAG} -I${INCLUDE_DIR} -Xcompiler -fPIC -pg -g -c -O3  -Xcudafe --diag_suppress=177  $$< -o $$@
	
endef

$(foreach p,$(PARTITIONS),$(eval $(call PARTITION_RULE,$p)))

define GET_OBJECTS
	$(patsubst $(DIR)/partition_$1/%.cu,$(BUILD_DIR)/partition_$1/%.o,$(wildcard $(DIR)/partition_$1/*.cu))
endef


define LIB_RULE
$(OUTPUT_DIR)/function_map_$1.so: $(call GET_OBJECTS,$1) $(BUILD_DIR)/common_function.o
	$(NVCC) -D__FNAME__="${FNAME}" -ccbin ${CXX} ${CUDA_GENCODE_FLAG} -shared  $$^ -lcudart -o $$@
endef

$(foreach p,$(PARTITIONS),$(eval $(call LIB_RULE,$p)))

define LL_FUNCTION_RULE
$(OUTPUT_DIR)/functions$1: $(DIR)/partition_$1/ll.functions
	cp $$< $$@
endef

define INPUTS_RULE
$(OUTPUT_DIR)/inputs$1: $(DIR)/partition_$1/inputs$1
	cp $$< $$@
endef

define INPUTS_REMAPABLE_RULE
$(OUTPUT_DIR)/inputs_remapable$1: $(DIR)/partition_$1/inputs_remapable$1
	cp $$< $$@
endef

$(foreach p,$(PARTITIONS),$(eval $(call LL_FUNCTION_RULE,$p)))
$(foreach p,$(PARTITIONS),$(eval $(call INPUTS_RULE,$p)))
$(foreach p,$(PARTITIONS),$(eval $(call INPUTS_REMAPABLE_RULE,$p)))

$(OUTPUT_DIR)/program_inputs: $(DIR)/program_inputs
	cp $< $@

$(OUTPUT_DIR)/remapable_inputs: $(DIR)/remapable_inputs
	cp $< $@

$(OUTPUT_DIR)/calls: $(DIR)/calls
	cp $< $@

$(OUTPUT_DIR)/compile_config: $(DIR)/compile_config
	cp $< $@


.PHONY: tt
tt:
	$(info $(call GET_OBJECTS,0))

.PHONY: info
info:
	@echo "directory: $(DIR)"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Output Directory: $(OUTPUT_DIR)"
	@echo "GPUS: $(GPUS)"
	@echo "LAST: $(LAST)"
	@echo "partitions: $(PARTITIONS)"
	@echo "Library: $(LIB)"
