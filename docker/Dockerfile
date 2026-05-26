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

FROM nvidia/cuda:13.3.0-devel-ubuntu22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

# Base build tooling and repository helpers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        build-essential \
        ca-certificates \
        cmake \
        git \
        g++-12 \
        gnupg \
        make \
        m4 \
        software-properties-common \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SEAL from the repo-local helper script.
COPY scripts /scripts
RUN bash /scripts/install_seal.sh

# Python headers used by the build.
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10-dev \
        python3.12-dev \
        python3.10-venv \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Deadsnakes does not publish versioned pip packages; bootstrap pip per interpreter.
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3.10 /tmp/get-pip.py \
    && python3.12 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

RUN apt-get update \
    && apt-get install -y --allow-change-held-packages --no-install-recommends \
        libnccl2=2.30.7-1+cuda13.3 \
        libnccl-dev=2.30.7-1+cuda13.3 \
    && rm -rf /var/lib/apt/lists/*
