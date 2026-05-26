#!/bin/bash
set -e

SEAL_VERSION=v4.1.1

git clone https://github.com/microsoft/SEAL.git SEAL
cd SEAL
git checkout ${SEAL_VERSION}
cmake -S . -B build
cmake --build build
cmake --install build
