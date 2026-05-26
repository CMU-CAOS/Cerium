// SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <complex>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Cerium {
namespace Runtime {
namespace Utils {
using MessageType = std::variant<std::vector<double>,
                                 std::vector<std::complex<double>>, double>;
// (Vector[double] | Vector[complex] | double , scale)
using RawInputType = std::pair<MessageType, double>;
class RawInputsWrapper {
private:
  std::unordered_map<std::string, RawInputType> raw_inputs_;

public:
  RawInputsWrapper() = default;
  RawInputsWrapper(
      const std::unordered_map<std::string, RawInputType> &raw_inputs)
      : raw_inputs_(raw_inputs) {}
  const auto &get_raw_inputs() const { return raw_inputs_; }
};
using RawInputsWrapperPtr = std::shared_ptr<RawInputsWrapper>;
} // namespace Utils
} // namespace Runtime
} // namespace Cerium
