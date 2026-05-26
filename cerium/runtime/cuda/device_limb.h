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

#include "cerium/runtime/cuda/device_pointer.h"
#include "cerium/runtime/memory/pointer.h"
#include "cerium/runtime/polynomial/limb.h"

class Context;

namespace Cerium {
namespace Runtime {

using DeviceLimb = LimbBase<DevicePointer>;
using DeviceLimbPtr = std::shared_ptr<DeviceLimb>;

inline auto allocate_uint_device(const size_t count) {
  return allocate_device<DeviceLimb::Element_t>(count);
}

inline auto allocate_uint_device(const size_t count, const bool is_uvm) {
  return allocate_device<DeviceLimb::Element_t>(count, is_uvm);
}

} // namespace Runtime
} // namespace Cerium
