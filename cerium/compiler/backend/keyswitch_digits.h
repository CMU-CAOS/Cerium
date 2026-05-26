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
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace Cerium {
namespace Backend {
namespace KeySwitch {

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_aggregation_28bit(uint16_t level, uint16_t num_partitions,
                                     uint16_t current_partition_size);

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_broadcast_28bit(uint16_t level, uint16_t num_partitions,
                                   uint16_t current_partition_size);

enum KeySwitchType { Broadcast, Aggregation, Naive };

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_28bit(KeySwitchType key_switch_type, uint16_t level,
                         uint32_t mod_partitions,
                         uint32_t current_partition_size);

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum(KeySwitchType key_switch_type, uint16_t level,
                   uint32_t mod_partitions, uint32_t current_partition_size);
} // namespace KeySwitch
} // namespace Backend
} // namespace Cerium