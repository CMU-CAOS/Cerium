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

#include "cerium/compiler/backend/keyswitch_digits.h"

#include <string>

namespace Cerium {
namespace Backend {
namespace KeySwitch {

[[noreturn]] void throwInvalidKeyswitchConfig(const char *function_name,
                                              uint16_t level,
                                              uint32_t mod_partitions,
                                              uint32_t current_partition_size) {
  throw std::runtime_error(
      std::string("Unsupported keyswitch digit configuration in ") +
      function_name + ": level=" + std::to_string(level) +
      ", mod_partitions=" + std::to_string(mod_partitions) +
      ", current_partition_size=" + std::to_string(current_partition_size));
}

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_aggregation_28bit(uint16_t level, uint32_t mod_partitions,
                                     uint32_t current_partition_size) {
  // # <= 32: 1
  // # <= 42: 2
  // # <= 48: 3
  // # <= 51: 4
  // # <= 53: 5
  // # <= 54: 6
  // # <= 56: 7
  // # <= 56: 8
  // # <= 57: 9
  // # <= 58: 10
  // # <= 59: 12
  // # <= 60: 19
  // # <= 61: 30
  // # <= 62: 61
  // # <= 63: 63
  if (mod_partitions == 1) {
    if (level <= 32) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 42) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 48) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 1 || mod_partitions == 4) {
    if (current_partition_size == 1) {
      if (level <= 13) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 39) {
        return std::pair<uint16_t, uint16_t>(1, 3);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 2) {
      if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(2, 1);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(2, 2);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(2, 3);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(2, 4);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(2, 5);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(2, 6);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(2, 10);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(2, 15);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(2, 31);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(2, 32);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 4) {
      if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(4, 1);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(4, 2);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(4, 3);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(4, 5);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(4, 8);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(4, 16);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 2) {
    if (current_partition_size == 1) {
      if (level <= 21) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 42) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 2) {
      if (level <= 42) {
        return std::pair<uint16_t, uint16_t>(2, 1);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(2, 2);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(2, 3);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(2, 4);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(2, 5);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(2, 6);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(2, 10);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(2, 15);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(2, 31);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(2, 32);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 8) {
    if (current_partition_size == 1) {
      if (level <= 13) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 39) {
        return std::pair<uint16_t, uint16_t>(1, 3);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 2) {
      if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(2, 1);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(2, 2);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(2, 4);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(2, 5);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(2, 6);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(2, 10);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(2, 15);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(2, 31);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(2, 32);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 4) {
      if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(4, 1);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(4, 2);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(4, 3);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(4, 5);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(4, 8);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(4, 16);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 8) {
      if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(8, 1);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(8, 2);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(8, 3);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(8, 4);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(8, 8);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 16) {
    if (current_partition_size == 1) {
      if (level <= 13) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 39) {
        return std::pair<uint16_t, uint16_t>(1, 3);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 2) {
      if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(2, 1);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(2, 2);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(2, 4);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(2, 5);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(2, 6);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(2, 10);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(2, 15);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(2, 31);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(2, 32);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 4) {
      if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(4, 1);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(4, 2);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(4, 3);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(4, 5);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(4, 8);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(4, 16);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 8) {
      if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(8, 1);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(8, 2);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(8, 3);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(8, 4);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(8, 8);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 16) {
      if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(16, 1);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(16, 2);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(16, 4);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 12) {
    if (current_partition_size == 1) {
      if (level <= 13) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 39) {
        return std::pair<uint16_t, uint16_t>(1, 3);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 2) {
      if (level <= 26) {
        return std::pair<uint16_t, uint16_t>(2, 1);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(2, 2);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(2, 4);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(2, 5);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(2, 6);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(2, 10);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(2, 15);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(2, 31);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(2, 32);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 4) {
      if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(4, 1);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(4, 2);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(4, 3);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(4, 5);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(4, 8);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(4, 16);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 8) {
      if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(8, 1);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(8, 2);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(8, 3);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(8, 4);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(8, 8);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else if (current_partition_size == 12) {
      if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(12, 1);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(12, 2);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(12, 3);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(12, 6);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                              current_partition_size);
  return std::pair(0, 0);
}

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_broadcast_28bit(uint16_t level, uint32_t mod_partitions,
                                   uint32_t current_partition_size) {

  // # <= 32: 1
  // # <= 42: 2
  // # <= 48: 3
  // # <= 51: 4
  // # <= 53: 5
  // # <= 54: 6
  // # <= 56: 7
  // # <= 56: 8
  // # <= 57: 9
  // # <= 58: 10
  // # <= 59: 12
  // # <= 60: 19
  // # <= 61: 30
  // # <= 62: 61
  // # <= 63: 63

  if (current_partition_size == 1) {
  
    #if 1
    if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 22) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 30) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 42) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 48) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
    #else
   if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 26) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 39) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
    #endif
  }

  if (mod_partitions == 4) {
    if (current_partition_size == 2) {
      if (level <= 32) {
        return std::pair<uint16_t, uint16_t>(1, 1);
      } else if (level <= 42) {
        return std::pair<uint16_t, uint16_t>(1, 2);
      } else if (level <= 48) {
        return std::pair<uint16_t, uint16_t>(1, 3);
      } else if (level <= 51) {
        return std::pair<uint16_t, uint16_t>(1, 4);
      } else if (level <= 53) {
        return std::pair<uint16_t, uint16_t>(1, 5);
      } else if (level <= 54) {
        return std::pair<uint16_t, uint16_t>(1, 6);
      } else if (level <= 56) {
        return std::pair<uint16_t, uint16_t>(1, 7);
      } else if (level <= 57) {
        return std::pair<uint16_t, uint16_t>(1, 9);
      } else if (level <= 58) {
        return std::pair<uint16_t, uint16_t>(1, 10);
      } else if (level <= 59) {
        return std::pair<uint16_t, uint16_t>(1, 12);
      } else if (level <= 60) {
        return std::pair<uint16_t, uint16_t>(1, 19);
      } else if (level <= 61) {
        return std::pair<uint16_t, uint16_t>(1, 30);
      } else if (level <= 62) {
        return std::pair<uint16_t, uint16_t>(1, 61);
      } else if (level <= 63) {
        return std::pair<uint16_t, uint16_t>(1, 63);
      } else {
        throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                    current_partition_size);
      }
    }

    if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 26) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 39) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (current_partition_size == 2) {
    if (level <= 10) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 42) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 48) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 8) {
    if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 26) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 39) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 12) {
    if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 26) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 39) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  if (mod_partitions == 16) {
    if (level <= 13) {
      return std::pair<uint16_t, uint16_t>(1, 1);
    } else if (level <= 26) {
      return std::pair<uint16_t, uint16_t>(1, 2);
    } else if (level <= 39) {
      return std::pair<uint16_t, uint16_t>(1, 3);
    } else if (level <= 51) {
      return std::pair<uint16_t, uint16_t>(1, 4);
    } else if (level <= 53) {
      return std::pair<uint16_t, uint16_t>(1, 5);
    } else if (level <= 54) {
      return std::pair<uint16_t, uint16_t>(1, 6);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 7);
    } else if (level <= 56) {
      return std::pair<uint16_t, uint16_t>(1, 8);
    } else if (level <= 57) {
      return std::pair<uint16_t, uint16_t>(1, 9);
    } else if (level <= 58) {
      return std::pair<uint16_t, uint16_t>(1, 10);
    } else if (level <= 59) {
      return std::pair<uint16_t, uint16_t>(1, 12);
    } else if (level <= 60) {
      return std::pair<uint16_t, uint16_t>(1, 19);
    } else if (level <= 61) {
      return std::pair<uint16_t, uint16_t>(1, 30);
    } else if (level <= 62) {
      return std::pair<uint16_t, uint16_t>(1, 61);
    } else if (level <= 63) {
      return std::pair<uint16_t, uint16_t>(1, 63);
    } else {
      throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                                  current_partition_size);
    }
  }

  throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                              current_partition_size);
  return std::pair(0, 0);
}


std::pair<uint16_t, uint16_t>
get_keyswitch_dnum_28bit(KeySwitchType key_switch_type, uint16_t level,
                         uint32_t mod_partitions,
                         uint32_t current_partition_size) {
  if (key_switch_type == KeySwitchType::Aggregation) {
    return get_keyswitch_dnum_aggregation_28bit(level, mod_partitions,
                                                current_partition_size);
  } else if (key_switch_type == KeySwitchType::Broadcast) {
    return get_keyswitch_dnum_broadcast_28bit(level, mod_partitions,
                                              current_partition_size);
  } else if (key_switch_type == KeySwitchType::Naive) {
    auto dnum = get_keyswitch_dnum_broadcast_28bit(level, mod_partitions,
                                                   current_partition_size);
    return dnum;
  }
  throwInvalidKeyswitchConfig(__func__, level, mod_partitions,
                              current_partition_size);
  return std::pair(0, 0);
}

std::pair<uint16_t, uint16_t>
get_keyswitch_dnum(KeySwitchType key_switch_type, uint16_t level,
                   uint32_t mod_partitions, uint32_t current_partition_size) {
  return get_keyswitch_dnum_28bit(key_switch_type, level, mod_partitions,
                                  current_partition_size);
}
} // namespace KeySwitch
} // namespace Backend
} // namespace Cerium