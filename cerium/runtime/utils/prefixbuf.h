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
#include <iostream>

namespace Cerium {
namespace Runtime {
namespace Utils {

// https://stackoverflow.com/questions/56521318/ostream-class-that-outputs-either-on-cout-or-on-a-file
class Prefixbuf : public std::streambuf {
  std::string prefix;
  std::string line_spefic_prefix;
  std::streambuf *sbuf;
  bool need_prefix;

  int sync() { return this->sbuf->pubsync(); }
  int overflow(int c) {
    if (c != std::char_traits<char>::eof()) {
      std::string write_prefix = this->line_spefic_prefix.empty()
                                     ? this->prefix
                                     : this->line_spefic_prefix + this->prefix;
      if (!write_prefix.empty()) {
        write_prefix += ": ";
      }
      if (this->need_prefix && !write_prefix.empty() &&
          write_prefix.size() !=
              this->sbuf->sputn(&write_prefix[0], write_prefix.size())) {
        return std::char_traits<char>::eof();
      }
      this->need_prefix = c == '\n';
    }
    return this->sbuf->sputc(c);
  }

public:
  Prefixbuf(std::string const &prefix, std::streambuf *sbuf)
      : prefix(prefix), sbuf(sbuf), need_prefix(true) {}

  Prefixbuf &
  update_line_specific_prefix(const std::string &new_line_specific_prefix) {
    this->line_spefic_prefix = new_line_specific_prefix;
    return *this;
  }
};

class OPrefixStream : private virtual Prefixbuf, public std::ostream {
public:
  OPrefixStream(std::string const &prefix, std::ostream &out)
      : Prefixbuf(prefix, out.rdbuf()),
        std::ios(static_cast<std::streambuf *>(this)),
        std::ostream(static_cast<std::streambuf *>(this)) {}

  OPrefixStream(std::string const &prefix)
      : Prefixbuf(prefix, std::cout.rdbuf()),
        std::ios(static_cast<std::streambuf *>(this)),
        std::ostream(static_cast<std::streambuf *>(this)) {}
};

} // namespace Utils
} // namespace Runtime
} // namespace Cerium
