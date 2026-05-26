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

#include "cerium/compiler/util/mkdir.h"


#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0755)
#endif


namespace Cerium {
namespace Util {

bool mkdir_p(const std::string& path){
    std::string adjusted_path = path;

    // Normalize path separators
    #ifdef _WIN32
    std::replace(adjusted_path.begin(), adjusted_path.end(), '\\', '/');
    #endif

    std::istringstream ss(adjusted_path);
    std::string token;
    std::string current_path;

    if (adjusted_path[0] == '/')
        current_path = "/";

    while (std::getline(ss, token, '/')) {
        if (token.empty()) continue;
        if (!current_path.empty() && current_path.back() != '/')
            current_path += "/";
        current_path += token;

        struct stat st;
        if (stat(current_path.c_str(), &st) != 0) {
            if (MKDIR(current_path.c_str()) != 0) {
                std::cerr << "Failed to create directory: " << current_path << std::endl;
                return false;
            }
        } else if (!S_ISDIR(st.st_mode)) {
            std::cerr << "Path exists and is not a directory: " << current_path << std::endl;
            return false;
        }
    }

    return true;
}

} // namespace Util
} // namespace Cerium
