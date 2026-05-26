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

try:
    from ._cerium_runtime import *
except ImportError as exc:
    import sys
    msg = """
Importing the Cerium Runtime C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how Cerium was
installed.

Please note and check the following:

  * The Python version is: Python%d.%d from "%s"

and make sure that they are the versions you expect.
Original error was: %s
""" % (sys.version_info[0], sys.version_info[1], sys.executable,
        exc)
    raise ImportError(msg)