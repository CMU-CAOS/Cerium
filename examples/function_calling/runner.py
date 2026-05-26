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

"""Execute main, which calls the compiled encrypted helper function."""

from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x = sample_vector(1)
    scale = 1 << 56
    run_program("function_calling", gpus, {"x": (x, scale)}, 2 * x, scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
