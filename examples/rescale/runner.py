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

"""Check the product after Cerium drops one CKKS modulus prime."""

from examples.common import LEVEL, run_cli, run_program, sample_vector
from examples.primes import Primes


def main(gpus):
    x, factor = sample_vector(1), sample_vector(2) + 1
    ciphertext_scale = 1 << 56
    plaintext_scale = 1 << 28
    # Rescaling divides the encoded product by the last active RNS prime.
    output_scale = ciphertext_scale * plaintext_scale / Primes[LEVEL - 1]
    run_program("rescale", gpus,
                {"x": (x, ciphertext_scale), "factor": (factor, plaintext_scale)},
                x * factor, output_scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
