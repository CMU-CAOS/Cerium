#!/usr/bin/env python3
"""Download the native Meta Llama 3.1 8B checkpoint bundle and verify it.

Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B, then
authenticate using ``huggingface-cli login`` or an ``HF_TOKEN`` environment
variable. The result is compatible with this repository torchrun loaders.
"""

import hashlib
import os
import shutil
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B"
MODEL_DIR = Path(__file__).resolve().parent / "model_weights" / "Llama3.1-8B"
CACHE_DIR = MODEL_DIR.parent / ".cache"
EXPECTED_MD5 = {
    "consolidated.00.pth": "7c46ceec43eef48f2d999813cf47d3ee",
    "params.json": "5d1e239b186cd12fac19afe77e8f9d45",
    "tokenizer.model": "08292403f8b173e7524d7fba7bbbd2d3",
}

os.environ.setdefault("HF_HOME", str(CACHE_DIR))

from huggingface_hub import hf_hub_download


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for filename, expected_digest in EXPECTED_MD5.items():
        downloaded_file = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=f"original/{filename}",
                cache_dir=CACHE_DIR,
                token=True,
            )
        )

        actual_digest = md5sum(downloaded_file)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_digest}, "
                f"got {actual_digest}"
            )

        shutil.copy2(downloaded_file, MODEL_DIR / filename)
        print(f"Verified and saved {filename}")

    print(f"Native checkpoint is ready in {MODEL_DIR}")


if __name__ == "__main__":
    main()
