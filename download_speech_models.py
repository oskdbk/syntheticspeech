#!/usr/bin/env python3
"""Download pinned XTTS-v2 and Afro-TTS inference files onto persistent scratch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL_SPECS = {
    "xtts": ("coqui/XTTS-v2", "XTTS-v2", "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"),
    "afro": ("intronhealth/afro-tts", "afro-tts", "f99ba77006ffcf2fb66ed12d7093832b552a3d26"),
}
REQUIRED_FILES = ("config.json", "model.pth", "vocab.json")


def workspace() -> Path:
    return Path(os.environ.get("NAS_HOME", Path(__file__).resolve().parents[2])).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--models-dir", type=Path, default=workspace() / "models")
    parser.add_argument("--token-file", type=Path, default=workspace() / "keys/hf_key.txt")
    parser.add_argument("--check-only", action="store_true", help="Validate local files without network access.")
    args = parser.parse_args()
    failed = False
    for name in dict.fromkeys(args.models):
        repo, folder, revision = MODEL_SPECS[name]
        target = args.models_dir.expanduser().resolve() / folder
        if not args.check_only:
            from huggingface_hub import snapshot_download

            token = args.token_file.read_text().strip() if args.token_file.is_file() else None
            # Offline flags belong to inference. Downloads explicitly enable networking.
            import huggingface_hub.constants as constants

            constants.HF_HUB_OFFLINE = False
            snapshot_download(
                repo_id=repo, revision=revision, local_dir=str(target), token=token,
                allow_patterns=[*REQUIRED_FILES, "speakers_xtts.pth", "README.md", "LICENSE.txt"],
                max_workers=2,
            )
            (target / "download.json").write_text(
                json.dumps({"repo_id": repo, "revision": revision, "files": list(REQUIRED_FILES)}, indent=2) + "\n"
            )
        missing = [f for f in REQUIRED_FILES if not (target / f).is_file() or not (target / f).stat().st_size]
        print(f"{name}: {target}: " + (f"MISSING {missing}" if missing else "ready"), flush=True)
        failed |= bool(missing)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
