#!/usr/bin/env python3
"""Clone every utterance for one AfriSpeech speaker with Qwen3-TTS Base.

The generated WAVs keep Qwen3-TTS's natural duration and pace. The source
recording duration is recorded for comparison, but is not passed to the model
or used to alter generated audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REFERENCE_ID = "226dfbe5c1f2c06ddab271f9d10334ba"
SPEAKER_ID = "3eb89635a92eb86a21014bad9155e77b"
MODEL_FOLDER = "Qwen3-TTS-12Hz-1.7B-Base"


def default_nas_home() -> Path:
    configured = os.environ.get("NAS_HOME")
    if configured:
        return Path(configured).expanduser()
    # This file lives at <workspace>/repos/syntheticspeech/clone_speaker.py.
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    nas_home = default_nas_home()
    speaker_dir = (
        nas_home
        / "data/afrispeech/chosen"
        / f"speaker_{SPEAKER_ID}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-dir", type=Path, default=speaker_dir)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Defaults to <speaker-dir>/metadata.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=nas_home / "models" / MODEL_FOLDER,
        help="Local Qwen model directory; no model download is performed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <speaker-dir>/cloning/qwen3_tts_base_english",
    )
    parser.add_argument("--reference-id", default=REFERENCE_ID)
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small pilot limit; omit to process all metadata rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even when the manifest marks them complete.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return cleaned or "utterance"


def read_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def find_reference(rows: list[dict[str, str]], reference_id: str) -> dict[str, str]:
    for row in rows:
        candidates = {
            row.get("audio_ids", ""),
            row.get("ID", ""),
            Path(row.get("chosen_audio_path", "")).stem,
            Path(row.get("audio_path", "")).stem,
        }
        if reference_id in candidates:
            return row
    raise ValueError(f"Reference ID {reference_id!r} was not found in metadata")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "audio_ids",
        "ID",
        "target_text",
        "source_duration",
        "reference_id",
        "output_path",
        "sample_rate",
        "generated_duration",
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    speaker_dir = args.speaker_dir.expanduser().resolve()
    metadata_path = (args.metadata or speaker_dir / "metadata.csv").expanduser().resolve()
    output_dir = (
        args.output_dir
        or speaker_dir / "cloning/qwen3_tts_base_english"
    ).expanduser().resolve()
    generated_dir = output_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    rows = read_rows(metadata_path)
    if not rows:
        raise ValueError(f"No rows in metadata: {metadata_path}")
    reference = find_reference(rows, args.reference_id)
    reference_rel = reference.get("chosen_audio_path", "")
    if not reference_rel:
        raise ValueError("Metadata is missing chosen_audio_path")
    reference_audio = (speaker_dir / reference_rel).resolve()
    if not reference_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
    reference_text = reference.get("transcript", "").strip()
    if not reference_text:
        raise ValueError("Reference transcript is empty")

    speaker_ids = {row.get("user_ids", "") for row in rows}
    if speaker_ids != {SPEAKER_ID}:
        raise ValueError(
            f"Expected only speaker {SPEAKER_ID}, found: {sorted(speaker_ids)}"
        )

    reference_record = {
        "reference_id": args.reference_id,
        "speaker_id": SPEAKER_ID,
        "audio_ids": reference.get("audio_ids"),
        "dataset_id": reference.get("ID"),
        "relative_audio_path": reference_rel,
        "resolved_audio_path": str(reference_audio),
        "transcript": reference_text,
        "target_language": args.language,
        "dataset_duration_seconds": float(reference["duration"]),
        "metadata_csv": str(metadata_path),
    }
    atomic_write_json(output_dir / "reference.json", reference_record)

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local Qwen model directory not found: {model_dir}")

    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32
    print(f"Loading local model from {model_dir} on {device_map}", flush=True)
    tts = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=dtype,
        local_files_only=True,
    )
    print("Preparing the reusable voice-clone prompt", flush=True)
    prompt = tts.create_voice_clone_prompt(
        ref_audio=str(reference_audio),
        ref_text=reference_text,
        x_vector_only_mode=False,
    )

    selected_rows = rows[: args.limit] if args.limit is not None else rows
    manifest_path = output_dir / "manifest.csv"
    manifest: list[dict[str, Any]] = []
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            manifest = list(csv.DictReader(stream))
    completed = {
        row.get("audio_ids", "")
        for row in manifest
        if row.get("status") == "ok"
        and row.get("output_path")
        and Path(row["output_path"]).is_file()
    }
    manifest_by_id = {row.get("audio_ids", ""): row for row in manifest}

    for index, row in enumerate(selected_rows, start=1):
        audio_id = row.get("audio_ids") or row.get("ID") or row.get("chosen_audio_path")
        text = row.get("transcript", "").strip()
        output_path = generated_dir / f"{safe_name(str(audio_id or 'utterance'))}.wav"
        try:
            if not audio_id:
                raise ValueError("Row has no audio_ids, ID, or chosen_audio_path")
            if not text:
                raise ValueError("Target transcript is empty")
            if not row.get("chosen_audio_path"):
                raise ValueError("Row is missing chosen_audio_path")
            if audio_id in completed and not args.overwrite:
                print(f"[{index}/{len(selected_rows)}] already done: {audio_id}", flush=True)
                continue

            source_duration = float(row["duration"])
            print(f"[{index}/{len(selected_rows)}] generating {audio_id}", flush=True)
            wavs, sample_rate = tts.generate_voice_clone(
                text=text,
                language=args.language,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
            )
            raw_wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            generated_duration = len(raw_wav) / sample_rate
            sf.write(output_path, raw_wav, sample_rate, subtype="PCM_16")
            result = {
                "audio_ids": audio_id,
                "ID": row.get("ID", ""),
                "target_text": text,
                "source_duration": source_duration,
                "reference_id": args.reference_id,
                "output_path": str(output_path),
                "sample_rate": sample_rate,
                "generated_duration": generated_duration,
                "status": "ok",
                "error": "",
            }
        except Exception as error:  # Preserve progress and keep processing other rows.
            result = {
                "audio_ids": audio_id,
                "ID": row.get("ID", ""),
                "target_text": text,
                "source_duration": row.get("duration", ""),
                "reference_id": args.reference_id,
                "output_path": str(output_path),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"ERROR {audio_id}: {result['error']}", file=sys.stderr, flush=True)
        manifest_by_id[audio_id] = result
        manifest = list(manifest_by_id.values())
        write_manifest(manifest_path, manifest)

    print(f"Done. Outputs: {generated_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
