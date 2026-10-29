#!/usr/bin/env python3
"""Compare every utterance from one speaker using one fixed cloning reference.

Preparation, dry runs, and Azure use only Python's standard library. Neural
backends load sequentially and only when an output actually needs generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

from download_speech_models import MODEL_SPECS, workspace
from speech_backends import make_backend

SPEAKER_ID = "3eb89635a92eb86a21014bad9155e77b"
REFERENCE_ID = "226dfbe5c1f2c06ddab271f9d10334ba"
FIELDS = ["audio_id", "model", "text", "reference_id", "output_path", "source_duration",
          "sample_rate", "generated_duration", "status", "fingerprint", "provenance", "error"]


def parse_args(argv=None):
    home = workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-dir", type=Path, default=home / f"data/afrispeech/chosen/speaker_{SPEAKER_ID}")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: <speaker-dir>/comparison/recordings")
    parser.add_argument("--reference-id", default=REFERENCE_ID)
    parser.add_argument("--models", nargs="+", choices=["qwen", "xtts", "afro", "azure"], default=["qwen", "xtts", "afro", "azure"])
    parser.add_argument("--qwen-model-dir", type=Path, default=home / "models/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--xtts-model-dir", type=Path, default=home / "models/XTTS-v2")
    parser.add_argument("--afro-model-dir", type=Path, default=home / "models/afro-tts")
    parser.add_argument("--azure-voices", nargs="+", choices=["en-NG-AbeoNeural", "en-NG-EzinneNeural"], default=["en-NG-AbeoNeural"])
    parser.add_argument("--azure-key-file", type=Path, default=home / "keys/az_key.txt")
    parser.add_argument("--azure-resource-file", type=Path, default=home / "keys/az_res.txt")
    parser.add_argument("--azure-endpoint", help="Override the resource file with an HTTPS endpoint or region")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Positive pilot limit; omit for all utterances")
    parser.add_argument("--prepare-only", action="store_true", help="Copy originals/reference, write text and plan; do not synthesize")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report plan without writing files or calling models")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate the selected model outputs")
    parser.add_argument("--no-reuse-qwen", action="store_true", help="Generate fresh Qwen outputs instead of importing the matching pilot")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def atomic_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def atomic_json(path, content):
    atomic_bytes(path, (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode())


def write_manifest(path, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records.values())
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wav_info(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnframes() <= 0:
            raise ValueError(f"Empty recording: {path}")
        # Reading verifies that a truncated WAV cannot be treated as completed.
        expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
        if len(wav.readframes(wav.getnframes())) != expected:
            raise ValueError(f"Truncated recording: {path}")
        return {"sample_rate": wav.getframerate(), "generated_duration": wav.getnframes() / wav.getframerate()}


def valid_wav(path):
    try:
        wav_info(path)
        return True
    except (OSError, EOFError, wave.Error, ValueError):
        return False


def copy_audio(source, target):
    if target.is_file() and digest(source) == digest(target):
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_inputs(args):
    speaker = args.speaker_dir.expanduser().resolve()
    metadata = (args.metadata or speaker / "metadata.csv").expanduser().resolve()
    rows = read_csv(metadata)
    if not rows or any(not r.get("user_ids") for r in rows) or len({r["user_ids"] for r in rows}) != 1:
        raise ValueError("Metadata must contain exactly one speaker with nonempty user_ids")
    seen = set()
    reference = None
    for row in rows:
        audio_id = row.get("audio_ids", "")
        if not audio_id or audio_id in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in audio_id):
            raise ValueError(f"Unsafe or missing audio_ids: {audio_id!r}")
        if audio_id in seen:
            raise ValueError(f"Duplicate audio_ids: {audio_id}")
        seen.add(audio_id)
        if not row.get("transcript", "").strip():
            raise ValueError(f"Empty transcript: {audio_id}")
        if not row.get("chosen_audio_path"):
            raise ValueError(f"Missing chosen_audio_path: {audio_id}")
        source = (speaker / row["chosen_audio_path"]).resolve()
        if not source.is_relative_to(speaker):
            raise ValueError(f"Recording path escapes speaker directory: {audio_id}")
        wav_info(source)
        row["source"] = source
        if args.reference_id in {audio_id, row.get("ID"), source.stem}:
            reference = row
    if reference is None:
        raise ValueError(f"Reference {args.reference_id!r} missing from metadata")
    output = (args.output_dir or speaker / "comparison/recordings").expanduser().resolve()
    # Prevent accidental writes into the input tree or legacy model outputs.
    if output == speaker or output.is_relative_to(speaker / "recordings") or output.is_relative_to(speaker / "cloning"):
        raise ValueError("Output must be a separate comparison directory")
    return speaker, metadata, rows, reference, output


def fingerprint(model, row, reference, args):
    spec = {"version": 1, "model": model, "text": row["transcript"], "reference_sha256": reference["sha256"],
            "reference_text": reference["transcript"], "seed": args.seed, "language": "English", "device": args.device}
    if model.startswith("azure_"):
        spec.update(reference_sha256=None, reference_text=None, seed=None, device=None, format="riff-48khz-16bit-mono-pcm")
    else:
        model_dir = getattr(args, f"{model}_model_dir").expanduser().resolve()
        # Persistent scratch is mounted under different prefixes locally and on RCP.
        spec["model_directory"] = model_dir.name
        spec["config_sha256"] = digest(model_dir / "config.json") if (model_dir / "config.json").is_file() else None
        if (model_dir / "download.json").is_file():
            spec["download"] = json.loads((model_dir / "download.json").read_text())
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def legacy_qwen(speaker, reference):
    folder = speaker / "cloning/qwen3_tts_base_english"
    try:
        ref = json.loads((folder / "reference.json").read_text())
        if ref.get("reference_id") not in {reference["reference_id"], reference["audio_ids"], reference["source"].stem} or ref.get("transcript") != reference["transcript"].strip():
            return {}
        if ref.get("target_language") != "English":
            return {}
        return {r["audio_ids"]: r for r in read_csv(folder / "manifest.csv")
                if r.get("status") == "ok" and r.get("reference_id") == ref["reference_id"]}
    except (OSError, ValueError):
        return {}


def run(args, backend_factory=make_backend):
    speaker, metadata, all_rows, reference, output = load_inputs(args)
    reference = dict(reference, sha256=digest(reference["source"]), reference_id=args.reference_id)
    rows = all_rows[:args.limit] if args.limit else all_rows
    models = []
    for model in dict.fromkeys(args.models):
        models.extend([f"azure_{v}" for v in dict.fromkeys(args.azure_voices)] if model == "azure" else [model])
    print(f"Speaker {reference['user_ids']}: {len(rows)}/{len(all_rows)} texts; reference {args.reference_id}", flush=True)
    print(f"Models: {', '.join(models)}\nOutputs: {output}", flush=True)
    if args.dry_run:
        print("Input WAVs validated. Dry run: no files written, no API/model calls.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    records = {(r["audio_id"], r["model"]): r for r in read_csv(manifest_path)} if manifest_path.is_file() else {}
    atomic_json(output / "reference.json", {
        "reference_id": args.reference_id, "audio_ids": reference["audio_ids"], "speaker_id": reference["user_ids"],
        "transcript": reference["transcript"], "source": str(reference["source"]), "sha256": reference["sha256"],
    })
    atomic_json(output / "run.json", {
        "metadata": str(metadata), "selected_texts": len(rows), "total_texts": len(all_rows), "models": models,
        "seed": args.seed, "device": args.device, "qwen_language": "English", "xtts_language": "en",
        "azure_locale": "en-NG", "azure_is_stock_voice": True,
        "model_sources": {k: {"repo_id": v[0], "revision": v[2]} for k, v in MODEL_SPECS.items()},
    })
    legacy = {} if args.no_reuse_qwen or args.overwrite else legacy_qwen(speaker, reference)
    for row in rows:
        audio_id = row["audio_ids"]
        folder = output / audio_id
        folder.mkdir(exist_ok=True)
        copy_audio(row["source"], folder / "original.wav")
        copy_audio(reference["source"], folder / "reference.wav")
        atomic_bytes(folder / "text.txt", (row["transcript"] + "\n").encode())
        atomic_json(folder / "metadata.json", {
            **{k: v for k, v in row.items() if k != "source"},
            "source": str(row["source"]), "reference_id": args.reference_id,
            "reference_transcript": reference["transcript"], "reference_sha256": reference["sha256"],
        })
        for model in models:
            key = (audio_id, model)
            path = folder / f"{model}.wav"
            signature = fingerprint(model, row, reference, args)
            previous = records.get(key, {})
            imported = previous.get("provenance", "").startswith("legacy_qwen")
            if not args.overwrite and not (args.no_reuse_qwen and imported) and previous.get("status") == "ok" and previous.get("fingerprint") == signature and valid_wav(path):
                continue
            record = {"audio_id": audio_id, "model": model, "text": row["transcript"],
                      "reference_id": "" if model.startswith("azure_") else args.reference_id,
                      "output_path": str(path.relative_to(output)), "source_duration": row.get("duration", ""),
                      "status": "pending", "fingerprint": signature, "error": "", "provenance": ""}
            old = legacy.get(audio_id)
            # The pilot manifest contains /lightscratch paths; resolve using its local layout.
            old_path = speaker / "cloning/qwen3_tts_base_english/generated" / f"{audio_id}.wav"
            if model == "qwen" and old and old.get("target_text") == row["transcript"].strip() and valid_wav(old_path):
                copy_audio(old_path, path)
                record.update(wav_info(path), status="ok", provenance="legacy_qwen_pilot; seed unknown")
            records[key] = record
    write_manifest(manifest_path, records)
    if args.prepare_only:
        print(f"Prepared {len(rows)} text folders; matching Qwen pilot outputs imported. No synthesis requested.")
        return 0
    failures = 0
    for model in models:
        pending = [r for r in rows if records[(r["audio_ids"], model)]["status"] != "ok"]
        if not pending:
            print(f"{model}: all selected outputs complete", flush=True)
            continue
        backend = None
        try:
            backend = backend_factory(model, args, reference["source"], reference["transcript"].strip())
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"{model}: {message}", file=sys.stderr, flush=True)
            for row in pending:
                records[(row["audio_ids"], model)].update(status="error", error=message)
            failures += len(pending)
            write_manifest(manifest_path, records)
            continue
        try:
            for index, row in enumerate(pending, 1):
                record = records[(row["audio_ids"], model)]
                path = output / record["output_path"]
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    print(f"{model} [{index}/{len(pending)}] {row['audio_ids']}", flush=True)
                    seed = (args.seed + int(hashlib.sha256(row["audio_ids"].encode()).hexdigest()[:8], 16)) % 2 ** 32
                    backend.generate(row["transcript"], temporary, seed)
                    info = wav_info(temporary)
                    temporary.replace(path)
                    record.update(info, status="ok", error="", provenance="generated")
                except Exception as error:
                    failures += 1
                    record.update(status="error", error=f"{type(error).__name__}: {error}")
                    print(f"ERROR: {record['error']}", file=sys.stderr, flush=True)
                finally:
                    temporary.unlink(missing_ok=True)
                write_manifest(manifest_path, records)
        finally:
            if hasattr(backend, "close"):
                backend.close()
    print(f"Finished: {failures} errors in this run. Manifest: {manifest_path}", flush=True)
    return int(bool(failures))


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
