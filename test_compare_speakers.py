"""CPU-only regression tests: python -m unittest -v test_compare_speakers.py."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from compare_speakers import parse_args, read_csv, run
from speech_backends import AzureBackend, azure_endpoint, azure_ssml


def wav(path, frames=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * frames)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.speaker = Path(self.tmp.name) / "speaker"
        self.output = Path(self.tmp.name) / "comparison"
        self.rows = [{"user_ids": "speaker", "audio_ids": key, "transcript": text,
                      "chosen_audio_path": f"recordings/{key}.wav", "duration": "0.01"}
                     for key, text in [("ref", "Reference."), ("target", "A & B < C.")]]
        for row in self.rows:
            wav(self.speaker / row["chosen_audio_path"])
        self.save_metadata()
        self.calls = []

    def save_metadata(self):
        with (self.speaker / "metadata.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def args(self, *extra):
        return parse_args(["--speaker-dir", str(self.speaker), "--output-dir", str(self.output),
                           "--reference-id", "ref", "--models", "xtts", *extra])

    def backend(self, name, args, reference, text):
        self.assertEqual(reference, self.speaker / "recordings/ref.wav")
        self.assertEqual(text, "Reference.")
        outer = self

        class Backend:
            def generate(self, text, path, seed):
                outer.calls.append((name, text, seed))
                wav(path)

        return Backend()

    def execute(self, args, factory=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run(args, factory or self.backend)

    def test_prepare_and_resume_and_text_change(self):
        self.assertEqual(self.execute(self.args("--prepare-only")), 0)
        self.assertFalse(self.calls)
        folder = self.output / "target"
        self.assertEqual((folder / "text.txt").read_text(), "A & B < C.\n")
        self.assertEqual((folder / "original.wav").read_bytes(), (self.speaker / "recordings/target.wav").read_bytes())
        self.assertEqual((folder / "reference.wav").read_bytes(), (self.speaker / "recordings/ref.wav").read_bytes())
        self.assertEqual(self.execute(self.args()), 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.execute(self.args()), 0)
        self.assertEqual(len(self.calls), 2)
        self.rows[1]["transcript"] = "Changed text."
        self.save_metadata()
        self.assertEqual(self.execute(self.args()), 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.calls[-1][1], "Changed text.")

    def test_reference_change_and_truncated_output_regenerate(self):
        self.execute(self.args())
        wav(self.speaker / "recordings/ref.wav", frames=200)
        self.execute(self.args())
        self.assertEqual(len(self.calls), 4)
        path = self.output / "target/xtts.wav"
        path.write_bytes(path.read_bytes()[:-2])
        self.execute(self.args())
        self.assertEqual(len(self.calls), 5)

    def test_partial_run_preserves_manifest_and_is_portable(self):
        self.execute(self.args())
        self.execute(self.args("--limit", "1", "--overwrite"))
        records = read_csv(self.output / "manifest.csv")
        self.assertEqual(len(records), 2)
        self.assertTrue(all(not Path(r["output_path"]).is_absolute() for r in records))
        self.assertEqual(len(self.calls), 3)
        moved = self.output.with_name("moved")
        shutil.move(self.output, moved)
        args = self.args()
        args.output_dir = moved
        self.execute(args)
        self.assertEqual(len(self.calls), 3)

    def test_generation_and_initialization_errors_retry_and_return_failure(self):
        def broken(*args):
            raise RuntimeError("model unavailable")
        self.assertEqual(self.execute(self.args(), broken), 1)
        self.assertTrue(all(r["status"] == "error" for r in read_csv(self.output / "manifest.csv")))
        outer = self

        class FailOne:
            def generate(self, text, path, seed):
                if text.startswith("Reference"):
                    raise RuntimeError("generation failed")
                outer.backend("xtts", None, outer.speaker / "recordings/ref.wav", "Reference.").generate(text, path, seed)

        self.assertEqual(self.execute(self.args(), lambda *a: FailOne()), 1)
        self.assertEqual([r["status"] for r in read_csv(self.output / "manifest.csv")], ["error", "ok"])
        self.assertEqual(self.execute(self.args()), 0)
        self.assertEqual(len(self.calls), 2)

    def test_dry_run_does_not_write(self):
        self.execute(self.args("--dry-run"))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)

    def test_invalid_metadata_is_rejected_before_writing(self):
        for change in [{"audio_ids": "../escape"}, {"chosen_audio_path": "../../escape.wav"},
                       {"user_ids": "other"}, {"transcript": ""}, {"audio_ids": "ref"}]:
            with self.subTest(change=change):
                old = self.rows[1].copy()
                self.rows[1].update(change)
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.execute(self.args())
                self.assertFalse(self.output.exists())
                self.rows[1] = old

    def test_legacy_qwen_import_and_explicit_fresh_generation(self):
        # AfriSpeech audio_ids differ from filenames (including this pilot's reference).
        (self.speaker / "recordings/ref.wav").rename(self.speaker / "recordings/reference-file.wav")
        self.rows[0]["chosen_audio_path"] = "recordings/reference-file.wav"
        self.save_metadata()
        legacy = self.speaker / "cloning/qwen3_tts_base_english"
        wav(legacy / "generated/target.wav")
        (legacy / "reference.json").write_text(json.dumps({"reference_id": "reference-file", "transcript": "Reference.", "target_language": "English"}))
        with (legacy / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_ids", "target_text", "reference_id", "status", "output_path"])
            writer.writeheader()
            writer.writerow({"audio_ids": "target", "target_text": "A & B < C.", "reference_id": "reference-file", "status": "ok", "output_path": "/lightscratch/old.wav"})
        def backend(name, args, reference, text):
            return self.backend(name, args, self.speaker / "recordings/ref.wav", text)
        self.execute(self.args("--models", "qwen", "--reference-id", "reference-file"), backend)
        self.assertEqual(len(self.calls), 1)
        records = read_csv(self.output / "manifest.csv")
        self.assertIn("legacy", records[1]["provenance"])
        self.execute(self.args("--models", "qwen", "--no-reuse-qwen", "--reference-id", "reference-file"), backend)
        self.assertEqual(len(self.calls), 2)
        self.execute(self.args("--models", "qwen", "--overwrite", "--no-reuse-qwen", "--reference-id", "reference-file"), backend)
        self.assertEqual(len(self.calls), 4)


class AzureTests(unittest.TestCase):
    def test_ssml_escapes_text_and_selects_nigerian_voice(self):
        text = 'A & B < C > D "test"'
        root = ElementTree.fromstring(azure_ssml(text, "en-NG-AbeoNeural"))
        voice = list(root)[0]
        self.assertEqual(voice.text, text)
        self.assertEqual(voice.attrib["name"], "en-NG-AbeoNeural")
        self.assertEqual(root.attrib["{http://www.w3.org/XML/1998/namespace}lang"], "en-NG")
        with self.assertRaises(ValueError):
            azure_ssml(text, "en-US-Other")

    def test_endpoint_forms(self):
        self.assertEqual(azure_endpoint("westeurope"), "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1")
        self.assertEqual(azure_endpoint("https://example.cognitiveservices.azure.com/"), "https://example.cognitiveservices.azure.com/tts/cognitiveservices/v1")
        self.assertEqual(azure_endpoint("https://example.cognitiveservices.azure.com/tts/cognitiveservices/v1"), "https://example.cognitiveservices.azure.com/tts/cognitiveservices/v1")
        for value in ["http://example.com", "https://example.com", "https://example.cognitiveservices.azure.com/wrong"]:
            with self.assertRaises(ValueError):
                azure_endpoint(value)

    def test_azure_headers_and_request_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            wav(path)
            payload = path.read_bytes()
            args = parse_args([])
            with patch.dict("os.environ", {"AZURE_SPEECH_KEY": "test-key", "AZURE_SPEECH_REGION": "westeurope"}):
                backend = AzureBackend(args, "en-NG-EzinneNeural")
            response = contextlib.nullcontext(io.BytesIO(payload))
            with patch("speech_backends.urlopen", return_value=response) as request:
                backend.generate("A & B", path, 0)
            sent = request.call_args.args[0]
            self.assertEqual(sent.get_header("Ocp-apim-subscription-key"), "test-key")
            self.assertEqual(sent.get_header("X-microsoft-outputformat"), "riff-48khz-16bit-mono-pcm")
            self.assertIn(b"A &amp; B", sent.data)


if __name__ == "__main__":
    unittest.main()
