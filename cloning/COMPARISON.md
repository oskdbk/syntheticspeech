# Same-speaker model comparison

`compare_speakers.py` synthesizes **all 419 metadata rows** for the selected
Igbo-accented English speaker unless `--limit` is supplied. The existing
`clone_speaker.py` and its outputs are preserved.

The fixed reference is the same recording used by the Qwen pilot:

- Filename/reference ID: `226dfbe5c1f2c06ddab271f9d10334ba`.
- Metadata `audio_ids`: `df7cf6d510841d0d7b8c525b1c4b4fbd` (these IDs differ).
- Transcript: `18 The function of IAPP seems to be the modulation or counter regulation of insulin secretion and function.`

All three cloning models receive this reference audio. Qwen also receives
its transcript. XTTS-v2 and Afro-TTS reuse cached conditioning latents computed
with their checkpoint's own reference-conditioning settings. The reference
utterance itself is included among the targets.

Azure uses the **stock Nigerian English voice** `en-NG-AbeoNeural` by default.
Pass `--azure-voices en-NG-AbeoNeural en-NG-EzinneNeural` to compare both voices
from the earlier `chosen/output/en-NG` experiment. These voices do not clone
the reference speaker or promise an Igbo-specific accent. Each request speaks
the exact target transcript, escaped as SSML text; old Azure WAVs contain a
different text and are not reused.

## Output layout

Default root, relative to `NAS_HOME`:

```text
data/afrispeech/chosen/speaker_3eb89635a92eb86a21014bad9155e77b/comparison/recordings/
├── reference.json
├── run.json
├── manifest.csv
└── <audio_ids>/                 # one folder for each target utterance
    ├── text.txt
    ├── metadata.json
    ├── original.wav            # original speaker saying this target text
    ├── reference.wav           # identical fixed cloning example in every folder
    ├── qwen.wav
    ├── xtts.wav
    ├── afro.wav
    ├── azure_en-NG-AbeoNeural.wav
    └── azure_en-NG-EzinneNeural.wav  # if requested
```

Preparation writes originals, reference, text, metadata, and pending manifest
rows. Model WAVs appear as they complete. Originals and the reference are
byte-for-byte copies. Generated WAVs preserve each model's natural duration
and sample rate: there is no duration matching or time stretching. XTTS text
splitting is enabled for long transcripts. Azure requests use 48 kHz PCM WAV.

The manifest records text, model, reference ID, source/generated duration,
sample rate, status/error, fingerprint, and provenance. Output paths are
relative to the root so the dataset can move between local and cluster mounts.
Azure entries have an empty reference ID because they use a stock voice.
Matching successful outputs resume without loading that model. Changed text,
reference audio, model config/revision, seed, or explicitly chosen device
invalidate the corresponding result. Missing/truncated WAVs are retried.
Each output and manifest update is atomic; failures are retained and the
command exits nonzero if synthesis failed. Use one writer per output directory.

The three matching Qwen pilot WAVs are imported from the existing manifest,
using local paths even if that manifest contains `/lightscratch` paths. They
are labeled `legacy_qwen_pilot; seed unknown`. Use `--no-reuse-qwen` to generate
fresh versions, including replacing previously imported pilot entries.
`--overwrite` regenerates the selected models and rows. Other completed rows
remain in the manifest when using `--limit` or selecting only some models.
Target-specific seeds are stable across partial runs; GPU bitwise
reproducibility is not guaranteed.

## Download and prepare

Run these commands from the repository root. On the host, `NAS_HOME` defaults
to the workspace containing `repos/` and `data/`. In a container, set it to
the mounted workspace, for example `/lightscratch/users/dabkowsk`.

```bash
python download_speech_models.py
python download_speech_models.py --check-only
python compare_speakers.py --dry-run
python compare_speakers.py --prepare-only --azure-voices en-NG-AbeoNeural en-NG-EzinneNeural
```

Download requires `huggingface-hub==0.36.2` and network access. It uses the
existing `keys/hf_key.txt` when present, or Hugging Face's normal token lookup.
Only inference files and model documentation/licenses are downloaded; training
assets such as `dvae.pth` are unnecessary for these inference APIs. Downloads
resume through Hugging Face's cache. The pinned sources are:

| Model | Local folder under `models/` | Hugging Face revision |
| --- | --- | --- |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | `XTTS-v2` | `6c2b0d75eae4b7047358e3b6bd9325f857d43f77` |
| [Afro-TTS](https://huggingface.co/intronhealth/afro-tts) | `afro-tts` | `f99ba77006ffcf2fb66ed12d7093832b552a3d26` |

The two model files total about 7.5 GB. Each downloaded folder includes
`download.json` with its revision. Qwen continues to use the existing
`models/Qwen3-TTS-12Hz-1.7B-Base` download. Override paths with
`--qwen-model-dir`, `--xtts-model-dir`, or `--afro-model-dir`.

## Pilot and complete run

First test actual checkpoint loading and generation in the new image:

```bash
python /opt/syntheticspeech/compare_speakers.py --models qwen xtts afro --limit 1 --no-reuse-qwen
```

Then generate every text with all three models and the default Azure voice:

```bash
python /opt/syntheticspeech/compare_speakers.py
```

For both Azure voices, use:

```bash
python /opt/syntheticspeech/compare_speakers.py --azure-voices en-NG-AbeoNeural en-NG-EzinneNeural
```

Models load sequentially to avoid keeping all checkpoints on the GPU at once.
Neural inference loads local files and works with the image's default
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The explicit download command
enables Hugging Face networking for its downloads. Azure always needs network
access, regardless of Hugging Face offline settings. Use `--models qwen xtts
afro` for an entirely local run, or `--models azure` to call only Azure.

Azure credentials default to the existing `NAS_HOME/keys/az_key.txt` and
`NAS_HOME/keys/az_res.txt`. The resource file can contain a region, regional
synthesis endpoint, or resource-specific HTTPS URL. Resource-specific URLs
use `/tts/cognitiveservices/v1`; regional URLs use `/cognitiveservices/v1`,
following the [Azure Speech REST interface](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech).
Overrides: `AZURE_SPEECH_KEY`/`SPEECH_KEY`, `AZURE_SPEECH_ENDPOINT`,
`AZURE_SPEECH_REGION`/`SPEECH_REGION`, `--azure-key-file`,
`--azure-resource-file`, and `--azure-endpoint`. Credentials are read at
runtime and excluded from output metadata and the Docker build context.

## Local image build changes

**Changed: `repos/syntheticspeech/Dockerfile`.**

- Honors the existing `BASE_IMAGE` compose build argument.
- Creates `/opt/speech-env` to isolate the inference packages from NGC's
  prerelease PyTorch and optional vision packages.
- Installs matching `torch==2.8.0` and `torchaudio==2.8.0` CUDA 12.8 wheels.
  This increases image size and the initial build download.
- Installs `requirements-inference.txt`: `qwen-tts==0.1.1`,
  **`coqui-tts==0.27.4`**, `transformers==4.57.3`,
  `huggingface-hub==0.36.2`, `numpy==1.26.4`, and `soundfile==0.13.1`.
  The maintained [Coqui fork](https://github.com/idiap/coqui-ai-TTS/tree/v0.27.4)
  supplies the `TTS` import namespace and supports Python 3.12. Do not install
  the older `TTS` distribution into the same environment.
- Installs system `sox` and `libsndfile1`, runs `pip check` and Qwen/XTTS import
  checks during the build, and copies the new Python scripts.
- Defaults to the comparison runner's help. The original Qwen script is still
  included and can be called explicitly.

**Unchanged: `LiGHT-cluster-template/installation/docker-amd64-cuda/compose2.yaml`**
and the template's generic Dockerfiles/requirements. Its existing
`/mnt/d/syntheticspeech` context and `.env` build arguments are preserved. Copy
the updated speech repository, including the new requirements file, into
that local build context before building. From that template directory:

```bash
docker compose --env-file .env -f compose2.yaml build syntheticspeech
```

Model weights, data, and keys stay on mounted scratch, outside the image.
Set `NAS_HOME` in the GPU job to the mounted user directory. To run locally
with Docker, explicitly bind-mount the workspace and set `NAS_HOME`; the build
compose file alone does not configure runtime mounts or GPU access.

## Validation performed on 2026-10-29

- Ten CPU-only regression tests pass with
  `python -m unittest -v test_compare_speakers.py`.
- All 419 original WAVs validated; all 419 comparison folders prepared.
- Both pinned model downloads completed and pass `--check-only`.
- Three matching Qwen pilot WAVs imported.
- Six live Azure WAVs generated successfully: first three transcripts, both
  Nigerian voices, 48 kHz mono PCM. These are in the main comparison folders.
- Full Python dependency resolution succeeded on Python 3.12 using pip's
  dry-run resolver, including torch/torchaudio 2.8.0.
- Neural checkpoint loading/generation and the Docker build remain to be
  tested in the new image; the host has no installed neural inference stack.
