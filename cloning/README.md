# Qwen3-TTS voice-cloning pilot

## Fixed reference

The selected AfriSpeech recording and its transcript are recorded in the
experiment's `reference.json`. The selected recording is:

- Reference ID: `226dfbe5c1f2c06ddab271f9d10334ba`
- Speaker ID: `3eb89635a92eb86a21014bad9155e77b`
- Source manifest: `data/afrispeech/chosen/speaker_3eb89635a92eb86a21014bad9155e77b/metadata.csv`
- Source recording (relative to the speaker folder):
  `recordings/train/fe45cd2b-adbf-4f36-93a4-2335531af4fd/226dfbe5c1f2c06ddab271f9d10334ba.wav`
- Reference transcript: `18 The function of IAPP seems to be the modulation or counter regulation of insulin secretion and function.`
- Recorded duration: about 9.042 seconds

The script synthesizes the English transcript of every row in this speaker's
metadata, including the reference utterance. It uses the reference transcript
and audio once to create a reusable Qwen voice-clone prompt.

## Files and persistent outputs

The input metadata and recordings remain in the speaker folder. Generated
audio and run records are written alongside them, under:

```text
data/afrispeech/chosen/speaker_3eb89635a92eb86a21014bad9155e77b/cloning/qwen3_tts_base_english/
├── reference.json
├── generated/<audio_ids>.wav
└── manifest.csv
```

The manifest records each source ID and transcript, source duration, reference
ID, model output sample rate, generated duration, and status/error. It is
atomically rewritten after each utterance so completed outputs survive job
interruption. Rerunning resumes successful entries; pass `--overwrite` to
regenerate them.

The script does not set or enforce `target_len`, and it does not time-stretch,
pad, or crop generated audio. Each WAV keeps the duration and speaking pace
Qwen produces for that transcript and reference voice. `source_duration` is
recorded only to compare lengths afterward.

## Local run

From the workspace root, after installing the Dockerfile dependencies in an
environment with the model packages and a compatible GPU:

```bash
python repos/syntheticspeech/clone_speaker.py --limit 3
```

Omit `--limit` to generate all rows. Defaults use the local workspace paths;
on RCP, set `NAS_HOME=/lightscratch/users/<GASPAR>` and use the Run:ai PVC as
described below.

## RCP / Run:ai notes

The supplied [LiGHT RCP guide](https://lighters-playground.github.io/LiGHT-doc/clusters/rcp/rcp/)
uses Run:ai and the `light-scratch` PVC for persistent data. RCP jobs should
read the model/data from that PVC and write outputs under the same mounted
scratch path. Do not write generated audio to the container's writable layer.

The guide's Docker workflow uses the LiGHT cluster template: build its generic
CUDA/PyTorch image, then the user image with the matching scratch UID/GID; push
both to the EPFL registry. This project Dockerfile captures the inference
dependencies and matching-user pattern. Before making an RCP image, integrate
it with the template's user-image build so the cluster's startup credential
hooks and file ownership conventions are retained. Build on a Linux/amd64
Docker host, not as part of a GPU job.

Example Run:ai submission after the image has been built and pushed (replace
the registry image and group ID with the values for your RCP project/account):

```bash
runai submit \
  --name qwen3-voice-clone \
  --image registry.rcp.epfl.ch/<project>/<image>:<tag> \
  --pvc light-scratch:/lightscratch \
  --large-shm \
  -e NAS_HOME=/lightscratch/users/<GASPAR> \
  -e HF_TOKEN_PATH=/lightscratch/users/<GASPAR>/keys/hf_key.txt \
  --run-as-gid <LIGHT_SCRATCH_GID> \
  --node-pool h100 \
  --gpu 1 \
  -- python /opt/syntheticspeech/clone_speaker.py --limit 3
```

`HF_TOKEN_PATH` is included for the credential hook documented by LiGHT, but
this script loads the already-downloaded model with `local_files_only=True`,
so inference itself does not need the Hugging Face token or network access.
The PVC mount plus `NAS_HOME` keeps the model, inputs, and outputs on persistent
scratch. If the template image is used, its supported optional hooks include
the W&B key, Git config/credentials, and VS Code server directory; this job
does not need those hooks.

The RCP guide asks users to set up their Run:ai project and push images to
`registry.rcp.epfl.ch`; it also warns that GPU jobs are subject to the cluster's
GPU-idle limit. Confirm the correct project, image path, UID/GID, and node pool
before submission.
