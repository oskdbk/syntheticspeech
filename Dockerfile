# RCP runs OCI images through Run:ai; build this image on a Linux/amd64
# machine with Docker, push it to registry.rcp.epfl.ch, then submit a Run:ai job.
# Pin the UID/GID at build time to the account/group that owns the mounted scratch.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.02-py3
FROM ${BASE_IMAGE}

ARG APP_UID
ARG APP_GID

ENV HOME=/home/clonejob \
    PATH=/opt/speech-env/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY requirements-inference.txt /tmp/requirements-inference.txt

# Isolate from the NGC prerelease torch and optional vision packages. XTTS needs
# torchaudio compiled for exactly the installed torch, not the latest PyPI wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/speech-env \
    && /opt/speech-env/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/speech-env/bin/python -m pip install --no-cache-dir \
       "torch==2.8.0" "torchaudio==2.8.0" \
       --index-url https://download.pytorch.org/whl/cu128 \
    && /opt/speech-env/bin/python -m pip install --no-cache-dir -r /tmp/requirements-inference.txt \
    && /opt/speech-env/bin/python -m pip check \
    && /opt/speech-env/bin/python -c "from qwen_tts import Qwen3TTSModel; from TTS.tts.models.xtts import Xtts" \
    && mkdir -p /home/clonejob /opt/syntheticspeech \
    && chown -R "${APP_UID}:${APP_GID}" /home/clonejob /opt/syntheticspeech \
    && chmod -R a+rX /opt/speech-env

WORKDIR /opt/syntheticspeech
COPY --chown=${APP_UID}:${APP_GID} *.py /opt/syntheticspeech/
COPY --chown=${APP_UID}:${APP_GID} cloning /opt/syntheticspeech/cloning

USER ${APP_UID}:${APP_GID}
CMD ["python", "/opt/syntheticspeech/compare_speakers.py", "--help"]
