# RCP runs OCI images through Run:ai; build this image on a Linux/amd64
# machine with Docker, push it to registry.rcp.epfl.ch, then submit a Run:ai job.
# Pin the UID/GID at build time to the account/group that owns the mounted scratch.
FROM nvcr.io/nvidia/pytorch:25.02-py3

ARG APP_UID
ARG APP_GID

ENV HOME=/home/clonejob \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN mkdir -p /home/clonejob /opt/syntheticspeech \
    && chown -R "${APP_UID}:${APP_GID}" /home/clonejob /opt/syntheticspeech \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       "qwen-tts==0.1.1" \
       "soundfile==0.13.1"

WORKDIR /opt/syntheticspeech
COPY --chown=${APP_UID}:${APP_GID} clone_speaker.py /opt/syntheticspeech/clone_speaker.py

USER ${APP_UID}:${APP_GID}
CMD ["python", "/opt/syntheticspeech/clone_speaker.py", "--help"]
