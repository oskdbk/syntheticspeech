# Build this Linux/amd64 image for Run:ai and the H100 node pool.
# Match the image user to the account that owns the mounted scratch files.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ARG APP_UID
ARG APP_GID

ENV HOME=/home/clonejob \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN mkdir -p /home/clonejob /opt/syntheticspeech \
    && chown -R "${APP_UID}:${APP_GID}" /home/clonejob /opt/syntheticspeech \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --upgrade \
       --index-url https://download.pytorch.org/whl/cu128 \
       "torch==2.7.0" \
       "torchaudio==2.7.0" \
    && python -m pip install --no-cache-dir \
       "qwen-tts==0.1.1" \
       "soundfile==0.13.1" \
    && python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"

WORKDIR /opt/syntheticspeech

COPY --chown=${APP_UID}:${APP_GID} clone_speaker.py /opt/syntheticspeech/clone_speaker.py

USER ${APP_UID}:${APP_GID}

CMD ["python", "/opt/syntheticspeech/clone_speaker.py", "--help"]
