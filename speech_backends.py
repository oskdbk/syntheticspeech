"""Lazy inference backends; preparation and Azure need no neural dependencies."""
from __future__ import annotations

import os
import re
import time
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape, quoteattr


def azure_ssml(text: str, voice: str) -> str:
    if voice not in {"en-NG-AbeoNeural", "en-NG-EzinneNeural"}:
        raise ValueError("Choose a Nigerian English voice: en-NG-AbeoNeural or en-NG-EzinneNeural")
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-NG">'
            f'<voice name={quoteattr(voice)}>{escape(text)}</voice></speak>')


def azure_endpoint(resource: str) -> str:
    resource = resource.strip().rstrip("/")
    if re.fullmatch(r"[a-z][a-z0-9-]*", resource):
        return f"https://{resource}.tts.speech.microsoft.com/cognitiveservices/v1"
    parsed = urlsplit(resource)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Azure resource must be a region or an HTTPS Azure Speech endpoint")
    if not host.endswith((".cognitiveservices.azure.com", ".speech.microsoft.com")):
        raise ValueError("Azure endpoint must use an Azure Speech hostname")
    if parsed.path.rstrip("/") not in {"", "/cognitiveservices/v1", "/tts/cognitiveservices/v1"}:
        raise ValueError("Use a resource base URL or a synthesis URL ending in /cognitiveservices/v1")
    prefix = "/tts" if host.endswith(".cognitiveservices.azure.com") else ""
    return f"https://{parsed.netloc}{prefix}/cognitiveservices/v1"


class AzureBackend:
    def __init__(self, args, voice: str):
        self.voice = voice
        self.key = os.environ.get("AZURE_SPEECH_KEY") or os.environ.get("SPEECH_KEY")
        if not self.key and args.azure_key_file.is_file():
            self.key = args.azure_key_file.read_text().strip()
        resource = (args.azure_endpoint or os.environ.get("AZURE_SPEECH_ENDPOINT")
                    or os.environ.get("AZURE_SPEECH_REGION") or os.environ.get("SPEECH_REGION"))
        if not resource and args.azure_resource_file.is_file():
            resource = args.azure_resource_file.read_text().strip()
        if not self.key or not resource:
            raise ValueError("Azure needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION/ENDPOINT, or keys/az_key.txt and keys/az_res.txt")
        self.endpoint = azure_endpoint(resource)

    def generate(self, text: str, path: Path, seed: int) -> None:
        body = azure_ssml(text, self.voice).encode("utf-8")
        request = Request(self.endpoint, data=body, headers={
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml; charset=utf-8",
            "X-Microsoft-OutputFormat": "riff-48khz-16bit-mono-pcm",
            "User-Agent": "syntheticspeech-comparison",
        }, method="POST")
        for attempt in range(4):
            try:
                with urlopen(request, timeout=90) as response:
                    audio = response.read()
                path.write_bytes(audio)
                with wave.open(str(path), "rb") as wav:
                    if not wav.getnframes():
                        raise ValueError("Azure returned an empty WAV")
                return
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    # Do not persist request headers, resource URLs, or response bodies.
                    raise RuntimeError(f"Azure Speech HTTP {error.code}") from None
                delay = error.headers.get("Retry-After", "")
                time.sleep(min(float(delay), 30) if delay.isdigit() else 2 ** attempt)
            except URLError:
                if attempt == 3:
                    raise RuntimeError("Azure Speech connection failed after four attempts") from None
                time.sleep(2 ** attempt)


class NeuralBackend:
    def __init__(self, name: str, args, reference_audio: Path, reference_text: str):
        import torch

        self.name = name
        self.torch = torch
        self.device = args.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_dir = getattr(args, f"{name}_model_dir").expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory missing: {model_dir}; run download_speech_models.py first")
        if name == "qwen":
            from qwen_tts import Qwen3TTSModel

            dtype = torch.float32
            if self.device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = Qwen3TTSModel.from_pretrained(
                str(model_dir), device_map=self.device, dtype=dtype, local_files_only=True,
            )
            self.prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(reference_audio), ref_text=reference_text, x_vector_only_mode=False,
            )
        else:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            config = XttsConfig()
            config.load_json(str(model_dir / "config.json"))
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True, use_deepspeed=False)
            self.model.to(self.device).eval()
            self.config = config
            self.latent, self.embedding = self.model.get_conditioning_latents(
                audio_path=[str(reference_audio)], gpt_cond_len=config.gpt_cond_len,
                gpt_cond_chunk_len=config.gpt_cond_chunk_len, max_ref_length=config.max_ref_len,
                sound_norm_refs=config.sound_norm_refs,
            )

    def generate(self, text: str, path: Path, seed: int) -> None:
        import random
        import numpy as np
        import soundfile as sf

        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            if self.name == "qwen":
                wavs, rate = self.model.generate_voice_clone(
                    text=text, language="English", voice_clone_prompt=self.prompt, non_streaming_mode=True,
                )
                audio = wavs[0]
            else:
                settings = {key: getattr(self.config, key) for key in
                            ("temperature", "length_penalty", "repetition_penalty", "top_k", "top_p")}
                result = self.model.inference(
                    text, "en", self.latent, self.embedding, enable_text_splitting=True, **settings,
                )
                audio, rate = result["wav"], self.config.audio.output_sample_rate
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError("Model returned empty or non-finite audio")
            sf.write(str(path), audio, rate, subtype="PCM_16", format="WAV")

    def close(self) -> None:
        import gc

        del self.model
        for attr in ("prompt", "latent", "embedding"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def make_backend(name, args, reference_audio, reference_text):
    if name.startswith("azure_"):
        return AzureBackend(args, name.removeprefix("azure_"))
    return NeuralBackend(name, args, reference_audio, reference_text)
