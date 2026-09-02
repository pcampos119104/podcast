# AI Podcast

Local MP3 podcast generator using VibeVoice 1.5B on an NVIDIA GPU through Docker Compose. The model supports long narration and voice cloning from a reference WAV sample.

## Requirements

- NVIDIA GPU with CUDA. The project is configured for an RTX 3060 with 12 GB of VRAM.
- Docker Engine, Docker Compose, and NVIDIA Container Toolkit.
- [just](https://github.com/casey/just).
- `rsync` only for `just send_server`.
- Disk space and an internet connection for the first run. Model weights use about 5.4 GB and are stored in `.cache/huggingface`.

Python, uv, and FFmpeg are not required on the host. The CUDA image contains the Python runtime, VibeVoice, FFmpeg, and test dependencies.

The container pins the community VibeVoice fork to commit `952326ddb264062466a888cf32a5b2f4e803e16e` and uses checkpoint `vibevoice/VibeVoice-1.5B` at revision `d374386b2a51d8e05277a64d85b296c89ec52376`.

### Docker GPU Setup

Docker must be configured with the NVIDIA Container Toolkit. After installing the package according to the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configure and restart the daemon:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirm the configuration before building the image:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Usage

Build the image and confirm GPU access:

```bash
just build
just gpu_check
```

Create an English UTF-8 script, such as `script.txt`, and generate the episode:

```bash
just create script.txt
```

VibeVoice receives the complete script as a single narrator and saves `output/faramir-cast-YYYYMMDD-HHMMSS.mp3`. Native audio is 24 kHz; by default, the MP3 is slowed to `0.8x`. Its title metadata is set to `Weekly podcast - YYYY-MM-DD`, using the Sunday that starts the episode's week.

On the first generation, Hugging Face downloads the checkpoint to `.cache/huggingface`. Later runs reuse this cache. The directory is ignored by Git.

### Voice

`just create` uses `assets/voices/alice.wav` as its reference. The file is passed to VibeVoice through `--voice-sample`; the project API only receives the WAV in `voice_samples`, so it does not require a reference transcript.

To use another reference voice, pass `--voice-sample` after the script. This overrides the default Alice reference:

```bash
just create script.txt --voice-sample assets/voices/my-voice.wav --output output/episode.mp3
```

When running `src/vibevoice_infer.py` directly without `--voice-sample`, VibeVoice still defaults to `en-Carter_man.wav`, included in the container.

Only use samples for which you have consent and usage rights.

### Options

Arguments after the script are passed through to the generation CLI:

```text
-o, --output FILE          Output MP3 or WAV path.
--voice-sample FILE        Reference WAV for voice cloning.
--sample-rate HZ           Resample 24 kHz audio to the specified rate.
--bit-depth {16,24,32}     Bit depth when the output is WAV.
--speed SPEED              Final speed from 0.5 to 2.0 (default: 0.8).
--seed N                   Optional seed for reproducibility.
--cfg-scale N              VibeVoice CFG scale (default: 1.3).
--ddpm-steps N             VibeVoice DDPM steps (default: 10).
--overwrite                Allow an existing output file to be replaced.
```

### Upload to Server

Uploads use host `rsync` and SSH authentication. This keeps SSH keys and configuration outside the container:

```bash
just send_server output/faramir-cast-YYYYMMDD-HHMMSS.mp3
```

The default destination is `homelab:/home/pcampos/audiobookshelf/faramir_cast/` and requires the `homelab` SSH alias to be configured on the host.

## Verification

```bash
just check
just gpu_check
```

For an initial smoke test, start with 1-2 minutes of text and monitor the container's peak VRAM. The model loads in BF16 and prefers FlashAttention 2; when unavailable, the container falls back to SDPA, which may be slower and use more memory.

## Limitations and Responsible Use

The full model is intended for English and Chinese; this project generates English only. Long generation can take longer than the audio duration on an RTX 3060. The model card states that generated content contains an audible AI disclosure and provenance watermark. Review content before publishing, and do not clone voices without explicit consent.
