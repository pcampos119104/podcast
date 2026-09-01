FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG VIBEVOICE_COMMIT=952326ddb264062466a888cf32a5b2f4e803e16e

ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=4 \
    PYTHONUNBUFFERED=1

# The NGC image is recommended upstream, but this equivalent CUDA image avoids
# depending on the NGC registry at build time.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/vibevoice-community/VibeVoice.git /opt/vibevoice \
    && git -C /opt/vibevoice checkout "${VIBEVOICE_COMMIT}" \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e /opt/vibevoice \
    && python -m pip install --no-cache-dir pytest==9.1.1 \
    && (python -c "import flash_attn" || python -m pip install --no-cache-dir flash-attn --no-build-isolation)

ENV PYTHONPATH=/opt/vibevoice
