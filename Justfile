set shell := ["bash", "-euo", "pipefail", "-c"]

upload_destination := "homelab:/home/pcampos/audiobookshelf/faramir_cast/"
voice_sample := "assets/voices/alice.wav"

# List available tasks.
default:
    @just --list

# Build the CUDA image used for generation and tests.
build:
    docker compose build vibevoice

# Validate the Compose configuration.
config:
    docker compose config -q

# Verify CUDA is available inside the generation image.
gpu_check:
    docker compose run --rm --entrypoint python vibevoice -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print(torch.cuda.get_device_name(0))'

# Run the unit tests inside the image.
test:
    docker compose run --rm --entrypoint python vibevoice -m pytest -p no:cacheprovider

# Run the fast configuration and unit-test checks.
check: config test

# Generate an episode from a UTF-8 script. Extra arguments are forwarded to the CLI.
create script="script.txt" *args:
    mkdir -p .cache/huggingface output
    docker compose run --rm vibevoice "{{ script }}" --voice-sample "{{ voice_sample }}" {{ args }}

# Send an existing audio file to the configured Audiobookshelf directory.
send_server arquivo:
    rsync -av --partial --mkpath "{{ arquivo }}" "{{ upload_destination }}"
