import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PODCAST_NAME = "Faramir Cast"


def weekly_title(generated_at: datetime) -> str:
    """Return the episode title dated with the Sunday that starts its week."""
    days_since_sunday = (generated_at.weekday() + 1) % 7
    week_start = generated_at.date() - timedelta(days=days_since_sunday)
    return f"Weekly podcast - {week_start:%Y-%m-%d}"


def inference_command(
    input_path: Path,
    raw_output_path: Path,
    voice_sample: Path | None,
    seed: int | None,
    cfg_scale: float,
    ddpm_steps: int,
) -> list[str]:
    """Build the VibeVoice inference command for a temporary WAV output."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("vibevoice_infer.py")),
        "--input",
        str(input_path),
        "--output",
        str(raw_output_path),
        "--cfg-scale",
        str(cfg_scale),
        "--ddpm-steps",
        str(ddpm_steps),
    ]
    if voice_sample is not None:
        command.extend(["--voice-sample", str(voice_sample)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def transcode(
    ffmpeg: str,
    source_path: Path,
    output_path: Path,
    sample_rate: int | None,
    bit_depth: int,
    speed: float,
    generated_at: datetime,
) -> None:
    """Convert a generated WAV to MP3 or PCM WAV with the requested settings."""
    command = [ffmpeg, "-nostdin", "-y", "-i", str(source_path)]
    if speed != 1.0:
        command.extend(["-filter:a", f"atempo={speed}"])
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    if output_path.suffix.lower() == ".mp3":
        # MP3 embeds podcast metadata, while WAV preserves a PCM bit depth.
        command.extend(
            [
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                "-metadata",
                f"title={weekly_title(generated_at)}",
                "-metadata",
                f"artist={PODCAST_NAME}",
                "-metadata",
                f"album={PODCAST_NAME}",
                "-id3v2_version",
                "3",
            ]
        )
    else:
        command.extend(["-codec:a", f"pcm_s{bit_depth}le"])
    command.append(str(output_path))
    subprocess.run(command, check=True)


def main() -> None:
    """Run inference and transcode the resulting audio from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate an MP3 file from a text script using VibeVoice."
    )
    parser.add_argument("input", type=Path, help="Input .txt file encoded as UTF-8.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .mp3 or .wav file (default: timestamped MP3 in output/).",
    )
    parser.add_argument(
        "--voice-sample",
        type=Path,
        default=None,
        help="Reference voice WAV (default: Carter voice included in the container).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate in Hz (default: model's 24000 Hz).",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(16, 24, 32),
        default=16,
        help="WAV PCM bit depth (default: 16).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.8,
        help="Narration speed: below 1 slows it down (default: 0.8).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional generation seed.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.3,
        help="VibeVoice CFG scale (default: 1.3).",
    )
    parser.add_argument(
        "--ddpm-steps",
        type=int,
        default=10,
        help="VibeVoice DDPM steps (default: 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    args = parser.parse_args()

    generated_at = datetime.now()
    if args.output is None:
        args.output = PROJECT_ROOT / f"output/faramir-cast-{generated_at:%Y%m%d-%H%M%S}.mp3"
    else:
        args.output = args.output.resolve()
    args.input = args.input.resolve()
    if args.voice_sample is not None:
        args.voice_sample = args.voice_sample.resolve()

    if args.input.suffix.lower() != ".txt":
        parser.error("input file must have a .txt extension")
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    if args.output.suffix.lower() not in {".mp3", ".wav"}:
        parser.error("output file must have a .mp3 or .wav extension")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output file already exists: {args.output}; use --overwrite")
    if args.voice_sample is not None and not args.voice_sample.is_file():
        parser.error(f"voice sample not found: {args.voice_sample}")
    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error("--sample-rate must be greater than zero")
    if not 0.5 <= args.speed <= 2.0:
        parser.error("--speed must be between 0.5 and 2.0")
    if args.cfg_scale <= 0:
        parser.error("--cfg-scale must be greater than zero")
    if args.ddpm_steps <= 0:
        parser.error("--ddpm-steps must be greater than zero")
    try:
        text = args.input.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        parser.error("input file must use UTF-8 encoding")
    if not text:
        parser.error("input file is empty")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg not found; install it to process the audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Generating audio with VibeVoice 1.5B on CUDA...")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary_dir:
        # Keep the model's raw WAV isolated until FFmpeg produces the final file.
        raw_output = Path(temporary_dir) / "vibevoice.wav"
        command = inference_command(
            args.input,
            raw_output,
            args.voice_sample,
            args.seed,
            args.cfg_scale,
            args.ddpm_steps,
        )
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            parser.error("VibeVoice could not generate the audio")
        if not raw_output.is_file():
            parser.error("VibeVoice finished without creating the expected WAV")
        try:
            transcode(
                ffmpeg,
                raw_output,
                args.output,
                args.sample_rate,
                args.bit_depth,
                args.speed,
                generated_at,
            )
        except subprocess.CalledProcessError:
            parser.error("FFmpeg could not process the audio")

    print(f"Audio file saved to: {args.output}")


if __name__ == "__main__":
    main()
