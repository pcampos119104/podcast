import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PODCAST_NAME = "Faramir Cast"
PORTUGUESE_MONTHS = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def weekly_title(generated_at: datetime) -> str:
    days_since_sunday = (generated_at.weekday() + 1) % 7
    week_start = generated_at.date() - timedelta(days=days_since_sunday)
    month = PORTUGUESE_MONTHS[week_start.month - 1]
    return f"Podcast da semana do dia {week_start.day} de {month}"


def inference_command(
    input_path: Path,
    raw_output_path: Path,
    voice_sample: Path | None,
    seed: int | None,
    cfg_scale: float,
    ddpm_steps: int,
) -> list[str]:
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
    command = [ffmpeg, "-nostdin", "-y", "-i", str(source_path)]
    if speed != 1.0:
        command.extend(["-filter:a", f"atempo={speed}"])
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    if output_path.suffix.lower() == ".mp3":
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
    parser = argparse.ArgumentParser(
        description="Gera um arquivo MP3 a partir de um roteiro em texto usando VibeVoice."
    )
    parser.add_argument("input", type=Path, help="Arquivo de entrada .txt em UTF-8.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Arquivo de saída .mp3 ou .wav (padrão: MP3 com timestamp em output/).",
    )
    parser.add_argument(
        "--voice-sample",
        type=Path,
        default=None,
        help="WAV de referência da voz (padrão: voz Carter incluída no container).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Taxa de amostragem em Hz (padrão: 24000 Hz do modelo).",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(16, 24, 32),
        default=16,
        help="Profundidade de bits PCM do WAV (padrão: 16).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.8,
        help="Velocidade da narração: menor que 1 desacelera (padrão: 0.8).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semente opcional da geração.")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.3,
        help="Escala CFG do VibeVoice (padrão: 1.3).",
    )
    parser.add_argument(
        "--ddpm-steps",
        type=int,
        default=10,
        help="Passos DDPM do VibeVoice (padrão: 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir um arquivo de saída existente.",
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
        parser.error("o arquivo de entrada deve ter extensão .txt")
    if not args.input.is_file():
        parser.error(f"arquivo de entrada não encontrado: {args.input}")
    if args.output.suffix.lower() not in {".mp3", ".wav"}:
        parser.error("o arquivo de saída deve ter extensão .mp3 ou .wav")
    if args.output.exists() and not args.overwrite:
        parser.error(f"arquivo de saída já existe: {args.output}; use --overwrite")
    if args.voice_sample is not None and not args.voice_sample.is_file():
        parser.error(f"amostra de voz não encontrada: {args.voice_sample}")
    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error("--sample-rate deve ser maior que zero")
    if not 0.5 <= args.speed <= 2.0:
        parser.error("--speed deve estar entre 0.5 e 2.0")
    if args.cfg_scale <= 0:
        parser.error("--cfg-scale deve ser maior que zero")
    if args.ddpm_steps <= 0:
        parser.error("--ddpm-steps deve ser maior que zero")
    try:
        text = args.input.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        parser.error("o arquivo de entrada deve usar codificação UTF-8")
    if not text:
        parser.error("o arquivo de entrada está vazio")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg não encontrado; instale-o para processar o áudio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Gerando áudio com VibeVoice 1.5B em CUDA...")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary_dir:
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
            parser.error("VibeVoice não conseguiu gerar o áudio")
        if not raw_output.is_file():
            parser.error("VibeVoice terminou sem criar o WAV esperado")
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
            parser.error("FFmpeg não conseguiu processar o áudio")

    print(f"Arquivo de áudio salvo em: {args.output}")


if __name__ == "__main__":
    main()
