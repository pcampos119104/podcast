import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import soundfile
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


LANGUAGE_ID = "en"
DEFAULT_MAX_TEXT_TOKENS = 400
PAUSE_SECONDS = 0.25
PODCAST_NAME = "Faramir Cast"
DEFAULT_UPLOAD_DESTINATION = "homelab:/home/pcampos/audiobookshelf/faramir_cast/"
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


def count_tokens(model: ChatterboxMultilingualTTS, text: str) -> int:
    return model.tokenizer.text_to_tokens(text, language_id=LANGUAGE_ID).shape[1]


def split_words(
    model: ChatterboxMultilingualTTS, text: str, max_text_tokens: int
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in text.split():
        if count_tokens(model, word) > max_text_tokens:
            raise ValueError(f"uma palavra excede o limite de {max_text_tokens} tokens")

        candidate = f"{current} {word}".strip()
        if current and count_tokens(model, candidate) > max_text_tokens:
            chunks.append(current)
            current = word
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def split_paragraph(
    model: ChatterboxMultilingualTTS, paragraph: str, max_text_tokens: int
) -> list[str]:
    if count_tokens(model, paragraph) <= max_text_tokens:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    if len(sentences) == 1:
        return split_words(model, paragraph, max_text_tokens)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence_parts = (
            [sentence]
            if count_tokens(model, sentence) <= max_text_tokens
            else split_words(model, sentence, max_text_tokens)
        )
        for part in sentence_parts:
            candidate = f"{current} {part}".strip()
            if current and count_tokens(model, candidate) > max_text_tokens:
                chunks.append(current)
                current = part
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def split_text(
    model: ChatterboxMultilingualTTS, text: str, max_text_tokens: int
) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for part in split_paragraph(model, paragraph, max_text_tokens):
            candidate = f"{current}\n\n{part}".strip()
            if current and count_tokens(model, candidate) > max_text_tokens:
                chunks.append(current)
                current = part
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def weekly_title(generated_at: datetime) -> str:
    days_since_sunday = (generated_at.weekday() + 1) % 7
    week_start = generated_at.date() - timedelta(days=days_since_sunday)
    month = PORTUGUESE_MONTHS[week_start.month - 1]
    return f"Podcast da semana do dia {week_start.day} de {month}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera um arquivo MP3 a partir de um roteiro em texto."
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
        "--sample-rate",
        type=int,
        default=None,
        help="Taxa de amostragem em Hz (padrão: taxa nativa do modelo).",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=(16, 24, 32),
        default=16,
        help="Profundidade de bits PCM do WAV (padrão: 16).",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=DEFAULT_MAX_TEXT_TOKENS,
        help=(
            "Máximo de tokens por bloco (padrão: 400). Valores menores reduzem "
            "o uso de VRAM."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.8,
        help="Velocidade da narração: menor que 1 desacelera (padrão: 0.8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir um arquivo de saída existente.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Envia o áudio para o homelab após gerá-lo.",
    )
    parser.add_argument(
        "--upload-destination",
        default=DEFAULT_UPLOAD_DESTINATION,
        help=f"Destino do rsync (padrão: {DEFAULT_UPLOAD_DESTINATION}).",
    )
    args = parser.parse_args()

    generated_at = datetime.now()
    if args.output is None:
        args.output = Path(f"output/faramir-cast-{generated_at:%Y%m%d-%H%M%S}.mp3")

    if args.input.suffix.lower() != ".txt":
        parser.error("o arquivo de entrada deve ter extensão .txt")
    if not args.input.is_file():
        parser.error(f"arquivo de entrada não encontrado: {args.input}")
    if args.output.suffix.lower() not in {".mp3", ".wav"}:
        parser.error("o arquivo de saída deve ter extensão .mp3 ou .wav")
    if args.output.exists() and not args.overwrite:
        parser.error(f"arquivo de saída já existe: {args.output}; use --overwrite")
    if args.sample_rate is not None and args.sample_rate <= 0:
        parser.error("--sample-rate deve ser maior que zero")
    if args.max_text_tokens <= 0:
        parser.error("--max-text-tokens deve ser maior que zero")
    if not 0.5 <= args.speed <= 2.0:
        parser.error("--speed deve estar entre 0.5 e 2.0")

    ffmpeg = shutil.which("ffmpeg") if args.output.suffix.lower() == ".mp3" or args.speed != 1.0 else None
    if (args.output.suffix.lower() == ".mp3" or args.speed != 1.0) and ffmpeg is None:
        parser.error("FFmpeg não encontrado; instale-o para gerar MP3 ou ajustar a velocidade")
    if args.upload and shutil.which("rsync") is None:
        parser.error("rsync não encontrado; instale-o para usar --upload")

    try:
        text = args.input.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        parser.error("o arquivo de entrada deve usar codificação UTF-8")

    if not text:
        parser.error("o arquivo de entrada está vazio")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado: {device}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Carregando o modelo Chatterbox Multilingual...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    chunks = split_text(model, text, args.max_text_tokens)
    print(f"Gerando {len(chunks)} bloco(s) em inglês...")
    pause = torch.zeros(1, round(model.sr * PAUSE_SECONDS))
    audio_parts: list[torch.Tensor] = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"Gerando bloco {index}/{len(chunks)} ({count_tokens(model, chunk)} tokens)...")
        audio_parts.append(model.generate(chunk, language_id=LANGUAGE_ID))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if index < len(chunks):
            audio_parts.append(pause)
    wav = torch.cat(audio_parts, dim=1)

    sample_rate = args.sample_rate or model.sr
    if sample_rate != model.sr:
        print(f"Reamostrando de {model.sr} Hz para {sample_rate} Hz...")
        wav = torchaudio.functional.resample(wav, model.sr, sample_rate)

    subtype = f"PCM_{args.bit_depth}"
    if args.speed == 1.0:
        if args.output.suffix.lower() == ".wav":
            soundfile.write(args.output, wav.squeeze(0).numpy(), sample_rate, subtype=subtype)
        else:
            assert ffmpeg is not None
            with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary_dir:
                source_path = Path(temporary_dir) / "source.wav"
                soundfile.write(source_path, wav.squeeze(0).numpy(), sample_rate, subtype=subtype)
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(source_path),
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
                        str(args.output),
                    ],
                    check=True,
                )
    else:
        assert ffmpeg is not None
        print(f"Ajustando a velocidade da narração para {args.speed}x com FFmpeg...")
        with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary_dir:
            temporary_dir_path = Path(temporary_dir)
            source_path = temporary_dir_path / "source.wav"
            processed_path = temporary_dir_path / f"processed{args.output.suffix.lower()}"
            soundfile.write(source_path, wav.squeeze(0).numpy(), sample_rate, subtype=subtype)
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(source_path),
                "-filter:a",
                f"atempo={args.speed}",
            ]
            if args.output.suffix.lower() == ".mp3":
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
                command.extend(["-c:a", f"pcm_s{args.bit_depth}le"])
            command.append(str(processed_path))
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError:
                parser.error("FFmpeg não conseguiu ajustar a velocidade do áudio")
            processed_path.replace(args.output)
    print(f"Arquivo de áudio salvo em: {args.output}")
    if args.upload:
        print(f"Enviando arquivo para: {args.upload_destination}")
        try:
            subprocess.run(
                [
                    "rsync",
                    "-av",
                    "--partial",
                    "--mkpath",
                    str(args.output),
                    args.upload_destination,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            parser.error("rsync não conseguiu enviar o arquivo para o homelab")
        print("Arquivo enviado para o homelab.")


if __name__ == "__main__":
    main()
