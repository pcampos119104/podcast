from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_tts import PROJECT_ROOT, inference_command, transcode, weekly_title


def test_inference_command_passes_voice_sample_and_generation_options() -> None:
    voice_sample = PROJECT_ROOT / "assets" / "voices" / "alice.wav"
    command = inference_command(
        PROJECT_ROOT / "script.txt",
        PROJECT_ROOT / "output" / "raw.wav",
        voice_sample,
        seed=42,
        cfg_scale=1.3,
        ddpm_steps=10,
    )

    assert Path(command[0]).name.startswith("python")
    assert Path(command[1]).name == "vibevoice_infer.py"
    assert command[2:] == [
        "--input",
        str(PROJECT_ROOT / "script.txt"),
        "--output",
        str(PROJECT_ROOT / "output" / "raw.wav"),
        "--cfg-scale",
        "1.3",
        "--ddpm-steps",
        "10",
        "--voice-sample",
        str(voice_sample),
        "--seed",
        "42",
    ]


def test_weekly_title_uses_previous_sunday_in_iso_format() -> None:
    assert weekly_title(datetime(2026, 8, 26)) == "Weekly podcast - 2026-08-23"


def test_transcode_disables_ffmpeg_stdin(monkeypatch, tmp_path: Path) -> None:
    command: list[str] = []

    def run(captured_command: list[str], check: bool) -> None:
        assert check is True
        command.extend(captured_command)

    monkeypatch.setattr("generate_tts.subprocess.run", run)

    transcode(
        "ffmpeg",
        tmp_path / "source.wav",
        tmp_path / "episode.mp3",
        sample_rate=None,
        bit_depth=16,
        speed=1.0,
        generated_at=datetime(2026, 8, 26),
    )

    assert command[:3] == ["ffmpeg", "-nostdin", "-y"]
