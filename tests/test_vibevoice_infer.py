from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibevoice_infer import load_processor, narration_script


def test_narration_script_preserves_text_as_one_speaker() -> None:
    assert narration_script("First paragraph.\n\nSecond paragraph.") == (
        "Speaker 1: First paragraph. Second paragraph."
    )


def test_load_processor_keeps_vibevoice_revision_away_from_qwen(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, list[str]]] = []

    def snapshot_download(
        model_id: str, revision: str, allow_patterns: list[str]
    ) -> str:
        calls.append((model_id, revision, allow_patterns))
        return "/cache/vibevoice"

    class Processor:
        @classmethod
        def from_pretrained(cls, path: str) -> str:
            return path

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    processor = load_processor(Processor, "vibevoice/VibeVoice-1.5B", "revision")

    assert processor == "/cache/vibevoice"
    assert calls == [
        ("vibevoice/VibeVoice-1.5B", "revision", ["preprocessor_config.json"])
    ]
