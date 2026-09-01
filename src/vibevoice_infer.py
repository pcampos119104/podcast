import argparse
from importlib import import_module
from pathlib import Path
import time
from typing import Any

MODEL_ID = "vibevoice/VibeVoice-1.5B"
MODEL_REVISION = "d374386b2a51d8e05277a64d85b296c89ec52376"
DEFAULT_VOICE_SAMPLE = "/opt/vibevoice/demo/voices/en-Carter_man.wav"
SAMPLE_RATE = 24000


def narration_script(text: str) -> str:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return f"Speaker 1: {normalized}"


def load_processor(processor_class: Any, model_id: str, revision: str) -> Any:
    from huggingface_hub import snapshot_download

    processor_path = snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=["preprocessor_config.json"],
    )
    # The fork forwards keyword arguments to Qwen's tokenizer. Loading the
    # VibeVoice config from a local snapshot keeps its revision separate.
    return processor_class.from_pretrained(processor_path)


def load_model(model_id: str, revision: str, torch: Any) -> tuple[Any, Any]:
    model_module = import_module("vibevoice.modular.modeling_vibevoice_inference")
    processor_module = import_module("vibevoice.processor.vibevoice_processor")
    model_class = model_module.VibeVoiceForConditionalGenerationInference
    processor_class = processor_module.VibeVoiceProcessor

    processor = load_processor(processor_class, model_id, revision)
    try:
        model = model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    except Exception as error:
        print(f"FlashAttention 2 indisponível ({error}); usando SDPA.")
        model = model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
    model.eval()
    return processor, model


def main() -> None:
    torch = import_module("torch")
    parser = argparse.ArgumentParser(
        description="Gera WAV de narrador único com VibeVoice 1.5B em CUDA."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice-sample", type=Path, default=Path(DEFAULT_VOICE_SAMPLE))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--ddpm-steps", type=int, default=10)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA não está disponível no container")
    if not args.input.is_file():
        parser.error(f"roteiro não encontrado: {args.input}")
    if not args.voice_sample.is_file():
        parser.error(f"amostra de voz não encontrada: {args.voice_sample}")
    if args.cfg_scale <= 0:
        parser.error("--cfg-scale deve ser maior que zero")
    if args.ddpm_steps <= 0:
        parser.error("--ddpm-steps deve ser maior que zero")

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        parser.error("o arquivo de entrada está vazio")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(f"Carregando {args.model} ({args.model_revision}) em CUDA/BF16...")
    processor, model = load_model(args.model, args.model_revision, torch)
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    inputs = processor(
        text=[narration_script(text)],
        voice_samples=[[str(args.voice_sample)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {
        name: value.to("cuda") if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }

    torch.cuda.reset_peak_memory_stats()
    started_at = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            is_prefill=True,
        )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started_at

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        parser.error("o modelo não retornou áudio")
    audio = outputs.speech_outputs[0]
    audio_duration = audio.shape[-1] / SAMPLE_RATE
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Áudio gerado: {audio_duration:.1f}s")
    print(f"Tempo de geração: {elapsed:.1f}s (RTF {elapsed / audio_duration:.2f}x)")
    print(f"Pico de VRAM alocado: {peak_memory_gib:.2f} GiB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processor.save_audio(audio, output_path=str(args.output))
    print(f"WAV bruto salvo em: {args.output}")


if __name__ == "__main__":
    main()
