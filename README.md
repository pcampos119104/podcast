# AI Podcast

Gerador local de episodios MP3 com VibeVoice 1.5B, executado em uma GPU NVIDIA por Docker Compose. O modelo suporta narracao longa e clonagem de voz por uma amostra WAV de referencia.

## Requisitos

- NVIDIA GPU com CUDA. O projeto foi configurado para uma RTX 3060 com 12 GB de VRAM.
- Docker Engine, Docker Compose e NVIDIA Container Toolkit.
- [just](https://github.com/casey/just).
- `rsync` somente para `just send_server`.
- Espaco em disco e conexao na primeira execucao. Os pesos do modelo ocupam cerca de 5,4 GB e ficam em `.cache/huggingface`.

Python, uv e FFmpeg nao sao necessarios no host. A imagem CUDA contem o runtime Python, o VibeVoice, o FFmpeg e as dependencias de teste.

O container fixa o fork comunitario do VibeVoice no commit `952326ddb264062466a888cf32a5b2f4e803e16e` e usa o checkpoint `vibevoice/VibeVoice-1.5B` na revisao `d374386b2a51d8e05277a64d85b296c89ec52376`.

### GPU no Docker

O Docker deve estar configurado com o NVIDIA Container Toolkit. Apos instalar o pacote conforme a [documentacao oficial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configure e reinicie o daemon:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirme a configuracao antes de construir a imagem:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Uso

Construa a imagem e confirme o acesso a GPU:

```bash
just build
just gpu_check
```

Crie um roteiro UTF-8 em ingles, por exemplo `roteiro.txt`, e gere o episodio:

```bash
just create roteiro.txt
```

O VibeVoice recebe todo o roteiro como um unico narrador e salva `output/faramir-cast-AAAAMMDD-HHMMSS.mp3`. O audio nativo e 24 kHz; por padrao, o MP3 e desacelerado para `0.8x`.

Na primeira geracao, o Hugging Face baixa o checkpoint para `.cache/huggingface`. As execucoes seguintes reutilizam esse cache. O diretorio e ignorado pelo Git.

### Voz

`just create` usa como referencia `assets/voices/nova_reference.wav`. O arquivo e encaminhado ao VibeVoice pelo argumento `--voice-sample`; a API usada pelo projeto recebe somente o WAV em `voice_samples`, portanto nao requer a transcricao da referencia.

Para usar outra voz de referencia, informe `--voice-sample` depois do roteiro. Esse argumento sobrescreve a referencia Nova:

```bash
just create roteiro.txt --voice-sample assets/voices/minha-voz.wav --output output/episodio.mp3
```

Ao executar `src/vibevoice_infer.py` diretamente sem `--voice-sample`, o padrao do VibeVoice continua sendo `en-Carter_man.wav` incluida no container.

Use somente amostras para as quais voce tem consentimento e direitos de uso.

### Opcoes

Os argumentos apos o roteiro sao encaminhados para a CLI de geracao:

```text
-o, --output ARQUIVO       Caminho do MP3 ou WAV de saida.
--voice-sample ARQUIVO     WAV de referencia para clonagem de voz.
--sample-rate HZ           Reamostra o audio de 24 kHz para a taxa informada.
--bit-depth {16,24,32}     Profundidade de bits quando a saida e WAV.
--speed VELOCIDADE         Velocidade final entre 0.5 e 2.0 (padrao: 0.8).
--seed N                   Semente opcional para reproducao.
--cfg-scale N              Escala CFG do VibeVoice (padrao: 1.3).
--ddpm-steps N             Passos DDPM do VibeVoice (padrao: 10).
--overwrite                Permite substituir um arquivo de saida existente.
```

### Envio ao servidor

O envio usa o `rsync` e a autenticacao SSH do host. Isso mantem chaves e configuracao SSH fora do container:

```bash
just send_server output/faramir-cast-AAAAMMDD-HHMMSS.mp3
```

O destino padrao e `homelab:/home/pcampos/audiobookshelf/faramir_cast/` e requer o alias SSH `homelab` configurado no host.

## Verificacao

```bash
just check
just gpu_check
```

Para a primeira prova de uso, comece com 1-2 minutos de texto e acompanhe o pico de VRAM reportado pelo container. O modelo e carregado em BF16 e prefere FlashAttention 2; se ele nao estiver disponivel, o container cai para SDPA, que pode ser mais lento e consumir mais memoria.

## Limitacoes e Uso Responsavel

O modelo completo e voltado para ingles e chines; este projeto usa ingles. A geracao longa pode levar mais tempo que a duracao do audio em uma RTX 3060. O cartao do modelo informa um aviso audivel de conteudo gerado por IA e uma marca-d'agua de procedencia. Revise o conteudo antes de publicar e nao use clonagem de voz sem consentimento explicito.
