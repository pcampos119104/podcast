# AI Podcast

Gerador local de episódios MP3 com Chatterbox Multilingual, usando português brasileiro.

## Requisitos

- Python 3.14.7
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` para codificar o MP3 e ajustar a velocidade da narração
- `rsync` para enviar o episódio ao homelab (somente com `--upload`)
- Espaço em disco e conexão para o primeiro download do modelo

O script usa CUDA automaticamente quando `torch.cuda.is_available()` for verdadeiro. Sem uma GPU CUDA compatível, ele tenta executar em CPU. O Chatterbox Multilingual tem aproximadamente 500M parâmetros, portanto a execução em CPU pode ser lenta e exigir bastante memória.

O Chatterbox usa `pt` como identificador de idioma para português; este é também o identificador oficial do modelo dedicado a português brasileiro.

## Instalação

```bash
uv sync
```

As dependências de PyTorch são declaradas pelo próprio `chatterbox-tts`. Para instalações CUDA que precisem de uma variante específica do PyTorch, siga as instruções atuais em [PyTorch Start Locally](https://pytorch.org/get-started/locally/) e então execute `uv sync` novamente.

O projeto também restringe `setuptools` a uma versão anterior à 81: a versão atual de `resemble-perth`, dependência do Chatterbox responsável pela marca d'água, ainda depende de `pkg_resources`, removido em versões mais recentes do `setuptools`.

## Execução

Crie um arquivo de roteiro em texto UTF-8, por exemplo `roteiro.txt`, e execute:

```bash
uv run python src/generate_tts.py roteiro.txt
```

Por padrão, o áudio é salvo em `output/faramir-cast-AAAAMMDD-HHMMSS.mp3`, com velocidade `0.8x`. O título gravado no MP3 é calculado usando o domingo mais recente, por exemplo, `Podcast da semana do dia 23 de agosto`.

Para gerar e enviar o episódio ao Audiobookshelf, use:

```bash
uv run python src/generate_tts.py roteiro.txt --upload
```

O envio usa o alias SSH `homelab` configurado localmente e cria, se necessário, o destino `homelab:/home/pcampos/audiobookshelf/faramir_cast/`.

### Opções de saída

```text
-o, --output ARQUIVO    Caminho do MP3 ou WAV de saída.
--sample-rate HZ        Reamostra o áudio para a taxa informada.
--bit-depth {16,24,32} Profundidade de bits PCM do WAV.
--speed VELOCIDADE      Velocidade da narração (padrão: 0.8).
--overwrite             Permite substituir um arquivo existente.
--upload                Envia o áudio ao Audiobookshelf com rsync.
--upload-destination    Destino rsync alternativo.
```

Exemplo com configurações explícitas:

```bash
uv run python src/generate_tts.py roteiro.txt \
  --output output/episodio.mp3 \
  --sample-rate 24000 \
  --speed 0.9 \
  --upload
```

Na primeira execução, o Chatterbox baixa os arquivos do modelo. O Audiobookshelf deve apontar sua biblioteca de podcasts para `/home/pcampos/audiobookshelf` no servidor.
