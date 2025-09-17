# Roteiro do Vídeo de Demonstração (3–5 min)

1) Abertura (10s)
- Objetivo: mostrar um modelo que responde perguntas sobre títulos de produtos da Amazon usando o contexto do `trn.json`.

2) Dataset e Preparação (40s)
- Explicar rapidamente o The AmazonTitles-1.3MM.
- Mostrar `trn.json` (colunas `title` e `content`).
- Rodar `prepare_data.py` e exibir quantos exemplos de treino/val foram gerados.

3) Teste Pré-Treinamento (40s)
- Rodar `inference.py` sem `--peft` com uma pergunta e um título.
- Mostrar a qualidade base.

4) Treinamento (60–90s)
- Explicar SFT + LoRA/QLoRA e por que escolhemos (VRAM e rapidez).
- Mostrar comando `train_sft.py` e principais hiperparâmetros.

5) Inferência com Adapter + RAG (60s)
- Rodar `inference.py` com `--peft outputs/...`.
- Comparar resposta agora (melhor estrutura/aderência ao contexto).

6) API/CLI (30s)
- Subir FastAPI e fazer uma requisição POST.
- Mostrar CLI `cli_demo.py`.

7) Encerramento (10s)
- Reforçar reprodutibilidade, próximos passos (avaliação automática, melhores filtros de títulos, modelos maiores).
