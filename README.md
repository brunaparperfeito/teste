# AmazonTitles-1.3MM — Fine-tuning + RAG (Title → Description QA)

Este repositório contém um scaffold completo para treinar (SFT + LoRA/QLoRA) e servir um modelo
que responde perguntas sobre produtos usando como contexto o arquivo `trn.json` do dataset
**The AmazonTitles-1.3MM**.

Conteúdo:
- `scripts/prepare_data.py` — preparação (gera train/val jsonl)
- `scripts/train_sft.py` — script de fine-tuning (SFT + LoRA/QLoRA com TRL/PEFT)
- `scripts/inference.py` — inferência + RAG por título (rapidfuzz)
- `scripts/app_demo.py` — FastAPI demo
- `scripts/cli_demo.py` — CLI de demonstração
- `scripts/prompt_template.txt` — template de prompt (em português)
- `data/trn_sample.json` — amostra com 3 itens para testes rápidos
- `docs/video_script.md` — roteiro do vídeo demonstrativo

Siga o README para executar: instalar dependências, preparar dados, testar pré-treino, treinar e inferir.
