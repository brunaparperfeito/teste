import json, argparse, os, random, re
from pathlib import Path

QUESTION_TEMPLATES = [
    "Faça um resumo detalhado da descrição.",
    "Quais são as principais características?",
    "Para que serve este produto?",
    "Liste os destaques e benefícios do produto.",
    "Quais especificações técnicas importantes ele possui?",
    "O que este produto inclui?",
    "Quem é o público ideal para este produto?",
]

INSTRUCTION_HEADER = open(os.path.join(os.path.dirname(__file__), "prompt_template.txt"), "r", encoding="utf-8").read()

def build_input(title, question, context):
    return INSTRUCTION_HEADER.format(title=title.strip(), question=question.strip(), context=context.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="caminho para trn.json")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-samples", type=int, default=None, help="útil para protótipos")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)  # espera lista de dicts com chaves 'title' e 'content'

    examples = []
    for row in data:
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        if not title or not content:
            continue
        # limpeza básica
        content = re.sub(r"\s+\n", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        q = random.choice(QUESTION_TEMPLATES)
        inp = build_input(title, q, content)
        tgt = content
        examples.append({"text": inp + tgt})

    if args.max_samples:
        examples = examples[: args.max_samples]

    random.seed(42)
    random.shuffle(examples)

    val_size = min(args.val_size, int(0.1 * len(examples)) or 1)
    val = examples[:val_size]
    train = examples[val_size:]

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as ftr:
        for ex in train:
            ftr.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(args.val_out, "w", encoding="utf-8") as fva:
        for ex in val:
            fva.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Gerados {len(train)} exemplos de treino e {len(val)} de validação.")

if __name__ == "__main__":
    main()
