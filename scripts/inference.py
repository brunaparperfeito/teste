import argparse, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from rapidfuzz import process, fuzz

PROMPT = open(os.path.join(os.path.dirname(__file__), "prompt_template.txt"), "r", encoding="utf-8").read()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_by_title(title, data, limit=5):
    titles = [row.get("title","") for row in data]
    matches = process.extract(title, titles, scorer=fuzz.WRatio, limit=limit)
    # matches: list of tuples (match, score, index)
    return [(m[0], m[2], m[1]) for m in matches]

def build_prompt(title, question, context):
    return PROMPT.format(title=title, question=question, context=context)

def answer(model_name, peft_path, title, question, data_path, max_new_tokens=256, temperature=0.2, top_p=0.9, qlora=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if qlora else None

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config,
    )

    if peft_path:
        model = PeftModel.from_pretrained(model, peft_path)

    data = load_json(data_path)
    matches = find_by_title(title, data, limit=3)

    if not matches:
        ctx = "Título não encontrado no contexto."
        picked_title = title
        info = {"match_title": None, "score": 0}
    else:
        picked_title = data[matches[0][1]].get("title","")
        ctx = data[matches[0][1]].get("content","")
        info = {"match_title": picked_title, "score": matches[0][2]}

    prompt = build_prompt(picked_title, question, ctx)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    # extrair parte da resposta se o template foi usado
    if "<|assistant|>" in prompt and "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    return text.strip(), info

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--peft", default=None, help="caminho do adapter (outputs/...)")
    p.add_argument("--data", required=True, help="caminho para trn.json")
    p.add_argument("--title", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--no-qlora", action="store_true")
    args = p.parse_args()

    ans, info = answer(
        model_name=args.model,
        peft_path=args.peft,
        title=args.title,
        question=args.question,
        data_path=args.data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        qlora=not args.no_qlora
    )

    print("=== Match ===")
    print(info)
    print("\n=== Resposta ===")
    print(ans)

if __name__ == "__main__":
    cli()
