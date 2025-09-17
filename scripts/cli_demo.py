import argparse
from .inference import answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--peft", default=None)
    p.add_argument("--data", default="data/trn.json")
    args = p.parse_args()

    text, info = answer(args.model, args.peft, args.title, args.question, args.data)
    print("Match:", info)
    print("Resposta:", text)

if __name__ == "__main__":
    main()
