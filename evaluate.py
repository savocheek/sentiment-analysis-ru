"""One-line scorecard for comparing models.

Usage: python evaluate.py MODEL_DIR [MODEL_DIR ...] [--hard dev|test]
  --hard dev  (default) for choosing between experiments; --hard test only for the final run.
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score
from transformers import pipeline

from augment import generate
from stress_test import CASES, FRESH
from train import LABELS, load_balanced


def acc(preds, gold):
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--hard", default="dev", choices=["dev", "test"])
    args = p.parse_args()

    _, _, test = load_balanced(12000)
    hard = [json.loads(x) for x in Path(f"data/hard_{args.hard}.jsonl").read_text().splitlines()]
    held = [(c, t, LABELS[l]) for c, t, l in generate(1)]

    for m in args.models:
        clf = pipeline("text-classification", model=m, truncation=True, max_length=256)
        top = lambda texts: [r["label"] for r in clf(list(texts), batch_size=32)]
        real = top(test["text"])
        hp = top(x["text"] for x in hard)
        hn = [p for p, x in zip(hp, hard) if x["label"] == 0]
        hpos = [p for p, x in zip(hp, hard) if x["label"] == 2]
        row = {
            "model": m,
            "real_f1": round(f1_score([LABELS[l] for l in test["label"]], real, average="macro"), 4),
            f"hard_{args.hard}_neg": round(acc(hn, ["negative"] * len(hn)), 3),
            f"hard_{args.hard}_pos": round(acc(hpos, ["positive"] * len(hpos)), 3),
            "fresh23": sum(a == b for a, b in zip(top(t for _, t, _ in FRESH), [e for _, _, e in FRESH])),
            "first25": sum(a == b for a, b in zip(top(t for _, t, _ in CASES), [e for _, _, e in CASES])),
            "templ141": sum(a == b for a, b in zip(top(t for _, t, _ in held), [e for _, _, e in held])),
        }
        row[f"hard_{args.hard}_bal"] = round((row[f"hard_{args.hard}_neg"] + row[f"hard_{args.hard}_pos"]) / 2, 3)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
