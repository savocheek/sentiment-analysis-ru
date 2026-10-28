"""Items for hand labelling in label.py (round 3).

- aspects: ~300 clauses from train reviews (never test/hard-test), 60 per star 1–5★; 45 of each 60 have a lexicon
  hit, 15 have none (the lexicon skips those today). Pre-labelled: aspects by the lexicon, clause sentiment by
  the model (polarity_label) — the person confirms or fixes every one.
- audit: 300 test reviews for the "neutral" noise audit: 150 model errors on the neutral border
  (3★ predicted as not-neutral, or non-3★ predicted as neutral) + 150 random others, shuffled together.
  Stars and model answers are stored but label.py never shows them (blind labelling).
Usage: python prepare_labels.py [--model model]  → data/label_items_{aspects,audit}.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from sentiment import LABELS, Analyzer, clauses, find_aspects, polarity_label
from train import load_balanced

PER_STAR, NO_ASPECT = 60, 15


def aspect_items(an, train, rng):
    by_star = {s: [] for s in range(1, 6)}
    for t, r in zip(train["text"], train["rating"]):
        by_star[int(r)].append(t)
    items = []
    for star, texts in by_star.items():
        rng.shuffle(texts)
        hit, miss, seen = [], [], set()
        for t in texts:  # one clause per review keeps phrases diverse
            parts = [c for c in clauses(t) if 3 <= len(c.split()) <= 25 and c.lower() not in seen]
            if not parts:
                continue
            c = rng.choice(parts)
            seen.add(c.lower())
            (hit if find_aspects(c) else miss).append((c, t))
            if len(hit) >= PER_STAR - NO_ASPECT and len(miss) >= NO_ASPECT:
                break
        for c, t in hit[:PER_STAR - NO_ASPECT] + miss[:NO_ASPECT]:
            items.append({"text": c, "review": t, "stars": star})
    p = an.probs([x["text"] for x in items])
    for i, (x, row) in enumerate(zip(items, p)):
        x["id"] = f"a{i:03d}"
        x["pre_aspects"] = find_aspects(x["text"])
        x["pre_label"] = polarity_label(row)[0]
    rng.shuffle(items)
    return items


def audit_items(an, test, rng):
    texts, gold = list(test["text"]), np.array(test["label"])
    p = an.probs(texts, temperature=an.temperature)
    pred = p.argmax(1)
    border = np.where((gold != pred) & ((gold == 1) | (pred == 1)))[0].tolist()
    rest = np.where(~np.isin(np.arange(len(texts)), border))[0].tolist()
    print(f"neutral-border errors in test: {len(border)} of {len(texts)}")
    chosen = [(i, "border_error") for i in rng.sample(border, 150)] + [(i, "random") for i in rng.sample(rest, 150)]
    rng.shuffle(chosen)
    return [{"id": f"r{k:03d}", "text": texts[i], "stars": int(test["rating"][i]), "gold": LABELS[gold[i]],
             "pred": LABELS[pred[i]], "conf": round(float(p[i].max()), 3), "kind": kind}
            for k, (i, kind) in enumerate(chosen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    a = ap.parse_args()
    an = Analyzer(a.model)
    train, _, test = load_balanced(12000, extra=("rating",))
    rng = random.Random(3)
    for name, items in (("aspects", aspect_items(an, train, rng)), ("audit", audit_items(an, test, rng))):
        path = Path(f"data/label_items_{name}.jsonl")
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
        print(path, len(items))


if __name__ == "__main__":
    main()
