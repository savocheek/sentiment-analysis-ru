"""Extra tasks on the same real reviews, no new labelling: exact 1–5★ rating and place category.

The split is the one from train.load_balanced (same train/val/test texts as the sentiment model), with
- stars:    rating − 1 (0..4)
- category: the first rubric of the place, top-15 by train frequency, the rest → "другое"
"""
import json
from collections import Counter
from pathlib import Path

from train import load_balanced

N_CATEGORIES = 15
OTHER = "другое"
CATEGORIES_FILE = Path("data/categories.json")


def first_rubric(r):
    return (r or "").split(";")[0].strip() or OTHER


def load_multitask(per_class=12000):
    train, val, test = load_balanced(per_class, extra=("rating", "rubrics"))
    top = [c for c, _ in Counter(map(first_rubric, train["rubrics"])).most_common() if c != OTHER][:N_CATEGORIES]
    cats = top + [OTHER]
    CATEGORIES_FILE.parent.mkdir(exist_ok=True)
    CATEGORIES_FILE.write_text(json.dumps(cats, ensure_ascii=False, indent=1), encoding="utf-8")
    idx = {c: i for i, c in enumerate(cats)}
    fix = lambda b: {"stars": [r - 1 for r in b["rating"]],
                     "category": [idx.get(first_rubric(r), idx[OTHER]) for r in b["rubrics"]]}
    return [d.map(fix, batched=True).remove_columns(["rating", "rubrics"]) for d in (train, val, test)], cats
