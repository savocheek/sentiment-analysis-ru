"""Out-of-domain test: film reviews from Kinopoisk (blinoff/kinopoisk, MIT), never seen in training.

The models learnt on Yandex Maps reviews of places; here the domain, the length (median ~1600 chars,
longer than the 256-token window) and the style are all different. Balanced sample: N per class.
Reviews are longer than the window, so two readings are scored: the start (as deployed) and the last
150 words (film reviews usually end with the verdict).

Usage: python kinopoisk_eval.py MODEL_DIR [MODEL_DIR ...] [--n 1000]  → reports/kinopoisk.{json,md}
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from sentiment import LABELS, RU, Analyzer, clean
from train import load_balanced

GRADE = {"Bad": 0, "Neutral": 1, "Good": 2}


def sample(n, seed=0):
    ds = load_dataset("blinoff/kinopoisk", split="train")
    parts = [ds.filter(lambda x, g=g: x["grade3"] == g).shuffle(seed=seed).select(range(n)) for g in GRADE]
    texts = [clean(t) for p in parts for t in p["content"]]
    gold = np.array([GRADE[g] for p in parts for g in p["grade3"]])
    return texts, gold


def score(gold, pred):
    cm = confusion_matrix(gold, pred, labels=[0, 1, 2])
    return {"f1_macro": round(f1_score(gold, pred, average="macro"), 4),
            "accuracy": round(float((gold == pred).mean()), 4),
            "recall": {l: round(float(cm[i, i] / cm[i].sum()), 3) for i, l in enumerate(LABELS)},
            "confusion": cm.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    texts, gold = sample(args.n)
    tail = [" ".join(t.split()[-150:]) for t in texts]
    out_json = Path("reports/kinopoisk.json")
    res = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}

    if "tfidf_logreg" not in res:  # the same baseline as baselines.py, trained on Yandex Maps
        train, _, _ = load_balanced(12000)
        vec = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), sublinear_tf=True)
        clf = LogisticRegression(max_iter=1000, C=4).fit(vec.fit_transform(train["text"]), train["label"])
        res["tfidf_logreg"] = {"start": score(gold, clf.predict(vec.transform(texts))),
                               "end": score(gold, clf.predict(vec.transform(tail)))}
    for m in args.models:
        an = Analyzer(m)
        res[m] = {"start": score(gold, an.probs(texts).argmax(1)), "end": score(gold, an.probs(tail).argmax(1))}
        print(m, json.dumps(res[m]["start"]["f1_macro"]), json.dumps(res[m]["end"]["f1_macro"]), flush=True)
        del an

    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Внешний тест: рецензии на фильмы с Кинопоиска ({len(gold)} шт., по {args.n} на класс)\n",
          "Модели учились только на отзывах о местах с Яндекс Карт. «Начало» — первые 256 токенов, как в демо; "
          "«конец» — последние 150 слов рецензии.\n",
          "| Модель | F1 (начало) | F1 (конец) | Recall нег / нейтр / поз (конец) |", "|---|---|---|---|"]
    for m, r in res.items():
        rc = r["end"]["recall"]
        md.append(f"| {m} | {r['start']['f1_macro']:.3f} | {r['end']['f1_macro']:.3f} | "
                  f"{rc['negative']:.2f} / {rc['neutral']:.2f} / {rc['positive']:.2f} |")
    for m, r in res.items():
        md += ["", f"**{m}**, матрица ошибок (конец рецензии): строки — истина, столбцы — модель\n",
               "| | " + " | ".join(RU[l] for l in LABELS) + " |", "|---|---|---|---|"]
        md += [f"| {RU[LABELS[i]]} | " + " | ".join(map(str, row)) + " |" for i, row in enumerate(r["end"]["confusion"])]
    Path("reports/kinopoisk.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
