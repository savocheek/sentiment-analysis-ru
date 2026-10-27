"""TF-IDF + logistic regression baselines for the extra tasks (CPU). Writes reports/baselines.json

Usage: python baselines.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tasks import load_multitask


def task_metrics(task, y, proba):
    """Shared by baselines.py and train_multi.py so both report the same numbers."""
    from sklearn.metrics import accuracy_score, f1_score
    y, pred = np.asarray(y), proba.argmax(1)
    m = {"accuracy": round(accuracy_score(y, pred), 4), "f1_macro": round(f1_score(y, pred, average="macro"), 4)}
    if task == "stars":
        m["mae"] = round(float(np.abs(pred - y).mean()), 3)
        m["within_1"] = round(float((np.abs(pred - y) <= 1).mean()), 4)
    if task == "category":
        top3 = np.argsort(-proba, 1)[:, :3]
        m["top3"] = round(float((top3 == y[:, None]).any(1).mean()), 4)
    return m


def main():
    (train, _, test), cats = load_multitask()
    vec = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), sublinear_tf=True)
    xtr, xte = vec.fit_transform(train["text"]), vec.transform(test["text"])
    out = {}
    for task in ("stars", "category"):
        clf = LogisticRegression(max_iter=2000, C=4, n_jobs=12).fit(xtr, train[task])
        out[task] = task_metrics(task, test[task], clf.predict_proba(xte))
        print(task, out[task], flush=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baselines.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
