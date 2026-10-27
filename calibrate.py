"""Probability calibration of the sentiment head: ECE + reliability diagram, temperature scaling fitted on val.

Does "0.9 confident" mean right 9 times out of 10? ECE = mean |accuracy − confidence| over 15 confidence
bins, weighted by bin size. Temperature T is fitted on val (min NLL) and checked on test.

Usage: python calibrate.py [MODEL_DIR]  → reports/calibration.json, reports/reliability.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from sentiment import Analyzer
from train import load_balanced

BINS = 15


def softmax(logits, t=1.0):
    x = logits / t
    x = np.exp(x - x.max(1, keepdims=True))
    return x / x.sum(1, keepdims=True)


def ece(p, gold):
    conf, right = p.max(1), p.argmax(1) == gold
    edges = np.linspace(0, 1, BINS + 1)
    rows, total = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            rows.append((float(conf[m].mean()), float(right[m].mean()), int(m.sum())))
            total += m.mean() * abs(conf[m].mean() - right[m].mean())
    return float(total), rows


def nll(p, gold):
    return float(-np.log(p[np.arange(len(gold)), gold] + 1e-12).mean())


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model"
    an = Analyzer(model)
    _, val, test = load_balanced(12000)
    # log-probabilities are logits up to a per-row constant, which softmax ignores
    lv, lt = (np.log(an.probs(list(d["text"])) + 1e-12) for d in (val, test))
    gv, gt = np.array(val["label"]), np.array(test["label"])
    t = minimize_scalar(lambda t: nll(softmax(lv, t), gv), bounds=(0.3, 5), method="bounded").x

    res = {"temperature": round(float(t), 3)}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="идеальная калибровка")
    for name, temp, color in (("до", 1.0, "#d9534f"), ("после T-scaling", t, "#3fa45b")):
        p = softmax(lt, temp)
        e, rows = ece(p, gt)
        res[name] = {"ece": round(e, 4), "nll": round(nll(p, gt), 4), "mean_confidence": round(float(p.max(1).mean()), 4),
                     "accuracy": round(float((p.argmax(1) == gt).mean()), 4)}
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=color, label=f"{name}: ECE {e:.3f}")
    ax.set_xlabel("уверенность модели"), ax.set_ylabel("доля верных ответов")
    ax.set_title(f"Калибровка тональности, тест ({len(gt)} отзывов)"), ax.legend(loc="upper left")
    ax.set_xlim(0.3, 1), ax.set_ylim(0.3, 1)
    fig.tight_layout()
    for d in {Path("reports") if model == "model" else Path(model) / "reports", Path(model) / "reports"}:
        d.mkdir(exist_ok=True)
        fig.savefig(d / "reliability.png", dpi=150)
        (d / "calibration.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
