"""Quality of ONNX exports on CPU: sentiment F1 on the full test split and balanced accuracy on hard-dev.

Usage: python onnx_quality.py MODEL_DIR ONNX_DIR  → adds a "quality" section to ONNX_DIR/onnx_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from train import load_balanced


def main():
    model, onnx_dir = sys.argv[1], Path(sys.argv[2])
    os.environ["DEVICE"] = "cpu"
    from sentiment import Analyzer
    _, _, test = load_balanced(12000)
    hard = [json.loads(x) for x in Path("data/hard_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    gold_h = np.array([h["label"] for h in hard])
    quality = {}
    for name in ("model.onnx", "model.int8.onnx"):
        os.environ["ONNX_MODEL"] = str(onnx_dir / name)
        an = Analyzer(model)
        pred = an.probs(list(test["text"])).argmax(1)
        ph = an.probs([h["text"] for h in hard]).argmax(1)
        bal = np.mean([(ph[gold_h == c] == c).mean() for c in (0, 2)])
        quality[name] = {"real_f1": round(f1_score(test["label"], pred, average="macro"), 4),
                         "hard_dev_bal": round(float(bal), 3)}
        print(name, quality[name], flush=True)
    path = onnx_dir / "onnx_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["quality"] = quality
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
