"""Export a model to ONNX (all heads), quantize to int8 and compare speed/agreement on CPU.

Usage: python export_onnx.py MODEL_DIR [OUT_DIR]   → OUT/model.onnx, OUT/model.int8.onnx, OUT/onnx_report.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from sentiment import Analyzer, encode
from train import load_balanced


class AllHeads(torch.nn.Module):
    def __init__(self, an):
        super().__init__()
        self.model, self.heads = an.model, an.heads

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits, pooled = encode(self.model, {"input_ids": input_ids, "attention_mask": attention_mask,
                                             "token_type_ids": token_type_ids})
        if self.heads is None:
            return (logits,)
        return logits, self.heads["stars"](pooled), self.heads["category"](pooled)


def main():
    src = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"{src}_onnx")
    out.mkdir(exist_ok=True)
    an = Analyzer(src)  # DEVICE=cpu for export
    an.model.float().cpu(), an.heads and an.heads.cpu()
    names = ["sentiment"] + (["stars", "category"] if an.heads is not None else [])
    enc = an.tok(["Пример отзыва для экспорта"], return_tensors="pt")
    enc.setdefault("token_type_ids", torch.zeros_like(enc["input_ids"]))
    dyn = {k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids")}
    fp32 = out / "model.onnx"
    torch.onnx.export(AllHeads(an).eval(), (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), fp32,
                      input_names=list(dyn), output_names=names, dynamic_axes=dyn | {n: {0: "batch"} for n in names},
                      opset_version=17, dynamo=False)
    int8 = out / "model.int8.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    an.tok.save_pretrained(out)
    for f in ("multitask.json", "config.json"):
        if (Path(src) / f).exists():
            (out / f).write_bytes((Path(src) / f).read_bytes())

    # Speed and agreement on 300 real test reviews, CPU, batch 1 and batch 32
    texts = list(load_balanced(12000)[2].shuffle(seed=0).select(range(300))["text"])
    torch.set_num_threads(8)

    def torch_run(batch):
        return an.probs(batch)

    def ort_runner(path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        def run(batch):
            e = an.tok(batch, truncation=True, max_length=256, padding=True, return_tensors="np")
            feed = {k: e[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
            feed["token_type_ids"] = e.get("token_type_ids", np.zeros_like(e["input_ids"])).astype(np.int64)
            x = sess.run(["sentiment"], feed)[0]
            x = np.exp(x - x.max(1, keepdims=True))
            return x / x.sum(1, keepdims=True)
        return run

    report, ref = {}, torch_run(texts)
    for name, run in (("pytorch_fp32", torch_run), ("onnx_fp32", ort_runner(fp32)), ("onnx_int8", ort_runner(int8))):
        run(texts[:8])  # warm-up
        t0 = time.perf_counter()
        for t in texts[:50]:
            run([t])
        single = (time.perf_counter() - t0) / 50 * 1000
        t0 = time.perf_counter()
        p = np.concatenate([run(texts[i:i + 32]) for i in range(0, len(texts), 32)])
        report[name] = {"ms_per_review": round(single, 1), "reviews_per_s": round(len(texts) / (time.perf_counter() - t0)),
                        "agree_with_pytorch": round(float((p.argmax(1) == ref.argmax(1)).mean()), 4)}
        print(name, report[name], flush=True)
    report["size_mb"] = {"onnx_fp32": round(fp32.stat().st_size / 2**20), "onnx_int8": round(int8.stat().st_size / 2**20)}
    (out / "onnx_report.json").write_text(json.dumps(report, indent=2))
    print(report["size_mb"])


if __name__ == "__main__":
    main()
