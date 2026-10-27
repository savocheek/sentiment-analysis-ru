"""Multi-task fine-tuning: one encoder, three heads — sentiment (3), exact stars 1–5, place category (16).

Plain PyTorch loop (AdamW, linear warmup/decay, fp16 autocast on GPU) — no Trainer, so it also runs on
AMD's ROCm wheels for Windows. Synthetic (--augment) and mined hard (--hard) reviews only have a sentiment
label: their stars/category are masked out of the loss.

Usage: python train_multi.py --base-model model_v3b --out model_multi [--epochs 2] [--lr 2e-5] [--batch 32]
Saves OUT/ as a standard sentiment model (works with pipeline/evaluate.py) + OUT/heads.pt + OUT/multitask.json
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from baselines import task_metrics
from sentiment import encode, make_heads
from tasks import load_multitask
from train import LABELS, SEED

TASKS = ("label", "stars", "category")


@torch.no_grad()
def predict(model, heads, loader, device):
    model.eval(), heads.eval()
    out = {t: [] for t in TASKS}
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        enc = {k: b[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
        with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=device != "cpu"):
            logits, pooled = encode(model, enc)
            extra = {k: h(pooled) for k, h in heads.items()}
        for t, x in (("label", logits), *extra.items()):
            out[t].append(torch.softmax(x.float(), -1).cpu().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def scores(data, proba):
    return {t: task_metrics(t, data[t], proba[t]) for t in TASKS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="model_v3b")
    p.add_argument("--out", default="model_multi")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--hard", action="store_true")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--extra-weight", type=float, default=1.0, help="loss weight of stars/category heads")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (train, val, test), cats = load_multitask()
    masked = lambda d: d.map(lambda b: {"stars": [-100] * len(b["text"]), "category": [-100] * len(b["text"])},
                             batched=True).cast(train.features)
    if args.augment:
        from augment import train_examples
        train = concatenate_datasets([train, masked(Dataset.from_dict(train_examples()))])
    if args.hard:
        train = concatenate_datasets([train, masked(Dataset.from_json("data/hard_train.jsonl"))])
    print(f"train={len(train)} val={len(val)} test={len(test)} categories={len(cats)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    enc = lambda b: tok(b["text"], truncation=True, max_length=args.max_length)
    train_t, val_t, test_t = (d.map(enc, batched=True, num_proc=4).remove_columns("text") for d in (train, val, test))
    coll = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(train_t, batch_size=args.batch, shuffle=True, collate_fn=coll, generator=g)
    vl, sl = (DataLoader(d, batch_size=128, collate_fn=coll) for d in (val_t, test_t))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    heads = make_heads(model.config.hidden_size, {"stars": 5, "category": len(cats)}).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr},  # fresh heads need a bigger step
                             {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = math.ceil(len(tl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best, best_val, step, t0 = None, -1, 0, time.time()
    for epoch in range(math.ceil(args.epochs)):
        model.train(), heads.train()
        for b in tl:
            if step >= steps:
                break
            b = {k: v.to(device) for k, v in b.items()}
            x = {k: b[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
            with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=device != "cpu"):
                logits, pooled = encode(model, x)
                loss = ce(logits.float(), b["labels"])
                for t in ("stars", "category"):
                    if (b[t] != -100).any():
                        loss = loss + args.extra_weight * ce(heads[t](pooled).float(), b[t])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        v = scores(val, predict(model, heads, vl, device))
        mean_f1 = np.mean([v[t]["f1_macro"] for t in TASKS])
        print(f"epoch {epoch + 1} val {json.dumps(v)} mean_f1 {mean_f1:.4f}", flush=True)
        if mean_f1 > best_val:
            best_val = mean_f1
            best = (copy.deepcopy(model.state_dict()), copy.deepcopy(heads.state_dict()))

    model.load_state_dict(best[0]), heads.load_state_dict(best[1])
    test_scores = scores(test, predict(model, heads, sl, device))
    print("test", json.dumps(test_scores), flush=True)

    out = Path(args.out)
    model.save_pretrained(out), tok.save_pretrained(out)
    torch.save(heads.state_dict(), out / "heads.pt")
    (out / "multitask.json").write_text(json.dumps(
        {"stars": [1, 2, 3, 4, 5], "categories": cats, "base_model": args.base_model, "val_mean_f1": round(best_val, 4)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "multitask_test.json").write_text(json.dumps(test_scores, indent=2))


if __name__ == "__main__":
    main()
