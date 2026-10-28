"""Continue training the multi-task large model on (almost) all real reviews, not just the balanced 28.8k.

Data: every Yandex Maps review except the main val/test split and hard-dev/hard-test (Kinopoisk is a different
dataset, so nothing to remove there), + mined hard-train and synthetic reviews as before (sentiment only).
All 1–2★ and 3★ reviews are taken; 4–5★ fill the rest up to --n. Sentiment is still very imbalanced
(neutral is the smallest class), so the sentiment and stars losses get class weights (inverse frequency)^0.5.
Starts from model_multi with its heads. Validation every half epoch (val mean macro-F1 of the three heads, plus
hard-dev balanced accuracy for information); the best one is kept. Every validation also writes a resumable
checkpoint (weights, optimizer, scheduler, position), so a crash costs at most half an epoch.

Usage: python train_big.py [--init model_multi] [--out model_multi_big] [--n 330000] [--epochs 2]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
    get_linear_schedule_with_warmup

from augment import train_examples
from sentiment import encode, make_heads
from tasks import OTHER, first_rubric, load_multitask
from train import SEED, rating_to_label
from train_multi import TASKS, predict, scores

TOKEN_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def big_train(n, cats, exclude):
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    idx = {c: i for i, c in enumerate(cats)}
    rows, seen = {0: [], 1: [], 2: []}, set(exclude)
    for t, r, rub in zip(ds["text"], ds["rating"], ds["rubrics"]):
        t = (t or "").replace("\\n", " ")
        if r >= 1 and len(t) > 20 and t not in seen:
            seen.add(t)
            lab = rating_to_label(r)
            rows[lab].append((t, lab, r - 1, idx.get(first_rubric(rub), idx[OTHER])))
    rng = random.Random(SEED)
    rng.shuffle(rows[2])
    rows[2] = rows[2][:max(0, n - len(rows[0]) - len(rows[1]))]
    data = [x for k in (0, 1, 2) for x in rows[k]]
    print("big train by sentiment:", {k: len(v) for k, v in rows.items()}, flush=True)
    return Dataset.from_dict({k: [x[i] for x in data] for i, k in enumerate(("text", "labels", "stars", "category"))})


def class_weights(y, k):
    counts = np.bincount(np.asarray(y)[np.asarray(y) >= 0], minlength=k).astype(float)
    w = (counts.sum() / (k * np.maximum(counts, 1))) ** 0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def hard_dev(model, tok, device):
    hard = [json.loads(x) for x in Path("data/hard_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    model.eval()
    pred = []
    for i in range(0, len(hard), 64):
        enc = tok([h["text"] for h in hard[i:i + 64]], truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            pred += encode(model, enc)[0].argmax(-1).tolist()
    gold, pred = np.array([h["label"] for h in hard]), np.array(pred)
    return round(float(np.mean([(pred[gold == c] == c).mean() for c in (0, 2)])), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="model_multi")
    p.add_argument("--out", default="model_multi_big")
    p.add_argument("--n", type=int, default=330_000)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4, help="heads are already trained, so a small step")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--ckpt", default="checkpoints/big")
    args = p.parse_args()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)

    (small_train, val, test), cats = load_multitask()
    exclude = set(val["text"]) | set(test["text"])
    for s in ("dev", "test"):
        exclude |= {json.loads(x)["text"] for x in Path(f"data/hard_{s}.jsonl").read_text(encoding="utf-8").splitlines()}
    train = big_train(args.n, cats, exclude)
    sent_only = Dataset.from_dict(train_examples()).select_columns(["text", "label"])
    hard = Dataset.from_json("data/hard_train.jsonl").select_columns(["text", "label"])
    extra = concatenate_datasets([sent_only, hard]).rename_column("label", "labels")
    extra = extra.map(lambda b: {"stars": [-100] * len(b["text"]), "category": [-100] * len(b["text"])}, batched=True)
    train = concatenate_datasets([train, extra.cast(train.features)])
    assert not set(train["text"]) & exclude
    w = {"label": class_weights(train["labels"], 3), "stars": class_weights(train["stars"], 5)}
    print(f"train={len(train)} class weights: label {w['label'].tolist()} stars {w['stars'].tolist()}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.init)
    enc = lambda b: tok(b["text"], truncation=True, max_length=256)
    train_t = train.map(enc, batched=True, num_proc=4).remove_columns("text")
    coll = DataCollatorWithPadding(tok)
    val_t = val.map(enc, batched=True).remove_columns("text")
    vl = DataLoader(val_t, batch_size=128, collate_fn=coll)

    model = AutoModelForSequenceClassification.from_pretrained(args.init).to(device)
    heads = make_heads(model.config.hidden_size, {"stars": 5, "category": len(cats)}).to(device)
    heads.load_state_dict(torch.load(Path(args.init) / "heads.pt", map_location=device))
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr},
                             {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    per_epoch = math.ceil(len(train_t) / args.batch)
    steps = math.ceil(per_epoch * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    ce = {"label": torch.nn.CrossEntropyLoss(weight=w["label"].to(device)),
          "stars": torch.nn.CrossEntropyLoss(weight=w["stars"].to(device), ignore_index=-100),
          "category": torch.nn.CrossEntropyLoss(ignore_index=-100)}
    eval_every = per_epoch // args.evals_per_epoch

    v = scores(val, predict(model, heads, vl, device))
    start = {"val": v, "mean_f1": round(float(np.mean([v[t]["f1_macro"] for t in TASKS])), 4),
             "hard_dev": hard_dev(model, tok, device)}
    print(f"start (={args.init}) {json.dumps(start)}", flush=True)
    state = {"step": 0, "best": start["mean_f1"], "history": [start]}
    if (ckpt / "state.json").exists():  # resume after a crash
        state = json.loads((ckpt / "state.json").read_text())
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        heads.load_state_dict(torch.load(ckpt / "heads.pt", map_location=device))
        o = torch.load(ckpt / "optim.pt", map_location=device)
        opt.load_state_dict(o["opt"]), sched.load_state_dict(o["sched"]), scaler.load_state_dict(o["scaler"])
        print(f"resumed at step {state['step']}", flush=True)

    out = Path(args.out)

    def save_best():
        model.save_pretrained(out), tok.save_pretrained(out)
        torch.save(heads.state_dict(), out / "heads.pt")

    t0, step = time.time(), state["step"]
    while step < steps:
        epoch = step // per_epoch
        order = np.random.default_rng(SEED + epoch).permutation(len(train_t))
        model.train(), heads.train()
        for pos in range(step % per_epoch, per_epoch):
            if step >= steps:
                break
            b = coll([train_t[int(i)] for i in order[pos * args.batch:(pos + 1) * args.batch]])
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
                logits, pooled = encode(model, {k: b[k] for k in TOKEN_KEYS if k in b})
                loss = ce["label"](logits.float(), b["labels"])
                for t in ("stars", "category"):
                    if (b[t] != -100).any():
                        loss = loss + ce[t](heads[t](pooled).float(), b[t])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % 500 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
            if step % eval_every == 0 or step == steps:
                v = scores(val, predict(model, heads, vl, device))
                rec = {"step": step, "epoch": round(step / per_epoch, 2), "val": v,
                       "mean_f1": round(float(np.mean([v[t]["f1_macro"] for t in TASKS])), 4),
                       "hard_dev": hard_dev(model, tok, device)}
                print(f"eval {json.dumps(rec)}", flush=True)
                state["history"].append(rec)
                if rec["mean_f1"] > state["best"]:
                    state["best"] = rec["mean_f1"]
                    save_best()
                    print(f"  new best → {out}", flush=True)
                state["step"] = step
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt / "model.pt")
                torch.save(heads.state_dict(), ckpt / "heads.pt")
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict()},
                           ckpt / "optim.pt")
                (ckpt / "state.json").write_text(json.dumps(state))
                model.train(), heads.train()

    if not (out / "heads.pt").exists():
        print("no evaluation beat the starting model; nothing saved as best", flush=True)
        return
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    heads.load_state_dict(torch.load(out / "heads.pt", map_location=device))
    sl = DataLoader(test.map(enc, batched=True).remove_columns("text"), batch_size=128, collate_fn=coll)
    test_scores = scores(test, predict(model, heads, sl, device))
    print("test", json.dumps(test_scores), flush=True)
    (out / "multitask.json").write_text(json.dumps(
        {"stars": [1, 2, 3, 4, 5], "categories": cats, "base_model": args.init, "val_mean_f1": state["best"],
         "big": {"n_reviews": args.n, "train_total": len(train), "epochs": args.epochs, "history": state["history"]}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "multitask_test.json").write_text(json.dumps(test_scores, indent=2))


if __name__ == "__main__":
    main()
