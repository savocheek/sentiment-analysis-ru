"""Knowledge distillation: the multi-task ruRoberta-large (teacher) → a ruBert-base student with the same heads.

1. Transfer set: ~100k Yandex Maps reviews that are in NO split we use (train/val/test of load_balanced,
   hard_*.jsonl), stratified by rating so negatives and 3★ are not drowned by 5★. The rating is used only
   for sampling — the targets are the teacher's probabilities for sentiment, 1–5★ and category.
2. Student training on transfer set + the usual train split (+ synthetic and mined hard reviews, sentiment
   only), all with the teacher's soft targets (temperature T). ~15% of the examples get typos
   (augment.typos) with the teacher's targets for the clean text, so the student learns to ignore them.
Best epoch by mean macro-F1 on the real val split (gold labels), as in train_multi.py.

Usage: python distill.py [--teacher model] [--student model_v3b] [--out model_student] [--n 100000]
Teacher outputs are cached in data/distill_teacher.npz, so a rerun skips step 1.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from augment import train_examples, typos
from sentiment import Analyzer, encode, make_heads
from tasks import load_multitask
from train import LABELS, SEED
from train_multi import TASKS, predict, scores

CACHE = Path("data/distill_teacher.npz")
TOKEN_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def transfer_texts(n, used):
    """n unused reviews: up to a third each from 1–2★, 3★ and 4–5★ (3★ is scarce, the rest goes to the others)."""
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    buckets, seen = {0: [], 1: [], 2: []}, set(used)
    for t, r in zip(ds["text"], ds["rating"]):
        t = (t or "").replace("\\n", " ")
        if r >= 1 and len(t) > 20 and t not in seen:
            seen.add(t)
            buckets[0 if r <= 2 else 1 if r == 3 else 2].append(t)
    rng = random.Random(SEED)
    for b in buckets.values():
        rng.shuffle(b)
    take = {1: min(len(buckets[1]), n // 3)}
    take[0] = min(len(buckets[0]), (n - take[1]) // 2)
    take[2] = n - take[0] - take[1]
    print("transfer set by rating bucket:", take, "available:", {k: len(v) for k, v in buckets.items()}, flush=True)
    texts = [t for k in (0, 1, 2) for t in buckets[k][:take[k]]]
    rng.shuffle(texts)
    return texts


@torch.no_grad()
def teacher_logprobs(path, texts, batch=128):
    """Log-probabilities of all three teacher heads, one encoder pass per batch, length-sorted batches."""
    an = Analyzer(path)
    order = np.argsort([len(t) for t in texts])
    out = {t: [None] * len(texts) for t in TASKS}
    t0 = time.time()
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = an.tok([texts[j] for j in idx], truncation=True, max_length=256, padding=True,
                     return_tensors="pt").to(an.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=an.device != "cpu"):
            logits, pooled = encode(an.model, enc)
            heads = {"label": logits, **{k: h(pooled) for k, h in an.heads.items()}}
        for t, x in heads.items():
            lp = torch.log_softmax(x.float(), -1).cpu().numpy()
            for j, row in zip(idx, lp):
                out[t][j] = row
        if (i // batch) % 100 == 0:
            print(f"teacher {i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    del an
    torch.cuda.empty_cache()
    return {t: np.stack(v).astype(np.float32) for t, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="model")
    p.add_argument("--student", default="model_v3b", help="initial weights (a ruBert-base sentiment model)")
    p.add_argument("--out", default="model_student")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--typo-share", type=float, default=0.15)
    args = p.parse_args()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (train, val, test), cats = load_multitask()
    synth = train_examples()
    hard = [json.loads(x) for x in Path("data/hard_train.jsonl").read_text(encoding="utf-8").splitlines()]
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        texts, n_sent = list(z["texts"]), int(z["n_sentiment_only"])
        lp = {t: z[t] for t in TASKS}
    else:
        used = set(train["text"]) | set(val["text"]) | set(test["text"]) | set(synth["text"])
        for s in ("train", "dev", "test"):
            used |= {json.loads(x)["text"] for x in Path(f"data/hard_{s}.jsonl").read_text(encoding="utf-8").splitlines()}
        transfer = transfer_texts(args.n, used)
        sent_only = synth["text"] + [h["text"] for h in hard]  # no real stars/category behind these
        texts, n_sent = transfer + list(train["text"]) + sent_only, len(sent_only)
        lp = teacher_logprobs(args.teacher, texts)
        np.savez(CACHE, texts=np.array(texts, dtype=object), n_sentiment_only=n_sent, **lp)
        print(f"== {time.strftime('%H:%M')} teacher done, cool-down 60 s", flush=True)
        time.sleep(60)  # one GPU job at a time, with a pause between them (this PC once crashed under load)
    assert len(set(texts) & (set(val["text"]) | set(test["text"]))) == 0

    rng = random.Random(SEED)
    noisy = [typos(t, rng) if rng.random() < args.typo_share else t for t in texts]
    print(f"student train={len(texts)} (sentiment-only {n_sent}), with typos: "
          f"{sum(a != b for a, b in zip(noisy, texts))}", flush=True)
    has_extra = np.array([1.0] * (len(texts) - n_sent) + [0.0] * n_sent, dtype=np.float32)

    tok = AutoTokenizer.from_pretrained(args.student)
    data = Dataset.from_dict({"text": noisy, "idx": list(range(len(texts)))})
    data = data.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True,
                    num_proc=4).remove_columns("text")

    def collate(rows):
        idx = [r["idx"] for r in rows]
        b = tok.pad([{k: r[k] for k in TOKEN_KEYS if k in r} for r in rows], return_tensors="pt")
        b.update({f"lp_{t}": torch.from_numpy(lp[t][idx]) for t in TASKS})
        b["has_extra"] = torch.from_numpy(has_extra[idx])
        return b

    tl = DataLoader(data, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(SEED))
    from transformers import DataCollatorWithPadding
    enc = lambda b: tok(b["text"], truncation=True, max_length=256)
    vl = DataLoader(val.map(enc, batched=True).remove_columns("text"), batch_size=128,
                    collate_fn=DataCollatorWithPadding(tok))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student, num_labels=3, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    heads = make_heads(model.config.hidden_size, {"stars": 5, "category": len(cats)}).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr},
                             {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    steps = math.ceil(len(tl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    T = args.temperature

    def soft_ce(student_logits, teacher_lp, weight=None):
        """Cross-entropy to the softened teacher distribution, × T² (Hinton et al.) so gradients keep their scale."""
        q = torch.softmax(teacher_lp / T, -1)
        loss = -(q * torch.log_softmax(student_logits.float() / T, -1)).sum(-1) * T * T
        return loss.mean() if weight is None else (loss * weight).sum() / weight.sum().clamp(min=1)

    best, best_val, step, t0 = None, -1, 0, time.time()
    for epoch in range(math.ceil(args.epochs)):
        model.train(), heads.train()
        for b in tl:
            if step >= steps:
                break
            b = {k: v.to(device) for k, v in b.items()}
            x = {k: b[k] for k in TOKEN_KEYS if k in b}
            with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=device != "cpu"):
                logits, pooled = encode(model, x)
                loss = soft_ce(logits, b["lp_label"])
                for t in ("stars", "category"):
                    loss = loss + soft_ce(heads[t](pooled), b[f"lp_{t}"], b["has_extra"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % 500 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        v = scores(val, predict(model, heads, vl, device))
        mean_f1 = np.mean([v[t]["f1_macro"] for t in TASKS])
        print(f"epoch {epoch + 1} val {json.dumps(v)} mean_f1 {mean_f1:.4f}", flush=True)
        if mean_f1 > best_val:
            best_val = mean_f1
            best = (copy.deepcopy(model.state_dict()), copy.deepcopy(heads.state_dict()))

    model.load_state_dict(best[0]), heads.load_state_dict(best[1])
    sl = DataLoader(test.map(enc, batched=True).remove_columns("text"), batch_size=128,
                    collate_fn=DataCollatorWithPadding(tok))
    test_scores = scores(test, predict(model, heads, sl, device))
    print("test", json.dumps(test_scores), flush=True)

    out = Path(args.out)
    model.save_pretrained(out), tok.save_pretrained(out)
    torch.save(heads.state_dict(), out / "heads.pt")
    (out / "multitask.json").write_text(json.dumps(
        {"stars": [1, 2, 3, 4, 5], "categories": cats, "base_model": args.student, "teacher": args.teacher,
         "distillation": {"transfer": args.n, "temperature": T, "typo_share": args.typo_share},
         "val_mean_f1": round(best_val, 4)}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports" / "multitask_test.json").write_text(json.dumps(test_scores, indent=2))


if __name__ == "__main__":
    main()
