"""Fine-tune rubert-tiny2 on Russian reviews (Yandex Maps) for 3-class sentiment.

Usage: python train.py [--per-class 12000] [--epochs 3] [--augment] [--hard] [--base-model ID] [--seed N] [--max-length N] [--batch N] [--resume DIR] [--lr X]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

if not torch.distributed.is_available():  # AMD's ROCm wheels for Windows ship without torch.distributed,
    import sys, types                     # but Trainer imports DTensor just to check isinstance()
    sys.modules["torch.distributed.tensor"] = types.SimpleNamespace(DTensor=type("DTensor", (), {}))

from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

BASE_MODEL = "cointegrated/rubert-tiny2"
LABELS = ["negative", "neutral", "positive"]
SEED = 42


def rating_to_label(r):
    return 0 if r <= 2 else 1 if r == 3 else 2


def load_balanced(per_class):
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    ds = ds.filter(lambda x: x["rating"] >= 1 and x["text"] and len(x["text"]) > 20)
    ds = ds.map(lambda x: {"label": rating_to_label(x["rating"]),
                           "text": x["text"].replace("\\n", " ")})
    # Balance classes: neutral (3★) is the smallest at ~21k, positive dominates at ~430k.
    parts = [ds.filter(lambda x, c=c: x["label"] == c).shuffle(seed=SEED).select(range(per_class))
             for c in range(3)]
    ds = concatenate_datasets(parts).shuffle(seed=SEED).select_columns(["text", "label"])
    split = ds.train_test_split(test_size=0.2, seed=SEED)
    val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)
    return split["train"], val_test["train"], val_test["test"]


def baseline(train, test):
    vec = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, C=4)
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    pred = clf.predict(vec.transform(test["text"]))
    return f1_score(test["label"], pred, average="macro")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=12000)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--out", default="model")
    p.add_argument("--augment", action="store_true", help="add synthetic sarcasm/negation reviews (augment.py) to train")
    p.add_argument("--hard", action="store_true", help="add real hard reviews data/hard_train.jsonl (mine_hard.py) to train")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--resume", help="checkpoint dir to continue an interrupted run from")
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()

    train, val, test = load_balanced(args.per_class)
    if args.augment:  # train only: val/test stay real reviews
        from augment import train_examples
        synth = Dataset.from_dict(train_examples()).cast(train.features)
        train = concatenate_datasets([train, synth]).shuffle(seed=SEED)
    if args.hard:
        hard = Dataset.from_json("data/hard_train.jsonl").cast(train.features)
        train = concatenate_datasets([train, hard]).shuffle(seed=SEED)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    base_f1 = baseline(train, test)
    print(f"TF-IDF + LogReg macro-F1: {base_f1:.4f}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    enc = lambda b: tok(b["text"], truncation=True, max_length=args.max_length)
    train_t, val_t, test_t = (d.map(enc, batched=True) for d in (train, val, test))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    src = {v.lower(): int(k) for k, v in AutoConfig.from_pretrained(args.base_model).id2label.items()}
    if set(src) == set(LABELS):  # backbone already has a sentiment head: reorder its rows to our label order
        order = [src[l] for l in LABELS]
        model.classifier.weight.data = model.classifier.weight.data[order]
        model.classifier.bias.data = model.classifier.bias.data[order]

    def metrics(ev):
        pred = np.argmax(ev.predictions, axis=-1)
        return {"f1_macro": f1_score(ev.label_ids, pred, average="macro"),
                "accuracy": float((pred == ev.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"checkpoints/{args.out}", num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
            learning_rate=args.lr, warmup_steps=0.06, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="f1_macro",
            logging_steps=100, report_to="none", seed=args.seed,
            fp16=torch.cuda.is_available()),  # ~2x faster on NVIDIA GPUs
        train_dataset=train_t, eval_dataset=val_t,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train(resume_from_checkpoint=args.resume)

    out = trainer.predict(test_t)
    pred = np.argmax(out.predictions, axis=-1)
    report = classification_report(test["label"], pred, target_names=LABELS, digits=4)
    print(report)

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    reports = Path("reports") if args.out == "model" else Path(args.out) / "reports"  # experiments keep their own
    reports.mkdir(exist_ok=True)
    (reports / "classification_report.txt").write_text(report)
    (reports / "metrics.json").write_text(json.dumps({
        "test_size": len(test),
        "baseline_tfidf_logreg_f1_macro": round(base_f1, 4),
        "rubert_tiny2_f1_macro": round(out.metrics["test_f1_macro"], 4),
        "rubert_tiny2_accuracy": round(out.metrics["test_accuracy"], 4),
    }, indent=2))
    disp = ConfusionMatrixDisplay.from_predictions(
        test["label"], pred, display_labels=LABELS, normalize="true", cmap="Blues")
    disp.figure_.savefig(reports / "confusion_matrix.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
