"""Publish the model to the Hugging Face Hub and the demo to a Space (free CPU).

The default model is the distilled ruBert-base student (./model, from distill.py) with its int8 ONNX
in model/onnx/: on the Space's CPU the demo runs on onnxruntime (~7 ms per review), PyTorch is only
needed for word explanations.
Needs your own token: `huggingface-cli login` or the HF_TOKEN environment variable. Nothing is uploaded without it.
Usage: python publish.py HF_USER [--model-dir model] [--model-repo ru-reviews-rubert] [--space ru-reviews-demo]
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

GITHUB = "https://github.com/savocheek/sentiment-analysis-ru"

SPACE_README = """---
title: Анализ отзывов на русском
emoji: 💬
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---
Тональность, оценка 1–5★, категория места и аспекты для русскоязычных отзывов.
Модель: [{model}](https://huggingface.co/{model}), код: [GitHub]({github}).
"""

# CPU-only torch keeps the Space image small; versions as in requirements.txt
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.9.1
transformers==5.21.0
gradio==6.30.0
onnxruntime==1.31.1
matplotlib==3.11.2
pandas
huggingface_hub
"""

MODEL_CARD = """---
language: ru
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: ai-forever/ruBert-base
datasets:
- d0rj/geo-reviews-dataset-2023
tags:
- sentiment-analysis
- russian
- reviews
- distillation
- onnx
---
# ru-reviews-rubert: тональность, 1–5★ и категория для русских отзывов

ruBert-base, полученная **дистилляцией** из multi-task ruRoberta-large на отзывах с Яндекс Карт.
Одна модель — три головы: тональность (негатив / нейтрально / позитив), точная оценка 1–5★ и
категория места (топ-15 рубрик + «другое»). Демо: [Space](https://huggingface.co/spaces/{space}), код и
подробности обучения: [GitHub]({github}).

| Метрика (отложенный тест) | Эта модель | Учитель (large) |
|---|---|---|
| Тональность, macro-F1 | **{f1}** | 0.820 |
| Сложные отзывы (hard-test) | 0.832 | 0.835 |
| 1–5★: MAE / ошибка ≤ 1★ | {mae}★ / {w1:.1%} | 0.353★ / 90.6% |
| Категория: accuracy / top-3 | {cat_acc} / {top3:.1%} | 0.753 / 95.2% |
| Точность при 10% опечаток | 0.782 | 0.770 |
| CPU, ONNX int8: мс на отзыв / размер | 7.4 / 171 МБ | 29 / 341 МБ |

## Использование

Тональность — обычный `pipeline` (головы 1–5★ и категории лежат в `heads.pt`, их читает `sentiment.Analyzer` из репозитория на GitHub):

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{model}")
clf("Еда вкусная, но официант нахамил.")
```

Уверенность модели завышена; температура для калибровки — в `reports/calibration.json` (T ≈ {t}, ECE {ece_before} → {ece_after}).
`onnx/model.int8.onnx` — все три головы в ONNX int8 для CPU (выходы `sentiment`, `stars`, `category`).

Ограничения: «нейтрально» размечено по звёздам (3★), вне домена (рецензии на фильмы) macro-F1 0.62; сарказм без явных маркеров — слабое место.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("user")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--model-repo", default="ru-reviews-rubert")
    p.add_argument("--space", default="ru-reviews-demo")
    a = p.parse_args()
    api = HfApi()
    model_id, space_id = f"{a.user}/{a.model_repo}", f"{a.user}/{a.space}"
    src = Path(a.model_dir)

    test = json.loads((src / "reports" / "multitask_test.json").read_text())
    calib = json.loads((src / "reports" / "calibration.json").read_text(encoding="utf-8"))
    card = MODEL_CARD.format(
        model=model_id, space=space_id, github=GITHUB, f1=round(test["label"]["f1_macro"], 3),
        mae=round(test["stars"]["mae"], 3), w1=test["stars"]["within_1"], cat_acc=round(test["category"]["accuracy"], 3),
        top3=test["category"]["top3"], t=round(calib["temperature"], 2), ece_before=calib["до"]["ece"],
        ece_after=calib["после T-scaling"]["ece"])

    api.create_repo(model_id, exist_ok=True)
    api.upload_folder(repo_id=model_id, folder_path=src, ignore_patterns=["training_args.bin"],
                      commit_message="Distilled ruBert-base: sentiment + stars + category, int8 ONNX")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=model_id)
    print("model:", f"https://huggingface.co/{model_id}")

    with tempfile.TemporaryDirectory() as d:
        for f in ("app.py", "sentiment.py"):
            shutil.copy(f, d)
        Path(d, "requirements.txt").write_text(SPACE_REQUIREMENTS)
        Path(d, "README.md").write_text(SPACE_README.format(model=model_id, github=GITHUB), encoding="utf-8")
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.add_space_variable(space_id, "MODEL_PATH", model_id)
        api.add_space_variable(space_id, "DEVICE", "cpu")
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=d)
    print("demo:", f"https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
