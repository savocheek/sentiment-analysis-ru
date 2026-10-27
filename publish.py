"""Publish the model to the Hugging Face Hub and the demo to a Space (free CPU).

Needs your own token: `huggingface-cli login` (or HF_TOKEN). Nothing is uploaded without it.
Usage: python publish.py HF_USER [--model-dir model] [--model-repo ru-reviews-rubert] [--space ru-reviews-demo]
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_README = """---
title: Анализ отзывов на русском
emoji: 💬
colorFrom: green
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---
Тональность, оценка 1–5★, категория места и аспекты для русскоязычных отзывов. Модель: {model}
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

    api.create_repo(model_id, exist_ok=True)
    api.upload_folder(repo_id=model_id, folder_path=a.model_dir, ignore_patterns=["training_args.bin"])
    print("model:", f"https://huggingface.co/{model_id}")

    with tempfile.TemporaryDirectory() as d:
        for f in ("app.py", "sentiment.py", "requirements.txt"):
            shutil.copy(f, d)
        Path(d, "README.md").write_text(SPACE_README.format(model=model_id), encoding="utf-8")
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.add_space_variable(space_id, "MODEL_PATH", model_id)
        api.add_space_variable(space_id, "DEVICE", "cpu")
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=d)
    print("demo:", f"https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
