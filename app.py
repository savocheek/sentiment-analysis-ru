"""Gradio demo: sentiment of a Russian review. Run: python app.py"""
import os

import gradio as gr
from transformers import pipeline

MODEL = os.getenv("MODEL_PATH", "model")  # local folder or a Hugging Face Hub id
RU = {"negative": "😠 Негатив", "neutral": "😐 Нейтрально", "positive": "😊 Позитив"}

clf = pipeline("text-classification", model=MODEL, top_k=None, truncation=True, max_length=256)


def predict(text):
    if not text or not text.strip():
        return {}
    return {RU[p["label"]]: p["score"] for p in clf(text)[0]}


EXAMPLES = [
    "Отличная кофейня! Бариста знает своё дело, круассаны свежие, обязательно вернёмся.",
    "Ждали заказ час, официант нахамил, суп холодный. Больше ни ногой.",
    "Обычный магазин. Цены средние, ассортимент так себе, но до дома близко.",
    "Номер чистый, но шумоизоляция никакая и завтрак скудный за такие деньги.",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Текст отзыва", placeholder="Вставьте отзыв о месте, товаре или сервисе…"),
    outputs=gr.Label(num_top_classes=3, label="Тональность"),
    examples=EXAMPLES,
    title="Тональность отзывов на русском",
    description="rubert-tiny2, дообученный на 36 000 отзывов с Яндекс Карт (негатив / нейтрально / позитив).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
