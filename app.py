"""Gradio demo: one review → sentiment with word highlights, exact 1–5★, place category, aspects;
CSV batch analysis; model card. Run: python app.py  (MODEL_PATH=…, DEVICE=cpu)"""
import json
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sentiment import LABELS, MODEL, RU, Analyzer

an = Analyzer()
MULTI = an.heads is not None
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
COLORS = {"negative": "#d9534f", "neutral": "#9e9e9e", "positive": "#3fa45b"}
ASPECT_COLS = ["Аспект", "Фрагмент", "Тональность", "Полярность"]


def predict(text):
    """{'😠 Негатив': p, ...} — kept for tests and simple callers."""
    return {f"{EMOJI[l]} {RU[l]}": p for l, p in an.predict(text).items()}


def analyse(text):
    scores = predict(text)
    if not scores:
        return {}, "", [], {}, "", {}, pd.DataFrame(columns=ASPECT_COLS), ""
    note = "⚠️ Модель не уверена: отзыв, скорее всего, смешанный." if max(scores.values()) < 0.55 else ""
    words = [(w + " ", "к позитиву" if x > 0.2 else "к негативу" if x < -0.2 else None) for w, x in an.explain(text)]

    stars, mean = an.stars(text)
    stars_label = {"★" * s + "☆" * (5 - s): p for s, p in stars.items()}
    stars_note = f"Ожидаемая оценка: **{mean} ★**" if mean else "Нужна multi-task модель (train_multi.py)."
    category = an.category(text)

    asp = an.aspects(text)
    rows = [[a["aspect"], a["fragment"], f"{EMOJI[a['label']]} {RU[a['label']]}", a["polarity"]] for a in asp]
    plus = sorted({a["aspect"] for a in asp if a["label"] == "positive"})
    minus = sorted({a["aspect"] for a in asp if a["label"] == "negative"})
    summary = (f"**Хвалят:** {', '.join(plus) or '—'}  \n**Ругают:** {', '.join(minus) or '—'}" if asp
               else "Аспекты не найдены: в отзыве нет слов из словаря (еда, персонал, цены…).")
    return scores, note, words, stars_label, stars_note, category, pd.DataFrame(rows, columns=ASPECT_COLS), summary


def batch(file, column):
    if file is None:
        raise gr.Error("Загрузите CSV-файл с отзывами.")
    df = pd.read_csv(file, sep=None, engine="python")
    col = column or next((c for c in df.columns if df[c].dtype == object), None)
    if col not in df.columns:
        raise gr.Error(f"Нет текстовой колонки «{col}». Колонки: {', '.join(map(str, df.columns))}")
    if len(df) > 5000:
        raise gr.Error("Не больше 5 000 отзывов за раз.")
    out = an.batch(df[col].fillna("").astype(str).tolist())
    df["sentiment"] = [RU.get(l, "") for l in out["labels"]]
    df["confidence"] = out["scores"]
    if "stars" in out:
        df["stars"], df["category"] = out["stars"], out["category"]

    share = out["share"]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([RU[l] for l in LABELS], [share[l] * 100 for l in LABELS], color=[COLORS[l] for l in LABELS])
    for i, l in enumerate(LABELS):
        ax.text(share[l] * 100 + 1, i, f"{share[l]:.0%}", va="center")
    ax.set_xlim(0, 110), ax.set_xlabel("% отзывов"), ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    summary = (f"**{out['analysed']}** отзывов проанализировано из {out['total']}. "
               f"Негатив: **{share['negative']:.0%}**, нейтрально: **{share['neutral']:.0%}**, "
               f"позитив: **{share['positive']:.0%}**.")
    if "mean_stars" in out:
        top = pd.Series([c for c in out["category"] if c]).value_counts().head(3)
        summary += (f"  \nСредняя ожидаемая оценка: **{out['mean_stars']} ★**. "
                    f"Чаще всего: {', '.join(f'{c} ({n})' for c, n in top.items())}.")
    complaints = pd.DataFrame([[c["aspect"], c["negative"], c["positive"], c["example"]] for c in out["complaints"]],
                              columns=["Аспект", "Жалоб", "Похвал", "Пример жалобы"])
    path = tempfile.NamedTemporaryFile(delete=False, suffix="_sentiment.csv").name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return summary, fig, complaints, path


def columns(file):
    if file is None:
        return gr.Dropdown(choices=[], value=None)
    df = pd.read_csv(file, sep=None, engine="python", nrows=50)
    text_cols = [c for c in df.columns if df[c].dtype == object]
    return gr.Dropdown(choices=list(df.columns), value=text_cols[0] if text_cols else None)


def model_card():
    md = [f"**Модель:** `{MODEL}`" + (" — multi-task: тональность + 1–5★ + категория" if MULTI else "")]
    for name, f in (("Тональность, 1–5★, категория (тест)", Path(MODEL) / "reports" / "multitask_test.json"),
                    ("Бейзлайн TF-IDF + LogReg (тест)", Path("reports/baselines.json")),
                    ("Аспекты", Path(MODEL) / "reports" / "aspects.json")):
        if f.exists():
            md.append(f"**{name}**\n```json\n{f.read_text(encoding='utf-8')[:1500]}\n```")
    return "\n\n".join(md)


EXAMPLES = [
    "Отличная кофейня! Бариста знает своё дело, круассаны свежие, обязательно вернёмся.",
    "Ждали заказ час, официант нахамил, суп холодный. Больше ни ногой.",
    "Еда вкусная и порции большие, но официант хамил, а цены завышены.",
    "Номер чистый, но шумоизоляция никакая и завтрак скудный за такие деньги.",
    "Ни разу не пожалели, что выбрали этот отель.",
    "Спасибо огромное за холодную пиццу и хамство, очень приятно.",
    "Записалась на маникюр, мастер опоздала на 20 минут, но покрытие держится уже три недели.",
]

with gr.Blocks(title="Тональность отзывов") as demo:
    gr.Markdown("# Анализ отзывов на русском\n"
                "ruBERT, дообученный на отзывах с Яндекс Карт: тональность, оценка 1–5★, категория места и аспекты.")
    with gr.Tab("Отзыв"):
        text = gr.Textbox(lines=4, label="Текст отзыва", placeholder="Вставьте отзыв о месте, товаре или сервисе…")
        btn = gr.Button("Анализировать", variant="primary")
        gr.Examples(EXAMPLES, text)
        with gr.Tab("Тональность"):
            label = gr.Label(num_top_classes=3, label="Тональность")
            note = gr.Markdown()
            words = gr.HighlightedText(label="Какие слова повлияли на ответ", show_legend=True,
                                       color_map={"к позитиву": "#3fa45b", "к негативу": "#d9534f"})
        with gr.Tab("Оценка 1–5★"):
            stars = gr.Label(num_top_classes=5, label="Какую оценку поставил автор")
            stars_note = gr.Markdown()
        with gr.Tab("Категория места"):
            category = gr.Label(num_top_classes=5, label="О каком заведении отзыв (топ-15 категорий + «другое»)")
        with gr.Tab("Аспекты"):
            aspect_summary = gr.Markdown()
            aspects = gr.Dataframe(label="Фразы отзыва: аспект и тональность (полярность от −1 до +1)", wrap=True)
        outs = [label, note, words, stars, stars_note, category, aspects, aspect_summary]
        btn.click(analyse, text, outs)
        text.submit(analyse, text, outs)
    with gr.Tab("Пакетный анализ CSV"):
        gr.Markdown("Загрузите CSV с колонкой отзывов: доля негатива, график, топ жалоб по аспектам"
                    + (", оценка и категория для каждого отзыва." if MULTI else "."))
        with gr.Row():
            file = gr.File(label="CSV", file_types=[".csv"])
            col = gr.Dropdown(label="Колонка с текстом", choices=[], allow_custom_value=True)
        run = gr.Button("Анализировать файл", variant="primary")
        summary = gr.Markdown()
        plot = gr.Plot(label="Распределение тональности")
        complaints = gr.Dataframe(label="Топ жалоб по аспектам", wrap=True)
        result = gr.File(label="Результат с разметкой")
        file.change(columns, file, col)
        run.click(batch, [file, col], [summary, plot, complaints, result])
    with gr.Tab("О модели"):
        gr.Markdown(model_card())

if __name__ == "__main__":
    demo.launch()
