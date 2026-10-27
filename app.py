"""Gradio demo: one review (sentiment, word highlights, aspects) and CSV batch analysis. Run: python app.py"""
import tempfile

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sentiment import LABELS, MODEL, RU, Analyzer

an = Analyzer()
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
COLORS = {"negative": "#d9534f", "neutral": "#9e9e9e", "positive": "#3fa45b"}


def predict(text):
    """{'😠 Негатив': p, ...} — kept for tests and simple callers."""
    return {f"{EMOJI[l]} {RU[l]}": p for l, p in an.predict(text).items()}


def analyse(text):
    scores = predict(text)
    if not scores:
        return {}, [], pd.DataFrame(columns=["Аспект", "Фрагмент", "Тональность", "Полярность"]), ""
    words = [(w + " ", "к позитиву" if x > 0.2 else "к негативу" if x < -0.2 else None) for w, x in an.explain(text)]
    rows = [[a["aspect"], a["fragment"], f"{EMOJI[a['label']]} {RU[a['label']]}", a["polarity"]]
            for a in an.aspects(text)]
    best = max(scores.values())
    note = "⚠️ Модель не уверена: отзыв, скорее всего, смешанный." if best < 0.55 else ""
    return (scores, words, pd.DataFrame(rows, columns=["Аспект", "Фрагмент", "Тональность", "Полярность"]), note)


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


EXAMPLES = [
    "Отличная кофейня! Бариста знает своё дело, круассаны свежие, обязательно вернёмся.",
    "Ждали заказ час, официант нахамил, суп холодный. Больше ни ногой.",
    "Еда вкусная и порции большие, но официант хамил, а цены завышены.",
    "Номер чистый, но шумоизоляция никакая и завтрак скудный за такие деньги.",
    "Ни разу не пожалели, что выбрали этот отель.",
    "Спасибо огромное за холодную пиццу и хамство, очень приятно.",
]

with gr.Blocks(title="Тональность отзывов") as demo:
    gr.Markdown("# Тональность отзывов на русском\n"
                f"ruBERT, дообученный на отзывах с Яндекс Карт · модель: `{MODEL}`")
    with gr.Tab("Один отзыв"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(lines=5, label="Текст отзыва", placeholder="Вставьте отзыв о месте, товаре или сервисе…")
                btn = gr.Button("Анализировать", variant="primary")
                gr.Examples(EXAMPLES, text)
            with gr.Column():
                label = gr.Label(num_top_classes=3, label="Тональность")
                note = gr.Markdown()
        words = gr.HighlightedText(label="Какие слова повлияли на ответ", show_legend=True,
                                   color_map={"к позитиву": "#3fa45b", "к негативу": "#d9534f"})
        aspects = gr.Dataframe(label="Аспекты: о чём отзыв и с какой тональностью (полярность от −1 до +1)", wrap=True)
        btn.click(analyse, text, [label, words, aspects, note])
        text.submit(analyse, text, [label, words, aspects, note])
    with gr.Tab("Пакетный анализ CSV"):
        gr.Markdown("Загрузите CSV с колонкой отзывов: получите долю негатива, график и топ жалоб по аспектам.")
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

if __name__ == "__main__":
    demo.launch()
