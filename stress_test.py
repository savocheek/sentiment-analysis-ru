"""Stress test under artificial conditions: hand-made tricky reviews, noise robustness, edge inputs, speed.

Usage: python stress_test.py [model_dir] [report.md]  (defaults: model, reports/stress_test.md)
"""
import random
import sys
import re
import time
from pathlib import Path

from transformers import pipeline

from augment import generate
from train import LABELS, load_balanced

N, U, P = "negative", "neutral", "positive"

# (category, text, expected label) — written by hand, not from the dataset
CASES = [
    ("отрицание", "Совсем не вкусно и совсем не дёшево.", N),
    ("отрицание", "Ни разу не пожалели, что пришли сюда.", P),
    ("отрицание", "Не могу сказать, что мне что-то не понравилось.", P),
    ("отрицание", "Не советую, не тратьте время.", N),
    ("сарказм", "Ну просто супер сервис: ждали заказ всего два часа.", N),
    ("сарказм", "Спасибо огромное за холодную пиццу и хамство, очень приятно.", N),
    ("сарказм", "Браво, умудрились испортить даже чай.", N),
    ("смешанный", "Еда вкусная, но обслуживание медленное, цены завышены.", U),
    ("смешанный", "Место неплохое, но ничего особенного. Есть и получше.", U),
    ("смешанный", "Персонал приветливый, а вот в номере было грязно.", U),
    ("сленг/эмодзи", "Это просто огонь 🔥🔥🔥 всем рекомендую", P),
    ("сленг/эмодзи", "Треш полный, кринж, больше не приду 🤮", N),
    ("сленг/эмодзи", "Кайф, вайб топовый, бариста красавчик", P),
    ("опечатки", "ужастный сервес, официант хомил, ни кагда больше", N),
    ("опечатки", "очнь вкуснно и уютнo, абязательно вирнемся", P),
    ("капс/пунктуация", "ОТВРАТИТЕЛЬНО!!! НИКОМУ НЕ СОВЕТУЮ!!!", N),
    ("капс/пунктуация", "ЛУЧШЕЕ МЕСТО В ГОРОДЕ!!!!!!", P),
    ("другой домен", "Кроссовки развалились через неделю, продавец не отвечает.", N),
    ("другой домен", "Телефон пришёл быстро, упакован отлично, всё работает.", P),
    ("другой домен", "Фильм затянутый, но актёры играют хорошо.", U),
    ("другой язык", "Great place, friendly staff, loved it!", P),
    ("другой язык", "Terrible service, never again.", N),
    ("короткий", "Отлично!", P),
    ("короткий", "Ужас.", N),
    ("короткий", "Пойдёт.", U),
]

# Second hand-written set: written after the augmentation templates, deliberately in other
# phrasings, and never used for training — the honest check that the model learned the idea.
FRESH = [
    ("сарказм", "Великолепно: курьер опоздал на три часа и ещё обиделся.", N),
    ("сарказм", "Какой сервис! Мне даже разрешили самому убрать за собой стол.", N),
    ("сарказм", "Прекрасно провели час в очереди, незабываемые впечатления.", N),
    ("сарказм", "Очень рекомендую, если хотите уйти голодными и без денег.", N),
    ("сарказм", "Чудесный отель, особенно плесень в душе — придаёт шарм.", N),
    ("сарказм", "Ну конечно, кто же отвечает на звонки в рабочее время.", N),
    ("сарказм", "Пять звёзд за креативность: перепутали заказ трижды.", N),
    ("сарказм", "Повар, видимо, большой любитель соли. Очень большой.", N),
    ("искренне", "Спасибо мастеру за аккуратную работу, всё идеально.", P),
    ("искренне", "Браво шеф-повару, давно не ели такой сочный стейк.", P),
    ("искренне", "Великолепно отдохнули, персонал предугадывал каждое желание.", P),
    ("отрицание → позитив", "Ни одной претензии к курьеру, всё привёз вовремя.", P),
    ("отрицание → позитив", "Ничуть не хуже дорогих ресторанов, а цены ниже.", P),
    ("отрицание → позитив", "Ожидал подвоха, но его не оказалось — всё честно.", P),
    ("отрицание → позитив", "Никогда ещё не встречал настолько внимательного персонала.", P),
    ("отрицание → позитив", "Не пожалела ни об одном потраченном рубле.", P),
    ("отрицание → позитив", "Не было ни минуты, чтобы детям стало скучно.", P),
    ("отрицание → негатив", "Не вкусно, не быстро и не дёшево.", N),
    ("отрицание → негатив", "Ни уюта, ни вкуса, ни нормального сервиса.", N),
    ("отрицание → негатив", "Не понимаю, за что тут хорошие оценки.", N),
    ("отрицание → негатив", "Нормальной еды здесь не дождёшься.", N),
    ("отрицание → негатив", "Ни разу персонал даже не поздоровался.", N),
    ("отрицание → негатив", "Не советую даже врагу.", N),
]

EDGE = ["😡😡😡", "👍", "5/5", "1/10", "....", "<div>&nbsp;</div>", "а", "Хорошо. " * 400]


def noise(text, kind, rng):
    if kind == "нижний регистр":
        return text.lower()
    if kind == "без пунктуации":
        return re.sub(r"[^\w\s]", "", text)
    if kind == "опечатки 10%":  # swap adjacent letters in ~10% of positions
        c = list(text)
        for i in range(len(c) - 1):
            if c[i].isalpha() and rng.random() < 0.1:
                c[i], c[i + 1] = c[i + 1], c[i]
        return "".join(c)
    if kind == "−30% слов":
        words = text.split()
        kept = [w for w in words if rng.random() > 0.3]
        return " ".join(kept or words[:1])
    if kind == "только первые 10 слов":
        return " ".join(text.split()[:10])
    raise ValueError(kind)


def score_cases(top, title, cases, show_rows=True):
    preds = top([t for _, t, _ in cases])
    cats, rows = {}, []
    for (cat, text, exp), got in zip(cases, preds):
        cats.setdefault(cat, []).append(exp == got)
        rows.append(f"| {cat} | {text} | {exp} | {got} | {'✅' if exp == got else '❌'} |")
    total = sum(sum(v) for v in cats.values())
    md = [f"## {title}: {total}/{len(cases)}\n", "| Категория | Верно |", "|---|---|"]
    md += [f"| {c} | {sum(v)}/{len(v)} |" for c, v in cats.items()]
    if show_rows:
        md += ["", "| Категория | Текст | Ожидалось | Модель | |", "|---|---|---|---|---|", *rows]
    return md + [""]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model"
    out = sys.argv[2] if len(sys.argv) > 2 else "reports/stress_test.md"
    clf = pipeline("text-classification", model=model, truncation=True, max_length=256)
    top = lambda texts: [r["label"] for r in clf(texts, batch_size=32)]
    md = [f"# Стресс-тест в искусственных условиях (`{model}`)\n"]

    # 1. Hand-made tricky reviews
    md += score_cases(top, "1. Сложные отзывы, написанные вручную", CASES)
    md += score_cases(top, "1б. Новый рукописный набор (сарказм и отрицания, в обучении нет)", FRESH)
    held = [(c, t, LABELS[l]) for c, t, l in generate(1)]
    md += score_cases(top, "1в. Отложенные шаблоны из augment.py (в обучении нет)", held, show_rows=False)

    # 2. Noise robustness on real test reviews
    _, _, test = load_balanced(12000)
    sample = test.shuffle(seed=0).select(range(600))
    texts, gold = list(sample["text"]), [LABELS[l] for l in sample["label"]]
    base = top(texts)
    acc = lambda p: sum(a == b for a, b in zip(p, gold)) / len(gold)
    md += ["## 2. Устойчивость к шуму (600 реальных отзывов из теста)\n",
           "| Искажение | Accuracy | Ответ изменился |", "|---|---|---|",
           f"| нет (оригинал) | {acc(base):.3f} | — |"]
    rng = random.Random(0)
    for kind in ["нижний регистр", "без пунктуации", "опечатки 10%", "−30% слов", "только первые 10 слов"]:
        p = top([noise(t, kind, rng) for t in texts])
        flip = sum(a != b for a, b in zip(p, base)) / len(base)
        md.append(f"| {kind} | {acc(p):.3f} | {flip:.1%} |")
    md.append("")

    # 3. Edge inputs: must not crash, scores must be valid
    md += ["## 3. Граничные входы (не должно падать)\n", "| Вход | Ответ | Уверенность |", "|---|---|---|"]
    for t in EDGE:
        r = clf(t)[0]
        assert r["label"] in LABELS and 0 <= r["score"] <= 1
        shown = t if len(t) < 30 else f"«Хорошо.» × 400 ({len(t)} симв.)"
        md.append(f"| `{shown}` | {r['label']} | {r['score']:.2f} |")
    md.append("")

    # 4. Speed
    one = "Еда вкусная, но обслуживание медленное."
    clf(one)
    t0 = time.perf_counter()
    for _ in range(50):
        clf(one)
    single = (time.perf_counter() - t0) / 50 * 1000
    t0 = time.perf_counter()
    top(texts)
    batch = len(texts) / (time.perf_counter() - t0)
    md += ["## 4. Скорость\n", f"- Один отзыв: **{single:.1f} мс** (устройство: {clf.device})",
           f"- Пачкой по 32: **{batch:.0f} отзывов/с**", ""]

    report = "\n".join(md)
    Path(out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
