"""Evaluate aspect sentiment (sentiment.py: clauses + aspect lexicon + model polarity per clause).

There are no aspect labels in the dataset, so two checks:
1. Proxy on 300 real test reviews: in 1★ reviews every aspect mention should come out negative, in 5★ — positive
   (agreement with the stars of the whole review; mixed 3★ reviews are reported separately).
2. Hand-written mixed reviews (never used for training) with the expected {aspect: sentiment}.

Usage: python aspect_eval.py [MODEL_DIR]  → reports/aspects.json (+ MODEL_DIR/reports/aspects.json)
"""
import json
import sys
from pathlib import Path

from sentiment import Analyzer
from tasks import load_multitask

N, P, U = "negative", "positive", "neutral"
PER_GROUP = 100  # real test reviews per star group (1★, 5★, 3★): 300 in total, fast enough on CPU
HAND = [
    ("Еда вкусная, но официант нахамил.", {"Еда и напитки": P, "Персонал": N}),
    ("Номер чистый и просторный, но завтрак скудный.", {"Проживание": P, "Чистота": P, "Еда и напитки": N}),
    ("Цены кусаются, зато персонал очень внимательный.", {"Цены": N, "Персонал": P}),
    ("Ждали заказ сорок минут, хотя пицца оказалась отличной.", {"Скорость и ожидание": N, "Еда и напитки": P}),
    ("Интерьер уютный, музыка приятная, но в туалете грязно.", {"Атмосфера": P, "Чистота": N}),
    ("Курьер привёз всё быстро, а вот упаковка была порвана.", {"Доставка": N}),
    ("Удобное расположение рядом с метро, но парковки нет совсем.", {"Расположение": N}),
    ("Ассортимент огромный, цены приятные.", {"Товар и ассортимент": P, "Цены": P}),
    ("Мастер сделала стрижку идеально, но администратор была груба.", {"Персонал": N}),
    ("Кофе холодный, бариста равнодушный.", {"Еда и напитки": N, "Персонал": N}),
    ("Очень чисто, персонал приветливый, обязательно вернёмся.", {"Чистота": P, "Персонал": P}),
    ("Еда так себе, зато атмосфера отличная.", {"Еда и напитки": N, "Атмосфера": P}),
    ("Обслуживание быстрое, но порции маленькие.", {"Персонал": P, "Еда и напитки": N}),
    ("В номере пахло сыростью, кондиционер не работал.", {"Проживание": N, "Чистота": N}),
    ("Цены ниже, чем у конкурентов, и выбор товаров хороший.", {"Цены": P, "Товар и ассортимент": P}),
    ("Очередь на кассе огромная, кассир медленный.", {"Скорость и ожидание": N, "Персонал": N}),
    ("Шумно, но кормят вкусно.", {"Атмосфера": N, "Еда и напитки": P}),
    ("Доставка опоздала на час, суп остыл.", {"Доставка": N, "Скорость и ожидание": N, "Еда и напитки": N}),
    ("Врач внимательный, всё объяснил, но ждали приёма долго.", {"Персонал": P, "Скорость и ожидание": N}),
    ("Товар качественный, но размер не подошёл и обменять отказались.", {"Товар и ассортимент": N}),
]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model"
    an = Analyzer(model)

    # 1. Proxy on real reviews: polarity of every aspect mention vs the stars of the whole review
    (_, _, test), _ = load_multitask()
    proxy = {}
    for stars, want in ((0, N), (4, P), (2, None)):
        texts = [t for t, s in zip(test["text"], test["stars"]) if s == stars][:PER_GROUP]
        found = [an.aspects(t) for t in texts]
        mentions = [a["label"] for f in found for a in f]
        row = {"reviews": len(texts), "with_aspects": round(sum(map(bool, found)) / len(texts), 3),
               "mentions": len(mentions)}
        if want:
            row["agree"] = round(sum(m == want for m in mentions) / len(mentions), 3)
            row["opposite"] = round(sum(m == (P if want == N else N) for m in mentions) / len(mentions), 3)
        else:  # 3★: how often a review mixes praise and complaints
            row["mixed"] = round(sum({N, P} <= {a["label"] for a in f} for f in found) / len(texts), 3)
        proxy[f"{stars + 1}★"] = row
        print(stars + 1, row, flush=True)

    # 2. Hand-written mixed reviews
    tp = fp = fn = right = 0
    misses = []
    for text, gold in HAND:
        got = {}
        for a in an.aspects(text):
            got.setdefault(a["aspect"], a["label"])  # first clause that mentions the aspect
        tp += len(got.keys() & gold.keys())
        fp += len(got.keys() - gold.keys())
        fn += len(gold.keys() - got.keys())
        for a in got.keys() & gold.keys():
            right += got[a] == gold[a]
            if got[a] != gold[a]:
                misses.append(f"{text} → {a}: {got[a]} (ожидалось {gold[a]})")
    hand = {"cases": len(HAND), "aspect_precision": round(tp / (tp + fp), 3), "aspect_recall": round(tp / (tp + fn), 3),
            "sentiment_accuracy": round(right / tp, 3), "sentiment_misses": misses}
    print(json.dumps(hand, ensure_ascii=False, indent=1))

    report = json.dumps({"model": model, "proxy_real_test": proxy, "hand_mixed": hand}, ensure_ascii=False, indent=2)
    for d in (Path("reports"), Path(model) / "reports"):
        d.mkdir(exist_ok=True)
        (d / "aspects.json").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
