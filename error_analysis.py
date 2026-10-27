"""Error analysis: where the sentiment model is wrong on real reviews, and which words misled it.

Usage: python error_analysis.py [MODEL_DIR]  → reports/error_analysis.md (+ MODEL_DIR/reports/)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from sentiment import LABELS, RU, Analyzer
from train import load_balanced

GROUPS = {  # heuristic slices of the test set, each checked separately
    "похвала в тексте (спасибо, отлично…)": re.compile(r"\b(спасибо|отличн|прекрасн|супер|браво|замечательн)", re.I),
    "отрицания (не/ни…)": re.compile(r"\b(не|ни|нет|никогда|ничего)\b", re.I),
    "«но / однако» — смешанный отзыв": re.compile(r"\b(но|однако|зато|хотя)\b", re.I),
    "«нормально / неплохо / так себе»": re.compile(r"\b(нормальн|неплох|так себе|сойд[её]т|средн|пойд[её]т)", re.I),
    "короткий (≤ 8 слов)": None,
}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model"
    an = Analyzer(model)
    _, _, test = load_balanced(12000)
    texts, gold = list(test["text"]), np.array(test["label"])
    p = an.probs(texts)
    pred = p.argmax(1)
    wrong = pred != gold
    md = [f"# Анализ ошибок (`{model}`)\n",
          f"Тест: {len(texts)} реальных отзывов, ошибок **{wrong.sum()}** ({wrong.mean():.1%}).\n",
          "## Куда уходят ошибки\n", "| Истина \\ Модель | " + " | ".join(RU[l] for l in LABELS) + " |",
          "|---|---|---|---|"]
    for g in range(3):
        md.append(f"| {RU[LABELS[g]]} | " + " | ".join(str(int(((gold == g) & (pred == q)).sum())) for q in range(3)) + " |")
    far = int(((gold == 0) & (pred == 2)).sum() + ((gold == 2) & (pred == 0)).sum())
    md += ["", f"Грубых ошибок (негатив ↔ позитив): **{far}** ({far / len(texts):.1%}); "
               f"остальные {wrong.sum() - far} — на границе с «нейтрально».\n"]

    md += ["## Ошибки по типам отзывов\n", "| Срез | Отзывов | Ошибок | Доля ошибок |", "|---|---|---|---|",
           f"| все | {len(texts)} | {wrong.sum()} | {wrong.mean():.1%} |"]
    slices = {}
    for name, rx in GROUPS.items():
        mask = np.array([len(t.split()) <= 8 if rx is None else bool(rx.search(t)) for t in texts])
        slices[name] = {"n": int(mask.sum()), "error_rate": round(float(wrong[mask].mean()), 3)}
        md.append(f"| {name} | {mask.sum()} | {wrong[mask].sum()} | {wrong[mask].mean():.1%} |")

    # The most confident mistakes between the extremes, with the words that drove them
    md += ["", "## Самые уверенные грубые ошибки и слова, которые к ним привели\n",
           "Вес слова — Integrated Gradients (sentiment.Analyzer.explain): `+` тянет к позитиву, `−` к негативу.\n"]
    idx = [i for i in np.argsort(-p.max(1)) if {gold[i], pred[i]} == {0, 2}][:8]
    for i in idx:
        w = sorted(an.explain(texts[i]), key=lambda x: -abs(x[1]))[:5]
        short = texts[i] if len(texts[i]) < 300 else texts[i][:300] + "…"
        md += [f"- **{RU[LABELS[gold[i]]]} → {RU[LABELS[pred[i]]]}** ({p[i].max():.2f}): {short}",
               "  - слова: " + ", ".join(f"«{x}» {y:+.2f}" for x, y in w)]

    # Mined hard reviews (dev): what still fails
    hard = [json.loads(x) for x in Path("data/hard_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    hp = an.probs([h["text"] for h in hard]).argmax(1)
    miss = [h for h, q in zip(hard, hp) if q != h["label"]]
    md += ["", f"## Сложные реальные отзывы (hard-dev): ошибок {len(miss)} из {len(hard)}\n",
           "Примеры, где модель ошиблась, и слова, на которые она опиралась:\n"]
    for h in miss[:6]:
        w = sorted(an.explain(h["text"]), key=lambda x: -abs(x[1]))[:4]
        short = h["text"] if len(h["text"]) < 250 else h["text"][:250] + "…"
        md += [f"- истина **{RU[LABELS[h['label']]]}**: {short}",
               "  - слова: " + ", ".join(f"«{x}» {y:+.2f}" for x, y in w)]

    report = "\n".join(md) + "\n"
    for d in (Path("reports"), Path(model) / "reports"):
        d.mkdir(exist_ok=True)
        (d / "error_analysis.md").write_text(report, encoding="utf-8")
        (d / "error_slices.json").write_text(json.dumps(slices, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
