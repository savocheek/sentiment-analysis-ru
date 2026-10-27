"""Mine real "hard" reviews where wording contradicts the rating, from reviews NOT in the main sample.

- 1–2★ reviews with ≥2 praise words ("отлично", "спасибо", "браво"…) → negative
- 4–5★ reviews with negation idioms ("ни разу не", "не пожалели", "без нареканий"…) → positive

Split 40/20/40 into train pool / dev (for choosing experiments) / test (opened once, at the end).
Usage: python mine_hard.py  → data/hard_{train,dev,test}.jsonl
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from train import load_balanced

PRAISE = re.compile(r"\b(спасибо|браво|молодцы|супер|отличн|прекрасн|замечательн|великолепн|шикарн|чудесн|восхитительн|лучш)", re.I)
NEGATION = re.compile(r"не пожалел|ни разу не|не разочаров|никаких (претензий|нареканий|проблем)|не к чему придраться|"
                      r"не подв[её]л|ничуть не|нисколько не|ни одного минуса|не хуже|без нареканий|не ошиблись|ни о ч[её]м не жале", re.I)


def main():
    used = set()
    for split in load_balanced(12000):
        used.update(split["text"])
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    neg, pos, seen = [], [], set()
    for t, r in zip(ds["text"], ds["rating"]):
        t = (t or "").replace("\\n", " ")
        if len(t) <= 20 or t in used or t in seen:
            continue
        seen.add(t)
        if r in (1, 2) and len(PRAISE.findall(t)) >= 2:
            neg.append({"text": t, "label": 0})
        elif r in (4, 5) and NEGATION.search(t):
            pos.append({"text": t, "label": 2})
    rng = random.Random(7)
    rng.shuffle(neg)
    rng.shuffle(pos)
    pos = pos[:len(neg)]  # balance: equal negative / positive
    out = Path("data")
    out.mkdir(exist_ok=True)
    n = len(neg)
    cuts = {"train": (0, int(n * .4)), "dev": (int(n * .4), int(n * .6)), "test": (int(n * .6), n)}
    for name, (a, b) in cuts.items():
        rows = neg[a:b] + pos[a:b]
        rng.shuffle(rows)
        (out / f"hard_{name}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows))
        print(name, len(rows))
    sets = [{x["text"] for x in map(json.loads, (out / f"hard_{s}.jsonl").read_text().splitlines())} for s in cuts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]) and not (used & set().union(*sets))


if __name__ == "__main__":
    main()
