"""Inference core shared by the Gradio demo (app.py) and the REST API (api.py).

- clean():    strip HTML, entities and escaped newlines before the model sees the text
- Analyzer:   probabilities, word-level explanations (occlusion), aspect sentiment, CSV batch summary
"""
import html
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.getenv("MODEL_PATH", "model")  # local folder or a Hugging Face Hub id
LABELS = ["negative", "neutral", "positive"]
RU = {"negative": "Негатив", "neutral": "Нейтрально", "positive": "Позитив"}

# Aspect = what a clause talks about, found by word stems (domain: places on Yandex Maps).
# Sentiment of each clause comes from the model, so the lexicon only decides "about what", never "how".
ASPECTS = {
    "Персонал": r"персонал|официант|администрат|сотрудни|продав|кассир|менеджер|мастер|врач|доктор|бариста|"
                r"охран|хостес|горнич|обслуж|хам|груб|вежлив|приветлив|внимательн|отзывчив",
    "Еда и напитки": r"еда|еду|еды|блюд|кухн|вкус|суп|салат|кофе|чай|пицц|мяс|стейк|рыб|порци|десерт|выпечк|"
                     r"завтрак|обед|ужин|меню|напит|пив|вин[оау]|коктейл|бургер|шаурм|ролл|суши|повар|хлеб|торт",
    "Цены": r"цен|дорог|дешев|дешёв|стоимост|руб|деньг|переплат|бюджетн|прайс|скидк|акци|обсчит|чек",
    "Чистота": r"чист|грязн|убор|убира|пыл|вон[ья]|запах|туалет|плесен|таракан|мусор|антисанитар|санитар",
    "Атмосфера": r"атмосфер|интерьер|уют|музык|шум|тих|дизайн|обстановк|душевн|освещен|вид из",
    "Скорость и ожидание": r"жд[аеё]|ожидан|очеред|долг|быстр|медлен|опозд|минут|час[аов]? |оперативн|задерж",
    "Расположение": r"располож|парковк|метро|добират|добрать|найти|вход|район|центр|остановк|рядом с домом",
    "Доставка": r"доставк|курьер|привезл|привез[её]|самовывоз|упаковк|посылк",
    "Проживание": r"номер|кроват|отел|гостиниц|хостел|заселен|засели|душ|ванн|полотенц|бель[её]|матрас|кондицион",
    "Товар и ассортимент": r"ассортимент|товар|выбор|качеств|брак|размер|модел|продукц|наличи",
}
_ASPECT_RE = {a: re.compile(rf"\b({p})", re.I) for a, p in ASPECTS.items()}
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?…;])\s+|\s*[,–—-]\s*(?=(?:но|а|однако|зато|хотя|правда)\s)|\s+(?=но\s)", re.I)
_WORD = re.compile(r"\S+")


def encode(model, enc):
    """Shared encoder pass → (sentiment logits, pooled vector the extra heads read)."""
    h = model.base_model(**enc)
    if model.config.model_type == "bert":  # BertForSequenceClassification: classifier(dropout(pooler))
        pooled = h.pooler_output
        return model.classifier(model.dropout(pooled)), pooled
    seq = h.last_hidden_state  # RoBERTa-style heads read the <s> token themselves
    return model.classifier(seq), seq[:, 0]


def make_heads(hidden, sizes):
    return torch.nn.ModuleDict({k: torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, n))
                                for k, n in sizes.items()})


def clean(text):
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text).replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clauses(text, max_words=12):
    """Sentences and contrast clauses ("…, но …"); long ones are cut further at commas."""
    parts = []
    for p in _CLAUSE_SPLIT.split(clean(text)):
        if p and len(p.split()) > max_words:
            parts += p.split(",")
        elif p:
            parts.append(p)
    parts = [p.strip(" ,.;!?…") for p in parts]
    return [p for p in parts if len(p.split()) >= 2]


def find_aspects(fragment):
    return [a for a, rx in _ASPECT_RE.items() if rx.search(fragment)]


def polarity_label(row, margin=0.1):
    """Short fragments lean "neutral" (the model learnt 3★ = neutral from whole reviews),
    so a clause is judged by polarity P(pos) − P(neg) instead of argmax."""
    pol = float(row[2] - row[0])
    return ("positive" if pol > margin else "negative" if pol < -margin else "neutral"), round(pol, 3)


class Analyzer:
    def __init__(self, model=MODEL, max_length=256):
        self.device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.max_length = max_length
        self.heads, self.meta = None, {}
        meta = Path(model) / "multitask.json"
        if meta.exists():  # multi-task model (train_multi.py): stars 1–5 and place category heads
            self.meta = json.loads(meta.read_text(encoding="utf-8"))
            self.heads = make_heads(self.model.config.hidden_size,
                                    {"stars": 5, "category": len(self.meta["categories"])})
            self.heads.load_state_dict(torch.load(Path(model) / "heads.pt", map_location="cpu"))
            self.heads.to(self.device).eval()

    @torch.inference_mode()
    def probs(self, texts, batch_size=64, task="label"):
        """(N, classes) probabilities: task "label" (LABELS order), "stars" (1..5) or "category"."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            if task == "label":
                logits = self.model(**enc).logits
            else:
                logits = self.heads[task](encode(self.model, enc)[1])
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def stars(self, text):
        """{1: p, …, 5: p} and the expected rating; empty for empty input or a sentiment-only model."""
        text = clean(text)
        if not text or self.heads is None:
            return {}, None
        p = self.probs([text], task="stars")[0]
        return {s: float(x) for s, x in zip(range(1, 6), p)}, round(float(p @ np.arange(1, 6)), 2)

    def category(self, text, top=5):
        text = clean(text)
        if not text or self.heads is None:
            return {}
        p = self.probs([text], task="category")[0]
        return {self.meta["categories"][i]: float(p[i]) for i in np.argsort(-p)[:top]}

    def predict(self, text):
        """{label: probability}; empty dict for empty input."""
        text = clean(text)
        if not text:
            return {}
        return dict(zip(LABELS, map(float, self.probs([text])[0])))

    def explain(self, text, max_words=150):
        """Occlusion: drop each word and see how polarity P(pos) − P(neg) moves.

        Returns [(word, weight)], weight > 0 — the word pushes towards positive, < 0 — towards negative,
        scaled so the strongest word is ±1.
        """
        words = _WORD.findall(clean(text))[:max_words]
        if not words:
            return []
        variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        p = self.probs(variants)
        polarity = p[:, 2] - p[:, 0]
        w = polarity[0] - polarity[1:]
        scale = max(np.abs(w).max(), 1e-6)
        return [(word, round(float(x / scale), 3)) for word, x in zip(words, w)]

    def aspects(self, text):
        """Per-clause sentiment for every aspect mentioned: [{aspect, fragment, label, polarity}]."""
        parts = clauses(text)
        tagged = [(a, c) for c in parts for a in find_aspects(c)]
        if not tagged:
            return []
        p = self.probs([c for _, c in tagged])
        out = []
        for (a, c), row in zip(tagged, p):
            label, pol = polarity_label(row)
            out.append({"aspect": a, "fragment": c, "label": label, "polarity": pol})
        return out

    def batch(self, texts):
        """Label every review and summarise: class shares and the most frequent complaints by aspect."""
        texts = [clean(t) for t in texts]
        keep = [i for i, t in enumerate(texts) if t]
        p = self.probs([texts[i] for i in keep])
        labels = [None] * len(texts)
        scores = [None] * len(texts)
        for i, row in zip(keep, p):
            labels[i], scores[i] = LABELS[int(row.argmax())], round(float(row.max()), 3)

        # Clause-level aspects for all reviews in one batched pass
        tagged = [(i, a, c) for i in keep for c in clauses(texts[i]) for a in find_aspects(c)]
        cp = self.probs([c for _, _, c in tagged]) if tagged else np.zeros((0, 3))
        neg, pos, examples = Counter(), Counter(), {}
        for (i, a, c), row in zip(tagged, cp):
            lab, pol = polarity_label(row)
            if lab == "negative":
                neg[a] += 1
                if a not in examples or pol < examples[a][1]:
                    examples[a] = (c, pol)
            elif lab == "positive":
                pos[a] += 1
        counts = Counter(l for l in labels if l)
        n = max(len(keep), 1)
        complaints = [{"aspect": a, "negative": k, "positive": pos[a], "example": examples[a][0][:160]}
                      for a, k in neg.most_common()]
        praise = [{"aspect": a, "positive": k, "negative": neg[a]} for a, k in pos.most_common()]
        out = {
            "labels": labels, "scores": scores, "total": len(texts), "analysed": len(keep),
            "share": {l: round(counts[l] / n, 3) for l in LABELS},
            "complaints": complaints, "praise": praise,
        }
        if self.heads is not None and keep:  # multi-task model: predicted stars and place category too
            kept = [texts[i] for i in keep]
            sp, gp = self.probs(kept, task="stars"), self.probs(kept, task="category")
            out["stars"], out["category"] = [None] * len(texts), [None] * len(texts)
            for i, s_row, g_row in zip(keep, sp, gp):
                out["stars"][i] = int(s_row.argmax()) + 1
                out["category"][i] = self.meta["categories"][int(g_row.argmax())]
            out["mean_stars"] = round(float((sp @ np.arange(1, 6)).mean()), 2)
        return out
