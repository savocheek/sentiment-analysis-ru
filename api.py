"""REST API over the sentiment model. Run: uvicorn api:app --port 8000  (docs at /docs)"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from sentiment import LABELS, Analyzer

app = FastAPI(title="Тональность отзывов на русском",
              description="ruBERT: негатив / нейтрально / позитив, объяснения и аспекты.")
an = Analyzer()


class Review(BaseModel):
    text: str = Field(..., max_length=20_000, examples=["Еда вкусная, но официант нахамил."])


class Reviews(BaseModel):
    texts: list[str] = Field(..., max_length=1000)


def top(scores):
    return {"label": max(scores, key=scores.get) if scores else None, "scores": scores}


@app.get("/health")
def health():
    return {"status": "ok", "device": an.device, "labels": LABELS}


@app.post("/predict")
def predict(r: Review):
    return top(an.predict(r.text))


@app.post("/predict/batch")
def predict_batch(r: Reviews):
    out = an.batch(r.texts)
    return [{"label": l, "score": s} for l, s in zip(out["labels"], out["scores"])]


@app.post("/explain")
def explain(r: Review):
    return {**top(an.predict(r.text)), "words": [{"word": w, "weight": x} for w, x in an.explain(r.text)]}


@app.post("/aspects")
def aspects(r: Review):
    return {**top(an.predict(r.text)), "aspects": an.aspects(r.text)}


@app.post("/summary")
def summary(r: Reviews):
    out = an.batch(r.texts)
    out.pop("labels"), out.pop("scores")
    return out
