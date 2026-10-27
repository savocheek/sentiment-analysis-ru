"""REST API smoke tests (needs ./model or MODEL_PATH)."""
from fastapi.testclient import TestClient

from api import app

c = TestClient(app)


def test_predict():
    r = c.post("/predict", json={"text": "Ужасный сервис, больше не придём."}).json()
    assert r["label"] == "negative" and abs(sum(r["scores"].values()) - 1) < 1e-3


def test_empty_text():
    assert c.post("/predict", json={"text": "  "}).json() == {"label": None, "scores": {}}


def test_batch_and_summary():
    texts = ["Отлично, всем советую!", "Официант нахамил, ужас."]
    assert [x["label"] for x in c.post("/predict/batch", json={"texts": texts}).json()] == ["positive", "negative"]
    s = c.post("/summary", json={"texts": texts}).json()
    assert s["analysed"] == 2 and s["complaints"][0]["aspect"] == "Персонал"


def test_explain_and_aspects():
    assert c.post("/explain", json={"text": "Ужасный сервис"}).json()["words"][0]["weight"] < 0
    assert c.post("/aspects", json={"text": "Цены завышены"}).json()["aspects"][0]["aspect"] == "Цены"
