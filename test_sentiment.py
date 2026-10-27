"""Tests for sentiment.py: text helpers without a model, model-backed features with ./model (or MODEL_PATH)."""
import pytest

from sentiment import Analyzer, clauses, clean, find_aspects, polarity_label


def test_clean_strips_html_and_escapes():
    assert clean("<p>Отлично &amp; уютно</p>\\nвернёмся") == "Отлично & уютно вернёмся"
    assert clean(None) == ""


def test_clauses_split_on_contrast():
    assert clauses("Еда вкусная, но официант хамил. Цены высокие!") == ["Еда вкусная", "но официант хамил", "Цены высокие"]


def test_find_aspects():
    assert find_aspects("официант хамил") == ["Персонал"]
    assert "Цены" in find_aspects("цены завышены")
    assert find_aspects("просто так") == []


def test_polarity_label():
    assert polarity_label([0.1, 0.6, 0.3]) == ("positive", 0.2)
    assert polarity_label([0.4, 0.55, 0.05])[0] == "negative"
    assert polarity_label([0.2, 0.6, 0.2])[0] == "neutral"


@pytest.fixture(scope="module")
def an():
    return Analyzer()


def test_explain_marks_sentiment_words(an):
    neg = dict(an.explain("Ужасный сервис и грязные столы"))
    pos = dict(an.explain("Очень вкусная еда и приятный персонал"))
    assert neg["Ужасный"] < 0 < pos["вкусная"]
    assert max(abs(x) for x in neg.values()) == pytest.approx(1)


def test_aspects_split_mixed_review(an):
    got = {a["aspect"]: a["label"] for a in an.aspects("Еда очень вкусная, но официант нахамил.")}
    assert got == {"Еда и напитки": "positive", "Персонал": "negative"}


def test_batch_summary(an):
    out = an.batch(["Отличное место, всем советую!", "Ужасно, официант хамил.", "", "Обычное кафе."])
    assert out["total"] == 4 and out["analysed"] == 3 and out["labels"][2] is None
    assert sum(out["share"].values()) == pytest.approx(1, abs=0.01)
    assert out["complaints"][0]["aspect"] == "Персонал"
