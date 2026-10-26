"""Smoke tests for the trained model. Needs ./model (run train.py first)."""
from app import predict
from train import rating_to_label


def test_rating_to_label():
    assert [rating_to_label(r) for r in (1, 2, 3, 4, 5)] == [0, 0, 1, 2, 2]


def test_obvious_reviews():
    pos = predict("Прекрасное место, всё очень понравилось, персонал вежливый!")
    neg = predict("Ужасный сервис, грязно, хамят. Никому не советую.")
    assert max(pos, key=pos.get).endswith("Позитив")
    assert max(neg, key=neg.get).endswith("Негатив")
    assert abs(sum(pos.values()) - 1) < 1e-3


def test_empty_input():
    assert predict("   ") == {}
