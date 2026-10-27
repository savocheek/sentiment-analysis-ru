"""Template-generated reviews with sarcasm and negation, to mix into training.

Every template and filler list is split 70/30 into train / held-out pools, so the held-out
set (used only by stress_test.py) never shares a template or a filler with training.
The same templates are filled with good AND bad facts, so "спасибо"/"супер"/"не"
alone can't predict the label — the model has to read what actually happened.
"""
import itertools
import random

NEG, POS = 0, 2

BAD_THINGS = [  # accusative, after "за"
    "холодный суп", "грязную посуду", "хамство официанта", "полуторачасовое ожидание",
    "пересоленную пасту", "сломанный кондиционер", "обсчёт на кассе", "тараканов в номере",
    "просроченный йогурт", "потерянную бронь", "липкий стол", "сырую курицу",
    "испорченную стрижку", "мятую коробку и разбитую кружку", "очередь на сорок минут",
    "чёрствый хлеб", "грубость администратора", "навязанные доп. услуги", "мокрые полотенца",
    "заказ, который так и не привезли",
]
GOOD_THINGS = [
    "вкусный ужин", "чистый номер", "внимательное обслуживание", "быструю доставку",
    "идеальную стрижку", "тёплый приём", "свежую выпечку", "аккуратную упаковку",
    "подарок к заказу", "помощь с выбором", "уютную атмосферу", "честные цены",
    "вкуснейший кофе", "заботу о детях", "бесплатный поздний выезд", "отличный массаж",
    "живую музыку", "щедрые порции", "терпение консультанта", "ремонт за один день",
]
BAD_EVENTS = [
    "ждали заказ полтора часа", "официант нахамил", "в супе плавал волос", "нас обсчитали",
    "кофе принесли холодным", "в номере не было горячей воды", "посылку потеряли",
    "товар пришёл сломанным", "бронь куда-то пропала", "мастер опоздал на час",
    "курьер швырнул пакет у двери", "в салате попалась ракушка", "администратор бросил трубку",
    "нам продали просрочку", "кондиционер не работал всю неделю", "пиццу привезли сырой",
    "сдачу так и не отдали", "на нас накричали при детях", "под кроватью нашли мусор",
    "заказ перепутали дважды",
]
GOOD_EVENTS = [
    "заказ принесли за десять минут", "официант всё подсказал", "суп был горячий и наваристый",
    "нам сделали скидку", "кофе сварили идеально", "в номере было тепло и чисто",
    "посылка пришла на день раньше", "товар полностью соответствует описанию",
    "нас пересадили к окну", "мастер пришёл минута в минуту", "курьер поднял пакеты до двери",
    "салат был свежайший", "администратор решил вопрос за минуту", "все продукты свежие",
    "в номере стоял кондиционер и он работал", "пицца горячая и с хрустящим тестом",
    "нам вернули деньги без вопросов", "детям дали раскраски", "убирались каждый день",
    "всё упаковали с любовью",
]
VISITS = [
    "пришли сюда", "заказали здесь торт", "остановились в этом отеле", "выбрали этот салон",
    "записались к этому мастеру", "взяли этот тур", "отметили здесь юбилей", "купили здесь диван",
    "отдали машину в этот сервис", "привели сюда детей",
]
GOOD_ADJ = ["вкусно", "уютно", "чисто", "вежливо", "быстро", "дёшево", "удобно", "тихо", "свежо", "приятно"]

# Positive wording around a fact: label depends only on the fact.
IRONY_N = [
    "Спасибо огромное за {t}, очень приятно.", "Отдельное спасибо за {t}!",
    "Благодарим за {t}, вечер удался.", "Пять звёзд за {t}.",
    "Низкий поклон за {t}.", "Браво персоналу за {t}!",
    "Особая благодарность за {t}, такое не забывается.", "Мерси за {t}, вы лучшие.",
    "Сердечное спасибо за {t}.", "Ставлю лайк за {t}.",
]
IRONY_C = [
    "Ну просто супер: {e}.", "Браво, {e}!", "Обожаю, когда {e}.", "Сервис на высоте: {e}.",
    "Шикарно, {e}. Спасибо!", "Восторг, {e}. Всем советую.", "Лучший вечер в жизни: {e}.",
    "Молодцы, ничего не скажешь: {e}.", "Ну вы и профессионалы, {e}.", "Какая прелесть, {e}.",
    "Высший класс: {e}.", "Вот это уровень — {e}.",
]
# Negation that flips to positive / stays negative.
NEG_POS_V = [
    "Ни капли не жалеем, что {v}.", "Нисколько не пожалели, что {v}.",
    "Ни на минуту не усомнились, что не зря {v}.", "Ничуть не разочарованы, что {v}.",
    "Не прогадали, когда {v}.", "Никаких сожалений, что {v}.",
]
NEG_POS_E = [
    "Не нашли ни одного минуса: {e}.", "Придраться не к чему, {e}.", "Никаких претензий, {e}.",
    "Не ожидали, но {e}.", "Ничего плохого сказать не можем, {e}.", "Не было ни единой проблемы, {e}.",
    "Нечего добавить, {e}.", "Ни одного замечания, {e}.",
]
NEG_NEG_E = [
    "Ничего хорошего сказать не могу: {e}.", "Не стоит своих денег, {e}.",
    "Нисколько не понравилось, {e}.", "Не рекомендую: {e}.", "Не вернёмся сюда, {e}.",
    "Ни одного плюса не нашли, {e}.", "Никому не посоветую, {e}.", "Не ожидали такого, {e}.",
]
NEG_NEG_A = [
    "Ни капли не {a}, да и не {b}.", "Вообще не {a} и далеко не {b}.",
    "Было не {a} и уж точно не {b}.", "Ни {a}, ни {b} — мимо.", "Абсолютно не {a}, ещё и не {b}.",
]


def _split(items, seed):
    items = items[:]
    random.Random(seed).shuffle(items)
    cut = round(len(items) * 0.7)
    return items[:cut], items[cut:]


def _fill(templates, fillers, key, label, pool, seed):
    t = _split(templates, seed)[pool]
    f = _split(fillers, seed + 1)[pool]
    return [(tpl.format(**{key: x}), label) for tpl, x in itertools.product(t, f)]


def generate(pool):
    """pool=0 → train examples, pool=1 → held-out. Returns list of (category, text, label)."""
    out = []
    groups = [
        ("сарказм", IRONY_N, BAD_THINGS, "t", NEG), ("искреннее «спасибо»", IRONY_N, GOOD_THINGS, "t", POS),
        ("сарказм", IRONY_C, BAD_EVENTS, "e", NEG), ("искреннее «спасибо»", IRONY_C, GOOD_EVENTS, "e", POS),
        ("отрицание → позитив", NEG_POS_V, VISITS, "v", POS),
        ("отрицание → позитив", NEG_POS_E, GOOD_EVENTS, "e", POS),
        ("отрицание → негатив", NEG_NEG_E, BAD_EVENTS, "e", NEG),
    ]
    for i, (cat, tpl, fill, key, label) in enumerate(groups):
        out += [(cat, text, lab) for text, lab in _fill(tpl, fill, key, label, pool, seed=10 * i)]
    t = _split(NEG_NEG_A, 99)[pool]
    pairs = _split(list(itertools.permutations(GOOD_ADJ, 2)), 100)[pool]
    out += [("отрицание → негатив", tpl.format(a=a, b=b), NEG) for tpl in t for a, b in pairs]
    return out


def train_examples(max_per_cat=400, seed=42):
    """Capped per category so synthetic data stays a small share (~5%) of training."""
    rng = random.Random(seed)
    by_cat = {}
    for cat, text, label in generate(0):
        by_cat.setdefault(cat, []).append((text, label))
    rows = [r for items in by_cat.values() for r in rng.sample(items, min(max_per_cat, len(items)))]
    return {"text": [t for t, _ in rows], "label": [l for _, l in rows]}


if __name__ == "__main__":
    from collections import Counter
    tr, ho = generate(0), generate(1)
    assert not {t for _, t, _ in tr} & {t for _, t, _ in ho}, "train/held-out overlap"
    print("train:", Counter(c for c, _, _ in tr), "\nheld-out:", Counter(c for c, _, _ in ho))
    print("used for training:", len(train_examples()["text"]))
    for row in random.Random(0).sample(ho, 6):
        print(row)
