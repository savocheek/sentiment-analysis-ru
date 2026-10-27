# План обучения на мощном ПК

Передаточная записка: где остановились и что запускать дальше. Claude Code на новом устройстве: «прочитай NEXT_STEPS.md и продолжай».

## Где остановились (27.10.2026)

| Модель | F1 реальные | hard-dev среднее | Статус |
|---|---|---|---|
| v1 rubert-tiny2 | 0.784 | 0.718 | готова |
| v2 + шаблоны | 0.784 | 0.740 | готова, сейчас в `model/` |
| v3a + шаблоны + реальные сложные | 0.780 | **0.774** | лучшая по hard-dev |
| v3b ruBert-base | — | — | на Mac прервано после 1 эпохи (val F1 0.794) |

Цель: поднять hard-dev (отзывы, где слова противоречат оценке), не уронив F1 на реальных отзывах ниже 0.779.
Hard-test (`data/hard_test.jsonl`) ещё ни разу не открывали — открыть **один раз** для финальной модели.

## 1. Подготовка

```bash
git clone <repo> && cd <repo>          # или распаковать архив
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # под свою версию CUDA, см. pytorch.org
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # должно быть True
python mine_hard.py                     # data/hard_*.jsonl, детерминированно (≈2 мин)
```

На NVIDIA обучение само включает fp16.

## 2. Эксперименты (по убыванию ожидаемой пользы)

Каждый результат дописывается в `experiments.jsonl`:

```bash
run() { python train.py --augment --hard "$@" && python evaluate.py "${@: -1}" | grep '^{' >> experiments.jsonl; }

run --out model_v3a_gpu                                                              # контроль: должно совпасть с v3a ±0.01
run --base-model ai-forever/ruBert-base --out model_v3b                               # крупная модель, полные настройки
run --base-model seara/rubert-base-cased-russian-sentiment --lr 3e-5 --out model_v3c  # уже обучена на тональность
run --base-model ai-forever/ruRoberta-large --lr 1e-5 --batch 16 --out model_v3d      # 355M, нужно ≥16 ГБ видеопамяти
```

`run` ставит `--out` последним аргументом, так что evaluate получает папку модели.

## 3. Шум и выбор

Для двух лучших по `hard_dev_bal` повторить с `--seed 1` и `--seed 2`. Победитель — лучший по среднему `hard_dev_bal` при `real_f1 ≥ 0.779`. Если разница между кандидатами меньше разброса по seed, взять меньшую модель.

## 4. Финал

```bash
python evaluate.py model_v1 model_v2 <победитель> --hard test | tee final_test.jsonl   # один раз!
rm -rf model && cp -r <победитель> model
python stress_test.py && pytest -q
```

Обновить таблицы в README и закоммитить. Веса (>100 МБ) в git не класть — выложить на Hugging Face Hub.
