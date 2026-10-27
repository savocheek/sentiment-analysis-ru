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

Целевой ПК: **AMD Radeon RX 7900 GRE (16 ГБ, gfx1100) + Ryzen 7 9800X3D**. Видеокарта AMD, поэтому нужна сборка PyTorch под **ROCm**, а не под CUDA.
В ROCm-сборке `torch.cuda.*` работает через HIP, поэтому код проекта менять не нужно: fp16 включится сам.

Варианты по надёжности:
1. **Linux (Ubuntu 22.04/24.04)**: ROCm + `pip install torch --index-url https://download.pytorch.org/whl/rocmX.Y` (актуальная версия — на pytorch.org).
2. **Windows + WSL2 (Ubuntu)**: драйвер Adrenalin с поддержкой WSL, затем ROCm для WSL по инструкции AMD и та же команда pip внутри WSL.
3. **Windows нативно**: сборка PyTorch для Windows от AMD (ROCm on Windows) — по актуальной инструкции AMD для серии RX 7000.

Перед установкой сверить версии с текущей документацией AMD ROCm и pytorch.org: они часто меняются.

```bash
git clone <repo> && cd <repo>          # или распаковать архив
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# установить torch под ROCm (см. выше), затем:
pip install -r requirements.txt         # torch>=2.4 уже выполнено ROCm-сборкой, pip её не заменит
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# ждём: True и 'AMD Radeon RX 7900 GRE'
python mine_hard.py                     # data/hard_*.jsonl, детерминированно (≈2 мин)
```

16 ГБ видеопамяти: base-модели — с запасом; ruRoberta-large — `--batch 8` (или 16 при `--max-length 128`).

## 2. Эксперименты (по убыванию ожидаемой пользы)

Каждый результат дописывается в `experiments.jsonl`:

```bash
run() { python train.py --augment --hard "$@" && python evaluate.py "${@: -1}" | grep '^{' >> experiments.jsonl; }

run --out model_v3a_gpu                                                              # контроль: должно совпасть с v3a ±0.01
run --base-model ai-forever/ruBert-base --out model_v3b                               # крупная модель, полные настройки
run --base-model seara/rubert-base-cased-russian-sentiment --lr 3e-5 --out model_v3c  # уже обучена на тональность
run --base-model ai-forever/ruRoberta-large --lr 1e-5 --batch 8 --out model_v3d       # 355M, впритык в 16 ГБ
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
