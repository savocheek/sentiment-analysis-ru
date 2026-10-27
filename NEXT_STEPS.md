# Где остановились (27.10.2026, 10:30 МСК)

Передаточная записка для следующей сессии. Полный отчёт о дне — [REPORT.md](REPORT.md), результаты — [README.md](README.md).

## Состояние

- Финальная модель — **multi-task на ruRoberta-large** (`model/` = `model_multi`, в git не лежит, 1.4 ГБ): тональность F1 0.820, hard-test 0.835 (открыт один раз, [final_test.jsonl](final_test.jsonl)), 1–5★ MAE 0.35, категория acc 0.753 / top-3 0.95.
- Все эксперименты — в [experiments.jsonl](experiments.jsonl); seed-повторы: large 0.820 ± 0.007 / hard-dev 0.809 ± 0.012, base 0.802 / 0.784.
- Чекпоинты и модели (`model_v1`…`model_v3d_s2`, `model_multi`, `onnx/`) лежат на Windows-ПК с RX 7900 GRE, в git их нет.
- **Hard-test уже открыт.** Новые эксперименты сравнивать только по hard-dev, иначе тест перестанет быть честным.

## Как воспроизвести на этом ПК (AMD RX 7900 GRE, Windows)

```powershell
# Python 3.12 + PyTorch от AMD (ROCm 7.2.1, колёса с repo.radeon.com — см. README)
.venv\Scripts\activate
python train.py --augment --hard --base-model ai-forever/ruRoberta-large --lr 1e-5 --batch 8 --out model_v3d   # ~30 мин
python train_multi.py --base-model model_v3d --augment --hard --out model_multi                                 # ~25 мин
```

Правила для этого ПК (был BSOD 0x1CA, когда две модели учились одновременно): **одно обучение на GPU за раз**, пауза 60 с между ними ([gpu_queue.sh](gpu_queue.sh) так и делает), CPU-задачи ≤ 8 потоков. Оценка large-модели на CPU в AMD-сборке torch очень медленная: оценивать на GPU, когда он свободен, или через ONNX.

## Что дальше (по убыванию пользы)

1. **Опубликовать:** `huggingface-cli login` → `python publish.py <логин>`. Для бесплатного CPU Space лучше ONNX (`ONNX_MODEL`) или дистиллированная base-модель: large на CPU медленная.
2. **Аспекты как отдельная голова:** разметить 200–300 фраз (аспект + тональность) и дообучить. Сейчас словарь + полярность фраз дают 81% на смешанных отзывах.
3. **Сарказм:** собрать живые примеры (3/8 в рукописном наборе).
4. **«Нейтрально»** размечено по звёздам — главный источник ошибок (96% ошибок на границе с нейтральным). Ручная переразметка 1–2 тыс. отзывов дала бы честную метрику.
5. **int8** меняет 5% ответов large-модели — проверить по F1 или квантовать только FFN.
6. Проверить Dockerfile (на ПК не было Docker).
