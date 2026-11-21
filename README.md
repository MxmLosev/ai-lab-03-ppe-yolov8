# Лабораторная работа №3. Промышленная видеоаналитика: обнаружение СИЗ с YOLOv8

Обнаружение средств индивидуальной защиты (СИЗ) на строительной площадке: подготовка документированного набора данных, обучение YOLOv8 в двух конфигурациях, сравнение метрик (mAP, ошибки по классам), измерение задержки полной цепочки обработки и разбор ошибок.

- **Дисциплина:** «Введение в искусственный интеллект», ПетрГУ, 2026
- **Ветка:** `lab/03-ppe`
- **Отчёт:** `reports/lab03-report.md`
- **Паспорт данных:** `data/README.md`

## Оборудование и окружение

| Компонент | Значение |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 (12 ГБ GDDR6X) |
| CPU | Intel Core i5-12600KF |
| RAM | 32 ГБ |
| ОС | Windows 11 (build 26300), Python 3.12.0 |
| Ускорение | CUDA 12.4, torch 2.6.0+cu124, fp32 |
| Библиотеки | см. `requirements.txt` (версии зафиксированы) |

## Структура репозитория

```
configs/ppe.yaml            описание датасета для Ultralytics (10 классов)
configs/train_e1.yaml       конфигурация эксперимента E1 (базовая)
configs/train_e2.yaml       конфигурация эксперимента E2 (гипотеза imgsz=960)
src/download_data.py        сбор пула из исходного набора (Kaggle/Roboflow v28)
src/prepare_split.py        групповое разбиение 70/15/15 без утечек (seed 42)
src/check_leaks.py          проверка имён, md5-хэшей и групп между частями
src/validate_labels.py      схема разметки: классы, координаты, статистика
src/make_contact_sheets.py  контрольные листы для ручной проверки разметки
src/e0_pretrained.py        E0: предобученная yolov8n.pt на 10 кадрах
src/train.py                запуск обучения по конфигурации (--config)
src/evaluate.py             метрики на тесте: P/R/AP по классам, mAP, матрица ошибок
src/sweep_confidence.py     зависимость P/R/F1 от порога уверенности
src/benchmark_latency.py    задержка по этапам: декод→ресайз→инференс→NMS, медиана/p95
src/error_galleries.py      галереи ошибок (6 типов)
src/export_onnx.py          экспорт лучшей модели в ONNX
src/compare_onnx.py         сравнение PyTorch vs ONNX: скорость и расхождение
tests/test_minimal.py       минимальные тесты приёмки (pytest)
data/README.md              паспорт данных
reports/lab03-report.md     отчёт по лабораторной
galleries/                  галереи ошибок с подписями
models/                     веса обученных моделей и ONNX
```

## Команды воспроизведения

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 1. Данные: скачанный zip кладётся в .cache/kaggle_dl и распаковывается
python src/download_data.py          # либо: curl -L -o .cache/kaggle_dl/css.zip \
#    https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow
python src/prepare_split.py          # разбиение 70/15/15 по группам, 821 изображение
python src/check_leaks.py            # утечек нет
python src/validate_labels.py        # схема разметки валидна
python src/make_contact_sheets.py    # контрольные листы для ручной проверки

# 2. Эксперименты
python src/e0_pretrained.py                                      # E0: предобученная модель
python src/train.py --config configs/train_e1.yaml               # E1: 50 эпох, imgsz 640
python src/train.py --config configs/train_e2.yaml               # E2: 50 эпох, imgsz 960

# 3. Оценка (тест заморожен до выбора конфигурации)
python src/evaluate.py --weights runs/ppe/e1-opora/weights/best.pt --imgsz 640 --tag e1
python src/evaluate.py --weights runs/ppe/e2-imgsz960/weights/best.pt --imgsz 960 --tag e2
python src/sweep_confidence.py --weights models/ppe_best.pt --imgsz <imgsz> --tag best
python src/benchmark_latency.py --weights models/ppe_best.pt --imgsz <imgsz> --tag best

# 4. Ошибки и доп. часть
python src/error_galleries.py --weights models/ppe_best.pt --imgsz <imgsz> --conf 0.25
python src/export_onnx.py --weights models/ppe_best.pt --imgsz <imgsz> --out models/ppe_best.onnx
python src/compare_onnx.py --weights models/ppe_best.pt --onnx models/ppe_best.onnx --imgsz <imgsz>

# 5. Тесты
.venv/Scripts/python -m pytest tests/ -v
```

## Схема данных (кратко)

10 классов: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle. Пары вида «каска/без каски» — два отдельных класса с рамками (отрицательный класс размечен как объект, не как отсутствие). Разбиение выполняется по группам-источникам (видео/аугментации одного кадра не пересекают части выборки). Подробности и ограничения — в паспорте `data/README.md`.

## Этические ограничения применения

Учебный детектор не может быть единственным механизмом допуска к опасной работе или автоматического наказания сотрудника: ошибка зависит от камеры, освещения, перекрытий и качества разметки. Критические решения требуют человеческого контроля и независимых мер безопасности (см. отчёт, раздел системной оценки).
