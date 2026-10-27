# bot-detector

Определяет, кто из двух участников диалога — бот. Модель — LightGBM на TF-IDF, статистиках текста и признаках из эмбеддингов `rubert-tiny2` с температурной калибровкой; разбор решения — в `notebooks/solution.ipynb`.

## Структура

```
bot_detector/        код модели: признаки, обучение, MLflow-модель, FastAPI-сервис
notebooks/
  eda.ipynb          исследование данных и первые модели
  solution.ipynb     итоговое решение для соревнования
scripts/
  embed.py           эмбеддинги rubert-tiny2 для solution.ipynb (отдельным процессом)
  load_test.py       сверка ответов сервиса и нагрузочный тест
submissions/         сабмиты; итоговый — submission.csv
monitoring/          конфиги Prometheus и Grafana
data/                данные соревнования
Dockerfile, docker-compose.yml
```

Все скрипты запускаются из корня проекта; ноутбуки сами переходят в корень, поэтому пути `data/`, `cache/`, `submissions/` в них общие.

## Стек

```
                 ┌──────────────┐  модель @champion  ┌─────────────┐
 POST /v1/predict│ api :8000    │◄───────────────────│ mlflow :5050│◄── train (python -m bot_detector.train)
 ───────────────►│ FastAPI      │                    │ реестр      │    регистрирует версию, ставит champion,
                 └──────┬───────┘                    └─────────────┘    если она лучше текущей
                        │ /metrics/
                 ┌──────▼───────┐      ┌──────────────┐
                 │ prometheus   │◄─────│ grafana :3000│  дашборд bot-detector
                 │ :9090        │      └──────────────┘
                 └──────────────┘
```

| Сервис | Адрес | Что там |
|---|---|---|
| api | http://localhost:8000/docs | Swagger, `POST /v1/predict`, `GET /health`, `/metrics/` |
| MLflow | http://localhost:5050 | эксперименты, метрики CV, реестр модели `bot-detector` |
| Grafana | http://localhost:3000 (admin / admin) | дашборд `bot-detector` |
| Prometheus | http://localhost:9090 | сырые метрики |

## Запуск

```bash
docker compose up -d --build     # MLflow, api, Prometheus, Grafana
docker compose run --rm train    # обучить и зарегистрировать модель (данные из ./data)
```

Сервис раз в 30 секунд проверяет алиас `champion` в реестре и подменяет модель без перезапуска. Пока модели нет, `/health` и `/v1/predict` отвечают 503.

Новая версия становится `champion`, только если её `cv_logloss_calibrated` лучше, чем у текущей. Вручную сменить версию можно в UI MLflow (Models → bot-detector → алиасы) — сервис подхватит её сам.

## Запрос

```bash
curl -s localhost:8000/v1/predict -H 'Content-Type: application/json' -d '{
  "messages": [
    {"participant_index": 0, "text": "Привет! Как дела?"},
    {"participant_index": 1, "text": "Привет! 😊 У меня всё отлично, спасибо, что спросил! Чем могу помочь?"}
  ]
}'
```

Ответ — вероятность `is_bot` для обоих участников и версия модели.

## Проверка и нагрузка

```bash
uv run python scripts/load_test.py --check                      # ответы сервиса совпадают с сабмитом из рана в MLflow
uv run python scripts/load_test.py --concurrency 4 --duration 60
```

## Устройство кода

- `bot_detector/features.py` — признаки; один код для обучения и сервиса, поэтому они не расходятся.
- `bot_detector/model.py` — модель MLflow (`pyfunc`): пайплайн + энкодер + температура в одном артефакте, сервису не нужен доступ к HuggingFace.
- `bot_detector/train.py` — обучение, метрики CV, регистрация в MLflow.
- `bot_detector/api.py` — FastAPI и метрики Prometheus.
- `monitoring/` — конфиги Prometheus и Grafana (источник данных и дашборд подключаются автоматически).

Один процесс сервиса обрабатывает запросы по очереди: модель вызывается под замком, torch и LightGBM — в один поток (`MODEL_THREADS`). Масштабируется числом реплик за балансировщиком; в этом compose одна реплика, потому что порт 8000 проброшен напрямую.

На macOS обучать вне Docker нельзя: torch и lightgbm в одном процессе роняют его из-за двух копий OpenMP. В Linux-контейнере проблемы нет. `notebooks/solution.ipynb` обходит это, считая эмбеддинги отдельным процессом (`scripts/embed.py`).
