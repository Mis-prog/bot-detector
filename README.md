# bot-detector

Определяет, кто из двух участников диалога — бот. Модель сервиса — LightGBM на статистиках текста и признаках из эмбеддингов `rubert-tiny2`. Разбор решения — в `notebooks/solution.ipynb`; там для соревнования есть ещё TF-IDF по буквенным n-граммам (LogLoss 0.195 против 0.200 без него), в сервисе он убран ради скорости.

## Структура

```
bot_detector/        код модели: признаки, обучение, MLflow-модель, FastAPI-сервис
notebooks/
  solution.ipynb     исследование данных, бейзлайн, модели, подбор параметров и итоговый сабмит
scripts/
  embed.py           эмбеддинги rubert-tiny2 для solution.ipynb (отдельным процессом)
  load_test.py       сверка ответов сервиса и нагрузочный тест
submissions/         сабмиты; итоговый — submission.csv
monitoring/          конфиги Prometheus и Grafana
data/                данные соревнования
Dockerfile, docker-compose.yml
deploy.sh            развернуть стек одной командой
presentation/        слайды с итогами (Slidev): cd presentation && npm install && npm run dev
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
./deploy.sh            # поднять стек и обучить модель, если её ещё нет в реестре; ждёт готовности сервиса
./deploy.sh --train    # то же, но обучить новую версию в любом случае
```

То же вручную:

```bash
docker compose up -d --build     # MLflow, api, Prometheus, Grafana
docker compose run --rm train    # обучить и зарегистрировать модель (данные из ./data)
```

Сервис раз в 30 секунд проверяет алиас `champion` в реестре и подменяет модель без перезапуска. Пока модели нет, `/health` и `/v1/predict` отвечают 503.

Новая версия становится `champion`, только если её `cv_logloss` лучше, чем у текущей. Сделать новую версию `champion` в любом случае: `docker compose run --rm train python -m bot_detector.train --promote`. Вручную сменить версию можно в UI MLflow (Models → bot-detector → алиасы) — сервис подхватит её сам.

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
- `bot_detector/model.py` — модель MLflow (`pyfunc`): пайплайн + энкодер в одном артефакте, сервису не нужен доступ к HuggingFace.
- `bot_detector/train.py` — обучение, метрики CV, регистрация в MLflow.
- `bot_detector/api.py` — FastAPI и метрики Prometheus.
- `monitoring/` — конфиги Prometheus и Grafana (источник данных и дашборд подключаются автоматически).

Один процесс сервиса обрабатывает запросы по очереди: модель вызывается под замком, torch и LightGBM — в один поток (`MODEL_THREADS`). Масштабируется числом реплик за балансировщиком; в этом compose одна реплика, потому что порт 8000 проброшен напрямую.

На macOS обучать вне Docker нельзя: torch и lightgbm в одном процессе роняют его из-за двух копий OpenMP. В Linux-контейнере проблемы нет. `notebooks/solution.ipynb` обходит это, считая эмбеддинги отдельным процессом (`scripts/embed.py`).
