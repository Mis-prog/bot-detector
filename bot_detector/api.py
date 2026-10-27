"""bot-detector: HTTP-сервис предсказаний.

    uvicorn bot_detector.api:app --host 0.0.0.0 --port 8000

Модель берётся из реестра MLflow по алиасу (по умолчанию models:/bot-detector@champion).
Фоновый поток раз в MODEL_POLL_SECONDS проверяет алиас и подменяет модель, если чемпион сменился.
"""
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get('MODEL_NAME', 'bot-detector')
MODEL_ALIAS = os.environ.get('MODEL_ALIAS', 'champion')
POLL_SECONDS = float(os.environ.get('MODEL_POLL_SECONDS', '30'))
BOT_THRESHOLD = 0.5

log = logging.getLogger('bot_detector')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

LATENCY_BUCKETS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5)
REQUESTS = Counter('bot_detector_requests_total', 'HTTP-запросы', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('bot_detector_request_duration_seconds', 'Время обработки запроса', ['endpoint'], buckets=LATENCY_BUCKETS)
IN_PROGRESS = Gauge('bot_detector_requests_in_progress', 'Запросы в обработке')
STAGE_LATENCY = Histogram('bot_detector_stage_duration_seconds', 'Время этапов модели', ['stage'], buckets=LATENCY_BUCKETS)
PROBABILITY = Histogram('bot_detector_prediction_probability', 'Предсказанная вероятность is_bot', buckets=[i / 10 for i in range(1, 11)])
PREDICTIONS = Counter('bot_detector_predictions_total', 'Предсказания по классам (порог 0.5)', ['label'])
DIALOG_MESSAGES = Histogram('bot_detector_dialog_messages', 'Сообщений в диалоге', buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89))
MODEL_INFO = Gauge('bot_detector_model_info', 'Загруженная модель', ['name', 'version', 'run_id'])
MODEL_RELOADS = Counter('bot_detector_model_reloads_total', 'Загрузки модели', ['result'])


class Message(BaseModel):
    participant_index: Literal[0, 1]
    text: str


class PredictRequest(BaseModel):
    dialog_id: str | None = Field(default=None, description='Если не задан, генерируется')
    messages: list[Message] = Field(min_length=1, description='Сообщения в порядке отправки')


class ParticipantPrediction(BaseModel):
    participant_index: int
    is_bot: float


class PredictResponse(BaseModel):
    dialog_id: str
    model_version: str
    participants: list[ParticipantPrediction]


class ModelHolder:
    """Текущая модель и её версия. Модель не потокобезопасна, поэтому вызовы под замком."""

    def __init__(self):
        self.model = None
        self.version = None
        self.run_id = None
        self.lock = threading.Lock()

    def refresh(self):
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        if mv.version == self.version:
            return
        log.info('Загружаю %s версии %s (run %s)', MODEL_NAME, mv.version, mv.run_id)
        model = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/{mv.version}').unwrap_python_model()
        with self.lock:
            old_version, old_run_id = self.version, self.run_id
            self.model, self.version, self.run_id = model, mv.version, mv.run_id
        if old_version is not None:
            MODEL_INFO.remove(MODEL_NAME, old_version, old_run_id)
        MODEL_INFO.labels(MODEL_NAME, mv.version, mv.run_id).set(1)
        MODEL_RELOADS.labels('ok').inc()
        log.info('Модель версии %s загружена', mv.version)

    def poll_forever(self, stop):
        while not stop.is_set():
            try:
                self.refresh()
            except Exception:
                MODEL_RELOADS.labels('error').inc()
                log.exception('Не удалось загрузить модель %s@%s, повторю через %.0f с', MODEL_NAME, MODEL_ALIAS, POLL_SECONDS)
            stop.wait(POLL_SECONDS)


holder = ModelHolder()


@asynccontextmanager
async def lifespan(app):
    stop = threading.Event()
    poller = threading.Thread(target=holder.poll_forever, args=(stop,), daemon=True)
    poller.start()
    yield
    stop.set()


app = FastAPI(title='bot-detector', version='0.1.0', lifespan=lifespan)
app.mount('/metrics', make_asgi_app())


@app.middleware('http')
async def track_requests(request: Request, call_next):
    endpoint = request.url.path
    if endpoint.startswith('/metrics'):
        return await call_next(request)
    IN_PROGRESS.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        IN_PROGRESS.dec()
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, str(status)).inc()


@app.get('/health')
def health():
    if holder.model is None:
        raise HTTPException(status_code=503, detail=f'Модель {MODEL_NAME}@{MODEL_ALIAS} ещё не загружена')
    return {'status': 'ok', 'model': MODEL_NAME, 'version': holder.version}


@app.post('/v1/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    if holder.model is None:
        raise HTTPException(status_code=503, detail=f'Модель {MODEL_NAME}@{MODEL_ALIAS} ещё не загружена')
    dialog_id = request.dialog_id or str(uuid.uuid4())
    msgs = pd.DataFrame([
        {'dialog_id': dialog_id, 'message': i, 'participant_index': m.participant_index, 'text': m.text}
        for i, m in enumerate(request.messages)
    ])
    with holder.lock:
        result, stages = holder.model.predict_dialogs(msgs)
        version = holder.version

    for stage, seconds in stages.items():
        STAGE_LATENCY.labels(stage).observe(seconds)
    DIALOG_MESSAGES.observe(len(request.messages))
    for p in result['is_bot']:
        PROBABILITY.observe(p)
        PREDICTIONS.labels('bot' if p >= BOT_THRESHOLD else 'human').inc()

    return PredictResponse(
        dialog_id=dialog_id,
        model_version=version,
        participants=[
            ParticipantPrediction(participant_index=int(r.participant_index), is_bot=float(r.is_bot))
            for r in result.itertuples()
        ],
    )
