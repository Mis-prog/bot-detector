"""Модель для MLflow: пайплайн LightGBM + энкодер rubert-tiny2 + температура в одном артефакте."""
import json
import os
import time

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

from bot_detector.features import KEYS, apply_temperature, build_features, participants_of

MESSAGE_COLUMNS = ['dialog_id', 'message', 'participant_index', 'text']


class BotDetectorModel(mlflow.pyfunc.PythonModel):
    """Вход — сообщения (dialog_id, message, participant_index, text).
    Выход — вероятность is_bot для обоих участников каждого диалога.
    """

    def load_context(self, context):
        # torch импортируем только здесь: обучающему коду он не нужен до сохранения энкодера
        import torch
        from sentence_transformers import SentenceTransformer

        # один поток на процесс: масштабируемся числом процессов, а не потоков внутри
        threads = int(os.environ.get('MODEL_THREADS', '1'))
        torch.set_num_threads(threads)
        self.pipeline = joblib.load(context.artifacts['pipeline'])
        self.pipeline.set_params(clf__n_jobs=threads)
        self.encoder = SentenceTransformer(context.artifacts['encoder'], device='cpu')
        with open(context.artifacts['config']) as f:
            self.config = json.load(f)

    def embed(self, msgs):
        return self.encoder.encode(msgs['text'].tolist(), batch_size=64, normalize_embeddings=True)

    def score(self, msgs, emb):
        feature_names = self.pipeline.named_steps['features'].feature_names_in_
        df = build_features(participants_of(msgs), msgs, emb)
        p = self.pipeline.predict_proba(df[feature_names])[:, 1]
        p = apply_temperature(p, self.config['temperature'])
        clip = self.config['clip']
        return df[KEYS].assign(is_bot=np.clip(p, clip, 1 - clip))

    def predict_dialogs(self, msgs):
        """Возвращает (предсказания, {этап: секунды}) — время этапов нужно сервису для метрик."""
        msgs = msgs[MESSAGE_COLUMNS].reset_index(drop=True)
        t0 = time.perf_counter()
        emb = self.embed(msgs)
        t1 = time.perf_counter()
        result = self.score(msgs, emb)
        t2 = time.perf_counter()
        return result, {'embed': t1 - t0, 'features_and_model': t2 - t1}

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.predict_dialogs(model_input)[0]
