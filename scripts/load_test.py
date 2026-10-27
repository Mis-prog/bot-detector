"""Проверка и нагрузочный тест bot-detector.

    uv run python scripts/load_test.py --check            # ответы сервиса == сабмит из MLflow-рана чемпиона
    uv run python scripts/load_test.py --concurrency 8 --duration 60

Запросы — настоящие диалоги из data/test.json.
"""
import argparse
import json
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

API = 'http://localhost:8000'
MLFLOW = 'http://localhost:5050'


def load_dialogs(path='data/test.json'):
    with open(path, encoding='utf-8') as f:
        dialogs = json.load(f)
    return {
        dialog_id: [{'participant_index': m['participant_index'], 'text': m['text']} for m in sorted(msgs, key=lambda m: m['message'])]
        for dialog_id, msgs in dialogs.items()
    }


def check(dialogs):
    """Сравнивает ответы сервиса с submission.csv, который train.py посчитал батчем для той же версии."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW)
    health = requests.get(f'{API}/health', timeout=10).json()
    mv = MlflowClient().get_model_version('bot-detector', health['version'])
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path='submission.csv', dst_path=tmp)
        expected = pd.read_csv(path).set_index('ID')['is_bot']

    diffs = []
    for dialog_id, messages in dialogs.items():
        if not messages:
            continue
        r = requests.post(f'{API}/v1/predict', json={'dialog_id': dialog_id, 'messages': messages}, timeout=30)
        r.raise_for_status()
        for p in r.json()['participants']:
            diffs.append(abs(p['is_bot'] - expected[f"{dialog_id}_{p['participant_index']}"]))
    diffs = np.array(diffs)
    exact = (diffs < 1e-9).mean()
    print(f'Версия {health["version"]}: сравнено {len(diffs)} участников, совпали точно {exact:.1%}, '
          f'макс. расхождение {diffs.max():.2e}, среднее {diffs.mean():.2e}')
    # Точного совпадения у всех не будет: train.py кодирует сообщения большими батчами, сервис — по диалогу,
    # из-за разного паддинга эмбеддинги отличаются на ~1e-7, и изредка LightGBM уходит в другую ветку.
    # Ошибка в признаках дала бы крупные расхождения у многих участников.
    assert exact > 0.9 and diffs.max() < 0.05, 'Сервис расходится с батчевым инференсом обучения'


def load(dialogs, concurrency, duration):
    payloads = [{'dialog_id': d, 'messages': m} for d, m in dialogs.items() if m]
    latencies, errors = [], 0
    lock = threading.Lock()
    deadline = time.perf_counter() + duration

    def worker():
        nonlocal errors
        session = requests.Session()
        while time.perf_counter() < deadline:
            payload = random.choice(payloads)
            start = time.perf_counter()
            try:
                ok = session.post(f'{API}/v1/predict', json=payload, timeout=30).status_code == 200
            except requests.RequestException:
                ok = False
            elapsed = time.perf_counter() - start
            with lock:
                if ok:
                    latencies.append(elapsed)
                else:
                    errors += 1

    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(worker)
    wall = time.perf_counter() - start

    lat = np.array(latencies) * 1000
    print(f'Параллельных клиентов: {concurrency}, {wall:.0f} с')
    print(f'  RPS: {len(lat) / wall:.1f}, ошибок: {errors}')
    print(f'  задержка: p50 {np.percentile(lat, 50):.1f} мс, p95 {np.percentile(lat, 95):.1f} мс, p99 {np.percentile(lat, 99):.1f} мс')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='сверить ответы с сабмитом из MLflow')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--duration', type=float, default=30, help='секунд нагрузки')
    args = parser.parse_args()
    dialogs = load_dialogs()
    if args.check:
        check(dialogs)
    else:
        load(dialogs, args.concurrency, args.duration)


if __name__ == '__main__':
    main()
