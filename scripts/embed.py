"""Эмбеддинги сообщений rubert-tiny2 для notebooks/solution.ipynb.

    uv run python scripts/embed.py   # из корня проекта

Запускается отдельным процессом: на macOS torch и lightgbm в одном процессе
конфликтуют из-за двух копий OpenMP и роняют ядро.
Порядок векторов совпадает с порядком сообщений в load_messages.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'cointegrated/rubert-tiny2'
OUT_DIR = Path('cache')


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    for part in ('train', 'test'):
        with open(f'data/{part}.json', encoding='utf-8') as f:
            dialogs = json.load(f)
        texts = [msg['text'] for messages in dialogs.values() for msg in messages]
        emb = encoder.encode(texts, batch_size=256, normalize_embeddings=True)
        np.save(OUT_DIR / f'emb_{part}.npy', emb)
        print(f'{part}: {emb.shape} на {device}')


if __name__ == '__main__':
    main()
