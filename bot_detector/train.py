"""Обучение модели и регистрация в MLflow.

    python -m bot_detector.train            # champion — только если метрика лучше
    python -m bot_detector.train --promote  # champion в любом случае

Повторяет solution.ipynb (без TF-IDF): GroupKFold по диалогам, обучение на всём train.
Новая версия получает алиас champion, только если её CV LogLoss лучше, чем у текущего чемпиона
или передан --promote.
"""
import argparse
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from bot_detector.features import build_features, feature_columns, load_messages, make_model
from bot_detector.model import BotDetectorModel

MODEL_NAME = os.environ.get('MODEL_NAME', 'bot-detector')
ENCODER_NAME = 'cointegrated/rubert-tiny2'
DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
SEED = 42
N_SPLITS = 5
CLIP = 0.001
CHAMPION = 'champion'
METRIC = 'cv_logloss'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--promote', action='store_true', help=f'сделать новую версию {CHAMPION}, даже если метрика хуже')
    args = parser.parse_args()

    mlflow.set_experiment(MODEL_NAME)
    train_msgs = load_messages(DATA_DIR / 'train.json')
    test_msgs = load_messages(DATA_DIR / 'test.json')
    ytrain = pd.read_csv(DATA_DIR / 'ytrain.csv')
    ytest = pd.read_csv(DATA_DIR / 'ytest.csv')

    encoder = SentenceTransformer(ENCODER_NAME, device='cpu')
    encode = lambda m: encoder.encode(m['text'].tolist(), batch_size=256, normalize_embeddings=True)
    train_emb, test_emb = encode(train_msgs), encode(test_msgs)

    train = build_features(ytrain, train_msgs, train_emb)
    test = build_features(ytest, test_msgs, test_emb)
    features = feature_columns(train_emb.shape[1])
    y = train['is_bot'].to_numpy()
    groups = train['dialog_id']
    cv = GroupKFold(n_splits=N_SPLITS)

    with mlflow.start_run() as run:
        mlflow.log_params({
            'encoder': ENCODER_NAME,
            'n_splits': N_SPLITS,
            'clip': CLIP,
            'n_features': len(features),
            'n_train': len(train),
            **{f'lgbm_{k}': v for k, v in make_model(features).named_steps['clf'].get_params().items()
               if k in ('n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda')},
        })

        oof = cross_val_predict(make_model(features, SEED), train[features], y, groups=groups, cv=cv, method='predict_proba')[:, 1]
        metrics = {
            METRIC: log_loss(y, oof),
            'cv_roc_auc': roc_auc_score(y, oof),
            'cv_f1_at_0_5': f1_score(y, oof >= 0.5),
        }
        mlflow.log_metrics(metrics)
        print(json.dumps(metrics, indent=2))

        pipeline = make_model(features, SEED).fit(train[features], y)
        preds = pipeline.predict_proba(test[features])[:, 1]
        submission = test[['ID']].assign(is_bot=np.clip(preds, CLIP, 1 - CLIP))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            submission.to_csv(tmp / 'submission.csv', index=False)
            mlflow.log_artifact(tmp / 'submission.csv')
            joblib.dump(pipeline, tmp / 'pipeline.joblib')
            encoder.save(str(tmp / 'encoder'))
            (tmp / 'config.json').write_text(json.dumps({'clip': CLIP, 'encoder': ENCODER_NAME}))
            example = test_msgs[test_msgs['dialog_id'] == test_msgs['dialog_id'].iloc[0]]
            model_info = mlflow.pyfunc.log_model(
                name='model',
                python_model=BotDetectorModel(),
                artifacts={'pipeline': str(tmp / 'pipeline.joblib'), 'encoder': str(tmp / 'encoder'), 'config': str(tmp / 'config.json')},
                code_paths=['bot_detector'],
                input_example=example[['dialog_id', 'message', 'participant_index', 'text']],
                pip_requirements=[f'{p}=={version(p)}' for p in (
                    'mlflow', 'lightgbm', 'scikit-learn', 'pandas', 'numpy', 'scipy', 'torch', 'sentence-transformers', 'transformers',
                )],
                registered_model_name=MODEL_NAME,
            )

    client = MlflowClient()
    new_version = model_info.registered_model_version
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
        champion_metric = client.get_run(champion.run_id).data.metrics[METRIC]
    except mlflow.exceptions.MlflowException:
        champion, champion_metric = None, float('inf')

    if args.promote or metrics[METRIC] < champion_metric:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, new_version)
        print(f'Версия {new_version} стала {CHAMPION}: {METRIC} {metrics[METRIC]:.4f} (было {champion_metric:.4f})')
    else:
        print(f'Версия {new_version} зарегистрирована, {CHAMPION} остаётся версия {champion.version}: '
              f'{metrics[METRIC]:.4f} не лучше {champion_metric:.4f}')


if __name__ == '__main__':
    main()
