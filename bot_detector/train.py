"""Обучение модели и регистрация в MLflow.

    python -m bot_detector.train

Повторяет solution.ipynb: GroupKFold по диалогам, температура по OOF, обучение на всём train.
Новая версия получает алиас champion, только если её CV LogLoss с температурой лучше текущего чемпиона.
"""
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
from scipy.optimize import minimize_scalar
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from bot_detector.features import apply_temperature, build_features, feature_columns, load_messages, make_model
from bot_detector.model import BotDetectorModel

MODEL_NAME = os.environ.get('MODEL_NAME', 'bot-detector')
ENCODER_NAME = 'cointegrated/rubert-tiny2'
DATA_DIR = Path(os.environ.get('DATA_DIR', 'data'))
SEED = 42
N_SPLITS = 5
CLIP = 0.001
CHAMPION = 'champion'
METRIC = 'cv_logloss_calibrated'


def fit_temperature(p, target):
    return minimize_scalar(lambda T: log_loss(target, apply_temperature(p, T)), bounds=(0.3, 3), method='bounded').x


def main():
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
    features, numeric = feature_columns(train_emb.shape[1])
    y = train['is_bot'].to_numpy()
    groups = train['dialog_id']
    cv = GroupKFold(n_splits=N_SPLITS)

    with mlflow.start_run() as run:
        mlflow.log_params({
            'encoder': ENCODER_NAME,
            'n_splits': N_SPLITS,
            'clip': CLIP,
            'n_numeric_features': len(numeric),
            'n_train': len(train),
            **{f'lgbm_{k}': v for k, v in make_model(numeric).named_steps['clf'].get_params().items()
               if k in ('n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda')},
        })

        oof = cross_val_predict(make_model(numeric, SEED), train[features], y, groups=groups, cv=cv, method='predict_proba')[:, 1]
        temperature = fit_temperature(oof, y)
        calibrated = np.zeros(len(y))
        for tr, va in cv.split(train, y, groups=groups):
            calibrated[va] = apply_temperature(oof[va], fit_temperature(oof[tr], y[tr]))
        metrics = {
            'cv_logloss': log_loss(y, oof),
            METRIC: log_loss(y, calibrated),
            'cv_roc_auc': roc_auc_score(y, oof),
            'cv_f1_at_0_5': f1_score(y, oof >= 0.5),
            'temperature': temperature,
        }
        mlflow.log_metrics(metrics)
        print(json.dumps(metrics, indent=2))

        pipeline = make_model(numeric, SEED).fit(train[features], y)
        preds = apply_temperature(pipeline.predict_proba(test[features])[:, 1], temperature)
        submission = test[['ID']].assign(is_bot=np.clip(preds, CLIP, 1 - CLIP))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            submission.to_csv(tmp / 'submission.csv', index=False)
            mlflow.log_artifact(tmp / 'submission.csv')
            joblib.dump(pipeline, tmp / 'pipeline.joblib')
            encoder.save(str(tmp / 'encoder'))
            (tmp / 'config.json').write_text(json.dumps({'temperature': temperature, 'clip': CLIP, 'encoder': ENCODER_NAME}))
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

    if metrics[METRIC] < champion_metric:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, new_version)
        print(f'Версия {new_version} стала {CHAMPION}: {METRIC} {metrics[METRIC]:.4f} (было {champion_metric:.4f})')
    else:
        print(f'Версия {new_version} зарегистрирована, {CHAMPION} остаётся версия {champion.version}: '
              f'{metrics[METRIC]:.4f} не лучше {champion_metric:.4f}')


if __name__ == '__main__':
    main()
