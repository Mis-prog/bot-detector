"""Признаки участника диалога. Один и тот же код для обучения и для сервиса.

Логика перенесена из solution.ipynb, но без TF-IDF: он давал 0.003 LogLoss и удваивал время predict_proba.
Тексты участников (text, partner_text) build_features по-прежнему собирает — их ждут старые версии модели с TF-IDF.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

KEYS = ['dialog_id', 'participant_index']
STAT_COLS = ['latin', 'upper', 'punct', 'question', 'emoji', 'mean_words', 'max_len', 'unique_ratio']
COH_COLS = ['coh_mean', 'coh_min', 'coh_max', 'self_sim_mean']


def load_messages(path):
    with open(path, encoding='utf-8') as f:
        dialogs = json.load(f)
    rows = [
        {'dialog_id': dialog_id, **msg}
        for dialog_id, messages in dialogs.items()
        for msg in messages
    ]
    return pd.DataFrame(rows, columns=['dialog_id', 'text', 'message', 'participant_index'])


def participants_of(msgs):
    """Оба участника каждого диалога — даже если один из них ничего не написал."""
    dialog_ids = msgs['dialog_id'].drop_duplicates()
    return pd.DataFrame([(d, p) for d in dialog_ids for p in (0, 1)], columns=KEYS)


def aggregate(msgs):
    msgs = msgs.sort_values(['dialog_id', 'message'])
    return (
        msgs.groupby(KEYS)
        .agg(
            text=('text', lambda s: ' \n '.join(s)),
            n_messages=('text', 'size'),
            mean_len=('text', lambda s: s.str.len().mean()),
        )
        .reset_index()
    )


def add_partner(df):
    partner = (
        df[KEYS + ['text', 'n_messages', 'mean_len']]
        .assign(participant_index=lambda d: 1 - d['participant_index'])
        .rename(columns={'text': 'partner_text', 'n_messages': 'partner_n_messages', 'mean_len': 'partner_mean_len'})
    )
    return df.merge(partner, on=KEYS, how='left')


def text_stats(msgs):
    s = msgs['text']
    length = s.str.len().clip(lower=1)
    return (
        msgs.assign(
            latin=s.str.count(r'[A-Za-z]') / length,
            upper=s.str.count(r'[A-ZА-ЯЁ]') / length,
            punct=s.str.count(r'[.,!?;:]') / length,
            question=s.str.contains(r'\?').astype(int),
            emoji=s.str.contains(r'[)(]').astype(int),
            n_words=s.str.split().str.len(),
        )
        .groupby(KEYS)
        .agg(
            latin=('latin', 'mean'),
            upper=('upper', 'mean'),
            punct=('punct', 'mean'),
            question=('question', 'mean'),
            emoji=('emoji', 'mean'),
            mean_words=('n_words', 'mean'),
            max_len=('text', lambda t: t.str.len().max()),
            unique_ratio=('text', lambda t: t.nunique() / len(t)),
        )
        .reset_index()
    )


def add_stats(df, msgs):
    stats = text_stats(msgs)
    partner = (
        stats.assign(participant_index=lambda d: 1 - d['participant_index'])
        .rename(columns={c: 'partner_' + c for c in STAT_COLS})
    )
    return df.merge(stats, on=KEYS, how='left').merge(partner, on=KEYS, how='left')


def embedding_features(msgs, emb):
    rows = []
    for dialog_id, d in msgs.reset_index(drop=True).groupby('dialog_id', sort=False):
        d = d.sort_values('message')
        who = d['participant_index'].to_numpy()
        e = emb[d.index.to_numpy()]
        for p in (0, 1):
            own = np.flatnonzero(who == p)
            coh, self_sim = [], []
            for k, i in enumerate(own):
                prev_partner = np.flatnonzero(who[:i] != p)
                if len(prev_partner):
                    coh.append(e[i] @ e[prev_partner[-1]])
                if k > 0:
                    self_sim.append(e[i] @ e[own[k - 1]])
            mean_emb = e[own].mean(axis=0) if len(own) else np.zeros(emb.shape[1])
            rows.append({
                'dialog_id': dialog_id,
                'participant_index': p,
                'coh_mean': np.mean(coh) if coh else np.nan,
                'coh_min': np.min(coh) if coh else np.nan,
                'coh_max': np.max(coh) if coh else np.nan,
                'self_sim_mean': np.mean(self_sim) if self_sim else np.nan,
                **{f'emb_{j}': v for j, v in enumerate(mean_emb)},
            })
    return pd.DataFrame(rows)


def add_embedding_features(df, msgs, emb):
    feats = embedding_features(msgs, emb)
    cols = [c for c in feats.columns if c not in KEYS]
    partner = (
        feats.assign(participant_index=lambda d: 1 - d['participant_index'])
        .rename(columns={c: 'partner_' + c for c in cols})
    )
    return df.merge(feats, on=KEYS, how='left').merge(partner, on=KEYS, how='left')


def build_features(participants, msgs, emb):
    """participants — строки (dialog_id, participant_index), к которым дописываются признаки.

    msgs и emb выровнены: emb[i] — эмбеддинг сообщения msgs.iloc[i].
    """
    msgs = msgs.reset_index(drop=True)
    df = participants.merge(aggregate(msgs), on=KEYS, how='left')
    df = df.fillna({'text': '', 'n_messages': 0, 'mean_len': 0})
    df = add_partner(df)
    df = add_stats(df, msgs)
    return add_embedding_features(df, msgs, emb)


def feature_columns(emb_dim):
    emb_cols = [f'emb_{j}' for j in range(emb_dim)]
    return (
        ['n_messages', 'mean_len', 'partner_n_messages', 'partner_mean_len']
        + STAT_COLS + ['partner_' + c for c in STAT_COLS]
        + COH_COLS + ['partner_' + c for c in COH_COLS]
        + emb_cols + ['partner_' + c for c in emb_cols]
    )


def make_model(features, seed=42):
    """Гиперпараметры подобраны Optuna в notebooks/solution.ipynb (для модели с TF-IDF)."""
    return Pipeline([
        ('features', ColumnTransformer([('num', 'passthrough', features)])),
        ('clf', lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.024, num_leaves=22, min_child_samples=18,
            subsample=0.94, subsample_freq=1, colsample_bytree=0.5, reg_lambda=0.53,
            random_state=seed, verbose=-1,
        )),
    ])
