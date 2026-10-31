"""Exploratory, added after the pre-specified evaluation: per-class scores for the comparators.

The protocol fixed only their macro-F1. Nothing here selects a threshold or a model.
"""
import csv, json
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from common import *

tr, te = read('KDDTrain+.txt'), read('KDDTest+.txt')
enc = encoders(tr)
sc = StandardScaler().fit(design(tr, enc))
Xtr, Xte, ytr, yte = sc.transform(design(tr, enc)), sc.transform(design(te, enc)), targets(tr), targets(te)
rows = []
for s in range(10):
    Xb, yb = SMOTE(random_state=s, k_neighbors=5).fit_resample(Xtr, ytr)
    p = XGBClassifier(random_state=s, eval_metric='logloss', verbosity=0, n_jobs=-1).fit(Xb, yb).predict(Xte)
    pr, rc, f, _ = precision_recall_fscore_support(yte, p, labels=range(5), zero_division=0)
    rows.append({'model': 'smote_all_classes', 'seed': s,
                 'macro_f1': round(100 * f1_score(yte, p, average='macro'), 2),
                 **{f'f1_{c}': round(100 * f[i], 2) for i, c in enumerate(CLASSES)},
                 'r2l_precision': round(100 * pr[IDX['R2L']], 2), 'r2l_recall': round(100 * rc[IDX['R2L']], 2),
                 'u2r_precision': round(100 * pr[IDX['U2R']], 2), 'u2r_recall': round(100 * rc[IDX['U2R']], 2)})
m = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1)
m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
p = m.predict(Xte)
pr, rc, f, _ = precision_recall_fscore_support(yte, p, labels=range(5), zero_division=0)
rows.append({'model': 'balanced_weights', 'seed': 42, 'macro_f1': round(100 * f1_score(yte, p, average='macro'), 2),
             **{f'f1_{c}': round(100 * f[i], 2) for i, c in enumerate(CLASSES)},
             'r2l_precision': round(100 * pr[IDX['R2L']], 2), 'r2l_recall': round(100 * rc[IDX['R2L']], 2),
             'u2r_precision': round(100 * pr[IDX['U2R']], 2), 'u2r_recall': round(100 * rc[IDX['U2R']], 2)})
with open('comparators_detail.csv', 'w', newline='') as f_:
    w = csv.DictWriter(f_, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
