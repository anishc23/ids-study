"""Selects routing thresholds using KDDTrain+ only. Writes thresholds.json."""
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import common
from common import *

common.TUNING_ONLY = True
SEEDS = range(10)


def type_holdout(df, y, seed):
    """Arm S: a third of attack types per category go to validation whole."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(df), bool)
    held = {}
    for cat in ('DoS', 'Probe', 'R2L', 'U2R'):
        types = sorted(df.loc[y == IDX[cat], 'label'].unique())
        n = max(1, round(len(types) / 3))
        pick = sorted(rng.choice(types, n, replace=False).tolist())
        held[cat] = pick
        val |= df['label'].isin(pick).to_numpy()
    normal = np.flatnonzero(y == IDX['normal'])
    val[rng.choice(normal, len(normal) // 5, replace=False)] = True
    return ~val, val, held


def stratified(df, y, seed):
    idx = np.arange(len(df))
    fit, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    a, b = np.zeros(len(df), bool), np.zeros(len(df), bool)
    a[fit], b[va] = True, True
    return a, b, None


def select(df, y, enc, fit, val, seed):
    X = design(df, enc)
    sc = StandardScaler().fit(X[fit])
    Xf, Xv = sc.transform(X[fit]), sc.transform(X[val])
    gate = fit_gate(Xf, y[fit], seed)
    spec, le = fit_specialist(Xf, y[fit], seed)
    gp, gc, sp = predict_parts(gate, spec, le, Xv)
    grid = []
    for low in LOW_GRID:
        for cert in CERTAIN_GRID:
            p, m = route(gp, gc, sp, low, cert)
            grid.append({'low': low, 'certain': cert,
                         'macro_f1': round(100 * f1_score(y[val], p, average='macro', zero_division=0), 2),
                         'routed': round(100 * m.mean(), 2)})
    ok = [g for g in grid if g['routed'] <= ROUTE_CAP]
    if ok:
        best = max(ok, key=lambda g: g['macro_f1'])  # max keeps the first on ties
    else:
        best = min(grid, key=lambda g: g['routed'])  # amendment 1 in PROTOCOL.md
    return best, grid


df = read('KDDTrain+.txt')
y = targets(df)
enc = encoders(df)
out = {}
for arm, split in (('S', type_holdout), ('V', stratified)):
    rows = []
    for s in SEEDS:
        fit, val, held = split(df, y, s)
        best, grid = select(df, y, enc, fit, val, s)
        rows.append({'seed': s, 'held_out': held, 'selected': best,
                     'fit_rows': int(fit.sum()), 'val_rows': int(val.sum()), 'grid': grid})
        print(arm, s, best, held, flush=True)
    out[arm] = rows
json.dump(out, open('thresholds.json', 'w'), indent=1, sort_keys=True)
