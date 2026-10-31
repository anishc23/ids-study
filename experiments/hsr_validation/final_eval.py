"""Scores frozen thresholds on KDDTest+. Run once, after thresholds.json is committed."""
import collections, csv, hashlib, json
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from common import *
from tune import stratified, type_holdout

th = json.load(open('thresholds.json'))
tr, te = read('KDDTrain+.txt'), read('KDDTest+.txt')
enc = encoders(tr)
Xtr_raw, Xte_raw, ytr, yte = design(tr, enc), design(te, enc), targets(tr), targets(te)
sc = StandardScaler().fit(Xtr_raw)
Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)


def mf1(y, p):
    return 100 * f1_score(y, p, average='macro', zero_division=0)


def per_class(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=range(5), zero_division=0)
    return {c: {'precision': round(100 * pr[i], 2), 'recall': round(100 * rc[i], 2),
                'f1': round(100 * f[i], 2)} for i, c in enumerate(CLASSES)}


def fast_macro(y, p):
    cm = np.bincount(5 * y + p, minlength=25).reshape(5, 5)
    tp = np.diag(cm).astype(float)
    d = cm.sum(0) + cm.sum(1)
    return 100 * np.mean(np.divide(2 * tp, d, out=np.zeros(5), where=d > 0))


def modal(rows):
    c = collections.Counter((r['selected']['low'], r['selected']['certain']) for r in rows)
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    mean_routed = lambda k: np.mean([r['selected']['routed'] for r in rows
                                     if (r['selected']['low'], r['selected']['certain']) == k])
    return min(tied, key=mean_routed), dict((str(k), v) for k, v in c.items())


def summary(v):
    return {'median': round(float(np.median(v)), 2), 'min': round(float(min(v)), 2),
            'max': round(float(max(v)), 2)}


per_seed, res = [], {}
for arm, split in (('S', type_holdout), ('V', stratified)):
    final, frozen, tuned, gates = [], [], [], []
    for r in th[arm]:
        s, low, cert = r['seed'], r['selected']['low'], r['selected']['certain']
        gate = fit_gate(Xtr, ytr, s)
        spec, le = fit_specialist(Xtr, ytr, s)
        gp, gc, sp = predict_parts(gate, spec, le, Xte)
        p, m = route(gp, gc, sp, low, cert)
        # optimism: the best test-tuned row on the same refit models
        rows = [(mf1(yte, route(gp, gc, sp, a, b)[0]), 100 * route(gp, gc, sp, a, b)[1].mean())
                for a in LOW_GRID for b in CERTAIN_GRID]
        best = max(x[0] for x in rows if x[1] <= ROUTE_CAP)
        # sensitivity: frozen partition-trained models
        fit, _, _ = split(tr, ytr, s)
        sf = StandardScaler().fit(Xtr_raw[fit])
        g2 = fit_gate(sf.transform(Xtr_raw[fit]), ytr[fit], s)
        s2, l2 = fit_specialist(sf.transform(Xtr_raw[fit]), ytr[fit], s)
        p2, _ = route(*predict_parts(g2, s2, l2, sf.transform(Xte_raw)), low, cert)
        f = f1_score(yte, p, average=None, labels=range(5), zero_division=0)
        row = {'arm': arm, 'seed': s, 'low': low, 'certain': cert,
               'gate_macro_f1': round(mf1(yte, gp), 2), 'macro_f1': round(mf1(yte, p), 2),
               'accuracy': round(100 * accuracy_score(yte, p), 2), 'routed': round(100 * m.mean(), 2),
               'f1_r2l': round(100 * f[IDX['R2L']], 2), 'f1_u2r': round(100 * f[IDX['U2R']], 2),
               'test_tuned_best': round(best, 2), 'frozen_macro_f1': round(mf1(yte, p2), 2)}
        per_seed.append(row)
        final.append(row['macro_f1']); frozen.append(row['frozen_macro_f1'])
        tuned.append(best); gates.append(row['gate_macro_f1'])
        print(row, flush=True)
    (low, cert), counts = modal(th[arm])
    gate = fit_gate(Xtr, ytr, 42)
    spec, le = fit_specialist(Xtr, ytr, 42)
    gp, gc, sp = predict_parts(gate, spec, le, Xte)
    p, m = route(gp, gc, sp, low, cert)
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(5000):
        i = rng.randint(0, len(yte), len(yte))
        diffs.append(fast_macro(yte[i], p[i]) - fast_macro(yte[i], gp[i]))
    gate_ok, hsr_ok = gp == yte, p == yte
    b, c = int((hsr_ok & ~gate_ok).sum()), int((gate_ok & ~hsr_ok).sum())
    res[arm] = {
        'per_seed_macro_f1': summary(final), 'per_seed_frozen_macro_f1': summary(frozen),
        'gate_macro_f1': summary(gates), 'test_tuned_best': summary(tuned),
        'optimism_median': round(float(np.median(np.array(tuned) - np.array(final))), 2),
        'threshold_counts': counts,
        'reported_model': {
            'low': low, 'certain': cert, 'macro_f1': round(mf1(yte, p), 2),
            'gate_macro_f1': round(mf1(yte, gp), 2), 'accuracy': round(100 * accuracy_score(yte, p), 2),
            'routed': round(100 * m.mean(), 2), 'per_class': per_class(yte, p),
            'gate_per_class': per_class(yte, gp),
            'bootstrap_diff_ci95': [round(float(x), 2) for x in np.percentile(diffs, [2.5, 97.5])],
            'diff': round(mf1(yte, p) - mf1(yte, gp), 2),
            'mcnemar': {'hsr_only_correct': b, 'gate_only_correct': c,
                        'p_exact': float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0}}}
comp = {'balanced_weights': [], 'smote_all_classes': []}
for s in range(10):
    m = XGBClassifier(random_state=s, eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    comp['balanced_weights'].append(mf1(yte, m.predict(Xte)))
    Xb, yb = SMOTE(random_state=s, k_neighbors=5).fit_resample(Xtr, ytr)
    m = XGBClassifier(random_state=s, eval_metric='logloss', verbosity=0, n_jobs=-1).fit(Xb, yb)
    comp['smote_all_classes'].append(mf1(yte, m.predict(Xte)))
res['comparators'] = {k: summary(v) for k, v in comp.items()}
res['sha256'] = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
                 for f in ('PROTOCOL.md', 'thresholds.json')}
json.dump(res, open('validated_results.json', 'w'), indent=1, sort_keys=True)
with open('validated_per_seed.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_seed[0]))
    w.writeheader(); w.writerows(per_seed)
print(json.dumps(res, indent=1, sort_keys=True))
