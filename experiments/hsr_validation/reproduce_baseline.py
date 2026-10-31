"""Reproduces the published numbers, including the test-set threshold choice."""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from common import *

tr, te = read('KDDTrain+.txt'), read('KDDTest+.txt')
enc = encoders(tr)
unseen = {c: sorted(set(te[c]) - set(tr[c])) for c in CATEGORICAL}
print('categories in test but not train:', unseen)
Xtr, Xte = design(tr, enc), design(te, enc)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
ytr, yte = targets(tr), targets(te)
print('unmapped labels:', int(np.isnan(tr['label'].map(ATTACK_MAP).map(IDX).astype(float)).sum()),
      'test class counts', np.bincount(yte))
gate = fit_gate(Xtr, ytr)
spec, le = fit_specialist(Xtr, ytr)
gp, gc, sp = predict_parts(gate, spec, le, Xte)
print('gate macro-F1', round(100 * f1_score(yte, gp, average='macro'), 2))
pred, mask = route(gp, gc, sp, 0.5, 0.999)
print('HSR macro-F1', round(100 * f1_score(yte, pred, average='macro'), 2),
      'routed', round(100 * mask.mean(), 2),
      'per class', dict(zip(CLASSES, np.round(100 * f1_score(yte, pred, average=None), 2))))
