"""Shared pieces: data loading, model definitions and the routing rule.

Only `load(split="train")` is used by tuning. The test file is opened solely by
final_eval.py and reproduce_baseline.py.
"""
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA = os.environ.get("IDS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
TUNING_ONLY = False     # set by tune.py; opening the test file then raises
COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
           'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
           'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
           'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
           'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
           'label', 'difficulty']
FEATURES = COLUMNS[:41]
CATEGORICAL = ['protocol_type', 'service', 'flag']
ATTACK_MAP = {
    'normal': 'normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe',
    'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'httptunnel': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R',
}
CLASSES = ['DoS', 'Probe', 'R2L', 'U2R', 'normal']      # sorted, as LabelEncoder gave
IDX = {c: i for i, c in enumerate(CLASSES)}
LOW_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
CERTAIN_GRID = [0.9, 0.95, 0.99, 0.999, 0.9999]
ROUTE_CAP = 15.0


def read(name):
    if TUNING_ONLY and "Test" in name:
        raise RuntimeError("tuning must not open the test file")
    return pd.read_csv(os.path.join(DATA, name), names=COLUMNS)


def encoders(train_df):
    """Fitted on training data only; a category unseen in training maps to -1."""
    return {c: {v: i for i, v in enumerate(sorted(train_df[c].unique()))} for c in CATEGORICAL}


def design(df, enc):
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].map(enc[c]).fillna(-1).astype(int)
    return X.to_numpy(dtype=float)


def targets(df):
    return df['label'].map(ATTACK_MAP).map(IDX).to_numpy()


def fit_gate(X, y, seed=42):
    m = XGBClassifier(random_state=seed, eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(X, y)
    return m


def fit_specialist(X, y, seed=42):
    keep = np.isin(y, [IDX['normal'], IDX['R2L'], IDX['U2R']])
    Xr, yr = X[keep], y[keep]
    k = min(5, np.bincount(yr, minlength=5)[IDX['U2R']] - 1,
            np.bincount(yr, minlength=5)[IDX['R2L']] - 1)
    Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xr, yr)
    le = LabelEncoder()
    m = XGBClassifier(random_state=seed, eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(Xb, le.fit_transform(yb))
    return m, le


def predict_parts(gate, spec, le, X):
    proba = gate.predict_proba(X)
    gp, gc = np.argmax(proba, axis=1), np.max(proba, axis=1)
    sp = le.inverse_transform(np.argmax(spec.predict_proba(X), axis=1))
    return gp, gc, sp


def route(gp, gc, sp, low, certain):
    mask = (gc < low) | ((gp == IDX['normal']) & (gc < certain))
    return np.where(mask, sp, gp), mask
