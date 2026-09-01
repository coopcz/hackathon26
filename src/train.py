"""Shared estimators, metrics and split helpers.

Used by both src/train_esp32.py (the deployed ESP32 model) and src/intel/demo.py
(the reference experiment).

THE RECURRING TRAP THIS MODULE EXISTS TO AVOID: windows cut from one continuous
recording are near-duplicates - same room, same people, same furniture, seconds
apart.  A random train/test split puts them on both sides and reports an accuracy
that is mostly memorisation.  Every honest evaluation here splits on a GROUPING
column instead: `site` for the Intel rooms, the source recording for the ESP32.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance

from .features import (FEATURE_NAMES, FEATURE_FAMILY, SCALE_FREE_FEATURES,
                       fit_site_baseline, apply_site_baseline)

RANDOM_STATE = 42
LABELS = {0: "AWAY (empty)", 1: "HOME (occupied)"}


def make_rf():
    # class_weight balances the 1:8 away/home imbalance so the minority AWAY class
    # is not simply ignored -- a model that always predicts HOME would score 89%.
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def make_gb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, random_state=RANDOM_STATE)


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # AWAY is the minority and the operationally risky class: a false AWAY
        # switches the AC off while somebody is home.
        "precision_away": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_away": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_away": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_home": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_home": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_home": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_confusion(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{title}")
    print("                    pred AWAY   pred HOME")
    print(f"  actual AWAY  {cm[0,0]:10d}  {cm[0,1]:10d}")
    print(f"  actual HOME  {cm[1,0]:10d}  {cm[1,1]:10d}")
    return cm


def evaluate_random_split(X, y, test_size=0.2):
    """Optimistic within-session split (stratified 80/20)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    rf = make_rf().fit(Xtr, ytr)
    return rf, (Xtr, Xte, ytr, yte)


def evaluate_leave_one_site_out(X, y, site):
    """Realistic cross-environment split: train on 2 rooms, test on the held-out one."""
    results = {}
    for held in np.unique(site):
        tr, te = site != held, site == held
        rf = make_rf().fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        results[held] = {"metrics": metrics(y[te], pred), "y_true": y[te], "y_pred": pred,
                         "n_train": int(tr.sum()), "n_test": int(te.sum())}
    return results


def family_importance(names, importances):
    """Roll per-feature importance up to the physical family it belongs to."""
    agg = {}
    for n, v in zip(names, importances):
        agg.setdefault(FEATURE_FAMILY[n], 0.0)
        agg[FEATURE_FAMILY[n]] += v
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def build_calibrated(X, y, site, feature_names):
    """Apply per-site baseline calibration to every room independently."""
    Xc = np.zeros((len(X), len(SCALE_FREE_FEATURES)))
    for r in np.unique(site):
        m = site == r
        # baseline is fitted on that room's own UNLABELLED windows only
        Xc[m] = apply_site_baseline(X[m], list(feature_names), fit_site_baseline(X[m], list(feature_names)))
    return Xc


def evaluate_calibrated_loso(Xc, y, site):
    """Leave-one-room-out on the calibrated scale-free features: the deployment model."""
    results = {}
    for held in np.unique(site):
        tr, te = site != held, site == held
        rf = make_rf().fit(Xc[tr], y[tr])
        pred = rf.predict(Xc[te])
        results[held] = {"metrics": metrics(y[te], pred), "y_true": y[te], "y_pred": pred}
    return results


def train_production_model(Xc, y):
    """Fit the shipped model on all three calibrated rooms."""
    return make_rf().fit(Xc, y)
