"""
Train and honestly evaluate the ESP32 occupancy model.

    python -m src.train_esp32              # evaluate, fit, save artifacts/esp32_model.joblib
    python -m src.train_esp32 --no-save    # report only
    python -m src.train_esp32 --augment-intel

THE ONLY EVALUATION THAT MEANS ANYTHING HERE IS GROUPED BY RECORDING.

Windows cut from one 30-second recording are near-duplicates of each other: same
room, same person, same pose, fractions of a second apart -- and at 75% overlap
they literally share packets.  A random train/test split puts those on both
sides and reports ~99%, which is a measurement of memorisation.  Every number
below comes from StratifiedGroupKFold over the source recording, so no window in
the test fold shares a recording -- let alone a packet -- with any training window.

WHAT THIS SCRIPT SEARCHES OVER, AND WHY EACH AXIS EXISTS
--------------------------------------------------------

  FEATURE SET
    raw             all 16 features, uncalibrated.  The control.
    scale_free      the 7 hardware-portable features, baseline-calibrated.  This
                    is what the Intel experiment showed is needed to survive a
                    change of ENVIRONMENT.
    all_calibrated  all 16, each expressed relative to this site's own quiet
                    baseline.  Cross-site portability is deliberately traded away
                    for accuracy at the one site we actually calibrated at -- if
                    you have a baseline recording, "std_amp is 6x my quiet floor"
                    is strong evidence that `scale_free` simply discards.

  BASELINE MODE      how the site's "quiet floor" is estimated
    quantile        5th percentile of all training windows.  Needs no labels,
                    only that the room is empty some of the time.
    empty_median    median of the training windows labelled `empty`.  Uses the
                    calibration recordings you are deliberately taking, so it is
                    both more accurate and still deployment-realistic.
    Baselines are always fitted on the TRAINING fold only and then applied to the
    test fold.  Fitting on everything would leak the test recordings' statistics
    into the calibration.

  SMOOTHING          consecutive-window vote
    Occupancy changes on the scale of minutes; a 2.56 s window is a very short
    look.  Averaging p(HOME) over the last k windows trades a few seconds of
    latency for a large drop in flicker, and it is free.

  THRESHOLD          how confident AWAY has to be
    Asymmetric on purpose: a false AWAY turns the AC off on somebody who is home.
    The threshold is chosen as the one maximising AWAY recall subject to HOME
    recall staying above HOME_RECALL_FLOOR.
"""

import argparse
import os
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import AWAY, HOME, OVERLAP, WINDOW_SECONDS, build, summarize
from .esp_csi import REFERENCE_SUBCARRIER_MAPS
from .features import (FEATURE_NAMES, SCALE_FREE_FEATURES, OFFSET_CALIBRATED,
                       BASELINE_PERCENTILE)
from .train import metrics, print_confusion, RANDOM_STATE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "artifacts", "esp32_model.joblib")

# A false AWAY switches the AC off on somebody who is home. Comfort failures are
# noticed instantly; a missed saving costs cents. So HOME recall is a constraint,
# not something to trade away for a better headline number.
HOME_RECALL_FLOOR = 0.98
# A model that never says AWAY trivially satisfies the HOME-recall floor and is
# completely useless -- it can only ever leave the AC running. Below this it is
# reported as not deployable rather than crowned "best".
MIN_USEFUL_AWAY_RECALL = 0.20
SMOOTHING_CHOICES = (1, 3, 5, 9)
THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)

FEATURE_SETS = ("raw", "scale_free", "all_calibrated")
BASELINE_MODES = ("quantile", "empty_median")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
# Two kinds of feature, and using the wrong rule on either is silent:
#   RATIO  - unbounded positive magnitudes. "5x my own quiet floor" is meaningful,
#            so divide.
#   OFFSET - already bounded (acf_lag in [-1,1], dop_ratio in [0,1]). Dividing by
#            a near-zero baseline flips signs and explodes magnitudes. Subtract.
# Every feature except the two OFFSET ones is a positive magnitude, so the
# full-calibration rule is just "divide unless bounded".
_OFFSET_IDX = [FEATURE_NAMES.index(f) for f in OFFSET_CALIBRATED]


def fit_baseline(X, condition, mode, feature_names=FEATURE_NAMES):
    """Estimate this site's quiet reference level for every one of the 16 features.

    Returned as a full-length vector so any feature subset can index into it.
    """
    if mode == "empty_median":
        empty = X[condition == "empty"]
        if len(empty) >= 5:
            base = np.median(empty, axis=0)
        else:                              # not enough calibration data -- fall back
            base = np.percentile(X, BASELINE_PERCENTILE, axis=0)
    elif mode == "quantile":
        base = np.percentile(X, BASELINE_PERCENTILE, axis=0)
    else:
        raise ValueError(f"unknown baseline mode {mode!r}")
    # guard divisors only; an offset baseline of zero is perfectly fine
    div = np.ones(len(base), dtype=bool)
    div[_OFFSET_IDX] = False
    base = np.where(div & (np.abs(base) < 1e-9), 1e-9, base)
    return base


def apply_calibration(X, baseline, feature_set):
    """Project the raw 16-feature matrix into the chosen feature space."""
    if feature_set == "raw":
        return X.copy(), list(FEATURE_NAMES)

    names = SCALE_FREE_FEATURES if feature_set == "scale_free" else FEATURE_NAMES
    idx = [FEATURE_NAMES.index(f) for f in names]
    out = np.empty((len(X), len(idx)), dtype=float)
    for j, f in enumerate(names):
        i = FEATURE_NAMES.index(f)
        if f in OFFSET_CALIBRATED:
            out[:, j] = X[:, i] - baseline[i]
        else:
            out[:, j] = X[:, i] / baseline[i]
    return out, list(names)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models():
    """Small, fast, interpretable. The task is data-limited, not model-limited."""
    return {
        "rf": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "extratrees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "histgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=RANDOM_STATE),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def n_splits_for(y, group, cap=5):
    """Most folds we can cut and still get both classes in every test fold."""
    per_class = [len(np.unique(group[y == c])) for c in (AWAY, HOME)]
    return max(2, min(cap, min(per_class)))


def out_of_fold_proba(ds, feature_set, baseline_mode, model_name, n_splits,
                      augment=None):
    """Out-of-fold p(HOME) for every window, never predicted by a model that saw
    its recording.

    `augment` is an optional (X_extra, y_extra) already in the SAME calibrated
    feature space; it is added to training folds only and never scored.
    """
    X, y, group, cond = ds["X"], ds["y"], ds["group"], ds["condition"]
    proba = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for tr, te in cv.split(X, y, groups=group):
        baseline = fit_baseline(X[tr], cond[tr], baseline_mode)
        Xtr, _ = apply_calibration(X[tr], baseline, feature_set)
        Xte, _ = apply_calibration(X[te], baseline, feature_set)
        ytr = y[tr]
        if augment is not None and feature_set == "scale_free":
            Xtr = np.vstack([Xtr, augment[0]])
            ytr = np.concatenate([ytr, augment[1]])
        model = make_models()[model_name]
        model.fit(Xtr, ytr)
        proba[te] = model.predict_proba(Xte)[:, HOME]
    return proba


def smooth_proba(proba, group, k):
    """Trailing mean of p(HOME) over k windows, reset at every recording boundary.

    Trailing, not centred: at inference you only have the past. Evaluating on a
    centred window would quietly use information the deployed system never has.
    """
    if k <= 1:
        return proba.copy()
    out = np.empty_like(proba)
    for g in np.unique(group):
        m = np.flatnonzero(group == g)      # dataset rows are in window order per file
        p = proba[m]
        out[m] = [p[max(0, i - k + 1):i + 1].mean() for i in range(len(p))]
    return out


def predict_at(proba_home, threshold):
    """AWAY only when p(AWAY) clears the threshold; otherwise fail safe to HOME."""
    return np.where((1.0 - proba_home) >= threshold, AWAY, HOME)


def pick_threshold(y, proba_home, floor=HOME_RECALL_FLOOR):
    """Best AWAY recall subject to HOME recall >= floor.

    Returns (threshold, metrics, met_floor). If no threshold reaches the floor,
    the least-bad one is returned and flagged rather than silently accepted.
    """
    best, best_any = None, None
    for t in THRESHOLDS:
        m = metrics(y, predict_at(proba_home, t))
        if best_any is None or m["recall_home"] > best_any[1]["recall_home"]:
            best_any = (t, m)
        if m["recall_home"] >= floor and (best is None or m["recall_away"] > best[1]["recall_away"]):
            best = (t, m)
    return (*best, True) if best else (*best_any, False)


def per_condition_recall(condition, y, pred):
    """Where the errors actually live. `occupied_still` is the hard one --
    a person sitting perfectly still barely modulates the channel."""
    rows = {}
    for c in np.unique(condition):
        m = condition == c
        rows[c] = {"n": int(m.sum()), "correct": float((pred[m] == y[m]).mean())}
    return rows


# ---------------------------------------------------------------------------
# Intel augmentation (optional)
# ---------------------------------------------------------------------------

def intel_augmentation():
    """The Intel windows, calibrated into the same 7 scale-free features.

    The premise of scale-free calibration is that "5x my own noise floor" means
    the same thing on any radio. If that holds, 1784 extra Intel windows are
    usable training data while the ESP32 set is still small. If it does not, the
    CV table will say so -- which is the point of making it a flag.
    """
    from .intel.build_dataset import build as build_intel
    from .train import build_calibrated
    Xi, yi, room, _, _ = build_intel()
    return build_calibrated(Xi, yi, room, FEATURE_NAMES), yi


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def run(save=True, augment_intel=False, window_seconds=None, overlap=None,
        force=False, data_dir=None, include_bad=False, save_anyway=False,
        model_path=MODEL_PATH):
    rule("1  DATASET")
    kw = {}
    if window_seconds is not None:
        kw["window_seconds"] = window_seconds
    if overlap is not None:
        kw["overlap"] = overlap
    if data_dir:
        kw["data_dir"] = data_dir
    ds = build(force=force, exclude_bad=not include_bad, **kw)
    if include_bad:
        print("\n  --include-bad: quality-flagged recordings were NOT excluded. Any")
        print("  accuracy below is contaminated by AGC steps and dropped packets.")
    print()
    print(summarize(ds))

    y, group, cond = ds["y"], ds["group"], ds["condition"]
    n_groups = len(np.unique(group))
    groups_away = len(np.unique(group[y == AWAY]))
    groups_home = len(np.unique(group[y == HOME]))

    if groups_away < 2 or groups_home < 2:
        print(f"\n  NOT ENOUGH DATA TO TRAIN OR EVALUATE.")
        print(f"  Have {groups_away} empty recording(s) and {groups_home} occupied.")
        print(f"  Grouped cross-validation needs at least 2 of each, and the site")
        print(f"  baseline needs empty windows to calibrate against. Keep recording;")
        print(f"  aim for >=10 of each condition, and re-run this script.")
        return None

    n_splits = n_splits_for(y, group)
    print(f"\n  {n_groups} recordings, splitting {n_splits}-fold BY RECORDING "
          f"(no window shares a recording across the split)")

    augment = None
    if augment_intel:
        augment = intel_augmentation()
        print(f"  + {len(augment[1])} Intel windows added to training folds only "
              f"(scale_free configurations only)")

    # ------------------------------------------------------------------
    rule("2  CONFIGURATION SEARCH  (grouped CV, argmax AWAY recall at fixed HOME recall)")
    print("  Every row is out-of-fold. `thr` is the AWAY decision threshold chosen")
    print(f"  as the best AWAY recall with HOME recall >= {HOME_RECALL_FLOOR:.0%}.\n")
    print(f"  {'features':<15} {'baseline':<14} {'model':<11} {'k':>2} {'thr':>5} "
          f"{'acc':>6} {'AWAYrec':>8} {'HOMErec':>8} {'macroF1':>8}")
    print("  " + "-" * 92)

    results = []
    for feature_set in FEATURE_SETS:
        modes = BASELINE_MODES if feature_set != "raw" else ("quantile",)
        for baseline_mode in modes:
            for model_name in make_models():
                proba = out_of_fold_proba(ds, feature_set, baseline_mode,
                                          model_name, n_splits, augment)
                for k in SMOOTHING_CHOICES:
                    sp = smooth_proba(proba, group, k)
                    thr, m, met = pick_threshold(y, sp)
                    results.append({"feature_set": feature_set, "baseline_mode": baseline_mode,
                                    "model": model_name, "k": k, "threshold": thr,
                                    "metrics": m, "met_floor": met, "proba": sp})
                # print only the best k per (features, baseline, model), else it is 200 rows
                best_k = max([r for r in results if r["model"] == model_name
                              and r["feature_set"] == feature_set
                              and r["baseline_mode"] == baseline_mode],
                             key=lambda r: (r["met_floor"], r["metrics"]["recall_away"]))
                m, bl = best_k["metrics"], ("-" if feature_set == "raw" else baseline_mode)
                print(f"  {feature_set:<15} {bl:<14} {model_name:<11} {best_k['k']:>2} "
                      f"{best_k['threshold']:>5.2f} {m['accuracy']:>6.3f} "
                      f"{m['recall_away']:>8.3f} {m['recall_home']:>8.3f} "
                      f"{m['f1_macro']:>8.3f}"
                      f"{'' if best_k['met_floor'] else '   <-- HOME recall floor not reached'}")

    # ------------------------------------------------------------------
    def rank(r):
        m = r["metrics"]
        deployable = r["met_floor"] and m["recall_away"] >= MIN_USEFUL_AWAY_RECALL
        return (deployable, m["recall_away"], m["f1_macro"])

    best = max(results, key=rank)
    deployable = rank(best)[0]
    rule("3  BEST CONFIGURATION")
    print(f"  features   {best['feature_set']}")
    print(f"  baseline   {best['baseline_mode'] if best['feature_set'] != 'raw' else 'none'}")
    print(f"  model      {best['model']}")
    effective_window_seconds = window_seconds or WINDOW_SECONDS
    effective_overlap = OVERLAP if overlap is None else overlap
    smoothing_span_seconds = ((best["k"] - 1) * effective_window_seconds
                              * (1.0 - effective_overlap))
    total_context_seconds = effective_window_seconds + smoothing_span_seconds
    print(f"  smoothing  {best['k']} windows "
          f"(~{total_context_seconds:.1f} s of context)")
    print(f"  threshold  {best['threshold']:.2f} on p(AWAY)")
    m = best["metrics"]
    print(f"\n  accuracy {m['accuracy']:.3f}   macro F1 {m['f1_macro']:.3f}")
    print(f"  AWAY  precision {m['precision_away']:.3f}  recall {m['recall_away']:.3f}"
          f"   <-- how much AC-off saving you actually get")
    print(f"  HOME  precision {m['precision_home']:.3f}  recall {m['recall_home']:.3f}"
          f"   <-- how often the AC stays on for someone who is home")
    if not deployable:
        print(f"\n  *** NOT DEPLOYABLE YET ***")
        if not best["met_floor"]:
            print(f"  No configuration reached {HOME_RECALL_FLOOR:.0%} HOME recall: the model")
            print(f"  would switch the AC off on somebody who is home.")
        else:
            print(f"  The best configuration that keeps HOME recall above "
                  f"{HOME_RECALL_FLOOR:.0%} has")
            print(f"  AWAY recall {m['recall_away']:.2f}, below the "
                  f"{MIN_USEFUL_AWAY_RECALL:.0%} needed to be worth")
            print(f"  shipping -- it almost never turns the AC off, so it saves nothing.")
        print(f"  Cause is usually one of: too few recordings, only one condition per")
        print(f"  block, or captures that the quality gate flagged. Collect more and re-run.")

    pred = predict_at(best["proba"], best["threshold"])
    print_confusion(y, pred, "  Out-of-fold confusion matrix")

    print("\n  Accuracy per recorded condition (this is the diagnostic that matters):")
    for c, r in sorted(per_condition_recall(cond, y, pred).items()):
        flag = "   <-- the hard case" if c == "occupied_still" and r["correct"] < 0.9 else ""
        print(f"    {c:<18} {r['correct']:6.1%} of {r['n']:5d} windows{flag}")

    print("\n  Effect of smoothing (same features/model/baseline, threshold re-picked):")
    for k in SMOOTHING_CHOICES:
        r = next(x for x in results if x["feature_set"] == best["feature_set"]
                 and x["baseline_mode"] == best["baseline_mode"]
                 and x["model"] == best["model"] and x["k"] == k)
        print(f"    k={k:<2} AWAY recall {r['metrics']['recall_away']:.3f}   "
              f"HOME recall {r['metrics']['recall_home']:.3f}   "
              f"macro F1 {r['metrics']['f1_macro']:.3f}")

    # ------------------------------------------------------------------
    rule("4  DRIFT CHECK  (train on the earliest recordings, test on the latest)")
    print("  Grouped CV shuffles recordings, so it cannot see whether the channel")
    print("  changed over the session. This split can: it is the same question as")
    print("  'will the model still work an hour after you calibrated it'.\n")
    order = ds["order"]
    cut = np.median(np.unique(order))
    tr, te = order <= cut, order > cut
    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        print("  Skipped: the chronological halves are not both two-class. Interleave")
        print("  conditions while recording (empty, occupied, empty, ...) rather than")
        print("  doing all of one condition and then all of the other.")
    else:
        baseline = fit_baseline(ds["X"][tr], cond[tr], best["baseline_mode"])
        Xtr, _ = apply_calibration(ds["X"][tr], baseline, best["feature_set"])
        Xte, _ = apply_calibration(ds["X"][te], baseline, best["feature_set"])
        mdl = make_models()[best["model"]].fit(Xtr, y[tr])
        p = smooth_proba(mdl.predict_proba(Xte)[:, HOME], group[te], best["k"])
        dm = metrics(y[te], predict_at(p, best["threshold"]))
        print(f"  accuracy {dm['accuracy']:.3f}   AWAY recall {dm['recall_away']:.3f}   "
              f"HOME recall {dm['recall_home']:.3f}")
        drop = m["recall_away"] - dm["recall_away"]
        if drop > 0.15:
            print(f"  AWAY recall drops {drop:.2f} versus shuffled CV -- the channel is")
            print(f"  drifting. Re-record empty blocks more often, or re-calibrate the")
            print(f"  baseline periodically at deploy time.")

    # ------------------------------------------------------------------
    rule("5  FINAL MODEL")
    baseline = fit_baseline(ds["X"], cond, best["baseline_mode"])
    Xc, used_names = apply_calibration(ds["X"], baseline, best["feature_set"])
    ytr = y
    if augment is not None and best["feature_set"] == "scale_free":
        Xc = np.vstack([Xc, augment[0]])
        ytr = np.concatenate([y, augment[1]])
    model = make_models()[best["model"]].fit(Xc, ytr)
    print(f"  Fitted {best['model']} on all {len(Xc)} windows "
          f"({len(used_names)} features, baseline from the full set).")

    if hasattr(model, "feature_importances_"):
        print("\n  Feature importance:")
        for i in np.argsort(-model.feature_importances_)[:10]:
            v = model.feature_importances_[i]
            print(f"    {used_names[i]:<18} {v:.4f}  {'#' * int(v * 120)}")

    fs_median = float(np.median(ds["fs"]))
    bundle = {
        "model": model,
        "baseline": baseline,                       # full 16-length vector
        "baseline_mode": best["baseline_mode"],
        "feature_set": best["feature_set"],
        "feature_names": used_names,
        "all_feature_names": list(FEATURE_NAMES),
        "threshold": float(best["threshold"]),
        "smoothing_windows": int(best["k"]),
        "smoothing_span_seconds": float(smoothing_span_seconds),
        "window_seconds": effective_window_seconds,
        "fs": fs_median,
        "window_packets": int(round(fs_median * effective_window_seconds)),
        "subcarriers": REFERENCE_SUBCARRIER_MAPS[128][1],
        "cv": {k: float(v) for k, v in best["metrics"].items()},
        "cv_met_home_floor": bool(best["met_floor"]),
        "deployable": bool(deployable),
        "n_recordings": int(n_groups),
        "n_windows": int(len(y)),
        "augmented_with_intel": bool(augment is not None
                                     and best["feature_set"] == "scale_free"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_dir": os.path.realpath(data_dir) if data_dir else None,
    }
    if save and not deployable and not save_anyway:
        print(f"\n  NOT SAVED: this model is not deployable (see section 3).")
        print(f"  Re-run with more data. Pass --save-anyway to override.")
    elif save:
        model_path = os.path.realpath(model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if os.path.exists(model_path):
            stem, ext = os.path.splitext(model_path)
            backup = f"{stem}.previous{ext}"
            shutil.copy2(model_path, backup)
            print(f"\n  previous model backed up -> {os.path.relpath(backup, _ROOT)}")
        temporary = f"{model_path}.tmp"
        joblib.dump(bundle, temporary)
        os.replace(temporary, model_path)
        print(f"\n  saved atomically -> {os.path.relpath(model_path, _ROOT)}")
        print(f"  The baseline ships WITH the model. A model without its site")
        print(f"  baseline is unusable -- predict_presence() raises rather than guess.")
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-save", action="store_true", help="report only, do not write the model")
    ap.add_argument("--augment-intel", action="store_true",
                    help="add calibrated Intel windows to training folds (scale_free only)")
    ap.add_argument("--window-seconds", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=None,
                    help="0.0-0.95; safe because splits are by recording")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--save-anyway", action="store_true",
                    help="write the model even if the evaluation says it is not deployable")
    ap.add_argument("--include-bad", action="store_true",
                    help="train on quality-flagged recordings too (contaminates results)")
    ap.add_argument("--force", action="store_true", help="rebuild the feature cache")
    ap.add_argument("--output", default=MODEL_PATH,
                    help="model bundle path (existing file is backed up before replacement)")
    a = ap.parse_args()
    run(save=not a.no_save, augment_intel=a.augment_intel,
        window_seconds=a.window_seconds, overlap=a.overlap,
        force=a.force, data_dir=a.data_dir, include_bad=a.include_bad,
        save_anyway=a.save_anyway, model_path=a.output)


if __name__ == "__main__":
    main()
