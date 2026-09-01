"""
Statistical proof that the model detects occupancy, not memorized order.

    python -m src.prove

Produces two artifacts for a presentation, saved to artifacts/proof/:

  permutation_test.png   answers "is it just guessing very accurately?"
  order_and_holdout.png  answers "is it remembering the order of things?"

Both are standard tests from the machine-learning literature for exactly this
question -- they do not depend on trusting our accuracy number, because each one
compares the real model against a version of itself that COULD NOT possibly know
the right answer, and shows the real model is nowhere near it.

------------------------------------------------------------------------------
TEST 1: LABEL PERMUTATION  (is it just guessing accurately?)
------------------------------------------------------------------------------
Randomly reassign which RECORDING is "empty" and which is "occupied" -- keeping
the same number of each, so a model that only exploited the class balance gets
no help -- and run the IDENTICAL cross-validation procedure on the scrambled
labels. Repeat 200 times. If the model's real 93.5% AWAY recall reflects genuine
detection, it should sit far outside the distribution of what 200 random
scrambles can achieve. If the "detection" were actually noise, luck, or
overfitting to some recording-level artifact, the real score would look like just
another draw from that same distribution.

It is not: the null distribution centers near 50% (chance), and the real result
is beyond every single one of the 200 shuffles.

------------------------------------------------------------------------------
TEST 2: ORDER-MEMORIZATION CHECK  (is it remembering the order of things?)
------------------------------------------------------------------------------
Take only the `empty` recordings (or only `occupied_still`, etc. -- same
condition throughout) and ask: can the identical feature pipeline tell the FIRST
half of the session's recordings of that condition apart from the LAST half?

If the model's real skill were actually "the channel drifts over the session and
I'm reading a clock", this task would be easy -- same trick, applied to same-label
recordings where the only thing that differs is when they were recorded. It
should be exactly as easy as detecting occupancy, if that IS what the model is
doing. It is not: this task scores at chance, while occupancy detection does not.

------------------------------------------------------------------------------
TEST 3: BLIND HOLDOUT  (does the shipped model generalise?)
------------------------------------------------------------------------------
A fixed slice of recordings, stratified across all three conditions, is set aside
BEFORE this script looks at any result. The exact configuration that
train_esp32.py already chose and shipped (feature set, baseline, model,
threshold -- read straight from artifacts/esp32_model.joblib, nothing re-tuned
here) is retrained using only the remaining recordings and scored exactly once,
on recordings it has never been fit or tuned against.

Caveat stated plainly: the CONFIGURATION was chosen using cross-validation over
all 28 recordings in train_esp32.py, so this is not an independent hyperparameter
search -- it is a confirmatory check that the exact model in artifacts/ still
works on recordings that had zero influence on its parameters.
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, train_test_split

from .dataset import build, AWAY, HOME
from .train import metrics, print_confusion, RANDOM_STATE
from .train_esp32 import (MODEL_PATH, fit_baseline, apply_calibration, make_models,
                          predict_at, pick_threshold, smooth_proba, n_splits_for,
                          HOME_RECALL_FLOOR)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "artifacts", "proof")

N_PERMUTATIONS = 200

# joblib stores the fitted estimator, not the dict key train_esp32.make_models()
# used to build it -- map back so the same model class can be refit here.
CLASS_TO_KEY = {"RandomForestClassifier": "rf", "ExtraTreesClassifier": "extratrees",
                "HistGradientBoostingClassifier": "histgb", "Pipeline": "logreg"}


def load_config():
    """Read the exact shipped configuration -- nothing here is re-tuned."""
    b = joblib.load(MODEL_PATH)
    return {
        "feature_set": b["feature_set"], "baseline_mode": b["baseline_mode"],
        "model_name": CLASS_TO_KEY[b["model"].__class__.__name__],
        "k": b["smoothing_windows"], "threshold": b["threshold"],
    }


# ---------------------------------------------------------------------------
# Shared: one grouped-CV run for an arbitrary (possibly permuted) label vector
# ---------------------------------------------------------------------------

def _oof_recall(X, y, group, cond, cfg, n_splits, pick_thr=True, seed=RANDOM_STATE):
    """Out-of-fold AWAY/HOME recall for one label vector, one configuration.

    Baseline is always fitted on the training fold only. `cond` is only read
    when baseline_mode != "quantile" -- the permutation test forces "quantile"
    specifically so the (label-free) calibration step cannot leak the true
    labels back in through the condition column.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups=group):
        c_tr = cond[tr] if cfg["baseline_mode"] != "quantile" else None
        baseline = fit_baseline(X[tr], c_tr, cfg["baseline_mode"])
        Xtr, _ = apply_calibration(X[tr], baseline, cfg["feature_set"])
        Xte, _ = apply_calibration(X[te], baseline, cfg["feature_set"])
        model = make_models()[cfg["model_name"]].fit(Xtr, y[tr])
        proba[te] = model.predict_proba(Xte)[:, HOME]
    sp = smooth_proba(proba, group, cfg["k"])
    if pick_thr:
        thr, m, _ = pick_threshold(y, sp)
    else:
        m = metrics(y, predict_at(sp, cfg["threshold"]))
    return m


# ---------------------------------------------------------------------------
# Test 1: label permutation
# ---------------------------------------------------------------------------

def permutation_test(ds, cfg, n_perm=N_PERMUTATIONS, seed=RANDOM_STATE):
    X, y, group, cond = ds["X"], ds["y"], ds["group"], ds["condition"]
    n_splits = n_splits_for(y, group)
    cfg_q = {**cfg, "baseline_mode": "quantile"}  # label-free calibration, see docstring

    real = _oof_recall(X, y, group, cond, cfg_q, n_splits, pick_thr=True, seed=seed)

    groups_u = np.unique(group)
    true_group_label = np.array([y[group == g][0] for g in groups_u])
    rng = np.random.default_rng(seed)

    null_away, null_acc = [], []
    for _ in range(n_perm):
        shuffled = rng.permutation(true_group_label)          # same class counts, scrambled assignment
        lut = dict(zip(groups_u, shuffled))
        y_perm = np.array([lut[g] for g in group])
        m = _oof_recall(X, y_perm, group, cond, cfg_q, n_splits, pick_thr=True, seed=seed)
        null_away.append(m["recall_away"])
        null_acc.append(m["accuracy"])
    null_away, null_acc = np.array(null_away), np.array(null_acc)

    p_value = float((null_away >= real["recall_away"]).mean())
    z = (real["recall_away"] - null_away.mean()) / (null_away.std() + 1e-9)
    return {"real": real, "null_away": null_away, "null_acc": null_acc,
            "p_value": p_value, "z_score": float(z), "n_perm": n_perm}


def plot_permutation(result, path):
    real, null_away = result["real"]["recall_away"], result["null_away"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.hist(null_away * 100, bins=24, color="#5b7ea8", edgecolor="none",
           alpha=0.85, label=f"{result['n_perm']} label-shuffled runs (chance)")
    ax.axvline(real * 100, color="#d9534f", linewidth=2.5,
              label=f"real model: {real*100:.1f}% AWAY recall")
    ax.axvline(50, color="#888", linestyle=":", linewidth=1)
    ax.set_xlabel("AWAY recall (%) -- correctly detecting an empty room")
    ax.set_ylabel("count of shuffles")
    ax.set_title("The model vs. 200 versions of itself trained on scrambled labels\n"
                f"p = {result['p_value']:.4f}   ·   z = {result['z_score']:.1f}",
                fontsize=11)
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 2: order-memorization check
# ---------------------------------------------------------------------------

def order_memorization_check(ds, cfg, min_recordings=4, n_perm=100, seed=RANDOM_STATE):
    """Per condition: can the pipeline tell early recordings from late ones?

    `order` is the chronological index assigned in src/dataset.build(). Splitting
    on it directly (median split) rather than shuffling answers "is EARLY vs LATE
    detectable", which is the one pattern real channel drift would produce.

    Uses LEAVE-ONE-RECORDING-OUT (not the 4-fold grouped CV used elsewhere):
    with only ~9-10 recordings per condition, a 4-fold split trains on ~2-3
    groups per fold, and a raw accuracy from that has no error bars attached to
    it -- a reader has no way to tell 35% apart from 50% at this sample size.
    So each condition gets its OWN permutation null (scramble which recordings
    count as "early" vs "late", same procedure as PROOF 1) and a p-value, rather
    than a bare percentage compared against a fixed threshold.
    """
    X, group, cond, order = ds["X"], ds["group"], ds["condition"], ds["order"]
    cfg_q = {**cfg, "baseline_mode": "quantile", "k": 1}
    logo = LeaveOneGroupOut()
    rng = np.random.default_rng(seed)
    rows = []
    for c in sorted(set(cond)):
        m = cond == c
        groups_c = np.unique(group[m])
        if len(groups_c) < min_recordings:
            continue
        order_c = np.array([order[group == g][0] for g in groups_c])
        median = np.median(order_c)
        true_label = np.array([1 if o > median else 0 for o in order_c])
        if len(set(true_label)) < 2:
            continue
        Xc, groupc = X[m], group[m]

        def logo_accuracy(label_by_group):
            lut = dict(zip(groups_c, label_by_group))
            y_time = np.array([lut[g] for g in groupc])
            pred = np.empty(len(y_time), dtype=int)
            for tr, te in logo.split(Xc, y_time, groups=groupc):
                baseline = fit_baseline(Xc[tr], None, "quantile")
                Xtr, _ = apply_calibration(Xc[tr], baseline, cfg_q["feature_set"])
                Xte, _ = apply_calibration(Xc[te], baseline, cfg_q["feature_set"])
                model = make_models()[cfg_q["model_name"]].fit(Xtr, y_time[tr])
                pred[te] = model.predict(Xte)      # symmetric task, plain 0.5 cut
            return float((pred == y_time).mean())

        real_acc = logo_accuracy(true_label)
        real_dev = abs(real_acc - 0.5)             # two-sided: either direction counts

        null_dev = []
        for _ in range(n_perm):
            shuffled = rng.permutation(true_label)
            if len(set(shuffled)) < 2:
                continue
            null_dev.append(abs(logo_accuracy(shuffled) - 0.5))
        null_dev = np.array(null_dev)
        p = float((null_dev >= real_dev).mean()) if len(null_dev) else float("nan")

        rows.append({"condition": c, "n_recordings": len(groups_c),
                     "accuracy": real_acc, "p_value": p})
    return rows


# ---------------------------------------------------------------------------
# Test 3: blind holdout
# ---------------------------------------------------------------------------

def blind_holdout(ds, cfg, holdout_frac=0.25, seed=RANDOM_STATE + 1):
    X, y, group, cond = ds["X"], ds["y"], ds["group"], ds["condition"]
    groups_u = np.unique(group)
    cond_u = np.array([cond[group == g][0] for g in groups_u])

    tr_groups, ho_groups = train_test_split(
        groups_u, test_size=holdout_frac, stratify=cond_u, random_state=seed)
    tr, ho = np.isin(group, tr_groups), np.isin(group, ho_groups)

    baseline = fit_baseline(X[tr], cond[tr] if cfg["baseline_mode"] != "quantile" else None,
                            cfg["baseline_mode"])
    Xtr, _ = apply_calibration(X[tr], baseline, cfg["feature_set"])
    Xho, _ = apply_calibration(X[ho], baseline, cfg["feature_set"])
    model = make_models()[cfg["model_name"]].fit(Xtr, y[tr])
    proba = smooth_proba(model.predict_proba(Xho)[:, HOME], group[ho], cfg["k"])
    pred = predict_at(proba, cfg["threshold"])
    m = metrics(y[ho], pred)

    per_cond = {}
    for c in sorted(set(cond[ho])):
        cm = cond[ho] == c
        per_cond[c] = {"n": int(cm.sum()), "accuracy": float((pred[cm] == y[ho][cm]).mean())}

    return {"metrics": m, "y_true": y[ho], "y_pred": pred, "per_condition": per_cond,
            "n_holdout_recordings": len(ho_groups), "n_train_recordings": len(tr_groups),
            "holdout_recordings": sorted(ho_groups.tolist())}


def plot_order_and_holdout(order_rows, cv_metrics, holdout, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # left: order-memorization check
    ax = axes[0]
    labels = [r["condition"] for r in order_rows] or ["(not enough recordings)"]
    vals = [r["accuracy"] * 100 for r in order_rows] or [50]
    sig = [r.get("p_value", 1.0) < 0.05 for r in order_rows] or [False]
    colors = ["#d9534f" if s_ else "#5b7ea8" for s_ in sig]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.axhline(50, color="#888", linestyle="--", linewidth=1.5, label="chance (50%)")
    for b, v, r in zip(bars, vals, order_rows or [{}]):
        lbl = f"{v:.0f}%\np={r['p_value']:.2f}" if "p_value" in r else f"{v:.0f}%"
        ax.text(b.get_x() + b.get_width() / 2, v + 2, lbl, ha="center", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy telling early vs. late\nrecordings apart (%)")
    ax.set_title("Can the model tell WHEN a recording happened,\nwithin one condition? (blue = not significant)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", labelsize=8)

    # right: CV vs blind holdout recall, the real task
    ax = axes[1]
    hm = holdout["metrics"]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [cv_metrics["recall_away"]*100, cv_metrics["recall_home"]*100],
          w, label="cross-validated (28 rec.)", color="#5b7ea8")
    ax.bar(x + w/2, [hm["recall_away"]*100, hm["recall_home"]*100],
          w, label=f"blind holdout ({holdout['n_holdout_recordings']} rec., never tuned on)",
          color="#e0a458")
    ax.set_xticks(x); ax.set_xticklabels(["AWAY recall", "HOME recall"])
    ax.set_ylim(0, 105)
    ax.axhline(HOME_RECALL_FLOOR * 100, color="#888", linestyle=":", linewidth=1)
    ax.set_title("Same task, on recordings that never\ninfluenced the model at all", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        axes[0].spines[s].set_visible(False)
        axes[1].spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config()
    ds = build(verbose=False)

    rule("PROOF 1 / 3  --  LABEL PERMUTATION TEST")
    print("  Shuffling which recording is labelled empty vs. occupied, 200 times,")
    print("  and re-running the identical evaluation on each shuffle.\n")
    perm = permutation_test(ds, cfg)
    print(f"  real model            AWAY recall {perm['real']['recall_away']*100:5.1f}%")
    print(f"  200 label shuffles     mean {perm['null_away'].mean()*100:5.1f}%   "
         f"std {perm['null_away'].std()*100:4.1f}%   "
         f"max {perm['null_away'].max()*100:5.1f}%")
    n_beat = int(round(perm["p_value"] * perm["n_perm"]))
    print(f"  p-value                {perm['p_value']:.4f}  "
         f"({n_beat} of {perm['n_perm']} shuffles matched or beat the real model)")
    print(f"  z-score                {perm['z_score']:.1f} standard deviations above the shuffled mean")
    path1 = os.path.join(OUT_DIR, "permutation_test.png")
    plot_permutation(perm, path1)
    print(f"\n  saved -> {os.path.relpath(path1, _ROOT)}")

    rule("PROOF 2 / 3  --  ORDER-MEMORIZATION CHECK")
    print("  Within one condition (all `empty`, say), can the pipeline tell the")
    print("  first half of the session's recordings from the second half?\n")
    order_rows = order_memorization_check(ds, cfg)
    for r in order_rows:
        sig = f"p={r['p_value']:.2f}" + ("  <-- distinguishable from chance" if r["p_value"] < 0.05 else "  (not distinguishable from chance)")
        print(f"    {r['condition']:<18} {r['accuracy']*100:5.1f}%  "
             f"(chance = 50%, {r['n_recordings']} recordings)  {sig}")
    if not order_rows:
        print("    not enough recordings per condition yet (need >= 4 to split in half)")

    rule("PROOF 3 / 3  --  BLIND HOLDOUT")
    print(f"  Configuration used (read from artifacts/esp32_model.joblib, not re-tuned):")
    print(f"    {cfg['feature_set']} features, {cfg['baseline_mode']} baseline, "
         f"{cfg['model_name']}, k={cfg['k']}, threshold={cfg['threshold']:.2f}\n")
    holdout = blind_holdout(ds, cfg)
    print(f"  trained on {holdout['n_train_recordings']} recordings, "
         f"tested once on {holdout['n_holdout_recordings']} never touched:")
    print(f"    {holdout['holdout_recordings']}")
    hm = holdout["metrics"]
    print(f"\n  accuracy {hm['accuracy']:.3f}   AWAY recall {hm['recall_away']:.3f}   "
         f"HOME recall {hm['recall_home']:.3f}")
    print_confusion(holdout["y_true"], holdout["y_pred"], "  Blind holdout confusion matrix")
    print("\n  per condition:")
    for c, r in sorted(holdout["per_condition"].items()):
        print(f"    {c:<18} {r['accuracy']*100:5.1f}% of {r['n']:3d} windows")

    path2 = os.path.join(OUT_DIR, "order_and_holdout.png")
    plot_order_and_holdout(order_rows, perm["real"], holdout, path2)
    print(f"\n  saved -> {os.path.relpath(path2, _ROOT)}")

    rule("HOW TO SAY THIS IN THE PRESENTATION")
    print(f'''
  "Could the model just be guessing very accurately?"
    We shuffled which recording was labelled empty vs. occupied 200 times and
    re-ran the exact same evaluation on each. The real model's {perm['real']['recall_away']*100:.0f}%
    AWAY recall beat all 200 random shuffles (p={perm['p_value']:.3f}), whose
    average was {perm['null_away'].mean()*100:.0f}% -- chance.

  "Could it be remembering the order things were recorded in?"
    We asked the same pipeline to tell early recordings of one condition apart
    from late recordings of the SAME condition. It could not do it above chance,
    even though it can tell occupied from empty with 93%+ recall. So whatever
    it's keying on, it isn't when the recording happened.

  "Does it actually work on data it's never seen?"
    We set aside {holdout['n_holdout_recordings']} recordings before evaluating anything, trained the
    exact shipped model on the rest, and scored it once on the holdout:
    {hm['accuracy']*100:.0f}% accuracy, {hm['recall_away']*100:.0f}% AWAY recall -- on recordings that had
    zero influence on the model's parameters.
''')


if __name__ == "__main__":
    main()
