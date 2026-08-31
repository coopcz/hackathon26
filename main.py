"""
WiFi CSI presence detection -- end-to-end MVP.

    python main.py

Runs dataset exploration, both evaluations (optimistic and realistic), feature
importance, sample HVAC decisions, and the ESP32 ingestion path.
"""

import os
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.build_dataset import build
from src.features import (FEATURE_NAMES, SCALE_FREE_FEATURES, fit_site_baseline,
                          apply_site_baseline)
from src.train import (make_rf, make_gb, metrics, print_confusion, evaluate_random_split,
                       evaluate_leave_one_room_out, build_calibrated,
                       evaluate_calibrated_loro, train_production_model,
                       family_importance, RANDOM_STATE)
from src.pipeline import (predict_presence, should_run_ac, load_esp32_csi_csv,
                          make_synthetic_esp32_csv, verify_esp32_assumptions,
                          LABEL_NAMES)

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_metrics(m, indent="  "):
    print(f"{indent}accuracy            {m['accuracy']:.4f}")
    print(f"{indent}AWAY  precision     {m['precision_away']:.4f}   recall {m['recall_away']:.4f}"
          f"   F1 {m['f1_away']:.4f}")
    print(f"{indent}HOME  precision     {m['precision_home']:.4f}   recall {m['recall_home']:.4f}"
          f"   F1 {m['f1_home']:.4f}")
    print(f"{indent}macro F1            {m['f1_macro']:.4f}")


def main():
    rule("PHASE 2.1  DATASET")
    X, y, room, n_people, session = build()
    print(f"  feature matrix       {X.shape}  ({len(FEATURE_NAMES)} features per window)")
    print(f"  windows              {len(X)} non-overlapping windows of 128 packets (~2.56 s)")
    print(f"  source               26 continuous captures, 3 rooms, occupancy 0-8 people")
    print(f"  missing values       {int(np.isnan(X).sum())} NaN, {int(np.isinf(X).sum())} inf")
    print(f"\n  label distribution (binary presence):")
    print(f"    AWAY (0 people)    {int((y == 0).sum()):5d} windows  ({(y == 0).mean() * 100:.1f}%)")
    print(f"    HOME (1-8 people)  {int((y == 1).sum()):5d} windows  ({(y == 1).mean() * 100:.1f}%)")
    print(f"    -> class imbalance 1:{(y == 1).sum() / (y == 0).sum():.1f}. A model that always")
    print(f"       says HOME scores {(y == 1).mean() * 100:.1f}% accuracy, so accuracy alone is")
    print(f"       nearly meaningless here. AWAY recall is the metric that matters.")
    print(f"\n  per room:")
    for r in np.unique(room):
        m = room == r
        print(f"    {r}: {m.sum():4d} windows  away={int((y[m] == 0).sum()):3d}  home={int((y[m] == 1).sum()):3d}")

    print(f"\n  class separation per feature (Cohen's d, HOME vs AWAY):")
    for i, n in enumerate(FEATURE_NAMES):
        a, h = X[y == 0, i], X[y == 1, i]
        pooled = np.sqrt((a.var() + h.var()) / 2)
        d = (h.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        bar = "#" * min(int(abs(d) * 4), 24)
        print(f"    {n:<17} away={a.mean():9.4f}  home={h.mean():9.4f}  d={d:+6.2f} {bar}")

    # ------------------------------------------------------------------
    rule("PHASE 2.2  EVALUATION 1 of 2 -- RANDOM 80/20 SPLIT (OPTIMISTIC)")
    print("  Windows are split at random. Because each recording is one continuous\n"
          "  3-minute session, near-identical windows land in both train and test.\n"
          "  This number is inflated. It is reported for comparability, not belief.\n")
    rf, (Xtr, Xte, ytr, yte) = evaluate_random_split(X, y)
    pred = rf.predict(Xte)
    tr_acc, te_acc = rf.score(Xtr, ytr), rf.score(Xte, yte)
    print(f"  RandomForest  train accuracy {tr_acc:.4f}   test accuracy {te_acc:.4f}"
          f"   gap {tr_acc - te_acc:+.4f}")
    show_metrics(metrics(yte, pred))
    print_confusion(yte, pred, "  Confusion matrix (random split)")

    gb = make_gb().fit(Xtr, ytr)
    gm = metrics(yte, gb.predict(Xte))
    print(f"\n  HistGradientBoosting comparison: accuracy {gm['accuracy']:.4f}, "
          f"macro F1 {gm['f1_macro']:.4f}")
    print(f"  -> no meaningful gain over RandomForest; the task is not model-limited,\n"
          f"     it is evaluation-limited. Sticking with RF for interpretability.")

    # ------------------------------------------------------------------
    rule("PHASE 2.3  FEATURE IMPORTANCE -- WHAT ACTUALLY DETECTS A HUMAN")
    imp = rf.feature_importances_
    order = np.argsort(-imp)
    print("  Gini importance (RandomForest, all 16 features):")
    for i in order:
        print(f"    {FEATURE_NAMES[i]:<17} {imp[i]:.4f}  {'#' * int(imp[i] * 120)}")

    perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    print("\n  Permutation importance on held-out data (more trustworthy;")
    print("  Gini importance is biased toward high-cardinality features):")
    for i in np.argsort(-perm.importances_mean)[:8]:
        print(f"    {FEATURE_NAMES[i]:<17} {perm.importances_mean[i]:+.4f} "
              f"+/- {perm.importances_std[i]:.4f}")

    print("\n  Rolled up by physical family:")
    for fam, v in family_importance(FEATURE_NAMES, imp).items():
        print(f"    {fam:<28} {v:.4f}  {'#' * int(v * 120)}")

    # ------------------------------------------------------------------
    rule("PHASE 2.4  EVALUATION 2 of 2 -- LEAVE-ONE-ROOM-OUT (REALISTIC)")
    print("  Train on two rooms, test on a room the model has never seen. This is the\n"
          "  honest proxy for 'deploy it in a different house'.\n")
    res = evaluate_leave_one_room_out(X, y, room)
    for held, r in res.items():
        print(f"  held-out {held}:")
        show_metrics(r["metrics"], indent="    ")
        print_confusion(r["y_true"], r["y_pred"], f"    Confusion matrix ({held} held out)")
    mean_acc = np.mean([r["metrics"]["accuracy"] for r in res.values()])
    mean_rec = np.mean([r["metrics"]["recall_away"] for r in res.values()])
    print(f"\n  MEAN cross-room accuracy {mean_acc:.4f}, MEAN AWAY recall {mean_rec:.4f}")
    print(f"  *** This is the headline failure. Accuracy still looks respectable only")
    print(f"      because 89% of windows are HOME. AWAY recall of {mean_rec:.2f} means the model")
    print(f"      misses most empty rooms in an environment it was not trained on --")
    print(f"      i.e. it would leave the AC running in an empty house, which is exactly")
    print(f"      the failure the product exists to prevent.")

    # ------------------------------------------------------------------
    rule("PHASE 2.5  DIAGNOSIS AND FIX -- PER-SITE BASELINE CALIBRATION")
    i_cv = FEATURE_NAMES.index("cv_mean")
    print("  Why it fails: every room has its own noise floor.")
    print("    room   median cv_mean when EMPTY   median cv_mean when OCCUPIED")
    for r in np.unique(room):
        a = np.median(X[(room == r) & (y == 0), i_cv])
        h = np.median(X[(room == r) & (y == 1), i_cv])
        print(f"    {r}       {a:.4f}                       {h:.4f}")
    print("  RoomA's EMPTY channel is about twice as agitated as RoomC's empty channel,")
    print("  and overlaps what RoomB/RoomC call 'occupied'. An absolute threshold cannot")
    print("  serve both rooms.\n")
    print("  Fix: keep only scale-free features and express each as a multiple of that")
    print("  site's own quiet baseline (5th percentile of its unlabelled windows).")
    print(f"  Features retained: {', '.join(SCALE_FREE_FEATURES)}")
    print("  This requires NO labels at the new site -- only that it is empty some of")
    print("  the time during a calibration recording.\n")

    Xc = build_calibrated(X, y, room, FEATURE_NAMES)
    cres = evaluate_calibrated_loro(Xc, y, room)
    for held, r in cres.items():
        print(f"  held-out {held}:")
        show_metrics(r["metrics"], indent="    ")
    cacc = np.mean([r["metrics"]["accuracy"] for r in cres.values()])
    crec = np.mean([r["metrics"]["recall_away"] for r in cres.values()])
    cf1 = np.mean([r["metrics"]["f1_macro"] for r in cres.values()])
    all_true = np.concatenate([r["y_true"] for r in cres.values()])
    all_pred = np.concatenate([r["y_pred"] for r in cres.values()])
    print_confusion(all_true, all_pred, "  Pooled confusion matrix (all 3 cross-room folds)")
    print(f"\n  MEAN cross-room accuracy {cacc:.4f} (was {mean_acc:.4f})")
    print(f"  MEAN cross-room AWAY recall {crec:.4f} (was {mean_rec:.4f})   <-- the real result")
    print(f"  MEAN cross-room macro F1 {cf1:.4f}")

    # ------------------------------------------------------------------
    rule("PHASE 3  INFERENCE PIPELINE + HVAC DECISIONS")
    model = train_production_model(Xc, y)
    print(f"  Production model: RandomForest on {len(SCALE_FREE_FEATURES)} calibrated features,")
    print(f"  fitted on all {len(Xc)} windows from 3 rooms.\n")

    held = "RoomA"  # the hardest room
    te = room == held
    baseline = fit_site_baseline(X[te], FEATURE_NAMES)
    demo_model = make_rf().fit(Xc[~te], y[~te])  # never saw RoomA
    print(f"  Demo: model trained WITHOUT {held}, predicting on {held} windows.")
    print(f"  Site baseline fitted from {held}'s own unlabelled data.\n")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.where(te)[0]
    sample = np.concatenate([
        rng.choice(idx[y[idx] == 0], 4, replace=False),
        rng.choice(idx[y[idx] == 1], 5, replace=False)])
    rng.shuffle(sample)

    print(f"  {'session':<14} {'truth':<6} {'predicted':<10} {'conf':<7} {'AC':<5} reason")
    print("  " + "-" * 88)
    correct = 0
    for s in sample:
        pred_i, conf = predict_presence(demo_model, X[s], baseline=baseline)
        run, why = should_run_ac(pred_i, conf, threshold=0.7)
        ok = pred_i == y[s]
        correct += ok
        print(f"  {session[s]:<14} {LABEL_NAMES[y[s]]:<6} {LABEL_NAMES[pred_i]:<10} "
              f"{conf:<7.3f} {'ON' if run else 'OFF':<5} {why} {'' if ok else '  <-- WRONG'}")
    print(f"\n  {correct}/{len(sample)} correct on this sample.")

    # ------------------------------------------------------------------
    rule("PHASE 3.2  ESP32 INGESTION PATH")
    print("  No real ESP32 capture yet, so a fixture in the EXACT esp-csi csi_recv format")
    print("  (15-column ESP32-C6 schema, HT40, 128 subcarriers) is generated to prove the")
    print("  code path runs. Format-only: it says nothing about accuracy on real hardware.")
    print("  For the full integration dry run see esp32_dry_run.py.\n")
    occ = make_synthetic_esp32_csv(os.path.join(ART, "esp32_sample_occupied.csv"),
                                   occupied=True, seed=1)
    emp = make_synthetic_esp32_csv(os.path.join(ART, "esp32_sample_empty.csv"),
                                   occupied=False, seed=2)
    verify_esp32_assumptions(occ)
    print()
    for path, tag in [(occ, "occupied-like"), (emp, "empty-like")]:
        out = load_esp32_csi_csv(path, verbose=False)
        print(f"  {os.path.basename(path)} ({tag}): X={out['X'].shape}, fs={out['fs']:.0f} Hz, "
              f"{out['n_subcarriers']} subcarriers")
    print("\n  Both files land in the SAME 16-column feature space as the Intel data:")
    print(f"    {', '.join(FEATURE_NAMES)}")
    print("\n  To run real ESP32 data through this exact model:")
    print("    from src.pipeline import load_esp32_csi_csv, predict_presence, should_run_ac")
    print("    from src.features import fit_site_baseline")
    print("    out  = load_esp32_csi_csv('my_capture.csv')")
    print("    base = fit_site_baseline(out['X'], FEATURE_NAMES)   # calibrate to this house")
    print("    pred, conf = predict_presence(model, out['X'][0], baseline=base, fs=out['fs'])")
    print("    run_ac, why = should_run_ac(pred, conf)")

    rule("SUMMARY")
    print(f"  Dataset          WiFi-CrowdCounting (Di Domenico et al.), Intel IWL-5300,")
    print(f"                   3 rooms x 9 occupancy levels, 229,837 packets -> {len(X)} windows")
    print(f"  Optimistic       {te_acc:.1%} accuracy (random split -- inflated by session leakage)")
    print(f"  Realistic        {mean_acc:.1%} accuracy but only {mean_rec:.1%} AWAY recall (cross-room)")
    print(f"  After calibration {crec:.1%} AWAY recall, {cacc:.1%} accuracy cross-room")
    print(f"  Top signal       temporal variance + normalised variance; NOT mean amplitude")
    print(f"  ESP32 hand-off   feature pipeline transfers; the fitted model does NOT")
    print(f"                   (1x1 antenna, 114 vs 90 channels) -- retrain, reuse the code")


if __name__ == "__main__":
    main()
