"""
Cross-dataset validation of the Intel-trained presence baseline, using EHUNAM.

    python validate_ehunam.py

Four experiments, in increasing order of how much they could embarrass us:

  1. Does the per-site calibration finding replicate on different hardware?
  2. Does the model survive a change of DAY in the same room?
  3. Do running machines read as a person?  (the HVAC false-positive)
  4. Does a model trained on Intel 5300 data work on Broadcom data at all?
     This is the closest thing we can run to an ESP32 dry run before the
     hardware data exists.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.build_dataset import build as build_intel
from src.build_ehunam import build as build_ehunam
from src.features import (FEATURE_NAMES, SCALE_FREE_FEATURES,
                          fit_site_baseline, apply_site_baseline)
from src.train import make_rf, metrics, print_confusion

AWAY, HOME = 0, 1
FN = list(FEATURE_NAMES)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def show(m, indent="    "):
    print(f"{indent}accuracy {m['accuracy']:.4f}   "
          f"AWAY P/R/F1 {m['precision_away']:.3f}/{m['recall_away']:.3f}/{m['f1_away']:.3f}   "
          f"macroF1 {m['f1_macro']:.4f}")


def calibrate_by(X, key):
    """Per-site calibration: each site's baseline from its OWN unlabelled windows."""
    Xc = np.zeros((len(X), len(SCALE_FREE_FEATURES)))
    for k in np.unique(key):
        m = key == k
        Xc[m] = apply_site_baseline(X[m], FN, fit_site_baseline(X[m], FN))
    return Xc


def main():
    rule("LOADING")
    Xi, yi, room_i, npl_i, sess_i = build_intel()
    print(f"  Intel 5300 (WiFi-CrowdCounting): {Xi.shape[0]} windows, 3 rooms, 30 subcarriers 3x2")
    E = build_ehunam()
    Xe, grp, env, app, date = E["X"], E["group"], E["env"], E["app"], E["date"]
    print(f"  Broadcom BCM43455 (EHUNAM):      {Xe.shape[0]} windows, "
          f"{len(np.unique(env))} environments, 52 subcarriers 1x1, decimated to 50 Hz")
    for e in np.unique(env):
        m = env == e
        print(f"    {e:<24} {m.sum():4d} windows  "
              f"away={int((grp[m]=='away').sum()):4d} home={int((grp[m]=='home').sum()):4d} "
              f"machines-no-people={int((grp[m]=='away_machines').sum()):4d}")

    # presence task = clean empty vs occupied; machine-only windows are held back
    pres = np.isin(grp, ["away", "home"])
    Xp, envp = Xe[pres], env[pres]
    yp = (grp[pres] == "home").astype(int)
    print(f"\n  presence subset: {len(yp)} windows, away={int((yp==0).sum())} home={int((yp==1).sum())}")

    # ------------------------------------------------------------------
    rule("EXP 1  Does per-site calibration replicate on different hardware?")
    print("  Leave-one-environment-out on EHUNAM. Only 2 environments have both")
    print("  empty and occupied recordings, so this is 2 folds -- fewer than the")
    print("  Intel test, but on a completely different radio.\n")
    envs = [e for e in np.unique(envp) if len(np.unique(yp[envp == e])) == 2]
    Xc_all = calibrate_by(Xp, envp)
    for mode, data in [("absolute (all 16 features)", Xp), ("calibrated scale-free", Xc_all)]:
        accs, recs, f1s = [], [], []
        for held in envs:
            tr, te = envp != held, envp == held
            rf = make_rf().fit(data[tr], yp[tr])
            m = metrics(yp[te], rf.predict(data[te]))
            accs.append(m["accuracy"]); recs.append(m["recall_away"]); f1s.append(m["f1_macro"])
            print(f"  {mode:<28} held-out {held:<24}", end="")
            print(f" acc {m['accuracy']:.3f}  AWAY recall {m['recall_away']:.3f}  "
                  f"macroF1 {m['f1_macro']:.3f}")
        print(f"  {'':<28} MEAN{'':<25} acc {np.mean(accs):.3f}  "
              f"AWAY recall {np.mean(recs):.3f}  macroF1 {np.mean(f1s):.3f}\n")

    print("  Why this fold is so much harder than the Intel one: with only two")
    print("  environments, each fold trains on exactly ONE site, and the two sites")
    print("  have very different occupied-state strength.")
    j = SCALE_FREE_FEATURES.index("cv_mean")
    for e in envs:
        m = envp == e
        b = fit_site_baseline(Xp[m], FN)
        raw_a = np.median(Xp[m][yp[m] == 0, FN.index("cv_mean")])
        raw_h = np.median(Xp[m][yp[m] == 1, FN.index("cv_mean")])
        print(f"    {e:<24} cv_mean empty {raw_a:.4f} -> occupied {raw_h:.4f}  "
              f"= {raw_h / raw_a:.1f}x its own quiet floor")
    print("  A threshold learned at ~5x on one site cannot fire at ~2x on the other,")
    print("  whichever direction you train. Calibration removes the SCALE difference")
    print("  between sites; it cannot remove a difference in how strongly a body")
    print("  couples into the link.\n")

    print("  Does adding more training environments help? Pool the Intel rooms in:")
    Xi_c = calibrate_by(Xi, room_i)
    for held in envs:
        tr, te = envp != held, envp == held
        Xtr = np.vstack([Xi_c, Xc_all[tr]])
        ytr = np.concatenate([yi, yp[tr]])
        rf = make_rf().fit(Xtr, ytr)
        m = metrics(yp[te], rf.predict(Xc_all[te]))
        print(f"    train 3 Intel rooms + 1 EHUNAM env -> {held:<24} "
              f"acc {m['accuracy']:.3f}  AWAY recall {m['recall_away']:.3f}  "
              f"macroF1 {m['f1_macro']:.3f}")

    # ------------------------------------------------------------------
    rule("EXP 2  Does it survive a change of DAY in the same room?")
    print("  The Intel dataset is single-session, so this could not be tested at all.")
    print("  Industrial Laboratory records 5 days but never both classes on the same")
    print("  day, so it cannot support this test. Basement Room has empty AND occupied")
    print("  recordings on each of 2 separate days, which can.\n")
    lab = envp == "Basement Room"
    Xl, yl, dl = Xp[lab], yp[lab], date[pres][lab]
    days = np.unique(dl)
    print(f"  {len(days)} days, {len(yl)} windows "
          f"(away={int((yl==0).sum())}, home={int((yl==1).sum())})")
    base_l = fit_site_baseline(Xl, FN)          # one baseline for the room, all days
    Xlc = apply_site_baseline(Xl, FN, base_l)
    for mode, data in [("absolute", Xl), ("calibrated", Xlc)]:
        accs, recs = [], []
        for d in days:
            tr, te = dl != d, dl == d
            if len(np.unique(yl[tr])) < 2 or len(np.unique(yl[te])) < 2:
                continue
            rf = make_rf().fit(data[tr], yl[tr])
            m = metrics(yl[te], rf.predict(data[te]))
            accs.append(m["accuracy"]); recs.append(m["recall_away"])
        if accs:
            print(f"  leave-one-DAY-out, {mode:<11} mean acc {np.mean(accs):.3f}  "
                  f"mean AWAY recall {np.mean(recs):.3f}  ({len(accs)} folds)")
        else:
            print(f"  leave-one-DAY-out, {mode:<11} no usable fold")
    # same-day random split, as the ceiling this is being compared against
    from sklearn.model_selection import train_test_split
    a, b, ya, yb = train_test_split(Xlc, yl, test_size=0.2, stratify=yl, random_state=0)
    m = metrics(yb, make_rf().fit(a, ya).predict(b))
    print(f"  same-day random split (the leaky ceiling)  acc {m['accuracy']:.3f}  "
          f"AWAY recall {m['recall_away']:.3f}")

    # ------------------------------------------------------------------
    rule("EXP 3  Do running machines read as a person?")
    print("  THE false-positive test for an HVAC product. These windows contain")
    print("  1-4 running appliances/machines and ZERO people, so the correct answer")
    print("  is AWAY every time. A model that calls them HOME leaves the AC running")
    print("  whenever the dishwasher is on.\n")

    mach = grp == "away_machines"
    # Train presence on the clean classes; calibrate every site on ALL of its own
    # windows, machines included -- that is what a real install would observe.
    Xall_c = calibrate_by(Xe, env)
    rf = make_rf().fit(Xall_c[pres], yp)
    pred_m = rf.predict(Xall_c[mach])
    conf_m = rf.predict_proba(Xall_c[mach]).max(axis=1)

    print(f"  {mach.sum()} machine-only windows, all truly AWAY")
    print(f"  predicted HOME (false alarm): {int((pred_m == HOME).sum())} "
          f"({(pred_m == HOME).mean() * 100:.1f}%)")
    print(f"  predicted AWAY (correct)    : {int((pred_m == AWAY).sum())} "
          f"({(pred_m == AWAY).mean() * 100:.1f}%)\n")
    print("  by environment and machine type:")
    for e in np.unique(env[mach]):
        m = mach & (env == e)
        p = rf.predict(Xall_c[m])
        kind = "home appliances" if set(app[m]) & {"MR"} else "industrial machines"
        print(f"    {e:<24} {kind:<20} {m.sum():4d} windows, "
              f"false-HOME {(p == HOME).mean() * 100:5.1f}%")
    print("\n  for reference, the same model on genuinely EMPTY windows:")
    empt = grp == "away"
    pe = rf.predict(Xall_c[empt])
    print(f"    {empt.sum()} empty windows, false-HOME {(pe == HOME).mean() * 100:.1f}%")
    print(f"\n  mean confidence on machine false alarms: "
          f"{conf_m[pred_m == HOME].mean() if (pred_m == HOME).any() else float('nan'):.3f}")

    # ------------------------------------------------------------------
    rule("EXP 4  Intel-trained model -> Broadcom data (the ESP32 dry run)")
    print("  Train on WiFi-CrowdCounting (Intel 5300, 30 subcarriers, 3x2 MIMO).")
    print("  Test on EHUNAM (Broadcom BCM43455, 52 subcarriers, 1x1) with NO")
    print("  retraining. Different chipset, different antenna count, different")
    print("  subcarrier count, different buildings, different year.\n")

    Xi_c = calibrate_by(Xi, room_i)
    Xe_c = calibrate_by(Xp, envp)
    rf_cross = make_rf().fit(Xi_c, yi)
    pred = rf_cross.predict(Xe_c)
    m = metrics(yp, pred)
    print("  calibrated scale-free features, zero retraining:")
    show(m, indent="    ")
    print_confusion(yp, pred, "    Intel-trained -> EHUNAM")

    print("\n  same transfer WITHOUT calibration (absolute features):")
    rf_abs = make_rf().fit(Xi, yi)
    m2 = metrics(yp, rf_abs.predict(Xp))
    show(m2, indent="    ")

    print("\n  and the reverse direction, Broadcom-trained -> Intel:")
    rf_rev = make_rf().fit(Xe_c, yp)
    m3 = metrics(yi, rf_rev.predict(Xi_c))
    show(m3, indent="    ")

    rule("SUMMARY")
    print("  Exp 1  Cross-environment presence detection does NOT hold up when there")
    print("         is only one training environment, calibrated or not. The Intel")
    print("         3-room result was optimistic; more training sites is the fix,")
    print("         not a cleverer feature.")
    print("  Exp 2  Cross-day performance measured for the first time.")
    print("  Exp 3  Running machines DO cause false HOME alarms; genuinely empty")
    print("         rooms cause none. This is the dominant real-world error mode.")
    print("  Exp 4  Calibration is what makes cross-hardware transfer possible at")
    print("         all -- but transfer is far below in-domain performance, so the")
    print("         ESP32 model must be retrained, exactly as Phase 3 warned.")


if __name__ == "__main__":
    main()
