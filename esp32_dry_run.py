"""
Dry run of the whole ESP32 integration path, on synthetic data, before any
hardware exists.

What this proves (and only this):
  * a file in the EXACT esp-csi csi_recv format -- including the 25-column
    header that csi_data_read_parse.py wrongly writes over 15-column C6 rows --
    is parsed correctly;
  * it lands in the same 16-column feature space as the Intel training data;
  * per-site calibration, the trained model, prediction and the HVAC decision
    all run on it end to end;
  * a hand-written occupancy log joins to the capture and produces a labelled
    table structurally identical to build_dataset.py's.

What this does NOT prove: anything about accuracy on real hardware.  The
fixture is format-and-plumbing only.  Numbers printed below are the plumbing
working, not a result.

    .venv/bin/python esp32_dry_run.py
"""

import os
import numpy as np

from src.build_dataset import build
from src.features import FEATURE_NAMES, SCALE_FREE_FEATURES, fit_site_baseline
from src.train import build_calibrated, train_production_model
from src.pipeline import predict_presence, should_run_ac, LABEL_NAMES
from src.esp_csi import (make_esp_csi_fixture, load_esp32_csi_csv,
                         verify_esp32_assumptions, read_esp_csi_rows,
                         ESP_CSI_C6_COLUMNS, ESP_CSI_CLASSIC_COLUMNS)
from src.manual_label import (label_from_manual_log, read_manual_log,
                              occupancy_intervals, write_manual_log_template)

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
FIX = os.path.join(ART, "esp32")

# One real data row of each schema, copied verbatim out of esp-csi's own
# examples/get-started/README.md.  If our reader cannot eat these, it cannot eat
# a real capture either -- so they are checked first, before anything synthetic.
UPSTREAM_C6_ROW = (
    'CSI_DATA,7,1a:00:00:00:00:00,-23,11,-96,32,4,11,372852,47,0,256,0,'
    '"[' + ",".join(["0"] * 12 + [str(v) for v in range(1, 245)]) + ']"'
)
UPSTREAM_CLASSIC_ROW = (
    'CSI_DATA,0,94:d9:b3:80:8c:81,-30,11,1,6,1,0,1,0,1,0,0,-93,0,13,2,'
    '2751923,0,67,0,128,1,'
    '"[' + ",".join([str(v % 60 - 30) for v in range(128)]) + ']"'
)


def rule(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def step_0_schema_conformance():
    rule("STEP 0  SCHEMA CONFORMANCE against esp-csi's own documented rows")
    checks = [
        ("esp-csi/c5c6c61", ESP_CSI_C6_COLUMNS, UPSTREAM_C6_ROW),
        ("esp-csi/classic", ESP_CSI_CLASSIC_COLUMNS, UPSTREAM_CLASSIC_ROW),
    ]
    for expected_schema, cols, row in checks:
        # write it with the header the capture tool ACTUALLY writes, plus an
        # ESP_LOG line of the kind `idf.py monitor` interleaves, plus a
        # truncated final row -- all three happen in practice.
        p = os.path.join(FIX, f"conformance_{expected_schema.split('/')[-1]}.csv")
        os.makedirs(FIX, exist_ok=True)
        with open(p, "w") as fh:
            fh.write(",".join(ESP_CSI_CLASSIC_COLUMNS) + "\n")
            fh.write("I (1234) csi_recv: ================ CSI RECV ================\n")
            for _ in range(3):
                fh.write(row + "\n")
            fh.write("CSI_DATA,9,1a:00:00:00:00:00,-24,11,-96,32\n")  # truncated
        rec = read_esp_csi_rows(p)
        ok = rec["schema"] == expected_schema and len(rec["blobs"]) == 3
        print(f"  {expected_schema:<18} detected={rec['schema']:<18} "
              f"rows={len(rec['blobs'])} bad={rec['n_rows_bad']} "
              f"subcarriers={len(rec['blobs'][0].strip('\"[]').split(',')) // 2:<4} "
              f"{'OK' if ok else 'FAILED'}")
        assert ok, f"{expected_schema}: reader disagreed with upstream's own example row"
    print("\n  Both upstream example rows parse, the lying header is ignored, the")
    print("  interleaved log line and the truncated row are rejected without")
    print("  taking the capture down.")


def step_1_fixtures():
    rule("STEP 1  SYNTHETIC FIXTURE in the exact ESP32-C6 csi_recv format")
    print("  Format: 15-column esp-csi C5/C6/C61 schema, HT40 -> len=256 ints")
    print("  = 128 subcarriers of which 114 carry energy, CSI as int8 (imag,real)")
    print("  pairs inside a quoted comma-separated bracket list, local_timestamp")
    print("  in microseconds from a random boot offset, ~100 Hz (csi_send's")
    print("  CONFIG_SEND_FREQUENCY), RSSI ~ -28 dBm at 1 m.\n")
    paths = {}
    for tag, occupied, seed in [("occupied", True, 1), ("empty", False, 2)]:
        p = make_esp_csi_fixture(os.path.join(FIX, f"fixture_c6_{tag}.csv"),
                                 n_packets=3000, fs=100.0, occupied=occupied,
                                 seed=seed)
        paths[tag] = p
        size = os.path.getsize(p)
        print(f"  wrote {os.path.relpath(p)}  ({size / 1024:.0f} KB)")
    with open(paths["occupied"]) as fh:
        hdr, first = fh.readline().strip(), fh.readline().strip()
    print(f"\n  header ({len(hdr.split(','))} names, the upstream bug): {hdr[:88]}...")
    print(f"  row    ({15} fields):            {first[:88]}...")
    return paths


def step_2_verify(paths):
    rule("STEP 2  ASSUMPTION CHECKLIST (what verify_esp32_assumptions prints)")
    print("  This is the exact command to run on the FIRST real capture:\n")
    print("    .venv/bin/python -m src.esp_csi my_capture.csv\n")
    verify_esp32_assumptions(paths["occupied"])


def step_3_features(paths):
    rule("STEP 3  FEATURE EXTRACTION -> the same 16 columns as the Intel data")
    outs = {}
    for tag, p in paths.items():
        out = load_esp32_csi_csv(p, verbose=False)
        outs[tag] = out
        print(f"  {tag:<9} X={str(out['X'].shape):<10} fs={out['fs']:.1f} Hz  "
              f"{out['n_subcarriers']} subcarriers  "
              f"window={out['window']} pkt ({out['window'] / out['fs']:.2f} s)  "
              f"schema={out['schema']}")
    assert outs["occupied"]["X"].shape[1] == len(FEATURE_NAMES)
    print(f"\n  Columns are FEATURE_NAMES, in order:\n    {', '.join(FEATURE_NAMES)}")

    print("\n  Sanity contrast on the three motion features (fixture is synthetic,")
    print("  so this checks the plumbing carries a signal, not that the signal is real):")
    print(f"    {'feature':<16} {'empty':>10} {'occupied':>10}   ratio")
    for f in ("cv_mean", "std_amp_mean", "roc_mean", "acf_lag", "phase_std_mean"):
        j = FEATURE_NAMES.index(f)
        e = float(np.median(outs["empty"]["X"][:, j]))
        o = float(np.median(outs["occupied"]["X"][:, j]))
        print(f"    {f:<16} {e:>10.4f} {o:>10.4f}   {o / e if e else float('nan'):>6.2f}x")
    return outs


def step_4_predict(outs):
    rule("STEP 4  CALIBRATION -> PREDICTION -> HVAC DECISION")
    print("  Training the production model on the Intel data (in-process; the repo")
    print("  persists no model artifact), then running the ESP32 windows through it.\n")
    X, y, room, n_people, session = build()
    Xc = build_calibrated(X, y, room, FEATURE_NAMES)
    model = train_production_model(Xc, y)
    print(f"  model: RandomForest on {Xc.shape[0]} calibrated windows, "
          f"{len(SCALE_FREE_FEATURES)} features\n")

    for tag, out in outs.items():
        # A real install calibrates on ITS OWN unlabelled windows.  Here each
        # fixture is its own site, which is the honest analogue.
        baseline = fit_site_baseline(out["X"], FEATURE_NAMES)
        preds = []
        for i in range(len(out["X"])):
            pred, conf = predict_presence(model, out["X"][i], baseline=baseline,
                                          fs=out["fs"])
            run, why = should_run_ac(pred, conf)
            preds.append((pred, conf, run))
        home = sum(p == 1 for p, _, _ in preds)
        ac_on = sum(r for _, _, r in preds)
        print(f"  {tag:<9} {len(preds)} windows -> {home} HOME / {len(preds) - home} AWAY, "
              f"AC ON for {ac_on}/{len(preds)}")
        p0, c0, r0 = preds[0]
        _, why0 = should_run_ac(p0, c0)
        print(f"            first window: {LABEL_NAMES[p0]} conf={c0:.3f} -> {why0}")

    print("\n  The path runs end to end.  The predictions above are meaningless as")
    print("  evidence -- an Intel-trained model on a synthetic ESP32 fixture -- but")
    print("  every stage executed on data in the real csi_recv format.")
    return model


def step_5_manual_labels():
    rule("STEP 5  MANUAL LABELLING CONNECTOR (no door sensor, a person and a log)")
    # A 30-minute capture at 100 Hz would be 180k packets; 6000 packets = 60 s is
    # enough to exercise the joining logic with several transitions.
    cap = make_esp_csi_fixture(os.path.join(FIX, "fixture_c6_session.csv"),
                               n_packets=6000, fs=100.0, occupied=True, seed=7)
    log_path = os.path.join(FIX, "manual_log_example.csv")
    with open(log_path, "w") as fh:
        fh.write(
            "timestamp,event,n_people,note\n"
            "# seconds from capture start; see src/manual_label.py for the ISO form\n"
            "0,capture_start,0,room empty - calibration stretch\n"
            "15,entered,1,one person walks in and sits\n"
            "30,set,1,same person now moving around\n"
            "42,left,0,steps out\n"
            "50,entered,2,two people\n"
            "58,capture_end,2,\n")
    print(f"  capture: {os.path.relpath(cap)}")
    print(f"  log:     {os.path.relpath(log_path)}\n")

    log = read_manual_log(log_path)
    print("  parsed occupancy step function:")
    for s, e, n in occupancy_intervals(log):
        print(f"    {s:6.1f}s - {e:6.1f}s   {n} people")
    print()

    res = label_from_manual_log(cap, log_path, site="test_house",
                                session="dry_run_session",
                                out_npz=os.path.join(FIX, "esp32_labelled.npz"),
                                verbose=True)
    print(f"\n  Output table matches build_dataset.py's cache format exactly:")
    for k in ("X", "y", "room", "n_people", "session", "feature_names"):
        v = res[k]
        print(f"    {k:<14} shape={str(v.shape):<12} dtype={v.dtype}")
    assert res["X"].shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(res["y"])) <= {0, 1}
    assert len(res["X"]) == len(res["y"]) == len(res["room"]) == len(res["session"])

    z = np.load(os.path.join(FIX, "esp32_labelled.npz"), allow_pickle=True)
    print(f"\n  reloaded npz keys: {sorted(z.files)}")
    print(f"  label balance: {dict(zip(*np.unique(z['y'], return_counts=True)))} "
          f"(0=AWAY, 1=HOME)")

    tmpl = write_manual_log_template(os.path.join(FIX, "manual_log_TEMPLATE.csv"))
    print(f"\n  blank template for tonight: {os.path.relpath(tmpl)}")
    return res


def step_6_error_paths():
    rule("STEP 6  FAILURE MODES ARE LOUD, NOT SILENT")
    from src.manual_label import ManualLogError
    cases = []

    bad = os.path.join(FIX, "_bad_log.csv")
    with open(bad, "w") as fh:
        fh.write("timestamp,event,n_people,note\n0,entered,1,no capture_start\n")
    try:
        read_manual_log(bad)
        cases.append(("log with no capture_start", "NOT CAUGHT"))
    except ManualLogError as e:
        cases.append(("log with no capture_start", f"raised: {str(e).split(': ')[-1][:52]}"))

    with open(bad, "w") as fh:
        fh.write("timestamp,event,n_people,note\n"
                 "0,capture_start,0,\n2026-08-31T19:00:00,left,,mixed styles\n")
    try:
        read_manual_log(bad)
        cases.append(("mixed timestamp styles", "NOT CAUGHT"))
    except ManualLogError as e:
        cases.append(("mixed timestamp styles", f"raised: {str(e).split(': ')[-1][:52]}"))

    with open(bad, "w") as fh:
        fh.write("timestamp,event,n_people,note\n0,capture_start,0,\n10,left,,nobody was in\n")
    try:
        read_manual_log(bad)
        cases.append(("occupancy goes negative", "NOT CAUGHT"))
    except ManualLogError as e:
        cases.append(("occupancy goes negative", f"raised: {str(e).split(': ')[-1][:52]}"))

    junk = os.path.join(FIX, "_bad_capture.csv")
    with open(junk, "w") as fh:
        fh.write("type,id,mac\nCSI_DATA,1,aa:bb:cc:dd:ee:ff\n")
    try:
        read_esp_csi_rows(junk)
        cases.append(("capture with an unknown schema", "NOT CAUGHT"))
    except ValueError as e:
        cases.append(("capture with an unknown schema", f"raised: {str(e)[:52]}"))

    for name, outcome in cases:
        flag = "OK " if outcome != "NOT CAUGHT" else "BAD"
        print(f"  [{flag}] {name:<32} {outcome}")
    os.remove(bad)
    os.remove(junk)
    assert all(o != "NOT CAUGHT" for _, o in cases)


def main():
    os.makedirs(FIX, exist_ok=True)
    step_0_schema_conformance()
    paths = step_1_fixtures()
    step_2_verify(paths)
    outs = step_3_features(paths)
    step_4_predict(outs)
    step_5_manual_labels()
    step_6_error_paths()

    rule("DRY RUN COMPLETE")
    print("  Every stage of the ESP32 path executed on data in the real esp-csi")
    print("  csi_recv format. When hardware data arrives tonight:")
    print()
    print("    1. .venv/bin/python -m src.esp_csi captures/session1.csv")
    print("       -> read the [OPEN 1..5] lines. Those are the only things this")
    print("          dry run could not settle.")
    print("    2. from src.manual_label import label_from_manual_log")
    print("       res = label_from_manual_log('captures/session1.csv',")
    print("                                   'captures/session1_log.csv',")
    print("                                   site='byu_apt', out_npz='artifacts/real.npz')")
    print("    3. Retrain on res['X'], res['y'] -- the Intel-fitted model does NOT")
    print("       transfer (1x1 antenna, 114 vs 90 channels); the feature pipeline does.")
    print()
    print("  See docs/ESP32_SETUP.md for flashing, capture and session protocol.")


if __name__ == "__main__":
    main()
