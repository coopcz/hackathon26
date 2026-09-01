"""Evaluate labelled holdout recordings through the exact deployed predictor.

Use recordings that were NOT passed to ``src.train_esp32``. One JSONL file is
one pure condition. The predictor resets between files exactly as it does when
the board reconnects, and consumes the original raw packets verbatim.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.predictor import PREDICT_EVERY, Predictor
from backend.replay import _packet_from
from src.dataset import _quality, label_of
from src.esp_csi import load_esp32_csi_csv


def longest_run(values, target):
    longest = current = 0
    for value in values:
        current = current + 1 if value == target else 0
        longest = max(longest, current)
    return longest


def score_recording(path, predictor):
    out = load_esp32_csi_csv(path, valid_subcarriers="reference", verbose=False)
    quality = _quality(out, out["amp"])
    condition = label_of(path, out["session_label"])
    expected = "AWAY" if condition == "empty" else "HOME"
    predictor.reset()
    verdicts = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                verdict = predictor.append(_packet_from(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if verdict:
                verdicts.append(verdict["presence"])
    wrong = "HOME" if expected == "AWAY" else "AWAY"
    fs = float(predictor.bundle["fs"])
    return {
        "filename": Path(path).name, "condition": condition, "expected": expected,
        "quality": quality, "verdicts": verdicts,
        "correct_rate": (verdicts.count(expected) / len(verdicts)) if verdicts else 0.0,
        "majority": Counter(verdicts).most_common(1)[0][0] if verdicts else "NONE",
        "longest_wrong_seconds": longest_run(verdicts, wrong) * PREDICT_EVERY / fs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory containing holdout JSONL recordings")
    parser.add_argument("--model", default="artifacts/esp32_model.joblib")
    parser.add_argument("--min-away", type=float, default=0.80)
    parser.add_argument("--min-home", type=float, default=0.95)
    parser.add_argument("--max-wrong-streak", type=float, default=3.0)
    args = parser.parse_args()

    predictor = Predictor(Path(args.model))
    if predictor.bundle is None:
        raise SystemExit(f"model did not load: {predictor.error or args.model}")
    paths = sorted(Path(args.data_dir).glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no JSONL recordings found in {args.data_dir}")

    totals = defaultdict(lambda: [0, 0])
    failed = False
    print(f"{'recording':<46} {'truth':<17} {'correct':>8} {'majority':>9} {'wrong run':>10}  status")
    print("-" * 112)
    for path in paths:
        result = score_recording(path, predictor)
        condition = result["condition"] or "UNKNOWN"
        rate = result["correct_rate"]
        minimum = args.min_away if condition == "empty" else args.min_home
        quality_ok = result["quality"]["usable"]
        passed = (quality_ok and result["majority"] == result["expected"]
                  and rate >= minimum
                  and result["longest_wrong_seconds"] <= args.max_wrong_streak)
        failed |= not passed
        totals[condition][0] += sum(v == result["expected"] for v in result["verdicts"])
        totals[condition][1] += len(result["verdicts"])
        reasons = []
        if not quality_ok:
            reasons.append("bad capture: " + "; ".join(result["quality"]["problems"]))
        if result["majority"] != result["expected"]:
            reasons.append("wrong majority")
        if rate < minimum:
            reasons.append(f"below {minimum:.0%} target")
        if result["longest_wrong_seconds"] > args.max_wrong_streak:
            reasons.append("wrong streak too long")
        print(f"{result['filename'][:45]:<46} {condition:<17} {rate:>7.1%} "
              f"{result['majority']:>9} {result['longest_wrong_seconds']:>8.2f}s  "
              f"{'PASS' if passed else 'FAIL: ' + '; '.join(reasons)}")

    print("\nAggregate exact-live-path accuracy:")
    for condition, (correct, total) in sorted(totals.items()):
        print(f"  {condition:<17} {correct / total:6.1%} ({correct}/{total})" if total
              else f"  {condition:<17} no predictions")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
