#!/usr/bin/env python3
"""
Live event marker for a manual-labelling capture session.

Start this at the SAME MOMENT you start csi_data_read_parse.py, in a second
terminal.  It stamps every keystroke into the CSV that src/manual_label.py reads,
so nobody has to read a clock and type a timestamp while a person is walking
through the door.

    python tools/mark.py ~/captures/session1_log.csv

Keys (press Enter after each):
    i / in / entered       someone entered   (occupancy + 1)
    o / out / left         someone left      (occupancy - 1)
    <number>               set occupancy to exactly that
    n <text>               note, no occupancy change
    q / quit               write capture_end and exit

Timestamps are written as seconds since this script started, which is exactly
the clock `label_from_manual_log` expects.  If you started this a few seconds
late, you do not need to retype anything -- pass `clock_offset=` when you join.

Stdlib only; nothing to install.
"""

import csv
import os
import sys
import time


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = os.path.expanduser(argv[1])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path):
        print(f"refusing to overwrite {path} -- move it aside first")
        return 1

    start = time.monotonic()
    wall = time.strftime("%Y-%m-%dT%H:%M:%S")
    occupancy = 0

    fh = open(path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["timestamp", "event", "n_people", "note"])

    def emit(event, n, note=""):
        t = round(time.monotonic() - start, 2)
        w.writerow([t, event, n, note])
        fh.flush()                      # every line hits disk immediately
        os.fsync(fh.fileno())           # a crash mid-session must not lose labels
        print(f"    {t:8.2f}s  {event:<14} n={n}  {note}")

    print(f"logging to {path}")
    print(f"capture_start at wall clock {wall}")
    print("  i=entered   o=left   <number>=set count   n <text>=note   q=quit\n")
    emit("capture_start", 0, f"wall clock {wall}")

    try:
        while True:
            try:
                raw = input("> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            # the first word is the command, everything after it is a free-text
            # note -- so "i person A sat down" works, not just "i"
            head, _, rest = raw.partition(" ")
            low = head.lower()
            rest = rest.strip()

            if low in ("q", "quit", "exit"):
                break
            if low in ("i", "in", "entered", "enter"):
                occupancy += 1
                emit("entered", occupancy, rest)
            elif low in ("o", "out", "left", "leave"):
                if occupancy == 0:
                    print("    (occupancy is already 0 -- ignoring, nothing to subtract)")
                    continue
                occupancy -= 1
                emit("left", occupancy, rest)
            elif low in ("n", "note"):
                emit("set", occupancy, rest or "note")
            elif head.isdigit():
                occupancy = int(head)
                emit("set", occupancy, rest)
            else:
                print("    ? use i / o / <number> / n <text> / q")
    except KeyboardInterrupt:
        print()
    finally:
        emit("capture_end", occupancy)
        fh.close()
        print(f"\nwrote {path}")
        print("Join it to the capture with:")
        print(f"  from src.manual_label import label_from_manual_log")
        print(f"  label_from_manual_log('<capture>.csv', '{path}', site='<site>')")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
