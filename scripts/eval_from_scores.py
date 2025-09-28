#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import yaml

def _load_gt_subjects_yaml(path: Path) -> Dict[str, List[int]]:
    with path.open("r") as f:
        y = yaml.safe_load(f)
    out = {}
    for subj, files in y.items():
        for fname, meta in files.items():
            vals = meta.get("times_wltoggle", [])
            vals = vals if isinstance(vals, list) else [vals]
            out[f"{subj}__{Path(fname).stem}.csv"] = [int(v) for v in vals]
    return out

def _first_spike_at_or_after(spikes: List[int], start: int, end: int | None) -> Tuple[int | None, List[int]]:
    """Return first spike >= start and <= end (if end), plus remaining spikes with that one removed."""
    idx = None
    for s in spikes:
        if s >= start and (end is None or s <= end):
            idx = s
            break
    if idx is None:
        return None, spikes
    rem = spikes.copy()
    rem.remove(idx)
    return idx, rem

def _evaluate_file(scores_path: Path,
                   toggles: List[int],
                   tp_window: int,
                   ignore_pre: bool,
                   rate_hz: float | None) -> Dict[str, float]:
    df = pd.read_csv(scores_path)
    spikes = df.loc[df["spike"] == 1, "step"].astype(int).tolist()

    # Optionally ignore all spikes before first toggle
    if ignore_pre and toggles:
        spikes = [s for s in spikes if s >= toggles[0]]

    used_spikes: List[int] = []
    tps = 0
    lags: List[int] = []

    rem = spikes[:]
    for t in sorted(toggles):
        s, rem = _first_spike_at_or_after(rem, start=t, end=(t + tp_window if tp_window > 0 else None))
        if s is not None:
            tps += 1
            used_spikes.append(s)
            lags.append(s - t)
            # ignore any *extra* spikes in the same window as duplicates (not FP)
            rem = [r for r in rem if not (t <= r <= (t + tp_window if tp_window > 0 else r))]
        # else: miss → no TP, lag not recorded

    fps = len(rem)  # remaining spikes that did not match any toggle window
    precision = tps / (tps + fps) if (tps + fps) else 1.0

    # For convenience keep the first lag and first lag in seconds
    lag1 = lags[0] if lags else None
    lag1_sec = (lag1 / rate_hz) if (lag1 is not None and rate_hz) else None

    return {
        "tp": tps,
        "fp": fps,
        "precision": precision,
        "detection_lag_steps": lag1 if lag1 is not None else float("nan"),
        "detection_lag_seconds": lag1_sec if lag1_sec is not None else float("nan"),
    }

def main():
    ap = argparse.ArgumentParser(description="Compute detection-lag and precision from per-file scores and GT.")
    ap.add_argument("--scores-dir", default="results/scores", help="Folder with per-file score CSVs.")
    ap.add_argument("--gt", required=True, help="Ground-truth file (csv or yaml).")
    ap.add_argument("--gt-format", choices=["csv", "yaml", "subjects_yaml"], default="subjects_yaml")
    ap.add_argument("--file-col", default="file", help="(csv GT) filename column.")
    ap.add_argument("--toggle-step-col", default="toggle_step", help="(csv GT) step column.")
    ap.add_argument("--toggle-ts-col", default=None, help="(csv GT) timestamp column (not used here).")
    ap.add_argument("--ts-col-in-scores", default=None, help="(unused here for now).")
    ap.add_argument("--rate-hz", type=float, default=None, help="For converting steps → seconds.")
    ap.add_argument("--tp-window", type=int, default=120, help="Steps after each toggle to accept the first spike as a TP.")
    ap.add_argument("--ignore-pre-toggle", action="store_true", help="Ignore spikes before the first toggle.")
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    if args.gt_format == "subjects_yaml":
        gt = _load_gt_subjects_yaml(Path(args.gt))
    else:
        raise NotImplementedError("Only --gt-format subjects_yaml is supported in this evaluator.")

    rows = []
    for f in sorted(scores_dir.glob("*.csv")):
        toggles = gt.get(f.name, [])
        m = _evaluate_file(f, toggles, args.tp_window, args.ignore_pre_toggle, args.rate_hz)
        rows.append({"file": f.name, **m})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f">> metrics -> {out}")

if __name__ == "__main__":
    main()
