#!/usr/bin/env python
"""
Compute detection-lag(s) and precision from per-file WL score CSVs.

Changes vs earlier:
- Supports multiple toggles per file (e.g., B and D). Emits semicolon-joined
  columns `lags_steps` and `lags_seconds` (rounded).
- Precision is TP / (TP + FP) where each toggle is paired with the first spike
  at or after that toggle; all *other* spikes are counted as FP.
- Optional long-format output (--long-out) with one row per toggle.

Usage examples:
  python -m scripts.eval_from_scores \
    --scores-dir results_demo/scores \
    --gt data_demo/subjects_testfiles_wltogglepoints.yaml \
    --gt-format subjects_yaml \
    --rate-hz 6.67 \
    --out results_demo/metrics.csv \
    --long-out results_demo/metrics_long.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse, re, csv
import math
import pandas as pd
import yaml


def _load_gt_subjects_yaml(path: Path) -> dict[str, list[int]]:
    """Map 'Subject__file.csv' -> [toggle_steps...] from nested YAML."""
    with path.open("r") as f:
        y = yaml.safe_load(f)
    out: dict[str, list[int]] = {}
    for subj, files in y.items():
        for fname, meta in files.items():
            vals = meta.get("times_wltoggle", [])
            vals = vals if isinstance(vals, list) else [vals]
            out[f"{subj}__{Path(fname).stem}.csv"] = [int(v) for v in vals]
    return out


def _load_gt_csv(path: Path, file_col: str, toggle_step_col: str) -> dict[str, list[int]]:
    """Simple 2-col CSV: file,toggle_step (can repeat file for multiple toggles)."""
    out: dict[str, list[int]] = {}
    with path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            file = row[file_col]
            t = int(float(row[toggle_step_col]))
            out.setdefault(file, []).append(t)
    # keep toggles sorted
    for k in out:
        out[k] = sorted(out[k])
    return out


def _subject_of(filename: str) -> str:
    m = re.match(r"(.+?)__.+\.csv$", filename)
    return m.group(1) if m else ""


def _pair_toggles_to_spikes(toggles: list[int], spikes: list[int]) -> tuple[list[int|None], list[int]]:
    """
    Greedy pairing: for each toggle (in order) take the first spike >= toggle
    that hasn't been used yet. Return (paired_spike_steps, unused_spikes).
    """
    paired: list[int|None] = []
    i = 0
    for t in toggles:
        while i < len(spikes) and spikes[i] < t:
            i += 1
        if i < len(spikes):
            paired.append(spikes[i])
            i += 1
        else:
            paired.append(None)
    unused = spikes[i:]
    return paired, unused


def main():
    ap = argparse.ArgumentParser(
        description="Compute detection-lag and precision from per-file scores and GT."
    )
    ap.add_argument("--scores-dir", default="results/scores", help="Folder with per-file score CSVs.")
    ap.add_argument("--gt", required=True, help="Ground-truth file (csv or yaml).")
    ap.add_argument("--gt-format", choices=["csv", "yaml", "subjects_yaml"], default="subjects_yaml")
    ap.add_argument("--file-col", default="file")
    ap.add_argument("--toggle-step-col", default="toggle_step")
    ap.add_argument("--toggle-ts-col", default=None)
    ap.add_argument("--ts-col-in-scores", default=None, help="Timestamp col name in scores (if using toggle_ts).")
    ap.add_argument("--rate-hz", type=float, default=None, help="For converting lag steps → seconds (optional).")
    ap.add_argument("--round-sec", type=int, default=2, help="Round detection_lag_seconds to this many decimals.")
    ap.add_argument("--out", default="results/metrics.csv")
    ap.add_argument("--long-out", default=None, help="Optional long-format CSV path (one row per toggle).")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        raise FileNotFoundError(f"No such scores-dir: {scores_dir}")

    # --- load ground truth ---
    if args.gt_format == "subjects_yaml":
        gt_map = _load_gt_subjects_yaml(Path(args.gt))
    elif args.gt_format == "csv":
        gt_map = _load_gt_csv(Path(args.gt), args.file_col, args.toggle_step_col)
    else:
        # Generic yaml (expects {file:[steps...]})
        with Path(args.gt).open() as f:
            raw = yaml.safe_load(f)
        gt_map = {k: sorted(list(map(int, v))) for k, v in raw.items()}

    # --- iterate score files ---
    rows_summary = []
    rows_long = []
    n_files = 0

    for score_path in sorted(scores_dir.glob("*.csv")):
        n_files += 1
        name = score_path.name
        df = pd.read_csv(score_path)

        # spikes & steps
        if "step" not in df.columns or "spike" not in df.columns:
            raise ValueError(f"{name} missing step/spike columns.")
        spikes = df.loc[df["spike"] == 1, "step"].astype(int).tolist()

        # resolve toggles
        toggles = gt_map.get(name, [])
        # Also support the case a user gave raw filename (no Subject__ prefix)
        if not toggles:
            toggles = gt_map.get(score_path.stem + ".csv", [])

        # pair toggles to spikes
        paired, unused_spikes = _pair_toggles_to_spikes(toggles, spikes)

        # compute per-toggle lags
        lags_steps: list[int|float] = []
        lags_seconds: list[float|None] = []
        for idx, t in enumerate(toggles):
            s = paired[idx]
            if s is None:
                lags_steps.append(math.nan)
                lags_seconds.append(math.nan if args.rate_hz else None)
            else:
                lag_steps = int(s - t)
                lags_steps.append(lag_steps)
                if args.rate_hz:
                    lsec = round(lag_steps / args.rate_hz, args.round_sec)
                    lags_seconds.append(lsec)
                else:
                    lags_seconds.append(None)

                # long row
                if args.long_out:
                    rows_long.append({
                        "file": name,
                        "subject": _subject_of(name),
                        "toggle_idx": idx + 1,
                        "toggle_step": t,
                        "spike_step": s,
                        "lag_steps": lag_steps,
                        "lag_seconds": (round(lag_steps/args.rate_hz, args.round_sec) if args.rate_hz else None),
                    })

        tp = sum(1 for s in paired if s is not None)
        fp = len(unused_spikes)
        total_spikes = len(spikes)
        precision = (tp / total_spikes) if total_spikes > 0 else (1.0 if tp == 0 else 0.0)

        rows_summary.append({
            "file": name,
            "subject": _subject_of(name),
            "spikes": total_spikes,
            "tp": tp,
            "fp": fp,
            "precision": round(precision, 4),
            # semicolon-joined for human-readability
            "lags_steps": ";".join("" if pd.isna(x) else str(int(x)) for x in lags_steps) if lags_steps else "",
            "lags_seconds": (
                ";".join("" if (x is None or pd.isna(x)) else str(x) for x in lags_seconds)
                if args.rate_hz else ""
            ),
        })

    if n_files == 0:
        raise RuntimeError(f"No score files in {scores_dir}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_summary).to_csv(outp, index=False)
    print(f"[summary] wrote {outp}")

    if args.long_out:
        longp = Path(args.long_out)
        longp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_long).to_csv(longp, index=False)
        print(f"[per-toggle] wrote {longp}")


if __name__ == "__main__":
    main()
