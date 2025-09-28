import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import argparse
from pathlib import Path
import pandas as pd
import yaml

from htm_wl.metrics import detection_lag_steps, precision

def _load_gt_csv(path: Path, file_col: str, step_col: str | None, ts_col: str | None):
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        key = str(r[file_col])
        item = {}
        if step_col and pd.notna(r.get(step_col)):
            item["toggle_step"] = int(r[step_col])
        if ts_col and pd.notna(r.get(ts_col)):
            item["toggle_ts"] = str(r[ts_col])
        out[key] = item
    return out

def _load_gt_yaml(path: Path):
    with path.open("r") as f:
        y = yaml.safe_load(f)
    # Expect mapping: { "file.csv": {"toggle_step": int} } or {"toggle_ts": "..."}
    return y

def _load_gt_subjects_yaml(path: Path):
    import yaml
    with path.open("r") as f:
        y = yaml.safe_load(f)
    out = {}
    # expects: { "Subject": { "file.csv": {"times_wltoggle":[step], ...}, ... }, ... }
    for subj, files in y.items():
        for fname, meta in files.items():
            step = int(meta["times_wltoggle"][0])
            key = f"{subj}__{fname}"   # matches scorer output names
            out[key] = {"toggle_step": step}
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute detection-lag and precision from per-file scores and GT.")
    ap.add_argument("--scores-dir", default="results/scores", help="Folder with per-file score CSVs.")
    ap.add_argument("--gt", required=True, help="Ground-truth file (csv or yaml).")
    ap.add_argument("--gt-format", choices=["csv","yaml","subjects_yaml"], default="csv")
    ap.add_argument("--file-col", default="file")
    ap.add_argument("--toggle-step-col", default="toggle_step")
    ap.add_argument("--toggle-ts-col", default="toggle_ts")
    ap.add_argument("--ts-col-in-scores", default=None, help="Timestamp col name in scores (if using toggle_ts).")
    ap.add_argument("--rate-hz", type=float, default=None, help="For converting lag steps → seconds (optional).")
    ap.add_argument("--out", default="results/metrics.csv")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    files = sorted(scores_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No score files in {scores_dir}")

    if args.gt_format == "csv":
        gt = _load_gt_csv(Path(args.gt), args.file_col, args.toggle_step_col, args.toggle_ts_col)
    elif args.gt_format == "yaml":
        gt = _load_gt_yaml(Path(args.gt))
    elif args.gt_format == "subjects_yaml":
        gt = _load_gt_subjects_yaml(Path(args.gt))

    rows = []
    for f in files:
        key = f.name if f.name in gt else f.stem + ".csv"  # match “file.csv” keys
        if key not in gt:
            print(f"[warn] no GT for {f.name}; skipping")
            continue
        g = gt[key]
        df = pd.read_csv(f)

        # Find toggle step
        if "toggle_step" in g:
            toggle_step = int(g["toggle_step"])
        elif "toggle_ts" in g and args.ts_col_in_scores:
            # first row with ts >= toggle_ts
            toggle_ts = pd.to_datetime(g["toggle_ts"])
            cand = pd.to_datetime(df[args.ts_col_in_scores])
            idx = cand.searchsorted(toggle_ts, side="left")
            toggle_step = int(idx + 1)  # steps are 1-based in our scoring
        else:
            print(f"[warn] GT for {f.name} is missing toggle_step and toggle_ts; skipping")
            continue

        spike_idx = df.index[df["spike"] == 1].to_list()
        spike_steps = [i+1 for i in spike_idx]  # to 1-based

        lag_steps = detection_lag_steps(toggle_step, spike_steps)
        lag_sec = (lag_steps / args.rate_hz) if (lag_steps is not None and args.rate_hz) else None
        prec = precision(toggle_step, spike_steps)

        rows.append({
            "file": f.name,
            "toggle_step": toggle_step,
            "spikes": len(spike_steps),
            "detection_lag_steps": lag_steps,
            "detection_lag_seconds": lag_sec,
            "precision": prec,
        })
        print(f"[done] {f.name}: spikes={len(spike_steps)} lag_steps={lag_steps} precision={prec:.3f}")

    if not rows:
        raise SystemExit("No metrics produced. Check your GT and score files.")
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"\nSaved metrics → {outp}")

if __name__ == "__main__":
    main()
