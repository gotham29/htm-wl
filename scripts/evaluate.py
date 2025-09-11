#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd

from htm_wl.io import (
    load_pipeline_yaml,
    extract_defaults,
    load_ground_truth,
    iter_rows,
)
from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector
from htm_wl.metrics import detection_lag_steps, precision


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce HTM-WL results: MWL (EMA of anomaly) -> spike detection -> detection lag & precision."
    )
    ap.add_argument("--config", default="run_pipeline.yaml", help="Path to pipeline YAML.")
    ap.add_argument("--data-root", default="data", help="Root folder containing subject directories.")
    ap.add_argument("--outdir", default="results", help="Where to write results/summary.csv.")
    ap.add_argument("--warmup-file", default="train.csv", help="Per-subject warmup file (e.g., train.csv).")
    ap.add_argument("--rate-hz", type=float, default=None, help="Override sampling rate; else taken from YAML.")
    # NEW verbosity & speed-control flags
    ap.add_argument("--verbose", action="store_true", help="Print detailed progress.")
    ap.add_argument("--limit-train", type=int, default=None, help="Max warmup steps per subject (for quick runs).")
    ap.add_argument("--max-files", type=int, default=None, help="Evaluate only this many test files total.")
    args = ap.parse_args()

    root = Path(".").resolve()
    data_root = root / args.data_root
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load config & defaults
    cfg = load_pipeline_yaml(root / args.config)
    defaults = extract_defaults(cfg)
    if args.rate_hz is not None:
        defaults["rate_hz"] = args.rate_hz

    # Ground truth map: { "Subject/file.csv": {"toggle_step": int, "total_steps": int} }
    gt_map = load_ground_truth(cfg)
    tests = list(gt_map.items())
    if args.max_files:
        tests = tests[: args.max_files]

    subjects = sorted(set(k.split("/")[0] for k, _ in tests))
    print(
        f"[info] subjects={len(subjects)} tests={len(tests)} rate_hz={defaults['rate_hz']} "
        f"enc_n_per_feature={defaults['enc_n_per_feature']} enc_w_per_feature={defaults['enc_w_per_feature']}"
    )
    if args.verbose and tests:
        first_key = tests[0][0]
        print(f"[info] example test key: {first_key}")

    # Build one HTM session (global), warmed by each subject's train.csv.
    # (We keep this simple for reproduction; per-subject reinit can be added later.)
    sess = HTMSession(
        feature_names=defaults["feature_names"],
        enc_n_per_feature=defaults["enc_n_per_feature"],
        enc_w_per_feature=defaults["enc_w_per_feature"],
        sp_params=defaults["sp_params"],
        tm_params=defaults["tm_params"],
        seed=defaults["seed"],
        anomaly_ema_alpha=0.2,
        feature_ranges=None,  # provide ranges if desired
    )

    # Warm-up pass
    for subj in subjects:
        train = data_root / subj / args.warmup_file
        if not train.exists():
            if args.verbose:
                print(f"[warmup] {subj}: {train.name} not found; skipping warmup.")
            continue
        if args.verbose:
            print(f"[warmup] {subj}: {train}")
        for i, feats in enumerate(iter_rows(train, defaults["feature_names"]), start=1):
            if args.limit_train and i > args.limit_train:
                if args.verbose:
                    print(f"[warmup] {subj}: limited to {args.limit_train} steps.")
                break
            sess.step(feats)
            if args.verbose and (i % 1000 == 0):
                print(f"[warmup] {subj}: {i} steps…")

    # Evaluate each test with ground-truth toggle
    rows = []
    for key, truth in tests:
        subj, fname = key.split("/", 1)
        test_path = data_root / subj / fname
        if not test_path.exists():
            print(f"[run] {key} … MISSING ({test_path}); skipping.")
            continue

        print(f"[run] {key} …")
        det = SpikeDetector(**defaults["spike_params"])
        spike_steps = []
        step_idx = 0

        for feats in iter_rows(test_path, defaults["feature_names"]):
            step_idx += 1
            res = sess.step(feats)
            s = det.update(res["mwl"])
            if s and s["spike"]:
                spike_steps.append(step_idx)
            if args.verbose and (step_idx % 1000 == 0):
                print(f"[run] {key}: {step_idx} steps…")

        lag_steps = detection_lag_steps(truth["toggle_step"], spike_steps)
        lag_s = None if lag_steps is None else (lag_steps / defaults["rate_hz"])
        prec = precision(truth["toggle_step"], spike_steps)

        rows.append(
            {
                "subject": subj,
                "file": fname,
                "toggle_step": truth["toggle_step"],
                "spikes": len(spike_steps),
                "detection_lag_steps": lag_steps,
                "detection_lag_seconds": lag_s,
                "precision": prec,
            }
        )
        print(f"[done] {subj}/{fname}: spikes={len(spike_steps)} lag_s={lag_s} precision={prec:.3f}")

    if not rows:
        print("[warn] No results produced (no tests found or all missing). "
              "Check run_pipeline.yaml and your data/ layout.")
        return

    df = pd.DataFrame(rows).sort_values(["subject", "file"])
    out_path = outdir / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()
