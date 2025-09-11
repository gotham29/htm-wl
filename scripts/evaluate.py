#!/usr/bin/env python
import argparse, time
from pathlib import Path
import pandas as pd

from htm_wl.io import load_pipeline_yaml, extract_defaults, load_ground_truth, iter_rows
from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector
from htm_wl.metrics import detection_lag_steps, precision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="run_pipeline.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--warmup-file", default="train.csv")
    ap.add_argument("--rate-hz", type=float, default=None, help="Override sampling rate; else from YAML")
    args = ap.parse_args()

    root = Path(".").resolve()
    cfg = load_pipeline_yaml(root / args.config)
    defaults = extract_defaults(cfg)
    if args.rate_hz is not None:
        defaults["rate_hz"] = args.rate_hz

    gt_map = load_ground_truth(cfg)  # key: "Subject/file.csv" -> {toggle_step,total_steps}
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Build model
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
    det = SpikeDetector(**defaults["spike_params"])

    # Warm-up once per subject (train.csv)
    subjects = set(k.split("/")[0] for k in gt_map.keys())
    for subj in sorted(subjects):
        train = Path(args.data_root) / subj / args.warmup_file
        if train.exists():
            for feats in iter_rows(train, defaults["feature_names"]):
                sess.step(feats)

    # Evaluate each test file with GT
    rows = []
    for key, truth in gt_map.items():
        subj, fname = key.split("/", 1)
        test_path = Path(args.data_root) / subj / fname
        if not test_path.exists():
            print(f"[skip] missing {test_path}")
            continue

        # reset detector buffer per file
        det = SpikeDetector(**defaults["spike_params"])
        spike_steps = []
        step_idx = 0

        for feats in iter_rows(test_path, defaults["feature_names"]):
            step_idx += 1
            res = sess.step(feats)
            s = det.update(res["mwl"])
            if s and s["spike"]:
                spike_steps.append(step_idx)

        lag_steps = detection_lag_steps(truth["toggle_step"], spike_steps)
        lag_s = None if lag_steps is None else lag_steps / defaults["rate_hz"]
        prec = precision(truth["toggle_step"], spike_steps)

        rows.append({
            "subject": subj,
            "file": fname,
            "toggle_step": truth["toggle_step"],
            "spikes": len(spike_steps),
            "detection_lag_steps": lag_steps,
            "detection_lag_seconds": lag_s,
            "precision": prec,
        })
        print(f"{subj}/{fname}: spikes={len(spike_steps)} lag_s={lag_s} precision={prec:.3f}")

    df = pd.DataFrame(rows).sort_values(["subject","file"])
    df.to_csv(outdir / "summary.csv", index=False)
    print(f"\nSaved results → {outdir / 'summary.csv'}")

if __name__ == "__main__":
    main()
