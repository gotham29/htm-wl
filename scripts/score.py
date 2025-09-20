#!/usr/bin/env python
import argparse, csv, glob, os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml

from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector

def _load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)

def _expand_globs(patterns: List[str]) -> List[Path]:
    inc, exc = [], []
    for p in patterns:
        (exc if p.startswith("!") else inc).append(p.lstrip("!"))
    files = []
    for p in inc:
        files += [Path(x) for x in glob.glob(p, recursive=True)]
    if exc:
        exset = set()
        for p in exc:
            exset |= set(Path(x) for x in glob.glob(p, recursive=True))
        files = [f for f in files if f not in exset]
    # keep stable, human-readable order
    return sorted(set(files))

def _iter_rows_mapped(csv_path: Path, feature_map: Dict[str, str], ts_col: Optional[str]):
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            feats = {}
            for name, col in feature_map.items():
                if col in row:
                    feats[name] = float(row[col])
                elif name == "PITCH_STIC" and "PITCH_STICK" in row:
                    feats[name] = float(row["PITCH_STICK"])
                else:
                    raise KeyError(f"Missing column {col} (feature {name}) in {csv_path}")
            ts = row[ts_col] if (ts_col and ts_col in row) else None
            yield ts, feats

def main():
    ap = argparse.ArgumentParser(description="Warm on train, score test CSVs, write per-file WL outputs.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(".").resolve()
    cfg = _load_yaml(root / args.config)
    ds = cfg["dataset"]; mdl = cfg["model"]; detcfg = cfg["detection"]

    # feature names and mapping
    feats = [f["name"] for f in ds["features"]]
    feat_map = {f["name"]: f["column"] for f in ds["features"]}
    ranges = {f["name"]: {"min": f.get("min",-1.0), "max": f.get("max",1.0)} for f in ds["features"]}
    ts_col = ds.get("timestamp_column")
    rate_hz = ds.get("rate_hz")

    # detector counts (seconds -> steps if needed)
    units = detcfg["windows"]["units"]
    if units == "seconds":
        if rate_hz is None:
            raise ValueError("windows units=seconds requires dataset.rate_hz in config.")
        recent_count = max(1, int(round(detcfg["windows"]["recent"] * rate_hz)))
        prior_count  = max(1, int(round(detcfg["windows"]["prior"]  * rate_hz)))
    else:
        recent_count = int(detcfg["windows"]["recent"])
        prior_count  = int(detcfg["windows"]["prior"])
    threshold_pct = float(detcfg["threshold_pct"])

    # Build session
    sess = HTMSession(
        feature_names=feats,
        enc_n_per_feature=int(mdl["encoder"]["n_per_feature"]),
        enc_w_per_feature=int(mdl["encoder"]["w_per_feature"]),
        sp_params=mdl["sp"],
        tm_params=mdl["tm"],
        seed=int(mdl["seed"]),
        anomaly_ema_alpha=float(mdl.get("anomaly_ema_alpha", 0.2)),
        feature_ranges=ranges,
    )

    # Warm-up on all train files
    train_files = _expand_globs(ds.get("train", []))
    if args.verbose:
        print(f"[warmup] files={len(train_files)}")
    for p in train_files:
        if args.verbose: print(f"[warmup] {p}")
        for _, feats_row in _iter_rows_mapped(p, feat_map, ts_col):
            sess.step(feats_row)

    # Score tests
    test_files = _expand_globs(ds.get("test", []))
    print(f"[score] test_files={len(test_files)}")
    out_root = root / args.outdir / "scores"
    out_root.mkdir(parents=True, exist_ok=True)

    for p in test_files:
        det = SpikeDetector(recent_count, prior_count, threshold_pct)
        rows = []
        step = 0
        for ts, feats_row in _iter_rows_mapped(p, feat_map, ts_col):
            step += 1
            res = sess.step(feats_row)
            s = det.update(res["mwl"])
            rec = {
                "step": step,
                "anomaly": res["anomaly"],
                "mwl": res["mwl"],
            }
            if s:
                rec.update({"spike": int(bool(s["spike"])), "mr": s["mr"], "mp": s["mp"], "growth_pct": s["growth_pct"]})
            else:
                rec.update({"spike": 0, "mr": None, "mp": None, "growth_pct": None})
            if ts_col:
                rec[ts_col] = ts
            rows.append(rec)
        subject = p.parent.name
        out_name = f"{subject}__{p.stem}.csv"
        out_path = out_root / out_name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[done] {p} → {out_path}")

if __name__ == "__main__":
    main()
