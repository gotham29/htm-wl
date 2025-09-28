#!/usr/bin/env python
"""
Live MWL + spike visualization (separate train/test modes).

Modes:
  - Train: step the model on training CSV(s) only (no spikes), optional live plot.
  - Test:  warm on training CSV(s) and then stream a test CSV with spikes.

Examples:
  # TRAIN ONLY (animate), using a specific training file
  python -m scripts.live_demo --mode train \
    --config config.demo.yaml \
    --train-file data_demo/DemoA/train.csv \
    --rate-hz 6.67

  # TEST ONLY (animate), warm on config globs and stream this test file
  python -m scripts.live_demo --mode test \
    --config config.demo.yaml \
    --file data_demo/DemoA/test_01.csv \
    --rate-hz 6.67 --speed 4.0

Notes:
  - MWL axis is fixed to [0, 1].
  - Top subplot shows z-scored inputs for the configured feature columns.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
from pathlib import Path
import argparse, csv, time
from collections import deque
from typing import Dict, Optional, List

import yaml
import numpy as np
import matplotlib.pyplot as plt

from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector


# ---------- helpers ----------
def _load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)

def _expand_globs(patterns: List[str]) -> List[Path]:
    import glob
    inc, exc = [], []
    for p in (patterns or []):
        (exc if p.startswith("!") else inc).append(p.lstrip("!"))
    files: List[Path] = []
    for p in inc:
        files += [Path(x) for x in glob.glob(p, recursive=True)]
    if exc:
        exset = set()
        for p in exc:
            exset |= set(Path(x) for x in glob.glob(p, recursive=True))
        files = [f for f in files if f not in exset]
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
            yield ts, feats, row  # include raw row for input plotting

def _setup_session(ds, mdl, ranges, feats):
    return HTMSession(
        feature_names=feats,
        enc_n_per_feature=int(mdl["encoder"]["n_per_feature"]),
        enc_w_per_feature=int(mdl["encoder"]["w_per_feature"]),
        sp_params=mdl["sp"],
        tm_params=mdl["tm"],
        seed=int(mdl["seed"]),
        anomaly_ema_alpha=float(mdl.get("anomaly_ema_alpha", 0.2)),
        feature_ranges=ranges,
    )

def _setup_detector(detcfg, rate_hz: float):
    units = detcfg["windows"]["units"]
    if units == "seconds":
        recent_count = max(1, int(round(detcfg["windows"]["recent"] * rate_hz)))
        prior_count  = max(1, int(round(detcfg["windows"]["prior"]  * rate_hz)))
    else:
        recent_count = int(detcfg["windows"]["recent"])
        prior_count  = int(detcfg["windows"]["prior"])
    return SpikeDetector(
        recent_count=recent_count,
        prior_count=prior_count,
        threshold_pct=float(detcfg["threshold_pct"]),
        min_delta=float(detcfg.get("min_delta", 0.0)),
        min_separation=int(detcfg.get("min_separation", 0)),
        edge_only=bool(detcfg.get("edge_only", True)),
        direction=str(detcfg.get("direction", "up")),
        eps=float(detcfg.get("eps", 1e-9)),
        min_mwl=float(detcfg.get("min_mwl", 0.0)),
    )

def _make_fig(title: str, window: int, show_raw: bool = True, include_growth: bool = True):
    plt.ion()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    fig.suptitle(title)

    # Top: raw inputs (z-scored)
    ax_top.set_ylabel("Inputs (z)")
    raw_lines = {}  # feature_name -> Line2D

    # Bottom: MWL + growth% + spikes
    ax_bot.set_xlabel("step")
    ax_bot.set_ylabel("MWL (EMA of anomaly)")
    ax_bot.set_ylim(0.0, 1.05)

    ax_right = None
    gpc_line = None
    spike_scatter = None

    # Colors: MWL = blue, growth% = orange, spikes = red
    mwl_line, = ax_bot.plot([], [], label="MWL", color="tab:blue")

    if include_growth:
        ax_right = ax_bot.twinx()
        ax_right.set_ylabel("growth %")
        gpc_line, = ax_right.plot([], [], alpha=0.9, label="growth%", color="tab:orange")
        spike_scatter = ax_bot.scatter([], [], s=22, label="Spike", color="tab:red")

        # unified legend
        h1, l1 = ax_bot.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_bot.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        # legend with MWL only
        h1, l1 = ax_bot.get_legend_handles_labels()
        ax_bot.legend(h1, l1, loc="upper right")

    return fig, ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter

def _update_plot(ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter,
                 step_series, raw_z_series_map, mwl_series, gpc_series, spike_x, spike_y):
    # top: raw inputs
    for name, line in raw_lines.items():
        zs = raw_z_series_map.get(name, [])
        line.set_data(step_series, zs)

    # bottom: mwl + growth + spikes
    mwl_line.set_data(step_series, mwl_series)

    if gpc_line is not None:
        gpc_line.set_data(step_series, gpc_series)
    if spike_scatter is not None and spike_x:
        spike_scatter.set_offsets(np.c_[spike_x, spike_y])

    ax_top.relim(); ax_top.autoscale_view()
    ax_bot.relim(); ax_bot.autoscale_view()
    ax_bot.set_ylim(0.0, 1.05)  # keep fixed
    if ax_right is not None:
        ax_right.relim(); ax_right.autoscale_view()

    plt.pause(0.001)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Live MWL + spike plot (separate train/test modes).")
    ap.add_argument("--mode", choices=["train", "test"], required=True, help="Run training-only or test streaming.")
    ap.add_argument("--config", default="config.yaml", help="YAML config.")
    ap.add_argument("--train-file", default=None, help="Training CSV to use (overrides config train globs).")
    ap.add_argument("--file", default=None, help="(Test mode) CSV to stream.")
    ap.add_argument("--rate-hz", type=float, default=None, help="Playback rate (Hz). If omitted, uses dataset.rate_hz.")
    ap.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (e.g., 4.0 = 4x faster than real-time).")
    ap.add_argument("--window", type=int, default=2000, help="Max points kept on screen (older data rolls off).")
    ap.add_argument("--show-raw", action="store_true", default=True, help="Show top subplot of input features (z).")
    ap.add_argument("--no-show-raw", action="store_false", dest="show_raw")
    ap.add_argument("--plot", action="store_true", default=True, help="Show live plot.")
    ap.add_argument("--no-plot", action="store_false", dest="plot")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(".").resolve()
    cfg = _load_yaml(root / args.config)
    ds = cfg["dataset"]; mdl = cfg["model"]; detcfg = cfg["detection"]

    # features + ranges + rate
    feats = [f["name"] for f in ds["features"]]
    feat_map = {f["name"]: f["column"] for f in ds["features"]}
    ranges = {f["name"]: {"min": f.get("min",-1.0), "max": f.get("max",1.0)} for f in ds["features"]}
    ts_col = ds.get("timestamp_column")
    rate_hz = float(args.rate_hz) if args.rate_hz is not None else float(ds.get("rate_hz", 1.0))
    period = (1.0 / max(rate_hz, 1e-9)) / max(args.speed, 1e-9)

    # choose train sources
    train_files: List[Path]
    if args.train_file:
        p = Path(args.train_file)
        if not p.exists():
            raise FileNotFoundError(f"--train-file not found: {p}")
        train_files = [p]
    else:
        train_files = _expand_globs(ds.get("train", []))
        if args.verbose:
            print(f"[train] {len(train_files)} file(s) from config")

    # session
    sess = _setup_session(ds, mdl, ranges, feats)

    # ---- TRAIN MODE ----
    if args.mode == "train":
        if not train_files:
            raise RuntimeError("No training files (pass --train-file or set dataset.train).")
        if args.plot:
            fig, ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter = _make_fig(
                title="Training (model warmup)", window=args.window, show_raw=args.show_raw, include_growth=False)
            # create a line per feature on top axes
            if args.show_raw:
                for name in feats:
                    (raw_lines[name],) = ax_top.plot([], [], label=name, alpha=0.8)
                ax_top.legend(loc="upper right")

            # rolling state
            X = deque(maxlen=args.window)
            MWL = deque(maxlen=args.window)
            GPC = deque(maxlen=args.window)  # unused in train
            spikes_x, spikes_y = [], []      # unused in train
            # z-score windows per feature
            z_win = {name: deque(maxlen=400) for name in feats}
            z_series = {name: deque(maxlen=args.window) for name in feats}

        step = 0
        for p in train_files:
            if args.verbose: print(f"[train] {p}")
            for _, feats_row, raw_row in _iter_rows_mapped(p, feat_map, ts_col):
                step += 1
                out = sess.step(feats_row)

                if args.plot:
                    X.append(step)
                    MWL.append(out["mwl"])
                    # per-feature z-score
                    if args.show_raw:
                        for name, col in feat_map.items():
                            v = float(raw_row.get(col, np.nan))
                            if np.isnan(v):  # fallback to feats_row
                                v = float(feats_row[name])
                            z_win[name].append(v)
                            m = float(np.mean(z_win[name])); s = float(np.std(z_win[name])) + 1e-9
                            z_series[name].append((v - m)/s)
                    # refresh occasionally
                    if step % 10 == 0:
                        # register raw lines once they have data
                        if args.show_raw:
                            for name in feats:
                                raw_lines[name].set_data(list(X), list(z_series[name]))
                        _update_plot(ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter,
                                     list(X), z_series, list(MWL), [np.nan]*len(X), spikes_x, spikes_y)
                        ax_bot.set_ylim(0.0, 1.05)
                        if period > 0:
                            time.sleep(period)

        if args.plot:
            print("Training finished. Close the window to exit.")
            plt.ioff(); fig.tight_layout(); plt.show()
        return

    # ---- TEST MODE ----
    if args.mode == "test":
        # warm on config (or --train-file) WITHOUT animation
        for p in train_files:
            if args.verbose: print(f"[warm] {p}")
            for _, feats_row, _ in _iter_rows_mapped(p, feat_map, ts_col):
                sess.step(feats_row)

        if not args.file:
            raise RuntimeError("Test mode requires --file (path to CSV to stream).")
        test_path = Path(args.file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        det = _setup_detector(detcfg, rate_hz)

        if args.plot:
            fig, ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter = _make_fig(
                title=f"Test stream: {test_path.name}", window=args.window, show_raw=args.show_raw, include_growth=True)
            if args.show_raw:
                for name in feats:
                    (raw_lines[name],) = ax_top.plot([], [], label=name, alpha=0.8)
                ax_top.legend(loc="upper right")

            X = deque(maxlen=args.window)
            MWL = deque(maxlen=args.window)
            GPC = deque(maxlen=args.window)
            spikes_x, spikes_y = [], []
            z_win = {name: deque(maxlen=400) for name in feats}
            z_series = {name: deque(maxlen=args.window) for name in feats}

        step = 0
        for _, feats_row, raw_row in _iter_rows_mapped(test_path, feat_map, ts_col):
            step += 1
            out = sess.step(feats_row)
            r = det.update(out["mwl"])

            if args.plot:
                X.append(step)
                MWL.append(out["mwl"])
                if r:
                    GPC.append(r["growth_pct"])
                    if r["spike"]:
                        spikes_x.append(step); spikes_y.append(out["mwl"])
                        print(f"SPIKE @ step {step} (mwl={out['mwl']:.3f}, growth%={r['growth_pct']:.1f})")
                else:
                    GPC.append(np.nan)

                if args.show_raw:
                    for name, col in feat_map.items():
                        v = float(raw_row.get(col, np.nan))
                        if np.isnan(v):
                            v = float(feats_row[name])
                        z_win[name].append(v)
                        m = float(np.mean(z_win[name])); s = float(np.std(z_win[name])) + 1e-9
                        z_series[name].append((v - m)/s)

                if step % 5 == 0:
                    for name in feats:
                        if name in raw_lines:
                            raw_lines[name].set_data(list(X), list(z_series[name]))
                    _update_plot(ax_top, ax_bot, ax_right, raw_lines, mwl_line, gpc_line, spike_scatter,
                                 list(X), z_series, list(MWL), list(GPC), spikes_x, spikes_y)
                    ax_bot.set_ylim(0.0, 1.05)
                    if period > 0:
                        time.sleep(period)

        if args.plot:
            print("Stream ended. Close the window to exit.")
            plt.ioff(); fig.tight_layout(); plt.show()
        return


if __name__ == "__main__":
    main()
