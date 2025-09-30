#!/usr/bin/env python
# app/app.py
"""
Streamlit demo UI for HTM-WL:
- Load or generate demo data
- Configure model & detector
- Train-only animation
- Test streaming with live MWL/growth%/spikes
- Export scores and (if GT present) metrics

Run:
  streamlit run app/app.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
from pathlib import Path
import time
import io
import glob
import csv
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector


# ---------- small helpers ----------
REPO_ROOT = Path(".").resolve()

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def expand_globs(patterns: List[str] | None) -> List[Path]:
    patterns = patterns or []
    inc, exc = [], []
    for p in patterns:
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

def iter_rows_mapped(csv_path: Path, feature_map: Dict[str, str], ts_col: Optional[str]):
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
            yield ts, feats, row

def build_session(cfg: dict) -> HTMSession:
    ds, mdl = cfg["dataset"], cfg["model"]
    feats = [f["name"] for f in ds["features"]]
    ranges = {f["name"]: {"min": f.get("min", -1.0), "max": f.get("max", 1.0)} for f in ds["features"]}
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

def build_detector(cfg: dict, rate_hz: float) -> SpikeDetector:
    detcfg = cfg["detection"]
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

def generate_demo_if_missing():
    demo_root = REPO_ROOT / "data_demo"
    if demo_root.exists():
        return
    # import and call the demo generator with defaults
    from scripts.make_demo import main as make_demo_main
    # mimic CLI defaults
    st.info("Generating demo dataset…")
    make_demo_main()
    st.success("Demo data generated at data_demo/")


# ---------- Streamlit UI ----------
st.set_page_config(page_title="HTM-WL Demo", layout="wide")
st.title("HTM-WL Demo")

with st.sidebar:
    st.header("Setup")
    # Config file
    default_cfg = (REPO_ROOT / "config.demo.yaml") if (REPO_ROOT / "config.demo.yaml").exists() \
                  else (REPO_ROOT / "config.yaml")
    cfg_choice = st.radio("Config source", ["Use repo config", "Upload YAML"], horizontal=False)
    if cfg_choice == "Upload YAML":
        up = st.file_uploader("Upload config YAML", type=["yaml", "yml"])
        if up:
            cfg = yaml.safe_load(up.getvalue())
        else:
            st.stop()
    else:
        if not default_cfg.exists():
            st.error("No config.demo.yaml or config.yaml found.")
            st.stop()
        cfg = load_yaml(default_cfg)

    # Data choice (demo vs BYOD)
    data_choice = st.radio("Data source", ["Demo (data_demo)", "BYOD"], horizontal=True)
    if data_choice == "Demo (data_demo)":
        if st.button("Generate demo (if missing)"):
            generate_demo_if_missing()
        data_root = REPO_ROOT / "data_demo"
    else:
        dr = st.text_input("Data root folder", "data")
        data_root = Path(dr)

    # Choose subject (folder under data root)
    subjects = sorted([p.name for p in data_root.glob("*") if p.is_dir()])
    subject = st.selectbox("Subject folder", subjects) if subjects else None

    # Train & Test file selection
    if subject:
        train_default = data_root / subject / "train.csv"
        train_file = st.text_input("Train CSV path", str(train_default if train_default.exists() else ""))
        test_candidates = [p for p in (data_root / subject).glob("*.csv") if p.name != "train.csv"]
        test_file = st.selectbox("Test CSV", [str(p) for p in test_candidates]) if test_candidates else ""

    # Playback
    rate_hz = float(cfg["dataset"].get("rate_hz", 6.67))
    rate_hz = st.number_input("Playback rate (Hz)", value=rate_hz)
    speed = st.slider("Speed multiplier", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    window = st.number_input("Plot window (points)", value=2000, step=100)

    # Detector controls (bind back into cfg)
    st.header("Detector")
    det = cfg["detection"]
    units = st.selectbox("Window units", ["steps", "seconds"], index=0 if det["windows"]["units"]=="steps" else 1)
    recent = st.number_input("Recent window", value=float(det["windows"]["recent"]), step=1.0)
    prior  = st.number_input("Prior window",  value=float(det["windows"]["prior"]),  step=1.0)
    thresh = st.number_input("Threshold %",   value=float(det["threshold_pct"]), step=10.0)
    edge_only = st.checkbox("Edge only", value=bool(det.get("edge_only", True)))
    min_sep   = st.number_input("Min separation (steps)", value=int(det.get("min_separation", 0)), step=1)
    min_mwl   = st.number_input("Min MWL to consider", value=float(det.get("min_mwl", 0.0)), step=0.01, format="%.2f")

    det["windows"]["units"] = units
    det["windows"]["recent"] = recent
    det["windows"]["prior"]  = prior
    det["threshold_pct"] = thresh
    det["edge_only"] = edge_only
    det["min_separation"] = int(min_sep)
    det["min_mwl"] = float(min_mwl)

    st.caption("Tip: tweak threshold/windows and re-run streaming without rebuilding the model.")

# Keep these in session
if "cfg" not in st.session_state:
    st.session_state.cfg = cfg
else:
    st.session_state.cfg = cfg

# Features & mapping
ds = cfg["dataset"]
features = [f["name"] for f in ds["features"]]
feat_map = {f["name"]: f["column"] for f in ds["features"]}
ts_col = ds.get("timestamp_column")

# Tabs
tab_train, tab_test, tab_export = st.tabs(["Train (animate)", "Test (stream)", "Export / Metrics"])

# --------- Train tab ----------
with tab_train:
    st.subheader("Train-only animation")
    if not subject or not train_file:
        st.info("Pick a subject and a train file in the sidebar.")
        st.stop()

    start_train = st.button("Build model & animate training")
    train_placeholder = st.empty()

    if start_train:
        # Build fresh session
        sess = build_session(cfg)
        st.session_state.sess = sess

        # Live Plotly figure: 2 rows (MWL top, inputs bottom)
        fig_train = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.45, 0.55]
        )
        # MWL trace (row 1)
        fig_train.add_trace(
            go.Scatter(name="MWL", mode="lines", line=dict(color="#3366ff", width=2)),
            row=1, col=1
        )
        fig_train.update_yaxes(title_text="MWL (0–1)", range=[0, 1], row=1, col=1)
        fig_train.update_xaxes(title_text="step", row=2, col=1)
        # Input traces (row 2)
        input_trace_idx = {}
        input_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
        for i, name in enumerate(features):
            fig_train.add_trace(
                go.Scatter(name=name, mode="lines",
                           line=dict(color=input_colors[i % len(input_colors)], width=1),
                           opacity=0.9),
                row=2, col=1
            )
            input_trace_idx[name] = 1 + i  # index in fig.data (0 is MWL)
        plot_train = train_placeholder.plotly_chart(fig_train, use_container_width=True)


        # Rolling z-score buffers
        z_win = {name: deque(maxlen=400) for name in features}
        z_series = {name: [] for name in features}
        steps = []

        for step, (_, feats_row, raw_row) in enumerate(iter_rows_mapped(Path(train_file), feat_map, ts_col), start=1):
            out = sess.step(feats_row)
            steps.append(step)

            # Update MWL trace
            fig_train.data[0].x = steps
            fig_train.data[0].y = [*fig_train.data[0].y, out["mwl"]] if len(fig_train.data[0].y) else [out["mwl"]]

            # Build/refresh inputs (z) table occasionally
            for name, col in feat_map.items():
                v = float(raw_row.get(col, np.nan))
                if np.isnan(v): v = float(feats_row[name])
                z_win[name].append(v)
                m = float(np.mean(z_win[name])); s = float(np.std(z_win[name])) + 1e-9
                z_series[name].append((v - m)/s)

            if step % 10 == 0:
                # Update all input traces with full history (no 200-cap)
                for name in features:
                    idx = input_trace_idx[name]
                    fig_train.data[idx].x = steps[:len(z_series[name])]
                    fig_train.data[idx].y = z_series[name]
                plot_train.plotly_chart(fig_train, use_container_width=True)
            time.sleep(max(0.0, (1.0 / max(rate_hz, 1e-9)) / max(speed, 1e-9)))

        st.success("Training pass complete. Switch to the Test tab to stream.")

# --------- Test tab ----------
with tab_test:
    st.subheader("Test streaming (live MWL, growth%, spikes)")
    if "sess" not in st.session_state:
        st.info("Build the model on the Train tab first (or reuse a previous session).")
    elif not test_file:
        st.info("Pick a test CSV in the sidebar.")
    else:
        start_test = st.button("Stream test file")
        test_placeholder_top = st.empty()
        test_placeholder_bot = st.empty()
        msg_placeholder = st.empty()

        if start_test:
            sess: HTMSession = st.session_state.sess
            detector = build_detector(cfg, rate_hz)

            # Prepare live charts
            top_df = pd.DataFrame(columns=features)
            top_chart = test_placeholder_top.line_chart(top_df)

            # Bottom (dual y-axes): MWL (left, 0–1) & growth% (right)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=[], y=[], name="MWL", mode="lines",
                           line=dict(color="#3366ff", width=2)),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=[], y=[], name="growth%", mode="lines",
                           line=dict(color="#ff7f0e", width=1.5)),
                secondary_y=True
            )
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="markers", name="Spike",
                           marker=dict(color="#d62728", size=7, symbol="circle")),
                secondary_y=False
            )
            fig.update_yaxes(title_text="MWL (0–1)", range=[0, 1], secondary_y=False)
            fig.update_yaxes(title_text="growth %", secondary_y=True, rangemode="tozero")
            fig.update_xaxes(title_text="step")
            plot_slot = test_placeholder_bot.plotly_chart(fig, use_container_width=True, key="bot-plot")
            xs, mwl_arr, growth_arr, sx, sy = [], [], [], [], []

            # Storage for export
            rows = []
            spikes = []
            step = 0

            # Rolling z stats for inputs
            z_win = {name: deque(maxlen=400) for name in features}
            z_series = {name: [] for name in features}

            for _, feats_row, raw_row in iter_rows_mapped(Path(test_file), feat_map, ts_col):
                step += 1
                out = sess.step(feats_row)
                r = detector.update(out["mwl"])

                # update inputs (z)
                z_row = {}
                for name, col in feat_map.items():
                    v = float(raw_row.get(col, np.nan))
                    if np.isnan(v): v = float(feats_row[name])
                    z_win[name].append(v)
                    m = float(np.mean(z_win[name])); s = float(np.std(z_win[name])) + 1e-9
                    z_val = (v - m)/s
                    z_row[name] = z_val
                    z_series[name].append(z_val)
                top_chart.add_rows(pd.DataFrame([z_row]))

                # bottom: update Plotly dual-axis figure
                gpc = (r["growth_pct"] if r else None)
                xs.append(step); mwl_arr.append(out["mwl"]); growth_arr.append(gpc)
                fig.data[0].x = xs;        fig.data[0].y = mwl_arr          # MWL (left axis)
                fig.data[1].x = xs;        fig.data[1].y = growth_arr       # growth% (right axis)
                fig.data[2].x = sx;        fig.data[2].y = sy               # spikes (markers)
                if step % 5 == 0:
                    plot_slot.plotly_chart(fig, use_container_width=True)

                spike_flag = int(bool(r and r["spike"]))
                if spike_flag:
                    spikes.append(step)
                    sx.append(step); sy.append(out["mwl"])
                    msg_placeholder.success(f"SPIKE @ step {step} (mwl={out['mwl']:.3f}, growth%={(gpc or 0):.1f})")
                rec = {"step": step, "anomaly": out["anomaly"], "mwl": out["mwl"], "growth_pct": gpc, "spike": spike_flag}
                if ts_col: rec[ts_col] = raw_row.get(ts_col, None)
                rows.append(rec)

                time.sleep(max(0.0, (1.0 / max(rate_hz, 1e-9)) / max(speed, 1e-9)))

            st.session_state.last_scores = pd.DataFrame(rows)
            st.session_state.last_spikes = spikes
            st.success("Streaming complete. See Export / Metrics tab for downloads.")

# --------- Export / Metrics tab ----------
with tab_export:
    st.subheader("Exports")
    if "last_scores" in st.session_state:
        df = st.session_state.last_scores
        st.dataframe(df.head(10))
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("Download scores CSV", data=csv_buf.getvalue(), file_name="scores.csv", mime="text/csv")
    else:
        st.caption("Run a test stream to enable exports.")

    # Optional quick metrics if GT YAML exists for this subject
    if subject:
        gt_yaml = data_root / "subjects_testfiles_wltogglepoints.yaml"
        if gt_yaml.exists() and "last_scores" in st.session_state:
            # Map toggles for this selected file (by stem)
            selected_stem = Path(test_file).stem if test_file else ""
            with gt_yaml.open("r") as f:
                y = yaml.safe_load(f)
            # Find toggles for this subject/file
            toggles = []
            if subject in y:
                for fname, meta in y[subject].items():
                    if Path(fname).stem == selected_stem:
                        vals = meta.get("times_wltoggle", [])
                        toggles = vals if isinstance(vals, list) else [vals]
                        break
            if toggles:
                df = st.session_state.last_scores
                spikes_idx = df.index[df["spike"] == 1].tolist()
                lags = []
                for t in toggles:
                    later_spikes = [i+1 for i in spikes_idx if (i+1) >= int(t)]
                    lags.append((later_spikes[0] - int(t)) if later_spikes else None)
                hz = cfg["dataset"].get("rate_hz", None)
                lags_sec = [ (None if (l is None or hz is None) else round(l/float(hz), 2)) for l in lags ]
                st.write("Detection lags (steps):", lags)
                st.write("Detection lags (seconds):", lags_sec)
                prec = (len(spikes_idx) and len([s for s in spikes_idx if any((s+1)>=int(t) for t in toggles)]))
                st.caption("For authoritative metrics across many files, use scripts.eval_from_scores.")
        else:
            st.caption("Ground-truth YAML not found (optional).")
