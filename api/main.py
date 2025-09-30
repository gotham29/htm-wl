#!/usr/bin/env python
# api/main.py
"""
Minimal FastAPI for HTM-WL demo

Run:
  uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /health
  GET  /version
  POST /score-file   (multipart: test_file; optional: train_file) -> CSV scores
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

from pathlib import Path
import csv
import io
from typing import Dict, Optional

import yaml
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from htm_wl.htm_session import HTMSession
from htm_wl.spike_detector import SpikeDetector


REPO_ROOT = Path(".").resolve()
app = FastAPI(title="HTM-WL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")
# drop a 16x16 favicon.ico in api/static and add:
from fastapi.responses import FileResponse
@app.get("/favicon.ico")
def favicon():
    return FileResponse("api/static/favicon.ico")

def load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def iter_rows_mapped_bytes(buf: bytes, feature_map: Dict[str, str], ts_col: Optional[str]):
    text = io.StringIO(buf.decode("utf-8"))
    rd = csv.DictReader(text)
    for row in rd:
        feats = {}
        for name, col in feature_map.items():
            if col in row:
                feats[name] = float(row[col])
            elif name == "PITCH_STIC" and "PITCH_STICK" in row:
                feats[name] = float(row["PITCH_STICK"])
            else:
                raise HTTPException(status_code=400, detail=f"Missing column {col} (feature {name})")
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

@app.get("/")
def root():
    return {"service": "htm-wl", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    # Optionally load from package metadata later
    return {"version": app.version}

@app.post("/score-file", response_class=PlainTextResponse)
async def score_file(
    test_file: UploadFile = File(..., description="CSV to score"),
    train_file: UploadFile | None = File(None, description="Optional CSV to warm on"),
    config_path: str = Form("config.demo.yaml"),
    rate_hz: float = Form(6.67),
):
    """
    Returns a CSV with columns: step, anomaly, mwl, growth_pct, spike, [<timestamp>]
    """
    cfg_file = REPO_ROOT / config_path
    if not cfg_file.exists():
        raise HTTPException(status_code=400, detail=f"Config not found: {cfg_file}")
    cfg = load_cfg(cfg_file)

    # Build session + optional warmup
    sess = build_session(cfg)
    ds = cfg["dataset"]
    feats = [f["name"] for f in ds["features"]]
    feat_map = {f["name"]: f["column"] for f in ds["features"]}
    ts_col = ds.get("timestamp_column")

    if train_file is not None:
        train_bytes = await train_file.read()
        for _, feats_row, _ in iter_rows_mapped_bytes(train_bytes, feat_map, ts_col):
            sess.step(feats_row)

    det = build_detector(cfg, rate_hz=float(rate_hz))

    test_bytes = await test_file.read()
    rows = []
    step = 0
    for _, feats_row, raw_row in iter_rows_mapped_bytes(test_bytes, feat_map, ts_col):
        step += 1
        out = sess.step(feats_row)
        r = det.update(out["mwl"])
        rec = {
            "step": step,
            "anomaly": out["anomaly"],
            "mwl": out["mwl"],
            "growth_pct": (r["growth_pct"] if r else None),
            "spike": int(bool(r and r["spike"])),
        }
        if ts_col:
            rec[ts_col] = raw_row.get(ts_col, None)
        rows.append(rec)

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
