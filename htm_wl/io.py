from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml, csv

def load_pipeline_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)

def extract_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Fallbacks chosen to match your previously shared defaults
    hz = float(cfg.get("hzs", {}).get("convertto", 6.67))
    enc_n = int(cfg.get("htm_config_model", {}).get("models_encoders", {}).get("n", 400))
    sp = cfg.get("htm_config_model", {}).get("models_params", {}).get("sp", {})
    tm = cfg.get("htm_config_model", {}).get("models_params", {}).get("tm", {})
    win = cfg.get("windows_ascore", {})
    recent = int(win.get("recent", 21))
    prior  = int(win.get("previous", 42))
    thresh = float(win.get("change_thresh_percent", 200.0))
    seed   = int(sp.get("seed", 0))

    return {
        "rate_hz": hz,
        "enc_n_per_feature": enc_n,
        "enc_w_per_feature": max(3, int(0.05 * enc_n)),  # simple width rule
        "sp_params": {
            "potentialPct": float(sp.get("potentialPct", 0.8)),
            "synPermActiveInc": float(sp.get("synPermActiveInc", 0.003)),
            "synPermInactiveDec": float(sp.get("synPermInactiveDec", 0.0005)),
            "synPermConnected": float(sp.get("synPermConnected", 0.2)),
            "boostStrength": float(sp.get("boostStrength", 0.0)),
            "columnCount": int(sp.get("columnCount", 2048)),
            "globalInhibition": bool(sp.get("globalInhibition", True)),
        },
        "tm_params": {
            "cellsPerColumn": int(tm.get("cellsPerColumn", 32)),
            "activationThreshold": int(tm.get("activationThreshold", 20)),
            "minThreshold": int(tm.get("minThreshold", 13)),
            "newSynapseCount": int(tm.get("newSynapseCount", 31)),
            "permanenceInc": float(tm.get("permanenceInc", 0.10)),
            "permanenceDec": float(tm.get("permanenceDec", 0.0)),
            "initialPerm": float(tm.get("initialPerm", 0.21)),
            "permanenceConnected": float(tm.get("permanenceConnected", 0.5)),
            "predictedSegmentDecrement": float(tm.get("predictedSegmentDecrement", 0.001)),
        },
        "spike_params": {
            "recent_count": recent,
            "prior_count": prior,
            "growth_threshold_pct": thresh,
        },
        "seed": seed,
        # Required features for the paper
        "feature_names": ["ROLL_STICK", "PITCH_STIC"],
    }

def load_ground_truth(cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Returns: { "<Subject>/<testfile.csv>": {"toggle_step": int, "total_steps": int} }
    """
    out = {}
    subjects = cfg.get("subjects_testfiles_wltogglepoints", {})
    for subject, files in subjects.items():
        for fname, meta in files.items():
            key = f"{subject}/{fname}"
            out[key] = {
                "toggle_step": int(meta["times_wltoggle"][0]),
                "total_steps": int(meta["time_total"]),
            }
    return out

def iter_rows(csv_path: Path, feature_names: List[str]):
    with csv_path.open() as f:
        rd = csv.DictReader(f)
        # Be permissive: accept PITCH_STIC or PITCH_STICK
        for row in rd:
            feats = {}
            for name in feature_names:
                if name in row:
                    feats[name] = float(row[name])
                elif name == "PITCH_STIC" and "PITCH_STICK" in row:
                    feats[name] = float(row["PITCH_STICK"])
                else:
                    raise KeyError(f"Missing column {name} in {csv_path.name}")
            yield feats
