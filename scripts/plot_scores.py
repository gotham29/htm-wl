#!/usr/bin/env python
"""
Plot MWL (EMA of anomaly), spike markers, growth %, and (optionally) raw inputs
for quick visual sanity checks.

Usage examples:
  python -m scripts.plot_scores --scores-dir results_demo/scores \
    --gt data_demo/subjects_testfiles_wltogglepoints.yaml --gt-format subjects_yaml \
    --data-root data_demo --outdir results_demo/plots --show-raw

  # BYOD (no GT, no raw)
  python -m scripts.plot_scores --scores-dir results/scores --outdir results/plots
"""
from pathlib import Path
import argparse
import re
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def _load_gt_subjects_yaml(path: Path) -> dict[str, list[int]]:
    with path.open("r") as f:
        y = yaml.safe_load(f)
    out = {}
    for subj, files in y.items():
        for fname, meta in files.items():
            vals = meta.get("times_wltoggle", [])
            vals = vals if isinstance(vals, list) else [vals]
            out[f"{subj}__{Path(fname).stem}.csv"] = [int(v) for v in vals]
    return out


def _try_load_raw(data_root: Path, subject: str, file_stem: str) -> pd.DataFrame | None:
    # Expect data_root/<subject>/<file_stem>.csv
    p = data_root / subject / f"{file_stem}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def main():
    ap = argparse.ArgumentParser(description="Plot MWL, spikes, growth %, and optional raw features.")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--gt", default=None, help="Ground truth file (subjects_yaml or csv/yaml).")
    ap.add_argument("--gt-format", choices=["subjects_yaml", "csv", "yaml"], default="subjects_yaml")
    ap.add_argument("--data-root", default=None, help="Root to find raw CSVs for plotting (optional).")
    ap.add_argument("--show-raw", action="store_true", help="Overlay raw ROLL_STICK/PITCH_STIC (z-scored).")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ground truth (toggle steps)
    gt_steps = {}
    if args.gt:
        if args.gt_format == "subjects_yaml":
            gt_steps = _load_gt_subjects_yaml(Path(args.gt))
        else:
            # Simple CSV/YAML formats left as future extension
            raise NotImplementedError("Only --gt-format subjects_yaml supported in this plotter for now.")

    data_root = Path(args.data_root) if args.data_root else None

    for f in sorted(scores_dir.glob("*.csv")):
        df = pd.read_csv(f)
        name = f.name
        m = re.match(r"(.+?)__(.+)\.csv$", name)
        subject, file_stem = (m.group(1), m.group(2)) if m else ("", f.stem)

        # Build the plot: MWL with spikes and growth% on a second axis
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.set_title(name)

        # MWL
        ax1.plot(df["step"], df["mwl"], label="MWL")
        ax1.set_xlabel("step")
        ax1.set_ylabel("MWL (EMA of anomaly)")

        # Spike markers
        spikes_idx = df.index[df["spike"] == 1].to_list()
        if spikes_idx:
            ax1.scatter(df.loc[spikes_idx, "step"], df.loc[spikes_idx, "mwl"], marker="o", s=18, label="Spike")

        # Toggle step (if provided)
        tsteps = gt_steps.get(name, [])
        for k, t in enumerate(tsteps):
            ax1.axvline(t, linestyle="--", label=("toggle" if k == 0 else f"toggle{1+k}") + f"={t}")

        # Growth % on a twin axis (helps debug thresholding)
        if "growth_pct" in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df["step"], df["growth_pct"], alpha=0.6, label="growth%")
            ax2.set_ylabel("growth %")

        # Optional raw inputs (z-scored)
        if args.show_raw and data_root:
            raw = _try_load_raw(data_root, subject, file_stem)
            if raw is not None:
                for col in [c for c in raw.columns if c.upper() in ("ROLL_STICK", "PITCH_STIC", "PITCH_STICK")]:
                    z = (raw[col] - raw[col].mean()) / (raw[col].std() + 1e-9)
                    ax1.plot(range(1, len(z) + 1), z, alpha=0.5, label=f"{col} (z)")

        # Legend (combine both axes)
        handles1, labels1 = ax1.get_legend_handles_labels()
        if "growth_pct" in df.columns:
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        outp = outdir / (f.stem + ".png")
        fig.tight_layout()
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        print(f"[plot] {name} → {outp}")

    print(f"Done. Plots are in: {outdir}")


if __name__ == "__main__":
    main()
