# HTM-WL

**HTM-WL** = HTM anomaly → EMA (MWL) → spike detection.

- Engine: **htm.core** (Spatial Pooler + Temporal Memory)
- MWL = **EMA(anomaly)** (smoothed TM anomaly)
- Spike detector: **recent vs prior mean growth %** with edge + separation controls
- Metrics: **Detection lag** (first spike ≥ toggle step) and **Precision**

This repo supports:

1) **Public Demo** – generate synthetic A→B→C→D sequences (known→novel→known→novel) to see the full pipeline.
2) **Bring-Your-Own-Data (BYOD)** – point to your CSVs, score runs, and (optionally) evaluate against ground truth.

---

## 0) Prereqs

- macOS or Linux
- Python 3.11 (Conda or `venv`)
- Build tools for **htm.core**

**macOS**
    
    xcode-select --install
    brew install cmake

**Ubuntu/Debian**
    
    sudo apt-get update && sudo apt-get install -y cmake build-essential

---

## 1) Environment

Create a virtual environment and install deps:
    
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Install **htm.core** into this env:
    
    git clone https://github.com/htm-community/htm.core.git
    cd htm.core
    python htm_install.py
    python -c "import htm; print('htm.core OK at:', htm.__file__)"
    cd ..

Optional (developer mode):
    
    pip install -e .

---

## 2) Quick Demo (no private data)

Generate synthetic data → score → evaluate → plots:
    
    make demo-clean
    make demo
    make plot-demo

Artifacts:

- `data_demo/` – synthetic CSVs and `subjects_testfiles_wltogglepoints.yaml` (two toggles per file: B and D)
- `results_demo/scores/*.csv` – per-run MWL/spike series
- `results_demo/metrics.csv` – lag/precision per file
- `results_demo/plots/*.png` – MWL, spike markers, growth %, and optional raw inputs (z-scored)

Tune thresholds/windowing in **`config.demo.yaml`** under `detection:*`.

---

## 3) BYOD (Bring-Your-Own-Data)

### 3.1 Create a config
    
    cp config.example.yaml config.yaml
    # edit:
    #   dataset.train/test globs
    #   dataset.features[].column names (your CSVs)
    #   dataset.rate_hz OR dataset.timestamp_column
    #   detection.windows (steps or seconds) and threshold_pct

Notes:

- Globs can exclude with `!` (e.g., `- "!data/*/train.csv"`).
- Each test CSV is treated as one run.

### 3.2 Score your files (produce MWL & spikes)
    
    python -m scripts.score --config config.yaml --outdir results --verbose

Outputs go to `results/scores/` as `Subject__file.csv` (prevents collisions).

**Columns produced**

| column      | meaning                               |
|-------------|----------------------------------------|
| step        | 1-based step index                     |
| anomaly     | TM anomaly                             |
| mwl         | EMA(anomaly), alpha from config        |
| spike       | 0/1 spike decision                     |
| mr, mp      | recent/prior MWL means (for debugging) |
| growth_pct  | % growth (mr vs mp)                    |
| `<timestamp>` | timestamp column if configured       |

### 3.3 Evaluate against ground truth (optional)

**Option A — nested subjects YAML**
    
    python -m scripts.eval_from_scores \
      --scores-dir results/scores \
      --gt data/subjects_testfiles_wltogglepoints.yaml \
      --gt-format subjects_yaml \
      --rate-hz 6.67 \
      --out results/metrics.csv

**Option B — simple CSV**
    
    # my_gt.csv (example)
    # file,toggle_step
    # DemoA__test_01.csv,400
    # DemoB__test_02.csv,500
    
    python -m scripts.eval_from_scores \
      --scores-dir results/scores \
      --gt my_gt.csv --gt-format csv \
      --file-col file --toggle-step-col toggle_step \
      --rate-hz 6.67 \
      --out results/metrics.csv

**Using timestamps instead of steps?** Put `toggle_ts` in your GT and pass `--ts-col-in-scores <your_ts_col>`; the evaluator maps to the first step with `ts ≥ toggle_ts`.

### 3.4 Live visualization (train/test)

Run interactive, step-by-step plots to illustrate how HTM-WL learns on training data and reacts to test streams.

**Train-only animation** (no spikes/growth% in the bottom panel):
    
    python -m scripts.live_demo --mode train \
      --config config.yaml \
      --train-file path/to/your/train.csv \
      --rate-hz 10.0

**Test stream** (warms on `dataset.train` or `--train-file`, then streams the test CSV):
    
    python -m scripts.live_demo --mode test \
      --config config.yaml \
      --file path/to/your/test.csv \
      --rate-hz 10.0 --speed 2.0

**What you’ll see**
- **Top subplot**: each configured feature from `dataset.features` as a **rolling z-score** (helps relate input changes to MWL spikes).
- **Bottom subplot** (fixed y-axis **0–1**):
  - **MWL** in **blue**
  - **growth%** in **orange** (test mode only)
  - **spikes** as **red** dots (test mode only)

**Tips**
- To warm on a single training file in test mode, add `--train-file path/to/train.csv`.
- Adjust detector behavior in your config via `detection.*` (e.g., `windows.recent/prior`, `threshold_pct`, `edge_only`, `min_separation`, `min_delta`, `eps`).
- CLI switches:
  - `--rate-hz <Hz>`: playback rate; `--speed <multiplier>`: speed up/slow down display
  - `--window <N>`: number of points kept on screen
  - `--no-show-raw`: hide the top inputs panel
  - `--no-plot`: disable plotting (useful for dry runs)

---

## 4) How the detector works

Let `nr = recent` window and `np = prior` window (steps, or seconds converted via `rate_hz`):

- `mr = mean(MWL[-nr:])`
- `mp = mean(MWL[-(nr+np):-nr])`
- `growth % = 100 * (mr - mp) / max(|mp|, eps)`
- Spike fires when `growth %` crosses the threshold (edge-only + min-separation are configurable).

Internals (`mr`, `mp`, `growth_pct`) are exported in score files.

---

## 5) Repo layout

    htm_wl/               # package (HTM session wrapper, detector, metrics)
    scripts/
      make_demo.py        # demo data generator (A→B→C→D)
      plot_scores.py      # quick plots for MWL/spikes/growth%
      score.py            # BYOD: warm + score runs -> per-run scores
      eval_from_scores.py # BYOD: detection lag + precision from scores
    config.example.yaml   # template for BYOD configs
    config.demo.yaml      # tuned for the public demo
    results*/             # GENERATED (scores, metrics, plots)
    data_demo/            # GENERATED (demo data + GT YAML)

---

## 6) Troubleshooting

**“No score files in results/scores”** — validate your test globs quickly:

    import glob, yaml
    c=yaml.safe_load(open('config.yaml')); pats=c['dataset']['test']
    inc=[p.lstrip('!') for p in pats if not p.startswith('!')]
    exc=[p[1:] for p in pats if p.startswith('!')]
    s=set(); [s.update(glob.glob(p)) for p in inc]; [s.difference_update(glob.glob(p)) for p in exc]
    print('matched', len(s)); print('\n'.join(sorted(list(s))[:10]))

**`ModuleNotFoundError: htm`** — install `htm.core` in this env:

    (cd htm.core && python htm_install.py)

**Looks “stuck”** — warmup may be long; reduce train size or adjust TM params; try smaller windows.

**Duplicate filenames across subjects** — handled by scorer naming as `Subject__file.csv`.

---

## 7) License

This repo is for BYOD evaluation and a public demo. Runtime depends on **htm.core** (AGPL-3.0). Do **not** vendor or modify `htm.core` here; install it into your environment.

---

## 8) Citation

If you use this repo for review or research, please cite the associated paper and this repository.


