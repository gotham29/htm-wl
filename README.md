<!-- # HTM-WL (Research & BYOD)

Reproducible pipeline for **HTM-based mental workload (MWL)**:
- Engine: **htm.core** (Spatial Pooler + Temporal Memory)
- MWL = **EMA(anomaly)** (smoothed TM anomaly)
- Spike detector: **recent vs prior window % growth**
- Metrics: **Detection lag** (first spike ≥ toggle step) and **Precision** (TP / (TP+FP))

This repo supports:
**Bring-Your-Own-Data (BYOD)** – point to your own CSVs, score runs, and (optionally) evaluate against your ground truth.

---

## 0) Prereqs

- macOS or Linux
- Python 3.11 (Conda *or* venv)
- Build tools for `htm.core`

**macOS**
~~~
xcode-select --install
brew install cmake
~~~

**Ubuntu/Debian**
~~~
sudo apt-get update && sudo apt-get install -y cmake build-essential
~~~

---

## 1) Environment setup

Using venv (works the same with conda):
~~~
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Install **htm.core** into this env:
~~~bash
git clone https://github.com/htm-community/htm.core.git
cd htm.core
python htm_install.py
python -c "import htm; print('htm.core OK at:', htm.__file__)"
cd ..
~~~

> Tip: from the repo root you can optionally `pip install -e .` to install the local package (`htm_wl`) in editable mode.

---

## 2) BYOD (Bring-Your-Own-Data)

## Quick Demo (no private data)

```bash
# create synthetic data -> score -> evaluate
make demo
# results: results_demo/metrics.csv and per-run scores in results_demo/scores/
```

### 2.1 Create a config
~~~
cp config.example.yaml config.yaml
# edit:
#   dataset.train/test globs
#   dataset.features[].column names (your CSVs)
#   dataset.rate_hz OR dataset.timestamp_column
#   detection.windows (steps or seconds) and threshold_pct
~~~

Notes:
- Globs can exclude with `!`, e.g. `- "!data/*/train.csv"`.
- Each test CSV is treated as one run.

### 2.2 Score your files (produce MWL & spikes)
~~~
python -m scripts.score --config config.yaml --outdir results --verbose
~~~

Outputs go to `results/scores/` with filenames like `Subject__file.csv` (prevents collisions).

**Columns produced**

| column        | meaning                                   |
|---------------|-------------------------------------------|
| step          | 1-based step index                        |
| anomaly       | TM anomaly                                |
| mwl           | EMA(anomaly), alpha from config           |
| spike         | 0/1 spike decision                        |
| mr, mp        | recent/prior MWL means (for transparency) |
| growth_pct    | % growth (mr vs mp)                       |
| `<timestamp>` | the timestamp column if you configured it |

### 2.3 Evaluate against ground truth (optional)

**Option A — your nested YAML (subject → files)**  
If you have `data/subjects_testfiles_wltogglepoints.yaml`:
~~~
python -m scripts.eval_from_scores \
  --scores-dir results/scores \
  --gt data/subjects_testfiles_wltogglepoints.yaml \
  --gt-format subjects_yaml \
  --rate-hz 6.67 \
  --out results/metrics.csv
~~~

**Option B — simple CSV**  
Create `my_gt.csv` like:
~~~
file,toggle_step
Crim__run01.csv,123
Moultrie__run07.csv,88
~~~

Then:
~~~
python -m scripts.eval_from_scores \
  --scores-dir results/scores \
  --gt my_gt.csv --gt-format csv \
  --file-col file --toggle-step-col toggle_step \
  --rate-hz 6.67 \
  --out results/metrics.csv
~~~

**Using timestamps instead of steps?**  
Put `toggle_ts` in your GT CSV and pass `--ts-col-in-scores <your_ts_col>`; the evaluator maps to the first step with `ts ≥ toggle_ts`.

---

## 3) How the detector works (transparent math)

Let `nr = recent` window length, `np = prior` window length (in steps or converted from seconds via `rate_hz`).

- `mr = mean(MWL[-nr:])`  
- `mp = mean(MWL[-(nr+np):-nr])`  
- `growth % = 100 * (mr - mp) / max(|mp|, eps)`  
- Spike fires when `growth % > threshold_pct` (configurable direction).

All internals (`mr`, `mp`, `growth_pct`) are exported in the score files to aid calibration.
---

## 4) Repo layout

~~~
htm_wl/               # package (HTM session wrapper, detector, metrics)
scripts/
  make_demo.py        # demo data generator (A→B→C→D)
  plot_scores.py      # quick plots for MWL/spikes/growth%
  score.py            # BYOD: warm + score runs -> per-run scores
  eval_from_scores.py # BYOD: detection lag + precision from scores
config.example.yaml   # template for BYOD
config.demo.yaml      # tuned for the public demo
results*/             # GENERATED (scores, metrics, plots)
data_demo/            # GENERATED (demo data + GT YAML)
~~~

---

## 5) Troubleshooting

**“No score files in results/scores”**  
Debug your test globs quickly:
~~~python
import glob, yaml
c=yaml.safe_load(open('config.yaml')); pats=c['dataset']['test']
inc=[p.lstrip('!') for p in pats if not p.startswith('!')]
exc=[p[1:] for p in pats if p.startswith('!')]
s=set(); [s.update(glob.glob(p)) for p in inc]; [s.difference_update(glob.glob(p)) for p in exc]
print('matched', len(s)); print('\n'.join(sorted(list(s))[:10]))
~~~

**`ModuleNotFoundError: htm`**  
Install **htm.core** in THIS env:
~~~
(cd htm.core && python htm_install.py)
~~~

**It looks “stuck”**  
Warmup may be long; try:
~~~
--limit-train 500
~~~

**Duplicate filenames across subjects**  
Handled by scorer naming as `Subject__file.csv`.

---

## 6) License note

This repo is for research reproduction and BYOD evaluation. The runtime depends on **htm.core** (AGPL-3.0). Do not vendor or modify `htm.core` in this repo; install it into your environment instead.

---

## 7) Citation

If you use this for review or research, please cite the associated paper and this repository. -->


# HTM-WL (BYOD + Demo)

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


