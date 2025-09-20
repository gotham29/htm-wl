# HTM-WL (Research & BYOD)

Reproducible pipeline for **HTM-based mental workload (MWL)**:
- Engine: **htm.core** (Spatial Pooler + Temporal Memory)
- MWL = **EMA(anomaly)** (smoothed TM anomaly)
- Spike detector: **recent vs prior window % growth**
- Metrics: **Detection lag** (first spike ≥ toggle step) and **Precision** (TP / (TP+FP))

This repo supports two flows:
1) **Paper Reproduction** – run on the provided dataset using `run_pipeline.yaml`.
2) **Bring-Your-Own-Data (BYOD)** – point to your own CSVs, score runs, and (optionally) evaluate against your ground truth.

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

## 2) Paper Reproduction

**Put files in place**
- `run_pipeline.yaml` at repo root (already here)
- Dataset under `./data/` (subject folders with `train.csv` and test CSVs)

**Run**
~~~
python -m scripts.evaluate \
  --config run_pipeline.yaml \
  --data-root data \
  --outdir results \
  --verbose \
  --limit-train 500 \
  --max-files 5
~~~

Remove the speed-limit flags for a full run.

**Output:** `results/summary.csv` with columns  
`subject,file,toggle_step,spikes,detection_lag_steps,detection_lag_seconds,precision`.

---

## 3) BYOD (Bring-Your-Own-Data)

### 3.1 Create a config
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

### 3.2 Score your files (produce MWL & spikes)
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

### 3.3 Evaluate against ground truth (optional)

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

## 4) How the detector works (transparent math)

Let `nr = recent` window length, `np = prior` window length (in steps or converted from seconds via `rate_hz`).

- `mr = mean(MWL[-nr:])`  
- `mp = mean(MWL[-(nr+np):-nr])`  
- `growth % = 100 * (mr - mp) / mp`  
- Spike fires when `growth % > threshold_pct`.

All internals (`mr`, `mp`, `growth_pct`) are exported in the score files to aid calibration.

---

## 5) Repo layout

~~~
htm_wl/               # package (engine wrapper, detector, metrics)
scripts/
  evaluate.py         # paper reproduction (uses run_pipeline.yaml)
  score.py            # BYOD: warm + score runs, writes per-run scores
  eval_from_scores.py # BYOD: compute detection lag + precision from scores
config.example.yaml   # template for BYOD configs (copy to config.yaml)
run_pipeline.yaml     # original paper pipeline config (repro flow)
results/              # GENERATED: per-run scores and summary metrics
data/                 # your datasets (not versioned)
~~~

---

## 6) Troubleshooting

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

## 7) License note

This repo is for research reproduction and BYOD evaluation. The runtime depends on **htm.core** (AGPL-3.0). Do not vendor or modify `htm.core` in this repo; install it into your environment instead.

---

## 8) Citation

If you use this for review or research, please cite the associated paper and this repository.
