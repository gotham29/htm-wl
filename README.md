# HTM-WL (Research)

This repo reproduces the HTM-WL pipeline and metrics reported in the paper:
- HTM core (SP/TM) via **htm.core**
- MWL = EMA(anomaly)
- Spike detector = recent/prior window % growth
- Metrics: detection-lag (first spike ≥ toggle_step), precision (TP / (TP+FP))

## 0) Prereqs

- macOS or Linux
- **Conda** (recommended) or Python 3.11
- Xcode CLT (macOS): `xcode-select --install`
- CMake (for htm.core): `brew install cmake` (macOS) or `sudo apt-get install cmake`

## 1) Create environment

```bash
conda env create -f environment.yml
conda activate htm_wl
