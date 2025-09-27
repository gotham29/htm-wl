#!/usr/bin/env python
"""
Generate a tiny synthetic dataset with clear pattern shifts.

- Subjects: DemoA, DemoB (configurable)
- Each subject gets:
  - train.csv (baseline dynamics)
  - test_01.csv, test_02.csv with a mid-run shift (toggle_step)
- GT written to: data_demo/subjects_testfiles_wltogglepoints.yaml
- Columns: ROLL_STICK, PITCH_STIC (compatible with current pipeline)
"""
from pathlib import Path
import argparse, math, random, yaml
import pandas as pd
import numpy as np

def _mk_run(n_steps, rate_hz, phase, amp, noise, kind="baseline"):
    # two correlated sinusoids with slight phase offset;
    # after toggle we change amplitude/freq/noise -> anomaly spikes expected
    t = np.arange(n_steps) / rate_hz
    freq = 0.4 if kind == "baseline" else 0.9
    ampR = amp if kind == "baseline" else amp * 1.8
    ampP = amp * 0.8 if kind == "baseline" else amp * 2.0
    ns = noise if kind == "baseline" else noise * 2.2
    roll  = ampR * np.sin(2*np.pi*freq*t + phase) + np.random.normal(0, ns, size=n_steps)
    pitch = ampP * np.sin(2*np.pi*freq*t + (phase+0.6)) + np.random.normal(0, ns, size=n_steps)
    df = pd.DataFrame({"ROLL_STICK": roll, "PITCH_STIC": pitch})
    return df

def make_subject(out_dir: Path, name: str, rate_hz: float, train_len: int, test_len: int, toggle_step: int, seed: int):
    rng = random.Random(seed)
    phase = rng.uniform(0, 2*math.pi)
    amp, noise = 0.7, 0.05

    subj_dir = out_dir / name
    subj_dir.mkdir(parents=True, exist_ok=True)

    # train (baseline only)
    train = _mk_run(train_len, rate_hz, phase, amp, noise, kind="baseline")
    train.to_csv(subj_dir / "train.csv", index=False)

    # test files with a mid-run shift
    gt = {}
    for i in [1, 2]:
        df1 = _mk_run(toggle_step-1, rate_hz, phase, amp, noise, kind="baseline")
        df2 = _mk_run(test_len - (toggle_step-1), rate_hz, phase, amp, noise, kind="shift")
        test = pd.concat([df1, df2], ignore_index=True)
        fname = f"test_{i:02d}.csv"
        test.to_csv(subj_dir / fname, index=False)
        gt[fname] = {"times_wltoggle": [toggle_step], "time_total": int(test_len)}
    return gt

def main():
    ap = argparse.ArgumentParser(description="Make a small synthetic dataset with pattern shifts.")
    ap.add_argument("--out", default="data_demo", help="Output root folder.")
    ap.add_argument("--subjects", type=int, default=2)
    ap.add_argument("--rate-hz", type=float, default=6.67)
    ap.add_argument("--train-len", type=int, default=1200)
    ap.add_argument("--test-len", type=int, default=1000)
    ap.add_argument("--toggle-step", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    subject_names = [f"Demo{chr(ord('A')+i)}" for i in range(args.subjects)]
    top_yaml = {}
    for i, name in enumerate(subject_names):
        gt = make_subject(
            out_root, name,
            rate_hz=args.rate_hz,
            train_len=args.train_len,
            test_len=args.test_len,
            toggle_step=args.toggle_step,
            seed=args.seed + i,
        )
        top_yaml[name] = gt

    # write GT file
    gt_path = out_root / "subjects_testfiles_wltogglepoints.yaml"
    with gt_path.open("w") as f:
        yaml.safe_dump(top_yaml, f, sort_keys=True)
    print(f"[demo] wrote ground truth → {gt_path}")

if __name__ == "__main__":
    main()
