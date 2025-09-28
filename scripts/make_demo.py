#!/usr/bin/env python
"""
Generate a synthetic, multi-phase dataset for the HTM-WL demo.

- Training contains two known patterns: A and C (concatenated into train.csv).
- Test contains 4 phases: A (known) -> B (novel) -> C (known) -> D (novel, different).
- Ground truth writes both departure steps: [start_B, start_D].

Usage:
  python -m scripts.make_demo --out data_demo \
    --subjects 2 --rate-hz 6.67 \
    --len-A 1200 --len-B 300 --len-C 900 --len-D 300
"""

from pathlib import Path
import argparse, math, random, yaml
import numpy as np
import pandas as pd


def _mix_pattern(
    n: int,
    rate: float,
    amp1: float, f1: float,
    amp2: float, f2: float,
    *,
    phase: float,
    noise: float,
    drift: float = 0.0,
    am=(0.0, 0.0),
    offset: float = 0.0,
) -> pd.DataFrame:
    """Two-tone sinusoid with optional AM, slow drift, offset, and noise."""
    t = np.arange(n) / rate
    am_depth, am_freq = am
    am_env = (1.0 + am_depth * np.sin(2*np.pi*am_freq*t)) if am_depth > 0 else 1.0

    roll  = (amp1*np.sin(2*np.pi*f1*t + phase)      + amp2*np.sin(2*np.pi*f2*t + 0.3)) * am_env
    pitch = (0.9*amp1*np.sin(2*np.pi*f1*t + phase + 0.6) + 1.1*amp2*np.sin(2*np.pi*f2*t - 0.2)) * am_env

    if drift:
        roll  += drift * (t - t.mean())
        pitch += drift * (t - t.mean())

    if offset:
        roll  += offset
        pitch += offset

    if noise:
        roll  += np.random.normal(0, noise, size=n)
        pitch += np.random.normal(0, noise, size=n)

    return pd.DataFrame({"ROLL_STICK": roll, "PITCH_STIC": pitch})


def _bursty_noise(n: int, p: float = 0.01, scale: float = 3.0) -> np.ndarray:
    """Occasional bursts/outliers to stress the model."""
    mask = np.random.rand(n) < p
    burst = np.zeros(n)
    burst[mask] = np.random.randn(mask.sum()) * scale
    return burst


def make_subject_multiphase(out_dir: Path, name: str, rate_hz: float, lengths: dict, seed: int):
    """
    Create one subject:
      - train.csv with A + C known patterns
      - test_01.csv and test_02.csv each with A->B->C->D
      - return a dict { 'test_XX.csv': {'times_wltoggle':[tB, tD], 'time_total': N} }
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    phase = rng.uniform(0, 2*math.pi)

    # Training lengths for A & C, plus test lengths for B & D.
    L_A = int(lengths["A"])
    L_B = int(lengths["B"])
    L_C = int(lengths["C"])
    L_D = int(lengths["D"])

    subj_dir = out_dir / name
    subj_dir.mkdir(parents=True, exist_ok=True)

    # ---- Known patterns for TRAIN -------------------------------------------------
    # A: slightly faster, higher amp, gentle drift & AM
    train_A = _mix_pattern(
        L_A, rate_hz, amp1=0.70, f1=0.45, amp2=0.25, f2=0.08,
        phase=phase, noise=0.04, drift=0.001, am=(0.15, 0.02)
    )
    # C: different known pattern (lower freq), mild drift, lighter AM
    train_C = _mix_pattern(
        L_C, rate_hz, amp1=0.60, f1=0.28, amp2=0.20, f2=0.05,
        phase=phase*0.7+0.4, noise=0.04, drift=0.0005, am=(0.10, 0.015)
    )

    pd.concat([train_A, train_C], ignore_index=True).to_csv(subj_dir / "train.csv", index=False)

    # ---- TEST: A (known) -> B (novel) -> C (known) -> D (novel different) --------
    # For test, use shorter A and C segments (so you see changes in one short plot)
    A_test_len = max(200, L_A // 3)
    C_test_len = max(200, L_C // 3)

    # Define pattern functions so we can re-generate with small randomness per file
    def gen_A(n):  # known
        return _mix_pattern(
            n, rate_hz, amp1=0.70, f1=0.45, amp2=0.25, f2=0.08,
            phase=phase + np.random.uniform(-0.05, 0.05),
            noise=0.04, drift=0.001, am=(0.15, 0.02)
        )

    def gen_B(n):  # novel #1 (more energetic + AM + bursty)
        df = _mix_pattern(
            n, rate_hz, amp1=1.10, f1=0.90, amp2=0.35, f2=0.18,
            phase=phase + np.random.uniform(-0.05, 0.05),
            noise=0.08, drift=0.002, am=(0.35, 0.03), offset=0.0
        )
        b = _bursty_noise(n, p=0.02, scale=2.5)
        df["ROLL_STICK"]  += b
        df["PITCH_STIC"]  += 0.6*b
        return df

    def gen_C(n):  # known (return to familiar)
        return _mix_pattern(
            n, rate_hz, amp1=0.60, f1=0.28, amp2=0.20, f2=0.05,
            phase=phase*0.7+0.4 + np.random.uniform(-0.05, 0.05),
            noise=0.04, drift=0.0005, am=(0.10, 0.015)
        )

    def gen_D(n):  # novel #2 (different characteristics)
        df = _mix_pattern(
            n, rate_hz, amp1=0.90, f1=0.62, amp2=0.27, f2=0.14,
            phase=phase + np.random.uniform(-0.05, 0.05),
            noise=0.07, drift=0.0015, am=(0.25, 0.04), offset=0.25
        )
        b = _bursty_noise(n, p=0.015, scale=3.0)
        df["ROLL_STICK"]  += 0.7*b
        df["PITCH_STIC"]  += b
        return df

    gt = {}
    for i in (1, 2):
        A0 = gen_A(A_test_len)
        B  = gen_B(L_B)
        C0 = gen_C(C_test_len)
        D  = gen_D(L_D)

        test = pd.concat([A0, B, C0, D], ignore_index=True)

        # Toggle steps are 1-based indices where the departures begin
        t_B = len(A0) + 1
        t_D = len(A0) + len(B) + len(C0) + 1
        N   = len(test)

        fname = f"test_{i:02d}.csv"
        (subj_dir / fname).write_text(test.to_csv(index=False))
        gt[fname] = {"times_wltoggle": [int(t_B), int(t_D)], "time_total": int(N)}

    return gt


def main():
    ap = argparse.ArgumentParser(description="Make a multi-phase synthetic dataset with two known patterns and two novelty departures.")
    ap.add_argument("--out", default="data_demo", help="Output root folder.")
    ap.add_argument("--subjects", type=int, default=2)
    ap.add_argument("--rate-hz", type=float, default=6.67)

    # Training lengths for A and C (known), and test lengths for B and D (novel)
    ap.add_argument("--len-A", type=int, default=1200, help="Training length for known pattern A")
    ap.add_argument("--len-B", type=int, default=300,  help="Test length for novel pattern B")
    ap.add_argument("--len-C", type=int, default=900,  help="Training length for known pattern C")
    ap.add_argument("--len-D", type=int, default=300,  help="Test length for novel pattern D")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    lengths = {"A": args.len_A, "B": args.len_B, "C": args.len_C, "D": args.len_D}
    subject_names = [f"Demo{chr(ord('A')+i)}" for i in range(args.subjects)]

    top_yaml = {}
    for i, name in enumerate(subject_names):
        meta = make_subject_multiphase(out_root, name, args.rate_hz, lengths, args.seed + i)
        top_yaml[name] = meta

    # write GT file
    gt_path = out_root / "subjects_testfiles_wltogglepoints.yaml"
    with gt_path.open("w") as f:
        yaml.safe_dump(top_yaml, f, sort_keys=True)
    print(f"[demo] wrote ground truth → {gt_path}")


if __name__ == "__main__":
    main()
