from typing import List, Optional

def first_spike_at_or_after(toggle_step:int, spike_steps:List[int]) -> Optional[int]:
    for s in spike_steps:
        if s >= toggle_step:
            return s
    return None

def detection_lag_steps(toggle_step:int, spike_steps:List[int]) -> Optional[int]:
    hit = first_spike_at_or_after(toggle_step, spike_steps)
    return None if hit is None else (hit - toggle_step)

def precision(toggle_step:int, spike_steps:List[int]) -> float:
    if not spike_steps:
        return 0.0
    tp = 1 if first_spike_at_or_after(toggle_step, spike_steps) is not None else 0
    fp = len(spike_steps) - tp
    denom = tp + fp
    return float(tp) / denom if denom > 0 else 0.0
