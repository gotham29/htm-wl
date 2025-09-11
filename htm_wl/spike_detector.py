from typing import Optional, Dict, List

class SpikeDetector:
    def __init__(self, recent_count:int, prior_count:int, threshold_pct:float):
        self.nr = max(1, int(recent_count))
        self.np = max(1, int(prior_count))
        self.th = float(threshold_pct)
        self.buf: List[float] = []

    def update(self, mwl_val: float) -> Optional[Dict[str, float]]:
        self.buf.append(float(mwl_val))
        if len(self.buf) < self.nr + self.np:
            return None
        recent = self.buf[-self.nr:]
        prior  = self.buf[-(self.nr + self.np):-self.nr]
        mr = sum(recent) / len(recent)
        mp = sum(prior)  / len(prior)
        if mp <= 1e-12:
            return None
        growth = 100.0 * (mr - mp) / mp
        return {"mr": mr, "mp": mp, "growth_pct": growth, "spike": growth > self.th}
