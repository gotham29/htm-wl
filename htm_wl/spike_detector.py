from typing import Optional, Dict

class SpikeDetector:
    """
    Compare recent vs prior mean of MWL and trigger spikes when growth crosses a threshold.

    Args:
        recent_count: length of the "recent" window (steps)
        prior_count:  length of the "prior"  window (steps)
        threshold_pct: percent growth threshold (positive number)
        min_delta: absolute MWL rise fallback (mr - mp >= min_delta) in addition to pct growth
        min_separation: min steps between spikes
        edge_only: only fire on threshold *crossings* (rising edge)
        direction: 'up' | 'down' | 'both'  (default 'up')
        eps: denominator guard for near-zero prior means
        min_mwl: minimum recent MWL required to allow a spike (filters low-energy blips)
    """
    def __init__(
        self,
        recent_count: int,
        prior_count: int,
        threshold_pct: float,
        min_delta: float = 0.0,
        min_separation: int = 0,
        edge_only: bool = True,
        direction: str = "up",
        eps: float = 1e-9,
        min_mwl: float = 0.0,
    ):
        self.recent_count = int(recent_count)
        self.prior_count = int(prior_count)
        self.threshold_pct = float(threshold_pct)
        self._min_delta = float(min_delta)
        self._buf = []
        self._step = 0
        self._prev_over = False
        self._prev_under = False
        self._last_spike_step = -10**9
        self._min_sep = int(min_separation)
        self._edge_only = bool(edge_only)
        self._dir = direction
        self._eps = float(eps)
        self._min_mwl = float(min_mwl)

    def update(self, mwl_value: float) -> Optional[Dict]:
        self._step += 1
        self._buf.append(float(mwl_value))
        n = self.recent_count + self.prior_count
        if len(self._buf) < n:
            return None

        recent = self._buf[-self.recent_count:]
        prior  = self._buf[-n:-self.recent_count]
        mr = sum(recent) / max(1, len(recent))
        mp = sum(prior)  / max(1, len(prior))

        # denom is strictly positive; MWL (EMA of anomaly) is >= 0
        denom = max(mp, self._eps)
        growth_pct = 100.0 * (mr - mp) / denom

        # gates
        enough_mwl   = mr >= self._min_mwl
        pct_up       = (mr > mp) and (growth_pct >  self.threshold_pct)
        pct_down     = (mr < mp) and (growth_pct < -self.threshold_pct)
        delta_ok     = (mr - mp) >= self._min_delta

        over  = (pct_up   or delta_ok) and enough_mwl
        under = (pct_down and enough_mwl)

        spike = False
        if self._dir in ("up", "both"):
            if over and (not self._edge_only or (self._edge_only and not self._prev_over)):
                if (self._step - self._last_spike_step) >= self._min_sep:
                    spike = True
                    self._last_spike_step = self._step
        if not spike and self._dir in ("down", "both"):
            if under and (not self._edge_only or (self._edge_only and not self._prev_under)):
                if (self._step - self._last_spike_step) >= self._min_sep:
                    spike = True
                    self._last_spike_step = self._step

        self._prev_over = over
        self._prev_under = under

        return {
            "mr": float(mr),
            "mp": float(mp),
            "growth_pct": float(growth_pct),
            "spike": bool(spike),
            "step": self._step,
        }
