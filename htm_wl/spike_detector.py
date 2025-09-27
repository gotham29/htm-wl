from typing import Optional, Dict, List


class SpikeDetector:
    """
    Recent/prior mean growth detector with optional:
      - rising-edge trigger (edge_only=True): fire only on False->True crossings
      - refractory period (min_separation): suppress follow-up spikes for N steps
    """
    def __init__(
        self,
        recent_count: int,
        prior_count: int,
        threshold_pct: float,
        min_separation: int = 0,
        edge_only: bool = True,
    ):
        self.recent_count = int(recent_count)
        self.prior_count = int(prior_count)
        self.threshold_pct = float(threshold_pct)

        self._buf: List[float] = []
        self._step: int = 0
        self._prev_over: bool = False
        self._last_spike_step: int = -10**9
        self._min_sep: int = int(min_separation)
        self._edge_only: bool = bool(edge_only)

    def update(self, mwl_value: float) -> Optional[Dict[str, float]]:
        self._step += 1
        self._buf.append(float(mwl_value))

        n = self.recent_count + self.prior_count
        if len(self._buf) < n:
            return None

        # Split into prior and recent windows
        recent_vals = self._buf[-self.recent_count:]
        prior_vals  = self._buf[-n:-self.recent_count]

        mr = sum(recent_vals) / len(recent_vals)
        mp = sum(prior_vals)  / len(prior_vals)

        # Avoid division by zero / tiny prior mean
        if abs(mp) <= 1e-12:
            growth_pct = 0.0
            over = False
        else:
            growth_pct = 100.0 * (mr - mp) / mp
            over = (growth_pct > self.threshold_pct)

        spike = False
        if over:
            edge_ok = (not self._edge_only) or (self._edge_only and not self._prev_over)
            sep_ok = (self._step - self._last_spike_step) >= self._min_sep
            if edge_ok and sep_ok:
                spike = True
                self._last_spike_step = self._step

        self._prev_over = over

        return {
            "mr": float(mr),
            "mp": float(mp),
            "growth_pct": float(growth_pct),
            "spike": bool(spike),
            "step": self._step,  # handy for debugging
        }
