from typing import Dict, List, Optional
import numpy as np

# htm.core
from htm.bindings.sdr import SDR
from htm.algorithms import SpatialPooler as SP
from htm.algorithms import TemporalMemory as TM

class _ScalarBucket:
    """
    Minimal scalar encoder. If you want RDSE, see comment in __init__ to
    switch to htm.bindings.encoders RDSE. Uses (n, w) per feature.
    """
    def __init__(self, n:int, w:int, vmin:float=-1.0, vmax:float=1.0):
        self.n, self.w = n, max(3, w)
        self.vmin, self.vmax = vmin, vmax
        self.range = max(1e-9, vmax - vmin)

    def encode(self, x: float) -> np.ndarray:
        x = min(self.vmax, max(self.vmin, x))
        frac = (x - self.vmin) / self.range
        center = int(frac * (self.n - 1))
        start = max(0, min(self.n - self.w, center - self.w // 2))
        sdr = np.zeros(self.n, dtype=np.int8)
        sdr[start:start+self.w] = 1
        return sdr

class HTMSession:
    """
    Builds encoders + SP + TM using htm.core. Produces:
      - anomaly (tm.anomaly)
      - mwl = EMA(anomaly) with alpha
    """
    def __init__(
        self,
        feature_names: List[str],
        enc_n_per_feature: int,
        enc_w_per_feature: int,
        sp_params: Dict,
        tm_params: Dict,
        seed: int,
        anomaly_ema_alpha: float = 0.2,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.feature_names = feature_names
        self.N = enc_n_per_feature * len(feature_names)
        self.C = int(sp_params.get("columnCount", 2048))
        self.alpha = anomaly_ema_alpha
        self._ema = None

        # Encoders: (switch to RDSE if you want exact NuPIC RDSE behavior)
        self.encoders = {}
        for name in feature_names:
            vmin = feature_ranges.get(name, {}).get("min", -1.0) if feature_ranges else -1.0
            vmax = feature_ranges.get(name, {}).get("max",  1.0) if feature_ranges else  1.0
            self.encoders[name] = _ScalarBucket(enc_n_per_feature, enc_w_per_feature, vmin, vmax)

        # Input SDR
        self.input_sdr = SDR((self.N,))

        # Spatial Pooler
        self.sp = SP(
            inputDimensions=(self.N,),
            columnDimensions=(self.C,),
            potentialPct=float(sp_params.get("potentialPct", 0.8)),
            globalInhibition=bool(sp_params.get("globalInhibition", True)),
            synPermActiveInc=float(sp_params.get("synPermActiveInc", 0.003)),
            synPermInactiveDec=float(sp_params.get("synPermInactiveDec", 0.0005)),
            synPermConnected=float(sp_params.get("synPermConnected", 0.2)),
            boostStrength=float(sp_params.get("boostStrength", 0.0)),
            seed=int(seed),
        )

        # Temporal Memory
        self.tm = TM(
            columnDimensions=(self.C,),
            cellsPerColumn=int(tm_params.get("cellsPerColumn", 32)),
            activationThreshold=int(tm_params.get("activationThreshold", 20)),
            initialPermanence=float(tm_params.get("initialPerm", 0.21)),
            connectedPermanence=float(tm_params.get("permanenceConnected", 0.5)),
            minThreshold=int(tm_params.get("minThreshold", 13)),
            maxNewSynapseCount=int(tm_params.get("newSynapseCount", 31)),
            permanenceIncrement=float(tm_params.get("permanenceInc", 0.1)),
            permanenceDecrement=float(tm_params.get("permanenceDec", 0.0)),
            predictedSegmentDecrement=float(tm_params.get("predictedSegmentDecrement", 0.001)),
            seed=int(seed),
        )

    def _encode(self, feats: Dict[str, float]) -> np.ndarray:
        chunks = [self.encoders[n].encode(float(feats[n])) for n in self.feature_names]
        return np.concatenate(chunks)

    def step(self, feats: Dict[str, float]) -> Dict[str, float]:
        x = self._encode(feats)
        self.input_sdr.sparse = np.nonzero(x)[0]
        active_columns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(self.input_sdr, True, active_columns)
        self.tm.compute(active_columns, learn=True)
        anomaly = float(self.tm.anomaly)
        self._ema = anomaly if self._ema is None else (1 - self.alpha) * self._ema + self.alpha * anomaly
        return {"anomaly": anomaly, "mwl": float(self._ema)}
