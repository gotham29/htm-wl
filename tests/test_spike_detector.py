from htm_wl.spike_detector import SpikeDetector

def test_spike_threshold():
    det = SpikeDetector(recent_count=2, prior_count=2, threshold_pct=100.0)
    # prior avg = 1.0, recent avg = 2.1 -> growth = 110% -> spike
    for x in [1.0, 1.0, 2.1, 2.1]:
        out = det.update(x)
    assert out and out["spike"] is True

def test_no_spike_below_threshold():
    det = SpikeDetector(recent_count=2, prior_count=2, threshold_pct=200.0)
    for x in [1.0, 1.0, 1.5, 1.5]:
        out = det.update(x)
    assert out and out["spike"] is False
