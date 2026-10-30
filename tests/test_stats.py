import numpy as np
import pytest
from dbs.stats import minimum_detectable_effect, paired, required_seeds, summarize


def test_summary_uses_sample_sd():
    v = [1.0, 2.0, 3.0]
    s = summarize(v)
    assert s.n == 3
    assert s.mean == pytest.approx(2.0)
    assert s.sd == pytest.approx(np.std(v, ddof=1))   # ddof=1, not population


def test_paired_requires_same_shape():
    with pytest.raises(ValueError):
        paired([1, 2, 3], [1, 2])


def test_paired_detects_a_constant_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    t = paired(a, a + 0.2)
    assert t.mean_diff == pytest.approx(0.2)
    assert t.se == pytest.approx(0.0, abs=1e-9)       # pairing removes all noise


def test_power_roundtrip():
    sd, effect = 1.0, 0.2
    n = required_seeds(sd, effect)
    assert minimum_detectable_effect(sd, n) <= effect * 1.001
