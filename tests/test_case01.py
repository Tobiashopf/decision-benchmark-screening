import numpy as np
from case import ActivePerceptionCase
from dbs.attackers import GreedyVOI, Lookahead, RandomSelection


def test_floor_is_analytic_and_not_measured():
    c = ActivePerceptionCase()
    inst = c.instance(np.random.default_rng(0))
    # no budget spent: under the prior nothing clears the blocking threshold
    assert c.floor_value(inst) == 0.0


def test_ceiling_bounds_every_strategy():
    c = ActivePerceptionCase()
    for i in range(5):
        inst = c.instance(np.random.default_rng(i))
        top = c.ceiling_value(inst)
        for s in (RandomSelection(), GreedyVOI(), Lookahead(2)):
            assert s.run(c, inst, np.random.default_rng(99)) <= top + 1e-9


def test_outcome_probabilities_sum_to_one():
    c = ActivePerceptionCase()
    inst = c.instance(np.random.default_rng(3))
    st = c.initial_state(inst)
    for a in c.actions(st)[:4]:
        assert sum(p for p, _ in c.outcomes(st, a)) == np.float64(1.0).item() or \
               abs(sum(p for p, _ in c.outcomes(st, a)) - 1.0) < 1e-9


def test_deeper_lookahead_is_not_worse_on_average():
    """Guards the horizon-clamp bug: unclamped, depth 3 scored below depth 2."""
    c = ActivePerceptionCase()
    d2, d3 = [], []
    for i in range(12):
        inst = c.instance(np.random.default_rng(i))
        d2.append(Lookahead(2).run(c, inst, np.random.default_rng(10**6 + i)))
        d3.append(Lookahead(3).run(c, inst, np.random.default_rng(10**6 + i)))
    assert np.mean(d3) >= np.mean(d2) - 0.25
