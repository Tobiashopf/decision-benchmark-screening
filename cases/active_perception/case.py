"""Case 01 -- active perception under a combinatorial belief.

The story: N accounts, an unknown subset of them fraudulent. Accounts are
linked by shared attributes. You have a budget of costly, noisy investigations
and must then decide which accounts to block.

The hypothesis under test: PAIR EVIDENCE creates complementarity between
measurements. A link only resolves when BOTH of its endpoints have been
investigated, so the value of investigating one account depends on what else
you investigated. That breaks adaptive submodularity, which is what gives the
greedy information-gain heuristic its (1-1/e) guarantee -- so non-myopic
planning should win, and a learned policy should be able to amortize it.

The screen is deliberately TRACTABLE: link components are small enough to
enumerate every assignment, and observations are binary, so beliefs and
expectations are exact. If the mechanism produces no gap with exact inference,
it will not produce one with approximate inference. Cheap filter.

Everything here is synthetic and self-contained. No external data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

P_INFECTED = 0.35   # prior that a link component is infected at all
Q_MEMBER = 0.60     # given infected, probability each member is fraudulent
A_IND = 0.66        # P(individual signal fires | fraudulent)  -- weak on purpose
B_IND = 0.34        # P(individual signal fires | legitimate)
V_CATCH = 1.0       # value of blocking a fraudulent account
C_FP = 1.0          # cost of blocking a legitimate one
THRESH = C_FP / (V_CATCH + C_FP)   # Bayes-optimal blocking threshold


@dataclass
class Component:
    """One link component: a handful of accounts that share attributes."""

    n: int
    edges: tuple[tuple[int, int], ...]
    rho: float                      # P(pair evidence fires | both fraudulent)
    configs: np.ndarray
    logprior: np.ndarray
    s_obs: dict = field(default_factory=dict)
    z_obs: dict = field(default_factory=dict)

    def copy(self) -> "Component":
        return Component(self.n, self.edges, self.rho, self.configs,
                         self.logprior, dict(self.s_obs), dict(self.z_obs))


@dataclass
class Instance:
    comps: list[Component]
    truths: list[np.ndarray]
    budget: int


@dataclass
class State:
    """Belief over every component, with per-component value cached.

    Successors share untouched components by reference and patch one cached
    value, so expanding a search tree costs one small marginal computation per
    branch instead of a full sweep. Without this, depth-3 planning on a toy
    problem took minutes per instance.
    """

    comps: list[Component]
    vals: list[float]

    def successor(self, ci: int, comp: Component, val: float) -> "State":
        comps = self.comps[:]
        vals = self.vals[:]
        comps[ci] = comp
        vals[ci] = val
        return State(comps, vals)


def _configs(n: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)


def _logprior(cfg: np.ndarray) -> np.ndarray:
    n = cfg.shape[1]
    k = cfg.sum(axis=1)
    clean = np.where(k == 0, 1.0 - P_INFECTED, 0.0)
    inf = P_INFECTED * (Q_MEMBER ** k) * ((1.0 - Q_MEMBER) ** (n - k))
    return np.log(clean + inf + 1e-300)


def _loglik(c: Component) -> np.ndarray:
    out = np.zeros(c.configs.shape[0])
    for i, s in c.s_obs.items():
        p = np.where(c.configs[:, i] == 1, A_IND, B_IND)
        out += np.log(p if s == 1 else 1.0 - p)
    for (i, j), z in c.z_obs.items():
        both = (c.configs[:, i] == 1) & (c.configs[:, j] == 1)
        p = np.where(both, c.rho, 1.0 - c.rho)
        out += np.log(p if z == 1 else 1.0 - p)
    return out


def _marginals(c: Component) -> np.ndarray:
    lp = c.logprior + _loglik(c)
    w = np.exp(lp - lp.max())
    return (w / w.sum()) @ c.configs


def _expected_decision_value(marg: np.ndarray) -> float:
    block = marg > THRESH
    return float((block * (marg * V_CATCH - (1 - marg) * C_FP)).sum())


def _decision_value(marg: np.ndarray, truth: np.ndarray) -> float:
    block = marg > THRESH
    return float((block * (truth * V_CATCH - (1 - truth) * C_FP)).sum())


class ActivePerceptionCase:
    name = "active_perception_fraud_rings"

    def __init__(self, rho: float = 0.90, n_components: int = 3,
                 comp_size: int = 4, budget: int = 6):
        self.rho = rho
        self.n_components = n_components
        self.comp_size = comp_size
        self._budget = budget
        self._cfg = _configs(comp_size)
        self._lp = _logprior(self._cfg)

    # -- Case ---------------------------------------------------------------
    def instance(self, rng) -> Instance:
        comps, truths = [], []
        n = self.comp_size
        for _ in range(self.n_components):
            edges = [(k, k + 1) for k in range(n - 1)]
            a, b = rng.choice(n, size=2, replace=False)
            if abs(int(a) - int(b)) > 1:
                edges.append((int(min(a, b)), int(max(a, b))))
            infected = rng.random() < P_INFECTED
            truth = ((rng.random(n) < Q_MEMBER).astype(np.int8) if infected
                     else np.zeros(n, np.int8))
            comps.append(Component(n, tuple(edges), self.rho, self._cfg, self._lp))
            truths.append(truth)
        return Instance(comps, truths, self._budget)

    def floor_value(self, inst: Instance) -> float:
        """Best decision under the prior, spending no budget. Analytic."""
        w = np.exp(self._lp - self._lp.max())
        marg = (w / w.sum()) @ self._cfg
        return sum(_decision_value(marg, t) for t in inst.truths)

    def ceiling_value(self, inst: Instance) -> float:
        """Oracle that knows the truth. Normalizer only; not attainable."""
        return float(sum((t * V_CATCH).sum() for t in inst.truths))

    # -- SequentialCase ------------------------------------------------------
    def initial_state(self, inst: Instance) -> State:
        comps = [c.copy() for c in inst.comps]
        return State(comps, [_expected_decision_value(_marginals(c)) for c in comps])

    def budget(self, inst: Instance) -> int:
        return inst.budget

    def actions(self, state: State):
        return [(ci, i) for ci, c in enumerate(state.comps)
                for i in range(c.n) if i not in c.s_obs]

    def _new_pairs(self, c: Component, i: int):
        done = set(c.s_obs)
        return [(min(a, b), max(a, b)) for (a, b) in c.edges
                if (a == i and b in done) or (b == i and a in done)]

    def outcomes(self, state: State, action):
        ci, i = action
        c = state.comps[ci]
        pairs = self._new_pairs(c, i)
        lp = c.logprior + _loglik(c)
        w = np.exp(lp - lp.max())
        post = w / w.sum()
        for s in (0, 1):
            for zs in itertools.product((0, 1), repeat=len(pairs)):
                p_ind = np.where(c.configs[:, i] == 1, A_IND, B_IND)
                ll = np.log(p_ind if s == 1 else 1.0 - p_ind)
                for (a, b), z in zip(pairs, zs):
                    both = (c.configs[:, a] == 1) & (c.configs[:, b] == 1)
                    pp = np.where(both, c.rho, 1.0 - c.rho)
                    ll = ll + np.log(pp if z == 1 else 1.0 - pp)
                prob = float(post @ np.exp(ll))
                if prob <= 1e-12:
                    continue
                nc = c.copy()
                nc.s_obs[i] = s
                for pair, z in zip(pairs, zs):
                    nc.z_obs[pair] = z
                yield prob, state.successor(ci, nc,
                                            _expected_decision_value(_marginals(nc)))

    def terminal_value(self, state: State) -> float:
        return float(sum(state.vals))

    def realize(self, state: State, action, inst: Instance, rng) -> State:
        ci, i = action
        c = state.comps[ci].copy()
        truth = inst.truths[ci]
        pairs = self._new_pairs(c, i)
        c.s_obs[i] = int(rng.random() < (A_IND if truth[i] == 1 else B_IND))
        for (a, b) in pairs:
            both = truth[a] == 1 and truth[b] == 1
            c.z_obs[(a, b)] = int(rng.random() < (c.rho if both else 1.0 - c.rho))
        return state.successor(ci, c, _expected_decision_value(_marginals(c)))

    def score(self, state: State, inst: Instance) -> float:
        return sum(_decision_value(_marginals(c), t)
                   for c, t in zip(state.comps, inst.truths))
