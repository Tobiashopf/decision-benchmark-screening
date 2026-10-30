"""Generic attackers that work on any `SequentialCase`.

These exist so you do not have to invent an attacker, because inventing one is
how screening goes wrong. A weak stand-in you wrote yourself will lose to your
method and tell you nothing: the design looks strong and is not.

Rule: the threat is the best cheap strategy over the SAME model class as your
intended method.
"""

from __future__ import annotations

import numpy as np

from ..protocols import Action, Instance, SequentialCase, State


class RandomSelection:
    """Spends the budget uniformly at random. A sanity floor, not a threat."""

    name = "random"

    def run(self, case: SequentialCase, inst: Instance, rng) -> float:
        state = case.initial_state(inst)
        for _ in range(case.budget(inst)):
            acts = case.actions(state)
            if not acts:
                break
            a = acts[int(rng.integers(len(acts)))]
            state = case.realize(state, a, inst, rng)
        return case.score(state, inst)


class GreedyVOI:
    """Myopic expected value of information: one step of exact lookahead.

    The standard heuristic, and usually the first thing a capable agent writes.
    If your method cannot beat this, stop.
    """

    name = "greedy_voi"

    def run(self, case: SequentialCase, inst: Instance, rng) -> float:
        state = case.initial_state(inst)
        for _ in range(case.budget(inst)):
            acts = case.actions(state)
            if not acts:
                break
            best, best_gain = None, -np.inf
            base = case.terminal_value(state)
            for a in acts:
                gain = sum(p * case.terminal_value(s)
                           for p, s in case.outcomes(state, a)) - base
                if gain > best_gain:
                    best_gain, best = gain, a
            state = case.realize(state, best, inst, rng)
        return case.score(state, inst)


class Lookahead:
    """Receding-horizon planner of depth k.

    Plans a sequence of depth `depth`, executes only the FIRST action, then
    replans with the new observation. Committing the whole planned sequence
    blind is a bug that makes deeper search look WORSE than shallow search --
    it cost a day of wrong conclusions once, so it is spelled out here.

    `beam` caps the branching so depth 3-4 stays affordable.
    """

    def __init__(self, depth: int = 2, beam: int | None = None):
        """beam=None searches every action. Keep it that way by default.

        TRAP, measured: pruning candidates by MYOPIC gain before the deep
        search defeats the deep search whenever complementarity is what you
        are testing -- the actions worth taking are precisely the ones that
        look worthless alone. With beam=3 this planner scored BELOW plain
        greedy on the complementarity case; with beam=None it beats it.
        Only set a beam when you have checked it does not invert your result.
        """
        if depth < 1:
            raise ValueError("depth >= 1")
        self.depth = depth
        self.beam = beam
        self.name = f"lookahead{depth}"

    def run(self, case: SequentialCase, inst: Instance, rng) -> float:
        state = case.initial_state(inst)
        total = case.budget(inst)
        for step in range(total):
            acts = case.actions(state)
            if not acts:
                break
            # Clamp the horizon to the budget that actually remains. Planning
            # 3 steps ahead with 2 left overvalues actions whose payoff needs
            # a step you will never take -- measured: it made depth 3 score
            # BELOW depth 2.
            horizon = min(self.depth, total - step)
            first = self._plan(case, state, horizon)
            state = case.realize(state, first, inst, rng)
        return case.score(state, inst)

    def _plan(self, case: SequentialCase, state: State, depth: int) -> Action:
        acts = case.actions(state)
        if self.beam is None or depth == 1 or len(acts) <= self.beam:
            cands = acts
        else:
            cands = sorted(acts, key=lambda a: -self._one_step(case, state, a))[: self.beam]
        scored = [(self._value(case, state, a, depth - 1), a) for a in cands]
        scored.sort(key=lambda t: -t[0])
        return scored[0][1]

    @staticmethod
    def _one_step(case: SequentialCase, state: State, action: Action) -> float:
        return sum(p * case.terminal_value(s) for p, s in case.outcomes(state, action))

    def _value(self, case: SequentialCase, state: State,
               action: Action, remaining: int) -> float:
        total = 0.0
        for p, nxt in case.outcomes(state, action):
            if p <= 0.0:
                continue
            if remaining <= 0:
                total += p * case.terminal_value(nxt)
            else:
                acts = case.actions(nxt)
                if not acts:
                    total += p * case.terminal_value(nxt)
                    continue
                # beam: rank successors cheaply by one-step gain, expand the top few
                ranked = (acts if self.beam is None else
                          sorted(acts, key=lambda a: -self._one_step(case, nxt, a))[: self.beam])
                total += p * max(self._value(case, nxt, a, remaining - 1)
                                 for a in ranked)
        return total
