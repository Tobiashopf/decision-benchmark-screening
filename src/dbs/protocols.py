"""The two interfaces a case implements.

`Case` is the minimum: generate an instance, say what a strategy scored, and
give the analytic floor. That is enough to screen any decision problem.

`SequentialCase` adds a belief/action/outcome interface. Implement it and the
generic attackers in `dbs.attackers.generic` work on your problem for free --
which is the point, because the single most common screening mistake is to
compare your method against a weak hand-rolled stand-in instead of against the
strongest cheap strategy over the same model class.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np

Instance = Any
State = Any
Action = Any


@runtime_checkable
class Case(Protocol):
    """A decision problem you want to screen."""

    name: str

    def instance(self, rng: np.random.Generator) -> Instance:
        """Draw one problem instance, including its hidden truth."""

    def floor_value(self, inst: Instance) -> float:
        """Value of the best decision available WITHOUT spending any budget.

        Analytic, derived from the prior. This is never a measured strategy's
        score: anchoring the floor on something you ran is the most damaging
        calibration error there is, and nothing mechanical catches it.
        """

    def ceiling_value(self, inst: Instance) -> float:
        """An upper bound on achievable value, e.g. an oracle that sees truth.

        Used only to normalize. It does not have to be attainable.
        """


@runtime_checkable
class SequentialCase(Case, Protocol):
    """A case whose decisions unfold over a budget of information-gathering steps."""

    def initial_state(self, inst: Instance) -> State:
        """Belief state before any observation. Must not read hidden truth."""

    def budget(self, inst: Instance) -> int:
        """Number of information-gathering actions allowed."""

    def actions(self, state: State) -> list[Action]:
        """Actions still available in this state."""

    def outcomes(self, state: State, action: Action) -> Iterable[tuple[float, State]]:
        """(probability, successor state) over possible observations.

        Exact enumeration where the outcome space is small; a sampled
        approximation is acceptable if you say so in the case's docstring.
        """

    def terminal_value(self, state: State) -> float:
        """Expected value of the best final decision under the CURRENT belief.

        This drives value-of-information attackers. It is a belief-side
        quantity and must not look at hidden truth.
        """

    def realize(self, state: State, action: Action,
                inst: Instance, rng: np.random.Generator) -> State:
        """Actually take the action: sample the observation from hidden truth."""

    def score(self, state: State, inst: Instance) -> float:
        """Value of the final decision, evaluated under the hidden truth.

        Score the DECISION, not the luck: use the expected payoff of the action
        the strategy chose given the hidden parameters, not a sampled reward.
        A sampled reward adds outcome noise that can swamp the difference
        between strategies.
        """


class Strategy(Protocol):
    """Anything that plays a case and returns the value it achieved."""

    name: str

    def run(self, case: Case, inst: Instance, rng: np.random.Generator) -> float:
        ...
