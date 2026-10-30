"""The screening harness: run strategies on shared seeds, then judge.

Usage is deliberately blunt. You name which strategy is the THREAT (the best
cheap thing an opponent would write) and which is the CEILING (the best you
know how to do). The harness reports threat/ceiling and refuses to call a tie
when the run lacked the power to see one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protocols import Case
from .stats import (Summary, minimum_detectable_effect, paired, required_seeds,
                    summarize)


@dataclass
class ScreenResult:
    case: str
    n_seeds: int
    floor: float
    ceiling: float
    values: dict[str, np.ndarray] = field(default_factory=dict)
    threat: str = ""
    ceiling_strategy: str = ""

    def summary(self, name: str) -> Summary:
        return summarize(self.values[name])

    def normalized(self, name: str) -> float:
        """Progress from the analytic floor toward the ceiling, in [0, 1]-ish."""
        span = self.ceiling - self.floor
        if span == 0:
            return float("nan")
        return (self.summary(name).mean - self.floor) / span

    @property
    def ratio(self) -> float:
        """threat / ceiling, both normalized. Near 1 means the design is dead."""
        top = self.normalized(self.ceiling_strategy)
        if top <= 0:
            return float("nan")
        return self.normalized(self.threat) / top

    def verdict(self, kill_above: float = 0.8) -> str:
        """The default threshold is a convention, not a derived result.

        Pick your own for your setting, and leave yourself margin: a ceiling
        computed with a planner is only a lower bound on what is achievable,
        so the true ratio is usually closer to 1 than the measured one.
        """
        gap = paired(self.values[self.threat], self.values[self.ceiling_strategy])
        mde = minimum_detectable_effect(gap.se * np.sqrt(gap.n), gap.n)
        r = self.ratio

        if not gap.significant:
            span = self.ceiling - self.floor
            sd = gap.se * np.sqrt(gap.n)
            observed = abs(gap.mean_diff)
            need = (required_seeds(sd, observed) if observed > 0 else None)
            msg = (f"INCONCLUSIVE: ceiling does not beat threat ({gap}). "
                   f"This run could only have reliably seen a gap of {mde:.4f} "
                   f"({100 * mde / span:.1f}% of the floor-to-ceiling span).")
            if need:
                msg += (f" To confirm a gap the size of the one observed "
                        f"({observed:.4f}) you would need about {need} seeds.")
            return msg + " Do not read this as a pass or a fail."
        if r >= kill_above:
            return (f"DEAD: threat reaches {r:.3f} of the ceiling "
                    f"(kill above {kill_above}).")
        return (f"SURVIVES: threat reaches {r:.3f} of the ceiling. "
                f"Not proof it is hard -- only that this attacker did not close it.")

    def table(self) -> str:
        w = max(len(k) for k in self.values) + 2
        lines = [f"case: {self.case}   seeds: {self.n_seeds}",
                 f"floor (analytic): {self.floor:+.4f}   "
                 f"ceiling (normalizer): {self.ceiling:+.4f}", ""]
        lines.append(f"{'strategy':<{w}}{'value':>22}{'normalized':>14}")
        for name in self.values:
            lines.append(f"{name:<{w}}{str(self.summary(name)):>22}"
                         f"{self.normalized(name):>14.4f}")
        lines += ["", f"threat   = {self.threat}",
                  f"ceiling  = {self.ceiling_strategy}",
                  f"ratio    = {self.ratio:.4f}", "", self.verdict()]
        return "\n".join(lines)


def screen(case: Case, strategies, n_seeds: int = 1500, *,
           threat: str, ceiling_strategy: str, seed0: int = 0) -> ScreenResult:
    """Run every strategy on the SAME instances and collect per-seed values.

    Shared seeds are not a detail: pairing is what makes a few hundred seeds
    informative when instance variance dominates.
    """
    names = [s.name for s in strategies]
    for required in (threat, ceiling_strategy):
        if required not in names:
            raise ValueError(f"{required!r} is not among {names}")

    vals = {s.name: np.empty(n_seeds) for s in strategies}
    floors = np.empty(n_seeds)
    ceils = np.empty(n_seeds)

    for i in range(n_seeds):
        inst = case.instance(np.random.default_rng(seed0 + i))
        floors[i] = case.floor_value(inst)
        ceils[i] = case.ceiling_value(inst)
        for s in strategies:
            # fresh rng per (seed, strategy) so observation noise is comparable
            vals[s.name][i] = s.run(case, inst, np.random.default_rng(10**6 + seed0 + i))

    return ScreenResult(
        case=case.name, n_seeds=n_seeds,
        floor=float(floors.mean()), ceiling=float(ceils.mean()),
        values=vals, threat=threat, ceiling_strategy=ceiling_strategy,
    )
