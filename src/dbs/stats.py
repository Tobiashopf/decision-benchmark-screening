"""Statistics for screening: summaries, paired tests, and power.

The whole point of this module is to make it hard to believe a difference that
the sample size cannot support. Screening runs are cheap; believing a noisy
result is expensive, because you then build on it for weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    """Mean of a strategy's per-seed values, with its uncertainty."""

    n: int
    mean: float
    sd: float
    se: float

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.96 * self.se
        return (self.mean - half, self.mean + half)

    def __str__(self) -> str:
        return f"{self.mean:+.4f} +/- {self.se:.4f}"


def summarize(values) -> Summary:
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("need at least 2 observations")
    sd = float(v.std(ddof=1))  # ddof=1: sample sd, not population
    return Summary(n=v.size, mean=float(v.mean()), sd=sd, se=sd / math.sqrt(v.size))


@dataclass(frozen=True)
class PairedTest:
    """Paired comparison of two strategies on the SAME seeds.

    Pairing matters more than it looks: instance-to-instance variance usually
    dwarfs the difference between strategies, so an unpaired comparison needs
    orders of magnitude more seeds to see the same effect.
    """

    n: int
    mean_diff: float
    se: float
    t: float

    @property
    def significant(self) -> bool:
        return abs(self.t) >= 1.96

    def __str__(self) -> str:
        return f"{self.mean_diff:+.4f} +/- {self.se:.4f} (t={self.t:+.2f})"


def paired(a, b) -> PairedTest:
    """Test b - a. Positive mean_diff means b beat a."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired() needs the same seeds in the same order")
    d = y - x
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(d.size)
    return PairedTest(
        n=int(d.size),
        mean_diff=float(d.mean()),
        se=se,
        t=float(d.mean() / se) if se > 0 else 0.0,
    )


def required_seeds(sd_of_differences: float, effect: float,
                   power: float = 0.80, alpha: float = 0.05) -> int:
    """Seeds needed to detect `effect` in a paired comparison.

    Normal approximation, two-sided. Use it BEFORE trusting a null result:
    "the attacker tied" means nothing if the run could not have seen the gap
    you care about.
    """
    if effect <= 0:
        raise ValueError("effect must be positive")
    z_alpha = 1.959963984540054 if alpha == 0.05 else _z(1 - alpha / 2)
    z_power = 0.8416212335729143 if power == 0.80 else _z(power)
    return int(math.ceil(((z_alpha + z_power) * sd_of_differences / effect) ** 2))


def minimum_detectable_effect(sd_of_differences: float, n: int,
                              power: float = 0.80, alpha: float = 0.05) -> float:
    """The smallest difference a run of `n` seeds could reliably have found."""
    z_alpha = 1.959963984540054 if alpha == 0.05 else _z(1 - alpha / 2)
    z_power = 0.8416212335729143 if power == 0.80 else _z(power)
    return (z_alpha + z_power) * sd_of_differences / math.sqrt(n)


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
