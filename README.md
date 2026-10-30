# Is your decision benchmark actually hard?

A screening harness for evaluation tasks aimed at autonomous decision-making
agents. You point it at a candidate design, it builds the strongest cheap
attacker you give it, and it tells you whether the design discriminates — before
you spend weeks implementing it.

Most of the literature is about *building* benchmarks. This is about killing bad
ones early, which is where most of the time actually goes.

---

## The problem

You want a task where a capable agent, given real compute and hours of wall
clock, cannot match an expert reference. Plausible-sounding designs usually
fail, and they fail for a small number of systematic reasons. The expensive
version of finding out is to build the whole environment, ship it, and watch the
agent beat your reference. The cheap version is this repo.

The discipline is one sentence:

> **Build the attacker before you build the task.** If the cheapest strong
> strategy over your own model class ties your intended method, the design is
> dead — and you can know that in an afternoon.

---

## What it does

```python
from dbs import screen
from dbs.attackers import GreedyVOI, Lookahead, RandomSelection

result = screen(
    MyCase(),
    [RandomSelection(), GreedyVOI(), Lookahead(depth=2), Lookahead(depth=3)],
    n_seeds=1500,
    threat="lookahead2",            # the cheap thing an opponent writes
    ceiling_strategy="lookahead3",  # the best you know how to do
)
print(result.table())
```

Four things the harness insists on, each because getting it wrong produced a
wrong answer at least once:

- **Score the decision, not the luck.** The value of the action the strategy
  chose, evaluated under the hidden parameters — not a sampled reward, whose
  noise can swamp the difference between strategies.
- **An analytic floor.** The best decision available with no budget spent,
  derived from the prior. Never a measured strategy's score.
- **Shared seeds and paired tests.** Instance variance usually dwarfs the
  difference between strategies.
- **A power check before any null result.** "The attacker tied" is meaningless
  if the run could not have seen the gap you care about, so the harness refuses
  to call a tie it could not have detected.

---

## A worked case, with its real output

`cases/active_perception/` — N accounts, an unknown subset fraudulent, accounts
linked by shared attributes. You have a budget of noisy investigations, then
decide whom to block.

The design hypothesis: a link only resolves when **both** endpoints have been
investigated, so measurements are complementary. That breaks adaptive
submodularity, which is what gives greedy information-gain its guarantee — so
non-myopic planning should win, and a learned policy should be able to amortize
that advantage.

The case was screened twice. The first run is the one that fits in ten minutes
on a laptop; the second is the one the first run said was needed.

**Run 1: 90 seeds.**

```
strategy                     value    normalized
random          +0.4333 +/- 0.1483        0.1660
greedy_voi      +0.4444 +/- 0.1360        0.1702
lookahead2      +0.5778 +/- 0.1611        0.2213
lookahead3      +0.6889 +/- 0.1946        0.2638

ratio    = 0.8387

INCONCLUSIVE: ceiling does not beat threat (+0.1111 +/- 0.1326 (t=+0.84)).
This run could only have reliably seen a gap of 0.3716 (14.2% of the
floor-to-ceiling span). To confirm a gap the size of the one observed
(0.1111) you would need about 1006 seeds. Do not read this as a pass or
a fail.
```

The ratio looked almost decisive and the harness still refused to call it.
It also looked as if `greedy_voi` barely beat `random`, which would have meant
the case was degenerate.

**Run 2: 1,200 seeds** (about an hour on two CPU cores; full output in
`results/01_active_perception.txt`).

```
case: active_perception_fraud_rings   seeds: 1200
floor (analytic): +0.0000   ceiling (normalizer): +2.5475

strategy                     value    normalized
random          +0.2533 +/- 0.0357        0.0994
greedy_voi      +0.4792 +/- 0.0450        0.1881
lookahead2      +0.6517 +/- 0.0472        0.2558
lookahead3      +0.7292 +/- 0.0475        0.2862

threat   = lookahead2
ceiling  = lookahead3
ratio    = 0.8937

DEAD: threat reaches 0.894 of the ceiling (kill above 0.8).
```

Paired differences on the same 1,200 instances:

| comparison | difference | t |
|---|---|---|
| greedy_voi − random | +0.226 ± 0.046 | 4.93 |
| lookahead2 − greedy_voi | +0.173 ± 0.039 | 4.42 |
| lookahead3 − lookahead2 | +0.078 ± 0.039 | 1.97 |

What the second run changed:

- **The design is dead, and for the reason the hypothesis predicts.**
  Complementarity between pairs of measurements is real: non-myopic planning
  clearly beats greedy (t = 4.4). But a depth-2 lookahead, the cheap thing an
  opponent writes, already captures 89% of what depth 3 achieves, and the
  remaining gap is barely distinguishable from zero (t = 1.97). Order-2
  complementarity is saturated by depth-2 lookahead. There is no room left for
  a learned policy to show an advantage.
- **The 90-seed "degenerate case" reading was noise.** With enough seeds,
  `greedy_voi` beats `random` by a wide margin (t = 4.9). Acting on the small
  run would have meant re-tuning a case that did not need it.

Both points are the argument for the tool: the cheap run tells you how many
seeds you need, and the expensive run is still an hour, not weeks of building.

---

## Three traps this harness exists to stop

Each of these is a mistake that is easy to make and silently wrong.

**1. Your attacker must not be a straw man.** The threat is the best cheap
strategy *over the same model class as your intended method* — the plug-in with
`argmax` instead of the posterior, not a weak heuristic you invented. A straw
man loses to your method and makes a dead design look strong.

**2. Pruning by myopic gain defeats non-myopic search.** `Lookahead` takes a
`beam` argument and defaults it to `None` on purpose. When complementarity is
what you are testing, the actions worth taking are precisely the ones that look
worthless alone — so ranking candidates by one-step gain and keeping the top few
prunes away the answer. With `beam=3` this planner scored *below* plain greedy
on the case above. With `beam=None` it beats it. The bug and the fix are both in
the code with comments.

**3. Clamp the planning horizon to the budget that remains.** Planning three
steps ahead with two left overvalues actions whose payoff needs a step you will
never take. Unclamped, depth 3 scored below depth 2; clamped, the ordering is
monotone in depth as it should be.

---

## Layout

```
src/dbs/
  stats.py         summaries, paired tests, power, minimum detectable effect
  protocols.py     Case and SequentialCase: what your problem must implement
  screen.py        the harness, the ratio, and the verdict
  attackers/
    generic.py     RandomSelection, GreedyVOI, Lookahead(depth, beam)
cases/
  active_perception/   complementarity between measurements
results/
tests/
```

Implement `SequentialCase` — belief state, actions, outcome distribution,
terminal value, and a score under hidden truth — and every generic attacker
works on your problem without you writing one.

## Running it

```bash
python -m pip install numpy
python cases/active_perception/run.py 90      # ~8 minutes; use 1200 for the full result
python -m pytest tests -q
```

Pure NumPy, no GPU, no services. The cases are small by design: if a mechanism
produces no gap with exact inference on a small instance, it will not produce
one with approximate inference on a large one.

## Limitations, stated plainly

- A ceiling you computed with a planner is a **lower bound** on what is
  achievable. The ratio tells you what *this* attacker could not close, not that
  the design is hard.
- The default kill threshold (0.8) is a convention, not a derived result. Set
  your own, and leave margin: planner ceilings tend to run optimistic.
- Exact enumeration bounds how big a case can be. Sampled outcome
  approximations are allowed but must be declared in the case docstring.

## Provenance

Synthetic generators are self-contained and original to this repo. The SEC case
(planned) uses the U.S. SEC Financial Statement Data Sets — U.S. Government
work, free to use and redistribute, also published on data.gov under CC0.

This repository was developed with substantial AI assistance (code and
documentation). I directed the design, reviewed the code, and ran and checked
the measurements.

## License

MIT.
