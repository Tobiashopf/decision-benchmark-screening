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

Running it (90 seeds, the number that fits in ten minutes on a laptop):

```
case: active_perception_fraud_rings   seeds: 90
floor (analytic): +0.0000   ceiling (normalizer): +2.6111

strategy                     value    normalized
random          +0.4333 +/- 0.1483        0.1660
greedy_voi      +0.4444 +/- 0.1360        0.1702
lookahead2      +0.5778 +/- 0.1611        0.2213
lookahead3      +0.6889 +/- 0.1946        0.2638

threat   = lookahead2
ceiling  = lookahead3
ratio    = 0.8387

INCONCLUSIVE: ceiling does not beat threat (+0.1111 +/- 0.1326 (t=+0.84)).
This run could only have reliably seen a gap of 0.3716 (14.2% of the
floor-to-ceiling span). To confirm a gap the size of the one observed
(0.1111) you would need about 1006 seeds. Do not read this as a pass or
a fail.
```

That output is the point of the tool. The ordering looks encouraging and the
ratio looks almost decisive, and the harness still refuses to call it, because
90 seeds cannot support the claim. A screening run that answered "0.84, design
is dead" here would have been guessing.

Two further things this run says out loud, both worth acting on:

- `greedy_voi` barely beats `random` (0.444 vs 0.433). At this size the case is
  close to degenerate — the floor-to-ceiling span is not really being traversed
  by anybody. Tune the instance before trusting any comparison on it.
- The honest next step is 1000+ seeds, which at ~5 s/seed is a few hours. Cheap
  compared to building the environment.

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
python cases/active_perception/run.py 90      # ~8 minutes
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
