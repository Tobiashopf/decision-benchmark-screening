"""Screen case 01. Usage: python cases/active_perception/run.py [n_seeds]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from case import ActivePerceptionCase          # noqa: E402
from dbs import screen                         # noqa: E402
from dbs.attackers import GreedyVOI, Lookahead, RandomSelection  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
res = screen(
    ActivePerceptionCase(rho=0.90),
    [RandomSelection(), GreedyVOI(), Lookahead(depth=2), Lookahead(depth=3)],
    n_seeds=n,
    threat="lookahead2",            # the cheap non-myopic attacker
    ceiling_strategy="lookahead3",  # the best planner we have
)
print(res.table())
