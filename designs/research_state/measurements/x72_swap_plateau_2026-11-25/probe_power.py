"""Power of X72's end exploitability probe (PREREG §10): X71's design with FIVE targets in the read (X71's four
banked targets and the one new one), computed THROUGH X71's OWN `simulate` (`../x71_exploitability_probe_2026-11-16/
x71_rule.py`), which takes the number of targets as an argument. Nothing of X71's is changed and no game is read.

    python designs/research_state/measurements/x72_swap_plateau_2026-11-25/probe_power.py > probe_power.log
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
X71 = HERE.parent / "x71_exploitability_probe_2026-11-16"
spec = importlib.util.spec_from_file_location("x71_rule", X71 / "x71_rule.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

a = R.r0_anchor(str((X71 / R.R0_CURVE).resolve()))
print(f"X72's probe: X71's design, 5 targets in the read (c0, x67, swap, x70 banked; one new), MULT = t(0.975, 5) = "
      f"{R.mult(5):.3f}; G = {R.G:,}, G0 = {R.G0:,}; seed 7100, 20,000 simulated reads a row; the anchor is R0's banked "
      f"curve (gap(0) {100 * a['g0']:+.2f} pp, learned at 5M {100 * a['learned5']:+.2f} pp)")
print(R._table(R.simulate(a, n_targets=5)))
