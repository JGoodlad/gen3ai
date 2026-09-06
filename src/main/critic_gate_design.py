"""main.critic_gate_design — WHAT THE DESIGN REGISTERS, in one declaration.

Every constant `main.critic_gate` gates on, separated from the code that gates, so that:

* the falsification clause can be asserted VERBATIM against the design file by
  ``critic_gate_test.py`` (a paraphrase would let the tool print a sentence the design never said);
* a bar that is the design's and a threshold that is only this TOOL's are visibly different things
  — ``STALL_RATE_SOURCE_NOTE`` exists because §4.3 registers G7 with no number, and a reader must
  never take our default for a pre-registration;
* the CLI and the renderer share one copy rather than two that can drift.

**G1's bar is deliberately absent.** It is the committed baseline artifact's own measured
resolution, read at run time from ``DEFAULT_BASELINE_DIR`` — hardcoding it here is exactly how a
bar and the record it came from drift apart.

Source: ``designs/ai_v12/design_winprob_only_critic.md`` §4.1 / §4.3 / §5.5.
"""
from __future__ import annotations

from typing import List, Tuple

from utils.paths import repo_path

#: The committed baseline this gate's G1 bars are READ from (never hardcoded). §4.1.
DEFAULT_BASELINE_DIR = str(repo_path("designs", "research_state", "measurements",
                                     "winprob_critic_baseline_2026-09-06"))
#: The artifact file inside it that is quotable — the SELECTION-REWEIGHTED one (§4.2). The raw
#: capture-quota table INVERTS the verdict, so it is not an alternative.
BASELINE_ARTIFACT = "selection_reweighted.json"
#: The design this gate implements; quoted in every report so a reader can check the bars.
DESIGN_DOC = "designs/ai_v12/design_winprob_only_critic.md"

#: §4.3's constant bars.
G2_MAX_RELIABILITY = 0.005
G3_MAX_ECE = 0.05

#: §5.4 chose the indicator terminal so a timeout ranks WITH a loss; G7 is the KILL condition that
#: watches the anti-stall defences §3.2/§3.3 removed. The design registers "no increase over the
#: era, pre-registered threshold" and names NO NUMBER — so these defaults are the TOOL's, and every
#: report says so rather than letting a reader take them for the design's.
DEFAULT_MAX_STALL_RATE = 0.05
DEFAULT_MAX_EP_LEN_RATIO = 1.25
STALL_RATE_SOURCE_NOTE = (
    "the design registers G7 as 'no increase over the era, pre-registered threshold' and names no "
    "number; this threshold is main.critic_gate's default, not the design's")

#: The strata G1-G4 are gated on. `all` is reported for context and deliberately NOT gated: it
#: averages two populations whose measured calibration bias has OPPOSITE SIGN
#: (designs/learning/win_prob_decomposition.md axis 3).
GATED_STRATA = ("bot", "pool")

#: §5.5, VERBATIM. `critic_gate_test.py` re-reads the design and fails if these drift apart.
FALSIFICATION_CLAUSE = (
    "What would falsify the design, stated before the data: G1 flat (resolution unmoved) with "
    "G2–G4 passing means the promotion bought calibration this head already had and nothing else "
    "— the wrong-meter trap, and the target/readout diagnosis of §2 would survive intact while "
    "*this* remedy for it would not. That must be reported as loudly as a pass."
)

#: §4.3 criteria this tool cannot compute. Printed every run — never quietly dropped, because
#: "a gate with three unrunnable criteria is a gate that will be quietly reduced to the runnable
#: ones under time pressure" is the design's own warning about itself.
NOT_RUNNABLE: List[Tuple[str, str, str]] = [
    ("G5", "sd_true_excess, floor-subtracted, per population", "gap M1 — not runnable from traces"),
    ("G6", "the MIRROR TABLE (no cell crossing 0.50)", "gap M2 — not runnable from traces"),
    ("G8", "win_mask coverage >= a pre-registered floor", "gap M3 — the run must record it"),
    ("G9", "capacity value_pooled participation ratio", "runnable, but by `python -m main.capacity`"),
]

#: The 95% normal multiplier, one copy.
Z95 = 1.959963984540054
