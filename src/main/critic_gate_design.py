"""main.critic_gate_design — WHAT THE DESIGN REGISTERS, in one declaration.

Every constant `main.critic_gate` gates on, separated from the code that gates, so that:

* the falsification clause can be asserted VERBATIM against the design file by
  ``critic_gate_test.py`` (a paraphrase would let the tool print a sentence the design never said);
* a bar the design REGISTERED and a bar that survives the 2026-09-06 owner ruling are visibly
  different things — ``G2_MAX_RELIABILITY`` / ``G3_MAX_ECE`` are still here and still §4.3's, but
  they are ASPIRATIONAL and the gate is ``RELATIVE_BARS`` (see ``OWNER_RULING_2026_09_06``);
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

from typing import List, NamedTuple, Tuple

from utils.paths import repo_path

#: The committed baseline this gate's G1 bars are READ from (never hardcoded). §4.1.
DEFAULT_BASELINE_DIR = str(repo_path("designs", "research_state", "measurements",
                                     "winprob_critic_baseline_2026-09-06"))
#: The artifact file inside it that is quotable — the SELECTION-REWEIGHTED one (§4.2). The raw
#: capture-quota table INVERTS the verdict, so it is not an alternative.
BASELINE_ARTIFACT = "selection_reweighted.json"
#: The design this gate implements; quoted in every report so a reader can check the bars.
DESIGN_DOC = "designs/ai_v12/design_winprob_only_critic.md"

#: §4.3's ABSOLUTE numbers for reliability and ECE. **ASPIRATIONAL — printed, never gated**
#: (owner ruling 2026-09-06, ``OWNER_RULING_2026_09_06`` below). They are kept, not deleted,
#: because the design registered them and a reader must be able to see how far an arm sits from
#: the number §4.3 wrote down; what they are NOT is a bar an arm can fail on.
G2_MAX_RELIABILITY = 0.005
G3_MAX_ECE = 0.05

#: The 2026-09-06 finding + ruling, as ONE quotable sentence — printed in every report so a reader
#: of the output never has to hold the design's history in their head. §4.3 carries the same
#: paragraph, dated.
OWNER_RULING_2026_09_06 = (
    "OWNER RULING 2026-09-06: §4.3's absolute G2 (reliability <= 0.005) and G3 (ECE <= 0.05) bars "
    "are ALREADY BREACHED by the committed baseline on the POOL stratum (reliability 0.0064 / "
    "0.0103, ECE 0.0667 / 0.0875 at 26M / 28M), while §4.3 called G3 a 'no-regression clause' the "
    "reweighted baseline already passes — true pooled and on bot, FALSE on pool, and both gates "
    "are registered over 'both classes'. As written the arm had to clear a bar its predecessor "
    "never cleared. G2 and G3 are therefore PER-STRATUM RELATIVE bars: no worse than the "
    "baseline's SAME-stratum value. The absolute numbers stay printed as aspirational targets.")

#: The relative rule, as ONE declaration read by the tool and asserted by the test — never prose.
#: Both metrics are LOWER-IS-BETTER, so "no worse" means "not detectably above".
RELATIVE_RULE = (
    "PASS if the arm's point estimate is <= the baseline's same-stratum value, OR the arm's "
    "cluster-bootstrap CI CONTAINS that value (non-inferiority — never a direction claim). FAIL "
    "only when the arm's whole CI sits ABOVE it.")

#: The label each row prints for WHICH clause decided it. One copy, shared by tool and renderer.
RULE_BETTER = "point <= base"
RULE_NONINFERIOR = "CI covers base"
RULE_WORSE = "CI above base"
RULE_NO_CI = "point only (no CI)"


class RelativeBar(NamedTuple):
    """One per-stratum non-inferiority bar, as DATA.

    ``metric`` is simultaneously the ``reliability_table`` key the arm is measured with and the
    committed artifact's key the baseline is read from — the same name on both sides is what makes
    "matched stratum, matched metric" checkable rather than asserted.
    """

    gate: str            #: "G2"
    metric: str          #: "reliability" — the key in BOTH the gauge's table and the artifact
    label: str           #: how it prints
    aspirational: float  #: §4.3's absolute number. PRINTED, never gated.


RELATIVE_BARS: Tuple[RelativeBar, ...] = (
    RelativeBar("G2", "reliability", "reliability", G2_MAX_RELIABILITY),
    RelativeBar("G3", "ece", "ECE", G3_MAX_ECE),
)

#: How the baseline's several steps reduce to ONE value per stratum, per gate family. G1's is
#: UNCHANGED by the ruling (``max`` = the strictest resolution to beat); the relative bars default
#: to the baseline's own LAST checkpoint. An explicit ``--baseline-reduce`` overrides BOTH.
G1_BASELINE_REDUCE = "max"
RELATIVE_BASELINE_REDUCE = "last"

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
