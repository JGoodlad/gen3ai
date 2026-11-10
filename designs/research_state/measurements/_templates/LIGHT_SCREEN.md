# <ID> — <one-line title>: LIGHT SCREEN (<YYYY-MM-DD>)

<!-- Copy to measurements/<id>_<slug>_<date>/SCREEN.md and fill every field. The rules are
     designs/research_state/README.md § "Registration tiers". The whole file stays ≤ 6 KB: when it cannot,
     the experiment is not a screen. Delete these comments. -->

**Tier: LIGHT SCREEN. Status: REGISTERED by this file's landing commit, before the first step, game or look.** No amendments: a change to anything below is a NEW screen in a new folder that names this one.
**This screen can only PROMOTE its candidate to a full registration or DROP it. Its result is never a verdict and
is never quoted as a claim.**

## 1. Question

<One or two sentences. What is the candidate, and what would make it worth a full registration?>

## 2. Eligibility (all four must be true, or write a full PREREG)

| check | value |
|---|---|
| GPU cost ≤ ~15 GPU-h | <N GPU-h, from which measured rate> |
| CPU cost ≤ ~50 lane-h | <N lane-h, from which measured rate> |
| adopts nothing (no recipe, architecture, baseline or default changes on its result) | yes |
| supports no goal claim, and no doc will quote its number as a finding | yes |

## 3. Arms and pin

| arm | what differs from the control (ONE lever) | run name(s) | start | pin (an EXISTING commit) |
|---|---|---|---|---|
| control | <banked runs, named; or trained here> | | | |
| candidate | | | | |

Launch proof (paste, do not paraphrase): `python -m main.checkargs` and `python -m main.launcher --dry-run` output
for each new argv, or the `main.reads check-spec` line for a read.

## 4. Seeds, checked disjoint

Seeds: <list>. Checked disjoint from every registered sibling's training seeds and unit seeds by <the command
run and its output, e.g. the kit's seed table or a grep of the sibling specs>. A seed shared with a control on
purpose (a paired design) is named here with its reason.

## 5. Meter

<ONE primary meter: the panel, the opponent, the units per final, the regime, and the interval. Any Kakuna
number carries the TEAM-CLUSTERED interval. Name the leak and determinism guards that exist for this meter
(`main.reads`' guards run unchanged; a search or label read names its leak audit) and state that they run.>

## 6. Decision rule (≤ 5 lines, ONE look)

<PROMOTE if …; DROP if …; anything else is DROP unless a line here says INCONCLUSIVE and what that does.
Every comparison is deterministic (root rule 8): state the exact statistic and threshold, and exclude an input
within rounding error of the boundary instead of tolerating it. A timeout is never an outcome.>

## 7. What each outcome does

- **PROMOTE:** <the full registration that gets written, and what it will test that this screen did not>.
- **DROP:** <the backlog row that closes; what, if anything, would reopen it>.

## 8. Cost and stop

<GPU-h and lane-h as budgeted in §2; the hard stop (steps, units or hours) at which the screen ends whatever it
shows. A screen that overruns its §2 bounds by more than ~25 % stops and is reported, never extended.>

## 9. Result (filled AFTER the look; the only section written after registration)

<The meter with its interval, the rule's line that fired, PROMOTE or DROP, and the ledger entry
`### YYYY-MM-DD · SCREEN · <ID> …` by its landing sha.>
