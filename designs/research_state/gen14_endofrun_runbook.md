# RUNBOOK — gen-14 end-of-run audit battery → the gen-15 config

**Pre-registered 2026-08-17, BEFORE gen-14 launches.** Every rule below is fixed while the number
it governs does not yet exist. Run from the RUN'S OWN pinned worktree; write reports into
`designs/research_state/measurements/` with provenance.

Gen-14 = the **frame-deletion** generation. Its one behavioural change is the removal of the 7×159
TurnDelta lag frames + the prev-turn action mask (−1124 obs dims), licensed by gen-13.5 §4
(`gen13_frames_arm_section4.json`: event_seats dV 2.7714 vs frames 1.3015, ratio 0.47). The frame
deletion **bumps `ARCH_SIGNATURE`**, so gen-14 is **fresh WEIGHTS** — gen-13's final is the eval
reference and pool seed, never a warm start.

Riding along (hygiene only, no behavioural claim): the `seed` readout + `seed_diagnostics` + the
`value_seeds/*` TB contract · the `hidden_opp_belief` **VF half** (the pi half KEEPS) ·
`value_intent` · `intent_threshold_value` · the `nmr` VF concat if its arm still reads 0.0.

---

## 0. Conventions inherited from gen-13's runbook (do not re-derive)

- **The ELO artifact rule (§0 there).** The headline verdict is `snapshot_ladder/ladder.json`
  tail-4, at matched snapshot COUNT, at run END. `main.elo`'s sparse fit is ORIENTATION and is
  never the verdict. A verdict disagreement between the two is itself reported.
- **The tie-break zone.** A route missing the bar by ≤25% is re-audited at ≥2× sample; deeper is a
  NULL with no tie-break.
- **INCONCLUSIVE is not a pass.** Proceeding anyway is recorded as a DECISION, not a verdict.
- **⚠️ n is specified on the SUBSET a reading actually lives on.** Gen-13.5 found §7b's `n ≥ 25` was
  written on DECISIONS while the confident-blind fraction lives on the win_prob ≥ 0.7 subset —
  which was 7 and 3. Every reading below that conditions on a subset states its own n.
- **The falsification set for any IRREVERSIBLE deletion** (now the standard, from §4): a POSITIVE
  CONTROL that must read large, a NULL arm that must read EXACTLY zero, and where possible an
  INDEPENDENT ROUTE to the same quantity. A deletion arm without all three is not evidence.

## 1. Non-inferiority vs gen-13 (the generation gate)

Tail-4 of the dense ladder, both runs, matched snapshot count. **NON_INFERIOR** iff `Δ ≥ −15.0`
AND `CI95-low > −40.0`. **INFERIOR** iff the whole CI sits below −15. Else INCONCLUSIVE.

Tie-break for INCONCLUSIVE, pre-authorised: add games to the frozen ladder (`load_games` SUMS
duplicate lines by design), **never** widen the tail-K.

## 2. THE FRAME DELETION — did it cost anything?

The deletion's licence was a DEPENDENCE reading on gen-13, where the seats were present. This is
the first run trained WITHOUT the frames, so it is the first test of the claim itself.

- **Primary:** §1's ladder verdict. A NON_INFERIOR gen-14 retires the frames permanently.
- **Mechanism:** the `event_seats` arm re-read on gen-14. If the seats' dV RISES relative to
  gen-13's 2.7714, they absorbed the frames' role — the substitution the deletion assumed.
- **The falsifier:** if gen-14 is INFERIOR **and** `event_seats` did not rise, the frames carried
  something the seats do not, and the deletion is REVERTED in gen-15 rather than explained away.

## 3. The two TIE-BREAK re-audits (registered, ≥2× sample)

Run `critic_route_audit` at **≥12000 states** (2× gen-13's 6000):

| route | gen-13 dV | rule |
|---|---|---|
| `intent_value_reduce` | 0.3826 | **< 0.39 here ⇒ DELETE, no appeal** |
| `value_clock` | 0.3370 | **< 0.39 here ⇒ DELETE, no appeal** |

`value_clock` context for the report ONLY, never an appeal path: C1's causal sweep showed the clock
CONTENT already reaches the critic through the trunk at ~83% of an HP-control's responsiveness, so
a low DIRECT-route reading may be substitution rather than deadness (the `nmr` pattern). That makes
the eventual deletion low-risk; it does not make the number mean something else.

## 4. `threat` — the DEADLINE fires here

`--value-threat-inject` was held at gen-13 with a registered deadline. **At this audit, threat
DELETES — no appeal — if its dV reads < 0.39.** At or above, it has proven independent content and
leaves the candidate list permanently.

## 5. `r` — the re-read

`r` passed gen-13's bar weakly (0.0571 vs a 0.0276 bar, sitting essentially AT the median 0.0551).
Same rule: ALIVE at ≥ 0.5 × median live family. A second weak pass with the frames gone is a KEEP;
a fail retires it with the consequence families.

## 6. THE RIDER ATTRIBUTION ARM — the decision-5 discharge

Gen-13.5 §5 exonerated the v84/v85 CELLS and said nothing about the RIDERS, so decision 5 is
**undischarged** and `--op-drop-renders` / `--op-believed-lean` / `--item-belief` stay conditional.

Build a rider-specific arm: ablate **each rider** one at a time (not the cells) on frozen
on-distribution decisions, and re-measure the mechanic pick-vs-prob gaps
(`tmp/mechanic_attribution.py` is the shape; it needs rider arms added).

**Discharge rule, fixed now:** a rider unconditionalizes in gen-15 **only if** its ablation moves
no mechanic's pick-vs-prob gap by more than the between-arm noise measured on the base arm. Any
rider that IS implicated stays conditional and gets its own investigation. Unconditionalization is
a one-way door; absence of a test is not absence of a defect.

## 7. Critic calibration — the §7 successor investigation

Gen-13.5's §7 FAILED with route liveness PROVEN: the critic is significantly over-confident on
stall losses (mean gap +0.358, CI [0.227, 0.504]; confident-blind 0.500, CI [0.290, 0.724]) and
gen-13 was NOT separably different from gen-12 on either reading. The delivery line is exhausted.

**The named successor, assigned to this window (NOT gating launch): measure the training
DISTRIBUTION of stall games.** What fraction of rollout decisions, and of value-loss mass, comes
from loss-side stall trajectories — against their share of eval losses. That number decides whether
the stall blindness is a COVERAGE problem the flywheel's thermostat can fix, or a representation
problem that needs input work.

If a §7-style calibration reading is repeated here: gen-13's number read FIRST, same script
version, battle-CLUSTERED bootstrap, **and n specified on the confident subset**. Report the
BETWEEN-RUN DIFFERENCE with its own CI — never two separate CIs, which is exactly how gen-13.5
nearly manufactured a "gen-13 got worse."

## 8. Re-entry condition for an α/β critic route (registered 2026-08-17)

`value_intent` is deleted in gen-14, which removes the ONLY α/β→critic route (C1's "one input with
no trunk substitute"). Rebuild cost via the v89 `_value_pooled_routes` seam is trivial, so the
rule-following deletion stands — but:

> **Any future proposal to restore an α/β critic route must first pass a C4-style OFFLINE HEAD
> GATE — frozen tokens, pre-registered metric — before ANY training spend.**

## The gen-15 draft (edited by the verdicts above)

```
gen-14's config
  ± the TD-consistency aux (--td-aux-coef), IF the C5 rung-2 forks pass their frozen gates
  - whichever of intent_value_reduce / value_clock / threat the re-audits condemned
  + unconditionalize whichever riders §6 discharged (and DELETE their legacy branches)
  - c1/c3/c5/x from the family string (+ c2/c4 if the G3 verdict says the CELLS carry it)
  + whatever the §7 stall-distribution probe indicates — coverage work, NOT another delivery route
```

---
*Verdicts are decision-support against these rules; this file is the registration of record.*
