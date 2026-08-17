# RUNBOOK — gen-13 end-of-run audit battery → the gen-14 config

**Pre-registered 2026-08-17, BEFORE gen-13 launches** — decision rules written before the numbers
exist (the concat-deletion precedent). Run from the RUN'S OWN pinned worktree; write reports into
`designs/research_state/measurements/` with provenance.

Gen-13 = `ai_v9_15_gen13_hb_events_stack_0817`, fresh-init at **v89** (`1fa4733`), 25M steps.
Delta from gen-12: `--history-events` (v81) · `h` graduates + `r` added to the family string ·
`--item-belief` (v83) · `--intent-threshold` (v84) · `--intent-conditional` (v85) ·
`--damage-matrices both` · `--op-drop-renders` + `--op-believed-lean` (v86) ·
`--value-entity-pool-full` (v82) · `--value-clock` + `--value-intent` (v87).

---

## 0. ⚠️ THE ARTIFACT RULE — decided in advance, because it was ambiguous last time

`python -m main.endofrun` §1 reads **`main.elo`'s fit over `eval_results.jsonl`** (the SPARSE
in-run fit, se ≈ 14.4/node). The frozen-vs-frozen `snapshot_ladder/ladder.json` is a DIFFERENT and
tighter artifact (se ≈ 10/node). On gen-12 the two disagreed materially — sparse Δ −24.8
CI [−65.4, +15.8] vs dense Δ −12.4 CI [−40.3, +15.5] — and the runbook prose ("dense offline
anchored ladder") pointed at one while `feedback_elo_reading_rules` pointed at the other. Choosing
after seeing the numbers is exactly the renegotiation pre-registration exists to prevent.

**RULE, fixed now:** the headline verdict is the **`snapshot_ladder/ladder.json` tail-4**, at
matched snapshot COUNT, at run END. The sparse `main.elo` number is reported alongside as
ORIENTATION and is never the verdict. If the two disagree in VERDICT, that disagreement is itself
reported as a finding and the dense one stands.

## 1. Non-inferiority vs gen-12 (the generation gate)

- Tail-4 mean of the dense ladder, both runs, matched snapshot count.
- **NON_INFERIOR** iff `Δ ≥ −15.0` **AND** `CI95-low > −40.0` (unchanged margins).
- **INFERIOR** iff the whole CI sits below −15. Otherwise **INCONCLUSIVE**.
- ⚠️ **An INCONCLUSIVE result is not a pass.** Gen-12 was INCONCLUSIVE and proceeded anyway on an
  owner call; that is a decision, not a verdict, and it must be recorded as one again if repeated.
- **Tie-break for INCONCLUSIVE, pre-authorized:** add games to the frozen ladder rather than
  re-slice the window. `load_games` SUMS duplicate lines by design ("independent samples of the
  SAME frozen matchup pool"), so 100 → 300 games/pair cuts per-node SE ~√3 and is pure variance
  reduction on a stationary Bernoulli. `play_pairs` currently skips measured pairs, so this needs
  a small force/extra-games flag. **Never** widen the tail-K to change a verdict.

## 2. The FIVE value routes — did they finally do anything? (THE headline of this generation)

Gen-13 is the **first run in which any of them can affect the critic** (v89 `1fa4733`; before it
`--value-from-dist` bypassed `vf_combined` entirely and gen-11/gen-12 trained them at exactly zero
gradient). So this is a genuine first measurement, not a re-read.

- **Liveness (necessary, not sufficient):** every route's zero-init projection must be off zero —
  `value_entity_pool.out_proj`, `intent_value_reduce.proj`, `value_clock_route`, `value_intent_route`,
  the v84 p_KO route. A 6k-step smoke already moved all five, so a 25M run reading zero on any of
  them means a REGRESSION IN THE WIRING, not a weak feature. `value_route_gradient_test.py` is the
  standing guard.
- **Effect:** `critic_route_audit` per-route |dV| — and note these arms are now MEANINGFUL for the
  first time. Compare against `threat` (the one route that was live all along) as the reference
  scale, not against zero.
- **Decision:** a route at ≥ half `threat`'s |dV| KEEPS. All five null ⇒ the v74/v80/v82/v84/v87
  critic-route program is a measured dead end and gen-14 deletes the lot (the honest outcome the
  two inert generations could never deliver).

## 3. `h` re-read + the `r` verdict

- `h` graduated on gen-12's §2 (|dV| 0.1618 vs median live family 0.0392 = 4.1×). Re-read it
  ALONGSIDE `r` and `--history-events`: `h` is compiled pair-history, the event seats are the same
  content in event form, so a large `h` drop when the seats are on is EVIDENCE THE SEATS CARRIED IT,
  not a regression.
- `r` (H-C reference edges) uses the same rule as `h` did: ALIVE at ≥ 0.5 × median live family.
  Zero-init ⇒ any nonzero is learned use.

## 4. H-B event seats — the gate on gen-14's frame deletion

The `event_seats` ablation arm (key-mask ALL H-B seats) + the seat usage audit.

- **Bar:** the seats must carry at least as much as the 7×159 TurnDelta frames they are meant to
  replace. Read the seat arm's |dV| + masked-KL against the frame content's own dependence.
- **KEEP + proceed:** seats at or above the bar ⇒ gen-14 deletes the 7×159 lag frames + the
  prev-turn action mask (−1124 dims), ALONE in its generation (the one non-zero-init deletion;
  `design_history_entity.md` row-2).
- **HOLD:** seats below the bar ⇒ the frames stay and H-B is re-examined before any deletion.

## 5. Mechanic usage (G2 — did the v84/v85 conditional cells move behavior?)

`python -m agents.model.mechanic_usage_baseline` on gen-13's traces vs
`measurements/gen12_mechanic_usage_baseline.json`. The cells exist to close the gap between a
mechanic's PICK rate and its own predicted probability. Gen-12's end-of-run reference:

| move | picked | mean prob |
|---|---|---|
| counter | 6.0% | 9.3% |
| pursuit | 3.9% | 7.3% |
| substitute | 1.9% | 4.7% |
| endure | 0.0% | 0.5% |
| protect | 24.5% | 21.0% |
| destinybond | 15.8% | 13.2% |

**Read the GAP, not the level** — a pick rate moving toward its own probability is the cells
working; both moving together is a policy shift, not conditional execution.

## 6. Wave-1 critic deletions — now HYGIENE, and re-derive the license here

`MultiSeedValueReadout` + `seed_diagnostics`, the `hidden_opp_belief` VF half, and the
`non_matchup_rest` VF concat sit in `vf_parts` beyond `value_pooled`, which `--value-from-dist`
still does not read. They are provably zero-gradient for the critic, so deleting them is code
removal, not a critic change — **do not report it as one**, and do not bundle it with a behavioral
arm. `--value-threat-inject` is NOT in this set (it writes into `value_pooled` and trained).

## The gen-14 draft (edited by the verdicts above)

```
gen-13's config
  + the 7x159 TurnDelta frame deletion        # §4 KEEP only — alone in its generation
  - the value routes that nulled in §2        # deletion, per §2's all-null branch
  + unconditionalize the riders gen-13 adopted (drop-renders / believed-lean / item-belief),
    deleting their legacy branches            # only if no regression is attributable to them
  - c1/c3/c5/x from the family string (+ c2/c4 if the G3 verdict says the CELLS carry it)
```
Re-read **d3** after believed-lean: a channel carrying DISTORTED content also reads low, so the
lean fix may REVIVE it — that is a KEEP signal, not a delete signal
(`design_opponent_intent` §7a(3)).

---
*Verdicts are decision-support against these rules; this file is the registration of record.*
