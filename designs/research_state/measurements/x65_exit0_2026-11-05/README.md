# X65 ExIt-0: the read (2026-11-06) — NOT-DELIVERED

**Class: NOT-DELIVERED** (PREREG §1, §5). The delivery gate's lower bound on Δagree is −1.40 pp, not above 0. No
Kakuna game of this read is played. The registration is [`PREREG.md`](PREREG.md), with Amendments 1–3, and the
review is [`REVIEW.md`](REVIEW.md). The design is
[`design_x65_exit0_search_distillation.md`](../../../endstate/design_x65_exit0_search_distillation.md).

**What it licenses (§1).** "At this recipe the students did not learn S′'s choices on unseen teams." X60-T is the
GPU's next candidate. ExIt-1 is not funded.

**What it does NOT license.** Anything about larger budgets, honest labels (ExIt-0H) or in-loop distillation.

## The gate (PREREG §5)

The gate scores every label row of each student's read panel. A label row is one of S′-LIVE's verified searched
decisions on the panel the student never trained on.
- a(student) = 1 when the exported student's greedy action equals S′'s pick (`played`);
- a(P\*) = 1 when P\*'s recorded greedy action (`pi_argmax`) equals `played`;
- Δagree is the mean of a(student) − a(P\*).

The interval is a bootstrap by deal within each panel: B = 4,000, `default_rng(20261105)`.

| panel | student | label rows | deals | agree, student | agree, P\* | Δagree |
|---|---|---|---|---|---|---|
| H | S_H | 32,020 | 900 | 0.6174 | 0.6280 | **−1.06 pp** |
| F | S_F | 33,192 | 895 | 0.6280 | 0.6374 | **−0.94 pp** |
| p72 | S_P | 32,183 | 900 | 0.6186 | 0.6365 | **−1.78 pp** |
| **pooled** (mean of the three) | | 97,395 | | | | **−1.26 pp [−1.40, −1.11]** |

- **LB = −0.01402**, so DELIVERED is false. It is not at the boundary: |LB| ≫ 0.0005.
- **By hand.** [`delivery_by_hand.py`](delivery_by_hand.py), with loops of its own, gives every Δagree, the pooled
  Δ, the LB and DELIVERED equal to the gate's file. See [`record/delivery_by_hand.json`](record/delivery_by_hand.json):
  `agrees: true`.
- **The rule.** `python -m main.reads decide` on the planned, unplayed look-1 state classes NOT-DELIVERED (Amendment 2's
  reading). See [`record/decision.json`](record/decision.json): 0 student units counted, and G-BANK is unplayed, as it
  should be.
- F has 895 deals with a label row, not 900. The searched unit `pstar_s__forum72__337__g19` is uncounted (Amendment
  2), and four more deals carry no label row (not root-caused; the bank's counts are Amendment 2's).

### Reported (§5), never decided

| panel | Δagree, `attribution_miss` ≥ 1 (rows) | `rewritten_moves` ≥ 1 (rows) | neither (rows) | student agrees at S′'s departures (rows) | agreement with S′ where S′ kept P\*'s pick | pre-turn-10 agreement with P\* | mean \|V_student − V_P\*\| |
|---|---|---|---|---|---|---|---|
| H | −1.03 (9,958) | −1.30 (12,190) | −0.87 (13,950) | 5.4 % (11,910) | 95.1 % | 92.5 % | 0.0035 |
| F | −1.19 (11,115) | −1.26 (12,846) | −0.66 (13,558) | 4.0 % (12,035) | 96.2 % | 95.5 % | 0.0022 |
| p72 | −2.16 (10,954) | −1.70 (10,976) | −1.76 (14,214) | 5.6 % (11,700) | 94.0 % | 92.9 % | 0.0042 |

How to read the departure columns:
- P\* agrees with S′ at none of S′'s departures (by definition) and at all of the rest. Each student recovers 4.0–5.6 %
  of the departures, and loses 3.8–6.0 % of the decisions where S′ kept P\*'s choice.
- There are about twice as many of the latter, so the net is negative.
- The students' moves off P\*'s greedy choice are close to unrelated to where S′ moved.
- No leak-exposure bucket is positive, so leak-exposed labels do not explain the deficit.

### The path check (operator, outcome-free; [`record/forward_check.py`](record/forward_check.py) → [`record/forward_check.json`](record/forward_check.json))

a(P\*) uses the recorded `pi_argmax`, while a(student) uses the gate's offline forward (`delivery._forward`). So a
forward that disagreed with the recorded choice would bias Δagree against the student (REVIEW R-M1's scenario). To
rule that out, the operator ran P\*'s own bytes through the same forward on every label row:

| panel | P\* offline = recorded `pi_argmax` | student = P\* offline |
|---|---|---|
| H | 100 % | 92.8 % |
| F | 100 % | 94.6 % |
| p72 | 100 % | 92.0 % |

The scoring path is exact, so the negative Δagree is the students', not the meter's.

### The training behind it (the GPU session's records; reported)

Each fold's 32 candidates are in `~/gen3ai_archive/x65_exit0_2026-11-05/students/<fold>/pick.json`.

**At lr 1e-5.**
- The held-out label KL falls only from ≈ 0.705 / 0.702 / 0.728 at the first quarter-epoch to ≈ 0.682 / 0.680 /
  0.702 after 4 epochs.
- Meanwhile the anchor KL rises to ≈ 0.022, so inner-validation L is flat to within ≈ 0.007.

**At lr 3e-5.** The label KL never falls below ≈ 0.689 / 0.680 / 0.701, the anchor KL reaches ≈ 0.046, and L rises.

**The picks.** Every pick is at 1e-5 (Amendment 3's table). That holds even on the TRAINING panels' held-out deals,
where the student sees the same teams: one supervised pass at this recipe moved P\*'s policy little toward π′.

## What ran

**Spec fills (Amendment 3, `ff6874fcc`).** The four specs carry the students' zips and sha256s, `pin: 320a535d4`
and the earlier looks' spec sha256s. `check-spec` exits 0 on all four.

**Engine.** At the play pin worktree (`320a535d4`, carrying main's untracked `src/rust_env/Cargo.lock`), the engine
stamp is E4, `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.

**The plan.** Look 1's state `~/gen3ai_archive/x65_exit0_2026-11-05/look1/` was planned from that worktree.
- Its `spec.json` has sha256 `1d4b6d40…5fcd`.
- It holds 135 G-BANK units and 900 student units on each panel.
- It copied four models: P\* `814823c5…`, S_H `7af676dd…`, S_F `e400efb5…` and S_P `ec767f51…`.
- Nothing in it was played.

**The gate.**
- `python -m main.x65.delivery score` ran for each panel, from a checkout at `ff6874fcc` with the CPU only, at
  nice 19 on cores 5-7,13-15 under `mem_cap` (≈ 2.4 min and 2.4 GB a panel). `gate` then wrote
  `~/gen3ai_archive/x65_exit0_2026-11-05/delivery.json` (sha256 `952c2b3e…5b3f`).
- The score files are in `…/delivery/<panel>.npz`: H `6f1a98b7…`, F `f72db4e6…`, p72 `c27147f7…`.

**Decide.** It ran from the same checkout and wrote `look1/decision.json` (sha256 `b4eee149…1987`).

**What did not run.** `record` refuses an unplayed state. Per Amendment 2, the read's record is `decision.json` plus
the delivery file, both copied to [`record/`](record/).

## FINDINGs

- **X65-R1 (the order).** The operator's brief listed B3's smoke game before the delivery gate. The smoke
  (`export.smoke`) is one game against a local Kakuna peer, and PREREG §5 reads the gate "before any game". The
  operator therefore ran the gate first, and on NOT-DELIVERED did not run the smoke. **No game of any kind was played
  under this read.** B3's smoke is unrun: `export.json` keeps `smoke: null`.
- **X65-R2 (what NOT-DELIVERED does not separate).** The labels may be hard to fit from the observation, which a
  leak-inclusive π′ predicts: a per-world best response that the information set cannot support. Or the recipe may be
  too weak: β = 1, ≤ 916 steps, lr ≤ 3e-5. The held-out label KL on the training panels' own deals moved ≈ 3 %, so
  the recipe did not fit π′ even on familiar teams. This read cannot tell the two apart. ExIt-0H (honest labels) or
  a recipe probe would be a new registration.
- **X65-R3 (the ExIt-1 control).** `rb_x65_c0_s6501`, ExIt-1's same-pin C0, was trained before this verdict, and
  its arm-end records landed at `bf94b3cb4`. A NOT-DELIVERED does not fund ExIt-1 (§1), so that run has no
  registered consumer under X65.
