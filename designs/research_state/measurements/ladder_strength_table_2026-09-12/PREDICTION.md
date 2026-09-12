# PRE-REGISTRATION — the critic ladder's fifteen 10M arms, read on STRENGTH as a family

**Registered 2026-09-12, committed BEFORE the matched-count refit was computed.** Zero GPU. No
file under `models/` is written by anything in this directory.

---

## 0. THE QUESTION

The win-prob critic ladder (`designs/research_state/winprob_critic_ladder_2026-09-08.md`) has
fifteen completed 10M arms. Every one of them was read on CRITIC rows — G1 resolution, identity
bias, the turn contrast, the conditioning decision row — and **none was ever read on STRENGTH as a
family**. §2b carries exactly one strength number for the whole campaign: "ladder by **45 Elo**"
between `ctrl10M` and `ctrl10M_b`, quoted once, as a floor.

So: does any lever on this ladder move the dense anchored rating at 10M, once the run-to-run floor
is measured rather than assumed?

## 1. THE ROSTER (fifteen completed arms; one live arm EXCLUDED)

`ai_v12_10_ladder_vf15`, `ai_v12_11_ladder_ctrl10M`, `ai_v12_12_ladder_cflabels`,
`ai_v12_13_ladder_tdaux`, `ai_v12_14_ladder_truevalue`, `ai_v12_15_ladder_ctrl10M_b`,
`ai_v12_16_ladder_ctrl10M_c`, `ai_v12_17_ladder_strata`, `ai_v12_19_ladder_lambda09`,
`ai_v12_20_ladder_denseaux`, `ai_v12_21_ladder_lambda09_b`, `ai_v12_22_ladder_lambda095`,
`ai_v12_23_ladder_rollout`, `ai_v12_24_ladder_strata_b`, `ai_v12_25_ladder_vf15_b`.

`ai_v12_26_ladder_ctrl10M_shaped` (the matched-pin SHAPED-critic control) is **LIVE on the GPU and
has no `snapshot_ladder/` directory at all** — it is excluded, not read, and nothing of its is
touched. The two `.DEAD_*` directories are dead relaunch stubs and are excluded.

## 2. THE INSTRUMENT, AND WHY

- **Headline = `<run>/snapshot_ladder/ladder.json`, the dense ±10 ladder — never `eval/elo`**
  (`UNDERSTANDING.md` §3.2 rule 1).
- **`python -m main.elo` has NO matched-count cross-run mode** — it takes a single positional
  `run_dir` and writes `<run_dir>/elo/`. It is therefore NOT used (it would also write under
  `models/`). `python -m main.critic_gate --at-snapshots N` *does* implement the matched-count rule,
  but only PAIRWISE (`run` vs `--parent`) and it drags four other endpoints and a GPU-free-but-slow
  untaught meter with it. This read calls the **same library function both of them sit on**:
  `agents.training.snapshot_ladder.fit_ladder(run_dir, first_n=4, steps=…, write=False)`, which
  `fit_ladder`'s own docstring names as the definition of "matched SNAPSHOT COUNT", and which
  cannot write (`first_n` forces `write = False`).
- **Matched count = 4, over the IDENTICAL step set `{4000032, 6000000, 8000016, 10000032}`.**
  Twelve arms rate exactly those four nodes. Three — `lambda09`, `lambda095`, `rollout` — rate a
  fifth, EARLIER node at `2000016` (they promoted at 2M; the self-play crossing is a per-arm
  lottery, rule 15's 2026-09-10 amendment). A naive `first_n=4` on those three would keep
  `{2M,4M,6M,8M}` and read their **8M** node against everyone else's **10M** — matching the count
  while un-matching the steps, which is the confound the rule exists to remove. So the three
  five-node arms are refit on the COMMON four steps, dropping their 2M node and its edges. Every
  arm is then the newest node of a four-node fit over the same four steps: matched **count**,
  matched **fit size**, matched **step set**.
- **Newest-node inflation (§3.2 rule 2) is handled by symmetry, and CHECKED.** All fifteen runs are
  FINISHED — no arm will gain a node, so no arm's 10M rating is the transient newest-node value of a
  run still growing. Every arm's 10M node occupies the identical structural position (newest of a
  four-node fit), so the inflation is **common-mode and cancels in a pairwise Δ to first order**.
  The registered cross-check: the **8M node of the same four-node fit** (the second-newest, already
  refit once). If the arm ORDERING and the floor verdict are the same on the 8M row as on the 10M
  row, inflation is not driving the read; if they disagree, the read is reported INCONCLUSIVE on the
  disagreeing arms rather than resolved in favour of either row.

## 3. THE FLOOR — registered definition

**floor = the widest pairwise |Δ| among the three controls** `ai_v12_11_ladder_ctrl10M` /
`ai_v12_15_ladder_ctrl10M_b` / `ai_v12_16_ladder_ctrl10M_c`, measured at the matched count by this
read (standing rule 3: *a floor is the MAX pairwise |Δ| over the replicates in hand, never the mean
and never the smaller single-pair bar*). The ledger's 45 Elo (2026-09-09 · *THE FLOOR*) is the PRIOR
this number reads against; it is not adopted.

Three controls ⇒ three pairwise deltas ⇒ the floor is the max of three. It bounds the CONTROL's
run-to-run variance only (rule 22) — a lever may change the variance it is read against.

## 4. THE DECISION RULE — registered before any refit exists

An arm is **OUTSIDE THE FLOOR** only if **all three** of the following hold:

1. |Δ vs `ctrl10M`| > floor, **and** |Δ vs `ctrl10M_b`| > floor, **and** |Δ vs `ctrl10M_c`| > floor
   (the three-control requirement given in the task and in §2b's practice); and
2. all three Δ have the **same sign** (an arm above one control and below another is not outside a
   floor, it is inside the control spread); and
3. **each Δ carries its own 95% CI and that CI excludes the floor in the same direction** — rule 6,
   *equivalence needs the DELTA's own CI inside the bar; a bar against a POINT estimate is vacuous*.
   Δ CI is formed from the two BT standard errors of the matched-count fit,
   `se(Δ) = sqrt(se_arm² + se_ctrl²)`, ±1.96·se(Δ). The bot anchors are pinned constants shared by
   every fit, so the two node SEs are treated as independent.

An arm meeting 1 and 2 but not 3 is a **CANDIDATE, not a detection** (rule 22). An arm meeting none
is **WITHIN FLOOR**. "Within floor" is reported as NOT DETECTED — never as "equal".

## 5. PREDICTIONS (both branches stated)

- **P1 — the headline.** **No arm is outside the floor.** Predicted floor 35–55 Elo; predicted arm
  range (max − min across all fifteen 10M ratings) 100–140 Elo, i.e. **wider than the floor** — so
  the family will LOOK spread while no individual arm clears the three-control test. Rationale: the
  three-control conjunction plus the Δ CI is a demanding bar at n = 1 per lever, and every one of
  these arms has already read NOT DETECTED on its own critic rows.
- **P2 — the alternative, stated so it can win.** If a lever does clear all three clauses, the
  prediction is that it is one of the two levers with a non-null critic-side history — `strata`
  (the one decision-row detection, since failed to replicate) or `vf15` (the one DOWN detection,
  replicate `vf15_b` landed 2026-09-12) — and that its SIGN matches its critic-row sign
  (`strata` up, `vf15` down).
- **P3 — the instrument's own resolution.** `se(Δ)` will come out **≈ 20–25 Elo, i.e. a 95% CI of
  roughly ±40–50 Elo**, which is *the same size as the predicted floor*. If that holds, clause 3
  can essentially never fire at this fit size, and the honest verdict is **the dense ladder at four
  nodes cannot resolve a 45-Elo run-to-run floor** — a statement about the instrument, registered
  here so it cannot be presented as a discovery about the arms.
- **P4 — the descriptors.** `eval_sentinel_greedy` will be `true` for all fifteen (all launched
  after the 2026-09-07 regime change), so `win_rate_vs_bots` is comparable within the family but
  not against anything pre-2026-09-07. Predicted `win_rate_vs_bots` at 10M: 0.93–1.00 for every
  arm, i.e. **SATURATED and unable to order the family** — which is the whole reason the dense
  ladder exists (`snapshot_ladder.py`'s module docstring: the bots "have SATURATED").
- **P5 — the ordering cross-check.** The Spearman rank correlation between the 10M row and the 8M
  row across the fifteen arms will be **positive but well below 1 (0.4–0.8)**, because a ~45 Elo
  floor on a ~120 Elo spread scrambles the middle of the table.

## 6. 🚨 DISCLOSURE — what had already been seen when this was registered

Honesty about the strength of this registration, because it changes how much the predictions above
are worth:

**While establishing the roster (which arms are complete, how many nodes each rates, and whether the
live arm has a ladder at all) I read the committed `ladder.json` of all fifteen arms, and therefore
SAW their committed 10M ratings, before writing this file.** I had not computed, and have not
computed, the matched-count refit, the floor, any Δ, or any SE.

For the twelve four-node arms the committed fit is *already* a four-node fit over the common step
set with the eval-sentinel edges dropped (`eval_sentinel_edges_dropped: 6` in every file), so the
refit is expected to reproduce their committed numbers closely — meaning P1 and P2 are, for those
twelve, **INFORMED predictions, not blind ones**. P3, P4, P5 and the three five-node arms' refits
were not informed by anything.

What this registration is genuinely worth, therefore, is **the DECISION RULE of §4 and the floor
DEFINITION of §3, fixed before the floor and the deltas existed** — which is the part that has
historically been fitted in this programme (the 2026-09-09 RETRACTION, rule 21's selection
inflation). It is not worth a claim that the numbers were unseen. A future read that wants a blind
registration on this table must register before the roster survey, not after.
