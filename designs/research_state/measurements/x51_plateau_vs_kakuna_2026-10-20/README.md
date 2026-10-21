# X51 — the plateau P\* against Metamon's Kakuna, on all four team sets — 2026-10-20 / 21

**Question** ([`design_beat_metamon_path.md`](../../../endstate/design_beat_metamon_path.md) §6.1, BM-1). X26's
plateau checkpoint P\* had never been played against Kakuna. Where does it stand on goal (a), our teams, and on goal
(b), unfamiliar teams? Is Kakuna's checkpoint 28 stronger than the checkpoint 34 every read used (the doc's FINDING 1)?
Does the toy T\*, which has no belief stack, lose less on unfamiliar teams?

Abbreviations: pp = percentage points; CI = 95 % interval; LB / UB = its lower / upper bound; P\* = `rb_x26_s2601`
@280,000,170; T\* = `rb_toy_ent_s4002_deep` @290,000,087; H = `heldout72_hpfix`; F = `forum72`; U = the three
unfamiliar panels H, F and P72; n.d. = not detected.

## Answer

- **Goal (a) is MET at the plateau.** On home72, P\* beats Kakuna ckpt 34 **0.598 [0.549, 0.644]** (n = 400, BETTER
  than 0.50). Against X26 @161M that is +0.023 [−0.046, +0.090], n.d.
- **Goal (b) is NOT met, and the plateau closed about a third of the gap.** The pooled unfamiliar score is **0.487
  [0.459, 0.515]** (equal weight over H / F / P72, Kish n_eff 1,208), up from 0.457 at 161M: **+0.030 [−0.010,
  +0.070], n.d.** It is **4.3 pp short of ≈ 0.53**, the score whose Wilson LB clears 0.50 at about 1,200 games.
- **Per panel the plateau moved H and F, not P72.** H 0.497 (+0.050, n.d.), F 0.530 (+0.060, n.d.; paired normal
  interval [−0.002, +0.122]), P72 0.433 (−0.020, n.d.). F's point estimate is above 0.50, its interval is not.
- **The familiarity premium is unchanged:** home72 − U = **+11.1 pp** at P\* (raw), against +11.8 pp at 161M.
- **Kakuna ckpt 28 is NOT detectably stronger than ckpt 34** in our regime: ckpt 28 scores **0.515 [0.466, 0.564]**
  against ckpt 34 (n = 400, a pair cell on home72). §6.1's switch rule (LB > 0.50) does not fire; ckpt 34 stays the
  goal's opponent. The cell cannot exclude a ±6 pp difference either way.
- **The toy T\* is WEAKER than P\* against Kakuna on both sets it played**, and it loses LESS between them:
  home72 **0.493 [0.444, 0.541]**, F **0.458 [0.409, 0.506]**; T\* − P\* = **−0.105 [−0.173, −0.036]** (home72) and
  **−0.073 [−0.141, −0.003]** (F), both WORSE. Its home72 → F drop is 3.5 pp (0.14 logits) against P\*'s 6.8 pp
  (0.28 logits); the difference of drops, +0.14 logits [−0.26, +0.53], is NOT detected (descriptive, X37c's
  question): this read neither supports nor kills "the belief stack is the memorisation channel".
- PROVISIONAL: one seed of one run; a symmetric cell mixes piloting and facing (X37a puts nearly all of the deficit
  in piloting).

## The table (greedy vs greedy, CPU, 20-game role-balanced units, paired seeds 0–19)

| cell | n | W–L–T | win rate [Wilson 95 %] | vs 0.50 | X26 @161M (banked, same seeds) | P\* − X26, Newcombe (headline) | paired descriptor: mean, normal / seed-bootstrap 95 %, discordant P\*-only / X26-only |
|---|---:|---|---|---|---|---|---|
| **home72** (72 / 72 in our pool) | 400 | 239–160–1 | **0.598 [0.549, 0.644]** | **BETTER** | 0.575 [0.526, 0.623] | +0.023 [−0.046, +0.090] n.d. | +0.022; [−0.040, +0.085] / [−0.035, +0.080]; 85 / 76 |
| **H** `heldout72_hpfix` | 400 | 199–201–0 | **0.497 [0.449, 0.546]** | n.d. | 0.448 [0.400, 0.496] | +0.050 [−0.019, +0.118] n.d. | +0.050; [−0.011, +0.111] / [−0.013, +0.113]; 87 / 67 |
| **F** `forum72` | 400 | 212–188–0 | **0.530 [0.481, 0.578]** | n.d. | 0.470 [0.422, 0.519] | +0.060 [−0.009, +0.128] n.d. | +0.060; [−0.002, +0.122] / [−0.025, +0.148]; 93 / 69 |
| **P72** | 400 | 173–227–0 | **0.433 [0.385, 0.481]** | **WORSE** | 0.453 [0.404, 0.501] | −0.020 [−0.088, +0.049] n.d. | −0.020; [−0.078, +0.038] / [−0.078, +0.040]; 67 / 75 |
| **U** (H, F, P72 equal weight; Kish) | n_eff 1,208 | — | **0.487 [0.459, 0.515]** | n.d. | 0.457 [0.429, 0.485] | **+0.030 [−0.010, +0.070] n.d.** (`main.x35.d1.newcombe_strat`) | — |
| **Kakuna ckpt 28 vs ckpt 34**, home72 (pair cell; reported from ckpt 28) | 400 | 206–194–0 | **0.515 [0.466, 0.564]** | n.d. | — | — | — |

- **Twins.** Every P\* game has an X26 @161M twin with the same unit seed, half and game index, and our team is equal
  in every twin (checked by `aggregate.py`'s `paired`: 400 / 400 per set). The paired descriptor agrees with the
  Newcombe headline on every set.
- **Per role** (we challenge / they challenge): home72 0.605 / 0.590, H 0.495 / 0.500, F 0.535 / 0.525, P72 0.400 /
  0.465; the pair cell 0.520 / 0.510.
- **The 250-turn forfeit** fired 1 / 1 / 3 / 2 times (home72 / H / F / P72). The pair cell has no forfeit rule (SOP
  §2). One tie, on home72. Median length 32 / 38 / 37 / 38 turns (39 in the pair cell). Our side drew 68–69 of the 72
  teams per cell, Kakuna 67–68 (from the capture).
- **Overlap** (`gen3_team_overlap_v1`, `cells.json`): home72 72 / 72 exact pool teams; H, F and P72 0 exact and 0
  sharing ≥ 4 mon sets with the pool (mon sets seen, strict: 45 / 100 / 35 of 432).

## The toy cells

`main.anchors` loads T\* at `63f97a57` (`bare`, config v158 like P\*), so FINDING 8 of the design doc is resolved.
T\* (`checkpoint_290000087_steps.zip`, sha256 `cfa7f7be…65e7`, a read-only copy) played the same paired seeds 0–19.

| cell | n | W–L–T | win rate [Wilson 95 %] | vs 0.50 | T\* − P\*, Newcombe | paired: mean, normal / seed-bootstrap, T\*-only / P\*-only |
|---|---:|---|---|---|---|---|
| **T\* home72** | 400 | 197–203–0 | **0.493 [0.444, 0.541]** | n.d. | **−0.105 [−0.173, −0.036] WORSE** | −0.105; [−0.169, −0.041] / [−0.168, −0.040]; 65 / 107 |
| **T\* F** | 400 | 183–216–1 | **0.458 [0.409, 0.506]** | n.d. | **−0.073 [−0.141, −0.003] WORSE** | −0.072; [−0.140, −0.005] / [−0.130, −0.013]; 81 / 110 |

- **The drop from our teams to unfamiliar ones** (home72 − F): T\* 3.5 pp, 0.140 logits; P\* 6.8 pp, 0.276 logits.
  The difference, +0.136 logits, has a delta-method 95 % interval of [−0.26, +0.53] (four independent cells,
  Var(logit p̂) ≈ 1 / (n p (1 − p))): NOT DETECTED. Descriptive only (no rule was registered for it).
- **Confounds.** T\* is a different run (the toy recipe), not P\* minus its belief stack; it is weaker everywhere,
  and a weaker pilot sits nearer 0.50, which shrinks any drop. T\* trained at pin `ae2d0901` and played here at
  `63f97a57` (both config v158; the network's identity across those two commits is UNVERIFIED for this run).
- Per role: home72 0.490 / 0.495, F 0.450 / 0.465. Forfeits 2 / 1; one tie (F). Median 31 / 37 turns.

## Regime stamp

- **Every P\* cell carries ONE stamp over its 400 rows**, equal to the X26 @161M cells' stamp except the model file:
  Kakuna ckpt 34 (`Kakuna@ckpt34`, Metamon `0a00a759`), greedy / greedy, `regime_matched` and
  `regime_verified_decisions` on every unit, CPU, `OMP_NUM_THREADS=1` (one torch thread on each side), Showdown pin
  `e0551883f`, the 250-turn forfeit, `--team-seed = --seed-base` = the unit seed, server
  `ws_frontend[in-process]@63f97a575+rust:cargo:src/rust_sim`, `rust_core_slot`, `bare` load.
- **P\*:** `checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`, a read-only copy (+ its `model_config.json`)
  re-hashed before every unit; the code is a fresh detached `63f97a57` (the X26 Kakuna reads' commit, FINDING 4 of
  the design doc), `gen3ai_torch28`. P\* trained at pin `9c033489`; nothing here changes its network (both config v158).
- **The pair cell:** commit `b9c75884` (the agent `policy` key, `Kakuna28` = `{"policy": "Kakuna", "checkpoint":
  28}`), `ws_frontend@b9c758841`, two Kakuna peers, reported from the ckpt-28 side (`our_side = metamon:Kakuna28`;
  the peer reports record `checkpoint` 28 and 34). Ckpt 28's weights came from the Hugging Face hub into Metamon's
  cache on 2026-10-20 (`policy_epoch_28.pt` sha256 `87c5450e…3ef6`; ckpt 34 `7e7474d5…f26c`; the same hub snapshot
  `ac8e88e7`).
- **Teams:** byte copies sha256-checked against the committed manifests (home72: `x35_breakers_stage_2026-10-17`;
  H: `heldout72_hpfix_kakuna_2026-10-18`; F and P72: `x37a_eval_matrix_2026-10-18`) on BOTH sides before every unit.

**Counting rule** (X37a's): `main.anchors.results.unit_counts` (undeclared, symmetric), both sides' content digests
equal, and the capture check (our packed team is the row's team file; Kakuna's is exactly one team of the same set;
in the pair cell both are home72 teams). **All 140 units counted on their first try** (`skipped_units` empty,
`fails.json` absent). Kakuna's argmax rate was 1.0000 everywhere.

## Cost and schedule

One lane on rule 5b's cores (`nice -n 19 taskset -c 5-7,13-15`), one HEAVY SLOT per ≤ 45-min chunk
(`mem_cap.sh 12`, FIFO; peak 3.7 GB), 8–14 units per chunk, 14 chunks plus a setup chunk. P\* took **11.8 s / game**,
the pair cell 10.5, T\* 7.1 (the toy network is cheaper); 8.0 h of play in all. Load1 mean 8.8 (max 20.4), beside the
live X35 trainers. The queue held each chunk 6–74 min before it ran (the setup chunk 2 h); the read took 2026-10-20
08:10 → 10-21 02:39 PT. The original 12-h stop (20:30) was lifted by the orchestrator mid-read; the first supervisor
stopped cleanly between chunks at 20:30 and a relay supervisor ran every cell to its registered 400 games.

## FINDINGS

1. **Goal (b) needs ≈ +4.3 pp more**, not the +6–8 pp the design doc planned on: the plateau took about a third of
   the gap (+3.0 pp, n.d.). The levers' arithmetic in §5 of the design doc shifts accordingly.
2. **P72 did not move with the plateau** (−2.0 pp, n.d.) while H and F gained +5–6 pp (n.d.). P72's teams are
   procedural (TSR: −146 Elo against forum teams); a weak-team panel compresses a symmetric cell toward 0.50, and
   P72 sits BELOW it. Descriptive; no rule reads it.
3. **Kakuna ckpt 28 vs 34 is not resolved to ±6 pp.** Its first 200 games read 0.460, its last 200 0.570
   (descriptive). Metamon's presets' "these gxe scores are incorrect" stands; ckpt 34 remains the opponent.
4. **The REVERSE cell (we pilot H, Kakuna pilots home72) and the backlog's "short PREREG" were not run / written.**
   This unit's brief named the four symmetric sets, the pair cell and the toy; the design doc §6.1 lists the reverse
   cell too. The rules applied are §6.1's, committed (`b3886800`) before any game.
5. **The heavy-slot queue, not the play, set the wall time** (18.5 h for 8.0 h of play). The brief's ≤ 12-h stop
   would have read about a third of the registered games; the orchestrator lifted it.
6. **The setup chunk's T\* and pair smokes were refused by the queue's own port check** (a false positive on the
   previous unit's TIME_WAIT connections); fixed (`SO_REUSEADDR` in the check), re-smoked in the first chunk, both
   passed. No unit was affected.
7. **The pair cell's rows name the ckpt-28 side only by its alias** (`our_side = metamon:Kakuna28`);
   `opponent_version` names the opponent (`Kakuna@ckpt34`). The record side's checkpoint is in the peer reports.
8. **Ckpt 28's weights were downloaded** from Hugging Face into Metamon's cache (not `models/`, not `data/`), sha256
   recorded above. The hub snapshot is the one ckpt 34 came from.
9. **The toy's harness is not its pin** (`63f97a57` vs `ae2d0901`; § The toy cells): its numbers carry that
   UNVERIFIED identity; P\*'s do not (the comparison the read is for runs both P\* and X26 @161M at `63f97a57`).
10. No other hazard hit: the T28 halt marker was clear before every unit; the queue's own server ran on 9551 (never
   8000 / 8001); nothing was written under `models/` or `data/`; no GPU (`CUDA_VISIBLE_DEVICES=""`); no public server.

## Files

- `cells.json` (every cell, `analysis`: the P\* − X26 blocks, goal (a), goal (b), the pair rule, the toy blocks;
  every unit's counting verdict; the overlap blocks), `rows.jsonl` (every counted game row + `read_cell`, `unit`,
  `unit_seed`, `their_team`).
- `scripts/x51q.py` (the detached, resumable one-lane queue: one heavy slot per ≤ 45-min chunk), `scripts/aggregate.py`
  (counting, capture check, every comparison), `scripts/make_configs.py` + `configs/` (the five per-cell anchors
  configs, sha256 in `x51q.py`).
- Not committed: the raw unit dirs, captures, logs, `units.jsonl` and the read-only model copies, in
  `/home/goodlad/dev/gen3ai-reads/x51_2026-10-20/`.
