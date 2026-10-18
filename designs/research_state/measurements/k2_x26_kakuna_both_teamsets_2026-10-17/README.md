# K2 and X26 vs Metamon's Kakuna, on BOTH team sets — 2026-10-17

**Question (owner, 2026-10-17).** X26@161M beat Kakuna on our teams (0.600, `x26_metamon_anchor_2026-10-17/`).
K2 (G0′), the previous lineage's plateau, was never read against Kakuna. Is X26 a jump over K2? Put both
models on both team sets.

**Answer.**

- **On Metamon's own teams (away), X26 is detectably ahead of K2: +0.143 [+0.074, +0.209]** (Newcombe, n = 400
  each). K2 LOSES to Kakuna there: 0.355 [0.310, 0.403], WORSE than even. X26 is level with it: 0.497
  [0.449, 0.546], NOT DETECTED.
- **On our teams (home72), X26 is also detectably ahead of K2: +0.087 [+0.018, +0.155]** (n = 400 each, after
  the follow-up top-up on paired seeds 12–19). K2 reads 0.487 [0.439, 0.536] (NOT DETECTED vs 0.50). X26 reads
  **0.575 [0.526, 0.623]** (BETTER), down from 0.600 at n = 240. So X26 still meets the goal bar on our teams and K2
  does not. At n = 240 the gap was +0.079 [−0.010, +0.166], NOT DETECTED.
- **The X26 away number came DOWN with more games:** 0.570 at n = 100 → **0.497 at n = 400**. The top-up's 300 new
  games read 0.473. Against the first 100 that is −0.097 [−0.205, +0.016], NOT DETECTED: the first 100 were the
  luckier draw, not a different regime. **X26 does not beat Kakuna on Metamon's teams.**
- **PROVISIONAL.** X26 is one checkpoint of a live, unplateaued run (161M of a 500M cap), one seed. K2 is a
  plateaued pre-era model read through its own pinned code (§ Comparability).

## The table (Kakuna ckpt 34, greedy vs greedy, CPU)

| model | team set | n | W–L–T | win rate [Wilson 95%] | vs 0.50 | source |
|---|---|---:|---|---|---|---|
| K2 `ai_v14_07_g0p_k2` @91,127,808 | home72 | 400 | 195–204–1 | 0.487 [0.439, 0.536] | NOT DETECTED | this read, seeds 0–19 (12–19 the follow-up) |
| X26 `rb_x26_s2601` @161,000,169 | home72 | 400 | 230–168–2 | **0.575 [0.526, 0.623]** | **BETTER** | seeds 0–11 from `x26_metamon_anchor_2026-10-17` `K_greedy_h72` + seeds 12–19 here (stamps identical) |
| K2 | away | 400 | 142–257–1 | **0.355 [0.310, 0.403]** | **WORSE** | this read, seeds 0–19 |
| X26 | away | 400 | 199–201–0 | 0.497 [0.449, 0.546] | NOT DETECTED | seeds 0–4 from the X26 read + seeds 5–19 here (stamps identical, below) |

| team set | X26 − K2, Newcombe 95% (headline) | verdict | paired descriptor (same teams, same seed, `paired.json`) |
|---|---|---|---|
| home72 | **+0.087 [+0.018, +0.155]** | **X26 BETTER** | +0.087; normal [+0.024, +0.151], seed bootstrap [+0.035, +0.140]; 102 / 67 discordant |
| away | **+0.143 [+0.074, +0.209]** | **X26 BETTER** | +0.142; normal [+0.080, +0.205], seed bootstrap [+0.075, +0.210]; 114 / 57 discordant |

The paired descriptor uses the design's pairing. Every K2 game has an X26 twin with the same unit seed, half and
index, and BOTH sides' team draws are identical in every twin (32 / 32 units, checked for our side and the peer's
`team_draws`; 40 / 40 units after the follow-up). It agrees with the Newcombe headline on both team sets. At
n = 240 the home72 paired lower bound sat at zero (+0.000 / +0.002), within rounding of the decision boundary. Under
root rule 8 it decided nothing then; that is why the follow-up was run.

Per role (our challenge / theirs): K2 home72 0.515 / 0.460; X26 home72 0.535 / 0.615; K2 away 0.330 / 0.380; X26
away 0.500 / 0.495. The 250-turn forfeit fired 1 (K2 home72), 1 (X26 home72), 0 (K2 away) and 4 (X26 away) times.
Median length was 33 / 30 / 36 / 37 turns.
Kakuna's argmax rate was 1.0000 in every unit.

**Context (Newcombe, same cell):**

- K2 − N0 on home72 = 0.487 − 0.460 = +0.027 [−0.057, +0.111], NOT DETECTED. The N0 number is from
  `kakuna_goal_read_2026-09-28`, on the same pre-era harness as K2 here.
- X26 − N0 at n = 400 = **+0.115 [+0.030, +0.198]**.
- So on our teams the gain over N0 against Kakuna belongs to the K2 → X26 step, not to K2's own lineage.
- X26's home72 top-up (seeds 12–19) read 86/160 = 0.538 against 0.600 for the first 240: −0.062 [−0.160, +0.036],
  NOT DETECTED.

## Regime stamp

| field | K2 cells | X26 cells |
|---|---|---|
| model file | `final_model.zip` @91,127,808, sha256 `cc5d5502ca23b0879c531b515993073d83b42613b5a63f19f63e48bbf58207c4`, copied read-only from `models/ai_v14_07_g0p_k2/` (arch `gen3_event_record_v2`, config v123, saved under torch 2.5.1+cu121) | `checkpoint_161000169_steps.zip`, sha256 `bc6050d1ca254abd9866945f3dde74a2189514a551e83c29abb79b2a20e52992`, the X26 read's own read-only copy (re-hashed before every unit) |
| code | K2's TRAINING PIN `a5b2fdba` (metadata `pin_history`: every step 83.07M–91.13M), a detached checkout at `~/.cache/gen3ai/k2_pin/pin_a5b2fdba` | `63f97a57`, the X26 read's commit, a detached checkout at `~/.cache/gen3ai/k2_pin/pin_63f97a57` |
| interpreter | `gen3ai_stable`, torch 2.5.1+cu121, SB3 2.8.0 | `gen3ai_torch28`, torch 2.8 |
| loader | `bare` (as K2's 2026-09-29 SmallRL read) | `bare` |
| our transport | the pin's `main.play` websocket client (vendored poke-env `RLPlayer`; rows carry no `our_transport` field) | `rust_core_slot` |
| server | `ws_frontend@a5b2fdba+rust:cargo:src/rust_sim` (a subprocess, port 9573) | `ws_frontend[in-process]@63f97a57+rust:cargo:src/rust_sim` (port 9573 here, 9571 in the X26 read) |
| shared | `metamon:Kakuna` ckpt 34 (`Kakuna@ckpt34`), metamon `0a00a759`, CPU, `OMP_NUM_THREADS=1`; greedy / greedy, `regime_matched` true, `regime_verified_decisions` true on every unit; Showdown pin `e0551883f`; 250-turn forfeit; 20-game role-balanced units, `--team-seed = --seed-base = unit seed`; home72 = `n0_endofrun_2026-09-27/teams_home72` via `anchors_home72_kakuna.json` (sha256 `e8a064db…`, the T55 manifest re-checked: 72 / 72 files on both sides); away = Metamon's 20 `competitive` teams | same |

**X26 away pooling.** Every one of the 400 X26 away rows carries ONE identical regime stamp (`cells.json`
`regime_stamps`: opponent, version, commit, regimes, team source and counts, model file and step, loader,
forfeit, Showdown pin, server impl and version, our side and transport, mirrored pairs). That is why the
top-up ran at `63f97a57` and not HEAD. The new seeds 5–19 are disjoint from the banked 0–4.

**Counting rule.** A unit counts iff `status == OK`, `n == 20`, `regime_verified_decisions`, no
`team_source_asymmetry` AND `peer_clean`. **All 63 units of this read (47 + the follow-up's 16) passed on their
first try**, and so did the 17 banked X26 units it pools. The X26 home72 cell (12 banked + 8 new units) also
carries ONE identical regime stamp over its 400 rows. No unit was skipped, no unit was retried, and no H17 occurred.

## Comparability

The X26 − K2 difference is a MODEL difference measured across TWO harnesses. That is the price of loading K2
faithfully: it does not load at HEAD (arch `gen3_event_record_v2` vs `gen3_mon_tied_gain_v1`), and nothing was
edited to make it load. Two things differ beside the model:

1. **Our transport.** K2 plays through the pin's poke-env websocket client; X26 plays through the in-process core
   slot. P3 (`pokeenv_p3_anchors_2026-10-07`) found the two byte-identical on 100 / 100 seeded battles for a
   current-architecture checkpoint. Nobody has run that check for a pre-era checkpoint (**UNVERIFIED** for K2).
2. **The torch build.** K2 runs torch 2.5.1, the build it was trained and recorded under. X26 runs torch 2.8.

The same split separated N0 (2.5.1, old client) from X26 in the X26 read. The Kakuna peer, its team draws, the
team files and the regimes are identical. The peer script differs between `a5b2fdba` and `63f97a57` only by the
mirrored-pairs path (unused) and comments.

## Cost

One lane on rule 5b's cores (`nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh 12`), next to the live X26 trainer
and the P0.1 E2 runner. Load1 had a mean of 5.1 and a max of 9.5. K2 took **4.5–5.8 s/game** and X26 took
**7.0–8.2 s/game**: the end-state network on one torch thread is the slower side. The wall time was
21:30–23:08 PT (1 h 38 min, 940 games), plus the follow-up at 23:14–23:49 PT (35 min, 320 games). Peak memory per
unit was 2.3 GB.

## FINDINGS

1. **X26's away 0.570 (n = 100) was a lucky first draw.** At n = 400 it reads 0.497, so "X26 vs Kakuna away" is
   level, not leaning ahead. The SOP's X26 row is updated.
2. **At n = 240, home72 did not resolve X26 − K2** (+0.079 [−0.010, +0.166], with the paired lower bound at zero).
   The coordinator then approved a top-up of BOTH home72 cells to n = 400 on paired seeds 12–19 (16 units, all
   OK on the first try, 23:14–23:49 PT). It resolves: +0.087 [+0.018, +0.155]. X26's home72 headline moves from
   0.600 (n = 240) to 0.575 [0.526, 0.623] (n = 400), and its lower bound still clears 0.50.
3. **The harness is confounded with the model** (§ Comparability). The P3 transport equivalence is UNVERIFIED for a
   pre-era checkpoint.
4. **K2's rows lack `our_transport` and write `mirrored_pairs = null`**, because the pin's tool predates both
   fields. Any reader that filters on them must treat null as "the legacy client, unmirrored".
5. **No hazards hit.** The T28 halt marker was clear before every unit (re-checked in `kq2.py`'s preflight). No
   server or process of anyone else's was touched, and the tool's own server bound 9573. Nothing was written
   under `models/` or `data/`. No GPU was used (`CUDA_VISIBLE_DEVICES=""`).

## Files

- `cells.json`: every cell pooled, with Wilson intervals, per-half splits, integrity counters, the distinct regime
  stamps with their row counts, the X26 − K2 Newcombe intervals and `skipped_units` (empty).
- `rows.jsonl`: every game row of every counted unit (1,600: the tool's full row plus `read_cell`, `unit`,
  `unit_seed`, `source_read`). It includes the X26 read's 240 home72 rows and its 100 away rows.
- `paired.json`: the paired descriptor.
- `scripts/kq2.py`: the detached, resumable, one-lane queue. `scripts/aggregate.py` pools the units, and
  `scripts/paired.py` computes the descriptor.
- The raw unit dirs, logs, `units.jsonl` and the read-only K2 copy live outside the repo, in
  `/home/goodlad/dev/gen3ai-reads/k2_x26_kakuna_both_teamsets_2026-10-17/`.
