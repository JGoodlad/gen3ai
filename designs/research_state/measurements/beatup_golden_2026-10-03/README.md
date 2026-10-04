# The K9 learner-golden re-bake proof, and the sim parity, for `gen3_beatup_exact_v1` (X5 Tier 0 F8 residual), 2026-10-03

**What changed.** `gen3_nonformula_damage_v1` (f0310ee7) priced Beat Up as ONE 10-BP Dark special hit — the
user's SpA against the target's SpD. The real gen-3 move (`deps/pokemon-showdown/data/mods/gen3/moves.ts`
`beatup`) is typeless and special, one hit per healthy party member, each hit the ordinary formula at BP 10 with
`A` = THAT ally's species BASE Atk and `D` = the TARGET's species BASE Def. Into a Blissey (base Def 10) the
old price was 8.7 HP of 651 (1.3 %); the real move into the pinned six-mon party below is 507 HP of 714 in the
sim and 511.5 HP in the op. Every kernel now prices it exactly (`damage_kinds.beatup_*`, `damage_tables`
kind `beatup_party`, `OP_SEMANTICS = gen3_beatup_exact_v1`).

## 1. Mechanics against the REAL sim (`out/sim_parity.{txt,json}`; `src/agents/model/beatup_sim_parity_test.py`)

The omniscient `utils/bridge/damage_probe.js` BattleStream, 11 constructed scenarios x 12-24 fixed seeds each
(`gen3customgame`, p1 lead Umbreon with Beat Up, p2 a lone target). Per scenario, three layers:

1. **Mechanics.** The sim's own log names the ally each hit is struck for (`-activate|…|move: Beat Up|[of] X`):
   the set is EXACTLY the healthy party (a fainted ally and a paralysed / sleeping ally are absent, a paralysed
   USER is absent, a healthy user is present), the hit count equals it, and EVERY non-crit hit lies in the
   integer band `floor((floor(floor(420*A/D)/50)*screen + 2) * r / 100)`, r in 85..100.
2. **The sim's mean** over the seeds sits on the exact integer mean of that process (3 standard errors).
3. **The op.** `DamageOperator._outgoing_block`'s smooth 0.925-mean price is within 1.5 HP per hit of that
   integer mean (the documented smooth-vs-floor bias; measured +0.9 .. +1.0 per hit, 5.6 over six hits).

| scenario | hits | sim mean (HP) | integer mean | op mean | what it pins |
|---|---|---|---|---|---|
| six healthy into Blissey | 6 | 507.2 | 505.9 | 511.5 | the base-stat sum, user included |
| typeless into Gengar / Skarmory / Tyranitar / Alakazam | 6 | 89.2 / 41.8 / 51.2 / 118.7 | 88.9 / 41.3 / 50.8 / 118.4 | 94.5 / 46.8 / 56.6 / 122.3 | no effectiveness, no STAB (Dark user), into Ghost / Steel / Dark / Psychic |
| Light Screen / Reflect / +Def target & +SpA user | 6 | 255.2 / 503.7 / 503.7 | 256.2 / 505.9 / 505.9 | 255.7 / 511.5 / 511.5 | the screen rule; no stage reaches either stat |
| paralysed + sleeping ally benched | 4 | 314.1 | 314.4 | 318.2 | statused allies excluded |
| fainted ally | 5 | 455.0 | 454.6 | 459.1 | fainted allies excluded |
| paralysed user | 5 | 453.2 | 454.6 | 459.1 | a statused user does not count itself |

Runs containing a crit are excluded from the band and mean checks (a crit doubles a hit and ignores a screen);
`out/sim_parity.txt`'s `n_clean` column gives each scenario's crit-free runs (per-hit crit chance 1/16, so about a
third of six-hit runs contain one). The paralysed-user scenario pins the 12 seeds on which the user is not fully
paralysed on turn 2 (seeds 1 and 2 are full-para turns — scanned once over 1..60 and pinned, asserted not branched).

## 2. The K9 learner golden (`src/agents/training/learner_golden.json`)

**Claim.** The golden moved ONLY because of Beat Up's table entries.

**Method** (`scripts/proof.py`, torch `2.8.0+cu126`, CPU, one thread, the golden's own `compute`): run the
golden update on the PARENT tree (`ed3a2bb5`) and on the FIX tree, each with and without Beat Up NEUTRALISED —
the same row on both trees: `MOVE_BP[beatup] := 0`, plus (fix tree only) `MOVE_BEATUP := 0` and
`MOVE_TYPE_IDX[beatup] := DARK` (the fix writes '???' there; the parent's row is Dark already, and the incoming
matrix's `type_mult` cell reads it). If the two neutralised runs are byte-identical, every kernel change is a
`where` / gather that selects exactly the old value for a non-Beat-Up cell, and every element the re-bake moves
is caused by Beat Up.

**Result on the committed buffer** (`out/{old,old_neutralised,new,new_neutralised}.json`):

| run | post_params_sha256 |
|---|---|
| parent | `ef8a1b7c9e5ced1c…` (= the banked golden) |
| parent, neutralised | `623c968108c75fcec…` |
| fix | `9ca337f3b250529b…` |
| fix, neutralised | `623c968108c75fcec…` |

**The rebuilt buffer.** The committed buffer's stored behaviour log-probs were the parent op's, so K9(b)'s probe
failed on it (`learner_gates_test`: max |Δ log π| 2.46e-4, bar 1e-4). `python -m agents.training.learner_golden
rebuild-buffer` re-recorded it: every obs field, action, mask, reward and episode start is BYTE-IDENTICAL (the same
games); only `values` (29 / 64 elements, max |Δ| 5.1e-5), `log_probs` (9 / 64, 3.4e-4), `advantages` (61 / 64,
4.3e-5) and `returns` (40 / 64, 1.9e-5) moved. The proof was repeated on it (`out/rebuilt_*.json`, `--buffer`):

| run (rebuilt buffer) | post_params_sha256 |
|---|---|
| parent | `1a206fc8607e474997c1…` |
| parent, neutralised | `bb7c20f740321fc80f2b…` |
| fix | `50f12a2357cc7b73…` (the recorded golden) |
| fix, neutralised | `bb7c20f740321fc80f2b…` |

Neutralised runs identical again: params, all 41 group hashes and all 19 losses. The un-neutralised move is
36 of 41 parameter groups and 16 of 19 pinned losses on the rebuilt buffer (36 / 41 and 15 / 19 on the old one);
init `f476942c…` is unchanged throughout. **Beat Up reaches the buffer** in our request slots, in the visible
move slots and in the opp active's believed top-K (`beatup_seen_in`, all three True).

**Reproduce:**

    git worktree add --detach <old> ed3a2bb5
    PYTHONPATH=<old>/src  python scripts/proof.py --label old [--neutralise] --out out/old[_neutralised].json
    PYTHONPATH=<fix>/src  python scripts/proof.py --label new [--neutralise] --out out/new[_neutralised].json
    # the rebuilt buffer: add --buffer <fix>/src/agents/training/learner_golden_buffer.npz (out/rebuilt_*.json);
    # the committed (parent) buffer is ed3a2bb5's src/agents/training/learner_golden_buffer.npz

`OMP_NUM_THREADS=1` (the golden builds and updates at one thread anyway).

**Cost** (`scripts/perf.py`, `out/perf_cpu.txt`): the production forward + backward on the 64 golden rows, CPU
eager fp32, one thread, the two trees interleaved: +0.9 % / +1.7 % on the median. The GPU compiled path is
**UNVERIFIED** (CPU only, by this unit's brief).
