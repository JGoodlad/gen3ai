# E2: the hidden-info study's COMPETENT-OPPONENT state bank (2026-10-16)

**Status.**
- **Pre-registered:** [`PREREGISTRATION.md`](PREREGISTRATION.md), shipped `5881ee83` before any E2 game was generated
  or any state drawn.
- **The harness and the bank's ids** are committed in this folder.
- **The arms** crawl detached under `queue_e2.py`; the read is `read_e2.py`, committed before any arm output existed.
- **Results: READ 2026-10-18 (§4).** The registered PRIMARY is **U-FIRST**, printed **SCORER-DEPENDENT** (the tie-free
  scorer reads SHARED), and SHARED on each checkpoint alone. PROVISIONAL at 15M. The GIGO audit is clean (§4.1).

**What it settles.** Tier 1 found that the never-seen Pokémon (class U) carry more of honest search's gap than the seen
mons' hidden moves (class M), by rule R3 on both sides. That held only on the registered mix (58 % scripted-bot
games). On the own-snapshot states it failed in both builds (`red_team.md` §2.6 (b); `third_party_comparison.md`).
E2 asks the same question, pre-registered, on 600 fresh states per checkpoint drawn ONLY from games against the
network's own snapshots.

Bulk lives in `~/gen3ai_archive/hidden_info_e2/<cp>/`: the generated games (`gen/`), the bank, `pool.npz`, the rows and
worlds per source, and the logs. This folder holds the scripts, the committed ids (`states_e2_<cp>.json`) and the read.

## 1. Pipeline

| step | script | where (`run_e2.sh <where>`) | output |
|---|---|---|---|
| games | `gen_games.sh` (the pin's `main.ops.eval_trace_gen`) | `gen_<cp>`: `~/.cache/gen3ai/hib_e2/pin_<cp>` (pin + Showdown submodule) | `gen/e2_<cp>_<step>/` |
| bank | `make_e2_bank.py` | `<cp>`: the vdp pins Tier 1 used | `bank/`, `pool.npz`, `states_e2_<cp>.json` |
| truth (O) | `phase1_e2.py` (the pilot's `run_playouts.one_state`) | `<cp>` | `rows/O/` |
| worlds | `build_e2_worlds.py` (`WorldSampler`, the guard on; `--limit` caps NEW work, `build_e2_worlds_test.py`) | `head`: `~/.cache/gen3ai/hib_e2/head` (the E2 HEAD) | `worlds/<arm>/` |
| playouts | `play_e2_worlds.py` (the pilot's `run_det_playouts.one_state`) | `<cp>_pad`: Tier 1's pad pins | `rows/<arm>/` |
| runner | `queue_e2.py` (launched by `launch_queue.sh`) | any; it calls `run_e2.sh` | `logs/queue_e2.log`, `QUEUE_DONE` |
| read | `read_e2.py` (+ `read_e2_test.py`, planted effects) | pure | `result_e2.json` |
| identity (§5) | `identity_e2.py` | pure | `identity_e2.json` |
| GIGO audit (before the read is believed) | `gigo_audit_e2.py` (counts, ids, provenance, finiteness; no outcome) | pure | `gigo_audit_e2.json` |
| pin re-gate (§5, descriptive) | `pin_regate_e2.py` | `<cp>_pad` (run by the queue) | `pin_regate_e2_<cp>_<arm>.json` |

Arm directories use `p` / `m` for `+` / `−`: `F`, `FpM`, `FpU`, `OmM`, `OmU`, and `V_oracle` for the identity check.

**CPU** follows root rule 5b:
- each worker runs one thread, under `nice -n 19` and SCHED_IDLE;
- workers are pinned with `taskset -c 5-7,13-15`, each in its own `mem_cap` scope;
- `queue_e2.py --slots 2` keeps at most 2 workers.

## 2. Reproduce / operate

```bash
F=designs/research_state/measurements/hidden_info_e2_competent_2026-10-16
bash $F/run_e2.sh gen_end2 2 8 12h gen_games.sh           # games (then gen_prod)
bash $F/run_e2.sh end2 1 14 3h make_e2_bank.py            # the bank (then prod); exit 4 = a short quintile (§1.3)
bash $F/launch_queue.sh                                   # the detached runner, from the E2 HEAD checkout
tail -f ~/gen3ai_archive/hidden_info_e2/logs/queue_e2.log # progress (counts only)
python -I $F/read_e2.py --partial                          # COUNTS only while it crawls; the read once QUEUE_DONE exists
python -I $F/gigo_audit_e2.py                              # the GIGO audit -> gigo_audit_e2.json (before believing the read)
python -I $F/identity_e2.py --out identity_e2.json        # the §5 identities -> identity_e2.json
python -I $F/read_e2.py                                    # the pre-registered read -> result_e2.json
```

## 3. Notes fixed before any arm output

- **Not blinded.** The arms carry their labels. Every rule is fixed as code in `read_e2.py` before any arm output exists,
  and the runner reads no outcome. Tier 1's blinding protected the frozen table and scripts, not the author's knowledge
  (Tier 1 README §3), so its cost buys nothing more here.
- **The builder's HEAD differs from Tier 1's.** Since Tier 1, the world builder gained the record-slot fix and the
  throwing guard (`8702a56f`) and the packed-move canonical reader (`631db7d8`). Tier 1's worlds were unaffected by
  either: no world changed, by those commits' own audits. E2's worlds are built with both.
- **The engine is unchanged.** `git diff 67f3f7fa <E2 HEAD> -- src/rust_sim/src src/rust_env/src` touches the same 7
  files as Tier 1's README §2 (oracle reveal, the gender pad, comments).
- **The `random`-bot games are a by-product.** `eval_trace_gen` cannot generate a sentinel-only cycle. Those games are
  never drawn.

## 4. Results (read 2026-10-18)

**How it was read.**
- **The queue:** `QUEUE_DONE` 2026-10-18 03:21, nothing parked.
- **The reader's self-test first:** `read_e2_test.py`, 7 / 7.
- **Then the registered scripts, unchanged:** `identity_e2.py`, then `read_e2.py`, as committed at `d5880c87` (the
  `--no-tier1` opt-out NOT used).
- **The GIGO audit:** `gigo_audit_e2.py`, written for this read. It computes no outcome.
- **The outputs, all in this folder:**
  - `result_e2.json` (the read);
  - `identity_e2.json`;
  - `gigo_audit_e2.json`;
  - the ten `pin_regate_e2_<cp>_<arm>.json`.
- **No deviation:** no rule, threshold, arm or script of the read was changed.

**Vocabulary** (§3 of the registration):
- IMP is the decision improvement: the win probability gained per decision by playing the scorer's pick instead of the
  policy's top choice, judged on the true world's held-out referee seeds.
- **O** is the oracle (the true world). **F** is the fully honest world.
- **M** is the seen Pokémon's unrevealed moves; **U** is the never-seen Pokémon.
- **F+X** tells class X; **O−X** hides only X.
- **LEAF-K8** is the mean outcome over 8 worlds played to the end, with ties going to the policy's order.

### 4.1 Validity: the GIGO audit (every check passed; nothing to list first)

| check | end2 | prod |
|---|---|---|
| the bank: states / per gap quintile / battles / max per battle; manifest sha = `states_e2_<cp>.json`; decision ids = its ids | 600 / 120 each / 365 / 2; ✓; ✓ | 600 / 120 each / 371 / 2; ✓; ✓ |
| the truth (Phase 1, arm O): rows, distinct ids, duplicates, foreign ids, torn lines, refused | 600, 600, 0, 0, 0, 0 | 600, 600, 0, 0, 0, 0 |
| … its pin / `set` / cp fields, 40 seeds, the same actions in outcome / critic / trace | all ✓ (0 mismatches) | all ✓ |
| … the shard file's row ORDER = sorted(bank ids) | ✓ | ✓ |
| worlds and playout rows, per arm: 600 rows each, 0 duplicates, 0 foreign ids, 0 torn lines | ✓ (5 arms × 2) | ✓ |
| every world built at the E2 HEAD `d5880c87`; every playout at the pad pin | ✓ (`67f3f7fa`) | ✓ (`9a698801`) |
| world ↔ row status agree; 8 worlds per ok row; the arm's actions = the truth's | ✓ | ✓ |
| exclusions (a `WorldError`): F / F+M / F+U / O−M / O−U | 40 / 45 / 39 / 0 / 1 | 52 / 50 / 51 / 0 / 1 |
| **guard DEFECTS** (`WorldConsistencyError`, §5) | **0** in every arm | **0** |
| F's exclusion share against the §6.1 10 % rule | 6.7 % (under) | 8.7 % (under) |
| the cross-commit identity: oracle worlds at HEAD, played at the pad pin = Phase 1 | 12 / 12 states, 92 actions, 0 mismatches, max \|ΔV\| 1e-5: PASS | 12 / 12, 94 actions, 0, 1e-5: PASS |
| the k = 6 identities: F+U ≡ F / O−U ≡ O, outcome for outcome | 103 + 10 both excluded / 113 states, 0 differ: PASS | 78 + 13 / 91, 0 differ: PASS |
| the pin re-gate, per arm (descriptive; recorded verdict re-checked against its criterion) | 5 / 5 PASS | 5 / 5 PASS |
| non-finite values (every float of every row, and every output) | 0 | 0 |

- **The pin re-gate in total:** 45,768 worlds re-gated at the pad pins' own engines, **0 failures**.
  - Every re-gate's state count equals its arm's ok world rows, at 8 worlds each.
  - The revealed-move plant was caught **5,586 / 5,586** times: the registered requirement.
  - The item-swap plant was caught 3,597 / 5,721 and the pad-strip plant 6,264 / 6,510. Both are descriptive, as in
    Tier 1.
- **The truth rows and the `--limit` fix (`6a1cf301`).**
  - Each checkpoint's first 30 truth rows were selected by `d5880c87`'s first call. The other 570 were selected by
    `6a1cf301`, from 2026-10-17 14:20 PDT.
  - Both versions take new work in sorted-id order, and the shard file's row order equals sorted(bank ids). So every
    post-fix call took exactly the next not-done ids: none was re-picked or skipped, and each bank id has exactly one O
    row.
  - The row computation is the same code in both versions. The stall had made 1,304 (end2) and 13 (prod) zero-work calls.
  - The world builder held the same `--limit` defect (`build_e2_worlds.py`'s `ids[: a.limit]`). It never ran with
    `--limit`, and is now fixed with `select_ids` + `build_e2_worlds_test.py`, which fails on a revert of either fix.

### 4.2 PRIMARY (§3.4): pooled end2 + prod, LEAF-K8 — **U-FIRST**, printed **SCORER-DEPENDENT**

| numerator | estimate [95 %] | states | clear direction |
|---|---|---|---|
| reveal-only **Δs** = IMP(F+U) − IMP(F+M) | **+0.0169 [+0.0065, +0.0268]** | 1,061 | **U**, MATERIAL (lower > +0.0055) |
| hide-only **Δn** = IMP(O−M) − IMP(O−U) | **+0.0114 [+0.0010, +0.0220]** | 1,198 | **U** (lower bound twice the ±0.0005 band) |

- **Both lower bounds clear +0.0005, so U-FIRST.** This is R3 met with X = U on fresh competent-opponent states.
- **The qualifiers:**
  - both classes are detected;
  - not SIDES-DISAGREE;
  - MATERIAL on the reveal-only side only;
  - not CLOSE;
  - GAP holds: G = IMP(O) − IMP(F) **+0.0718 [+0.0599, +0.0827]** (Tier 1's registered G was +0.070).
- **Validity:** VALID, quoted as "computed": the identities hold and there are no guard defects.
- **SCORER-DEPENDENT:** the tie-free scorer's verdict differs (§4.3 item 2).

### 4.3 SECONDARY reads (§4; none overrides the primary)

1. **Per checkpoint, LEAF-K8: end2 SHARED, prod SHARED.** On each, the reveal-only side is U and the hide-only side
   has no clear direction:
   - end2: Δs +0.0195 [+0.0045, +0.0348], Δn +0.0099 [−0.0053, +0.0244];
   - prod: Δs +0.0143 [+0.0018, +0.0268], Δn +0.0128 [−0.0023, +0.0275].
2. **The tie-free scorer LEAF-K8|RAND, pooled: SHARED.** Δs +0.0160 [+0.0065, +0.0253] is U; Δn +0.0100 [−0.0002,
   +0.0193] has no clear direction. Hence SCORER-DEPENDENT beside the primary. Every scorer, pooled (no verdict for any
   but the two above):

   | scorer | Δs | Δn | R3 conditions (s / n) |
   |---|---|---|---|
   | LEAF-K8 (primary) | +0.0169 [+0.0065, +0.0268] | +0.0114 [+0.0010, +0.0220] | U / U |
   | LEAF-K8\|RAND (tie-free) | +0.0160 [+0.0065, +0.0253] | +0.0100 [−0.0002, +0.0193] | U / none |
   | LEAF-K8\|CRITIC | +0.0164 [+0.0058, +0.0272] | +0.0112 [+0.0004, +0.0217] | U / none (lower inside the band) |
   | LEAF-K8\|TRUNC4 | +0.0166 [+0.0056, +0.0270] | +0.0095 [−0.0014, +0.0203] | U / none |
   | TRUNC-4-K8 | +0.0006 [−0.0100, +0.0108] | +0.0143 [+0.0041, +0.0239] | none / U |
   | LEAF-K4 | +0.0129 [+0.0029, +0.0232] | +0.0059 [−0.0049, +0.0162] | U / none |
   | LEAF-K1 | +0.0059 [−0.0021, +0.0138] | +0.0060 [−0.0027, +0.0141] | none / none |
   | CRITIC | −0.0027 [−0.0107, +0.0049] | −0.0069 [−0.0143, +0.0005] | none / none (no GAP: G +0.0068 [−0.0007, +0.0146]) |

   **No scorer or frame gives the moves a clear lead on either side.** Every scorer with a detected gap puts both
   point estimates on U's side. Here only the primary LEAF-K8 meets BOTH conditions at once. In Tier 1 its tie-free
   variants met them too (the red team, design §16.10).
3. **The shares** (pooled, LEAF-K8; F's exclusions are under 10 % on each checkpoint, so no share is flagged):

   | class | sufficiency (reveal-only) | necessity (hide-only) | absolute |
   |---|---|---|---|
   | seen mons' moves (M) | s_M **0.16** [0.02, 0.28] | n_M **0.58** [0.47, 0.70] | σ_M +0.0109 [+0.0010, +0.0213]; ν_M +0.0432 [+0.0340, +0.0523] |
   | never-seen mons (U) | s_U **0.40** [0.28, 0.50] | n_U **0.73** [0.62, 0.86] | σ_U +0.0286 [+0.0190, +0.0384]; ν_U +0.0541 [+0.0446, +0.0638] |
   | U − M | +0.24 [+0.09, +0.40] | +0.15 [−0.006, +0.30] | |

   - **B3 for M FAILS:** s_M is not ≥ 0.50.
   - **B4 for M HOLDS:** n_M ≥ 0.50 with lower bound ≥ 0.25. On competent play, hiding only the moves opens more than
     half the gap; on Tier 1's registered mix n_M was 0.42.
   - The moves are a necessary, material share and not a sufficient one. Descriptive, as registered: no new verdict on
     R-MOVES.
4. **Against Tier 1's populations** (descriptive; Tier 1's registered 600 + supplement, split by the red team's
   `cycle_opp` rule; independent samples):

   | population | states | Δs | Δn | E2 − it: Δs | E2 − it: Δn |
   |---|---|---|---|---|---|
   | Tier-1 bot-game states | 912 | +0.0256 [+0.0126, +0.0392] | +0.0394 [+0.0282, +0.0508] | −0.0087 [−0.0258, +0.0086] | **−0.0280 [−0.0438, −0.0131]** |
   | Tier-1 snapshot-game states | 708 | −0.0041 [−0.0183, +0.0109] | +0.0103 [−0.0028, +0.0239] | **+0.0210 [+0.0023, +0.0381]** | +0.0011 [−0.0161, +0.0175] |

   - **The reveal-only side:** E2 looks like the bot-game states and differs from Tier 1's snapshot-game states. The red
     team's reveal-only population attack does not replicate on fresh states.
   - **The hide-only side:** E2 looks like Tier 1's snapshot-game states and is clearly below the bot-game states. That
     half of the population effect does replicate.
   - **The k-adjusted E2 numerators** equal the raw ones by construction (FINDING 1 below), so these differences are
     NOT composition-adjusted.
   - **The volume regression's coefficients** (descriptive):
     - Δs: +0.031 per never-seen mon and +0.007 per hidden move slot;
     - Δn: +0.006 and −0.003.
5. **By k (opponent mons seen) and by traced outcome** (pooled, LEAF-K8):

   | group | k 1–2 | k 3 | k 4 | k 5 | k 6 | traced loss / draw | traced win |
   |---|---|---|---|---|---|---|---|
   | states | 273 | 207 | 241 | 275 | 204 | 812 | 388 |
   | Δs | +0.036 | +0.033 | +0.023 | +0.014 | −0.035 | +0.023 | +0.004 |
   | Δn | +0.027 | +0.028 | +0.023 | +0.014 | −0.043 | +0.012 | +0.009 |

   - **The U margin shrinks as the game reveals the team.** At k = 6, Δs = −σ_M and Δn = −ν_M by the identity (nothing
     is unseen), so only the moves carry value there.

### 4.4 What it means (the registered meaning, then the reader's qualification)

**The registered meaning of U-FIRST (§3.4):** "The never-seen Pokémon are the largest class on competent play too. The
red team's population attack does not replicate on fresh states. 'U first' then holds on both populations."

**The qualification:**
- **The verdict is the pooled primary scorer's only.**
  - It is SCORER-DEPENDENT: tie-free, it reads SHARED.
  - It is not met on either checkpoint alone; both read SHARED.
  - Both misses are on the hide-only side.
- **The reveal-only half is robust.** It is U on the primary, on each checkpoint, under the tie-free and the two other
  tie-broken leaves, and at LEAF-K4. It is MATERIAL, and it clearly exceeds Tier 1's snapshot-game states.
- **The hide-only half is marginal.**
  - Its primary lower bound is +0.0010.
  - Hiding the moves alone opens 0.58 of the gap, close to the never-seen mons' 0.73.
  - E2's hide-only margin sits where Tier 1's snapshot-game states put it.
- **Status:** PROVISIONAL. The 15M checkpoints are not plateaued (§10). It is computed on ONE implementation (the
  original harness) and replicated on neither checkpoint alone.

**For the study's claim:** "U > M in competent play" **HOLDS by the registered rule, scorer-dependent and PROVISIONAL.**
"The moves are THE binding constraint" stays refuted: B3 fails, and no frame gives M a lead. Detail and the roadmap
consequence: `design_hidden_info_binding.md` §16.12.

### 4.5 FINDINGS (hazards and things not verified)

1. **The registered "k-adjusted E2 Δs / Δn" is vacuous as coded.** `read_e2.by_groups` evaluates each draw's OLS fit at
   the frame's OWN mean volume. With an intercept, that reproduces the raw mean: +0.0169 / +0.0114, identical intervals.
   So §4.4's population differences carry no volume adjustment. A meaningful version evaluates E2's fit at Tier 1's
   population means. Not patched, because it is a reader change that would need its own test and a named deviation; the
   read is descriptive here.
2. **The reader has no duplicate-row guard.** `e2_common.read_dir` keeps the LAST row per id, so a duplicate would be
   silent. The audit counted raw rows and found 0 duplicates in every file.
3. **A latent twin in the playout driver** (`play_e2_worlds.py`, not fixed: out of this unit). It shards the NOT-DONE
   list, so two shards started at different times could pick the same id. The queue always ran one shard (`--of 1`), and
   the audit found 0 duplicates.
4. **The queue's re-gate ran while a playout unit was still going.** The prod O−U re-gate started at 02:57, and its last
   playout unit ended at 03:21. This is harmless: a re-gate reads worlds only, and those were all built on 2026-10-17 at
   00:59. But `queue_e2.py` gates the re-gate on the build, not on "after every playout unit" as its docstring says.
5. **Tier 1's populations here include the k-stratified supplement** (912 bot-game / 708 snapshot-game states), as
   registered. The red team's own split used the registered 600 only (697 / 503), so the two population figures are not
   the same sample.
6. **Not verified:** no playout or world was re-run, and the read is the original harness's alone. The exclusion
   reasons were not classified beyond `WorldError`.
7. **Rule 5b was followed:** nice 19, `taskset -c 5-7,13-15`, `mem_cap` (peaks 0.8 / 1.0 / 1.3 GB), no GPU, at most one
   heavy job at a time.

