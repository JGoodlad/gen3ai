# X61P / N1-P look 2 (final), read as registered: **NULL** (2026-10-30)

N1-P = C0 + `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` (12,000 procedural teams from the Smogon facade
alone), P\* + 20M at N1P_PIN `5f8f6cfa`, s3501–s3505. Control: C0 s3501–s3505 (production pin `f4b25238`). Rule: D1's,
verbatim, on F alone (`main.reads` rule `d1`, arm `X61P`, look 2; [`../PREREG.md`](../PREREG.md) §4–§9, AM-N1P-1 §14,
AM-N1P-2 §15). Record:
[`../../x35_breakers_stage_2026-10-17/reads/n1p_look2.json`](../../x35_breakers_stage_2026-10-17/reads/n1p_look2.json).
Look 1: [`../read/README.md`](../read/README.md) (CONTINUE).

## Result

| read | statistic | N1-P | C0 | N1-P − C0 | class |
|---|---|---|---|---|---|
| **F (primary)** | Kakuna ckpt 34, greedy vs greedy, `forum72`, 60 units = 1,200 games per seed final | 0.509 [0.497, 0.522] (3,056 / 6,000) | 0.504 [0.491, 0.516] (3,023 / 6,000) | **+0.55 pp [−1.26, +2.36]** (95 %, Newcombe; AM-5 φ̂ 1.02, df 8, ×1.010) | **NULL** (inside [−3.0, +3.0]; final) |
| **P (co-primary)** | S on `home72` vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 53.48 / 54.11 / 53.66 / 52.95 / 54.39 | 56.82 / 57.39 / 56.84 / 57.37 / 56.59 | **−3.28 pp [−4.14, −2.43]** (Welch 97.5 %, df 6.8) | **INFERIORITY-NOT-EXCLUDED** |

- **F per seed:** +1.0, −0.25, +1.5, +1.0, −0.5 pp (3 of 5 positive: the sign rule holds; it is read only for a GAIN).
- **The verdict (PREREG §6):** NULL rules out **"N1 at the plateau"** (share 0.5, 20M) beyond ±3 pp, and nothing
  else: not N1 from step 0 (the plasticity confound, §1), the toy screen (Option T) or a human-derived corpus. X53-N1's
  trigger (an N1-P GAIN) did not fire. The two-look false-GAIN rate does not apply (no GAIN).
- **Against the registered guess (§10):** Δ_F +1.5 [−1.5, +4.5] → +0.55 (inside); Δ_P −1.5 [−3.0, 0.0] → −3.28,
  below its 80 % interval (as at look 1). The final class NULL was given probability 0.30.
- **Guards** G-F / G-E / G-C: **not read** (F is not a GAIN; the A1 ruling (a)).
- **X63 (its PREREG §9, AM-X63-N1L2):** NULL ⇒ the "GAIN / NULL / UNDECIDED-final" row: **X63 unchanged.**

## REPORTED, never decided (PREREG §7; 7 units = 140 games per seed final and panel)

| panel | N1-P | C0 | Δ (95 %, Newcombe) | role |
|---|---|---|---|---|
| H `heldout72_hpfix` | 345 / 700 | 316 / 700 | +4.14 [−1.08, +9.34] | generator-adjacent (62 % of its full sets are in the corpus) |
| P72 | 349 / 700 | 329 / 700 | +2.86 [−2.37, +8.07] | the corpus's own population |
| home72 (vs Kakuna) | 379 / 700 | 392 / 700 | −1.86 [−7.05, +3.35] | the in-pool cost, against Kakuna |
| **U** (H, F, P72 equal weight) | 0.500 [0.482, 0.518] | 0.475 [0.457, 0.493] | +2.52 [−0.02, +5.05] (φ̂ 0.67, not widened) | goal (b)'s meter, **M4-hazarded** (§8): not a goal read |
| **U without H** (F + P72; by hand, [`by_hand.py`](by_hand.py)) | 0.504 | 0.487 | +1.70 [−1.06, +4.47] (φ̂ 0.96, not widened) | the panel-validity ruling (`4f555e57`) |
| **Δ_H − Δ_F** | | | **+3.59 pp** | the "learned the generator" line |

- **Δ_H − Δ_F rose from +0.92 (look 1) to +3.59.** The gain now sits on the generator-adjacent panel (H +4.14,
  P72 +2.86) and not on the human-built F (+0.55): the reading PREREG §7 registered as set familiarity, not piloting.
  Descriptive only; every reported interval spans 0.
- **M4 (§8):** U 0.500 is NOT read as goal (b).
- **By hand, descriptive** ([`by_hand.json`](by_hand.json)): N1-P − D1 on F (D1 look 2, 700 games, on E1)
  −1.92 pp [−5.81, +1.99]. The harness-drift line: C0 s3504 / s3505's forum72 units 0–6 against D1's banked C0 units
  (E1, `x35_d1_read/h_d`, `h_e`): **279 / 280 games identical**, wins 133 vs 134 (the one differing game is in
  `c0_s3505__forum72__05`; the E1 → E3 encoder fix, as look 1's 6 / 420).

## Pre-record checks: every registered check PASSES

| check | result |
|---|---|
| **VOID gate** (§9 guard 1; the module's, `void.json`) | K9(b) at arm end PASS on all five N1-P seeds (`k9b/rb_x35_n1p_s350{1..5}.json`) |
| **VOID guards 2–5** (operator, AM-N1P-1 §14) for s3504, s3505 (s3501–s3503: look 1's README) | **2:** launcher-log lines 54–55 of each: the exact `🧬 [TEAM-MIX] n1_proc_v1@ALL (12000 teams, tier sha d59f44f0b33b) share 0.50 on both sides; eval stays on the pool` banner, `0 hits over 5 panel(s) (registry sha ea3917c2f1e6)`, `0 corpus team(s) pack identically to a pool team`. **3:** `pin_history` = `5f8f6cfa…` alone (= `git_hash`); `go.pin.txt` pin-check identical in form to s3501's; corpus-check PASS (12,000 teams, 120,000 soak battles). **4:** `pool seed: PASS`. **5:** 0 `corpus_<n>` mentions in the launcher log or `launcher_child.full.log`; every `FATAL`-matching line is a startup banner or an armed guard's (s3505's u100 canary warning, FINDING 1) |
| **Engine identity** (G7) and one regime | every counted unit, own and imported, on **E3** (`port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126`), Kakuna's pack legacy, sides `their=legacy` / `eval`: primary 600 / 600 rows, reported 210 / 210; S on `env=393631b886965b0c/126`, 140 rows. Regime ids equal look 1's: primary `00c8c0114bdf0836`, reported `3d2cbdc0b1413bcb`, S `d4de0fd7c16a2658`. `cells_inconclusive` empty |
| **The import** (AM-N1P-1 §14.3) | look 1's state, spec sha256 `1266daad…`, rows 360 (primary) and 126 (reported) and S nodes, engine E3, sides equal: accepted by `plan` and by main's `assemble` (engine, pack and sides checks) |
| **G14, the re-derived forced-end and peer-fault guards** (`report` mode, the spec registered before 10-27) | 810 units checked (486 imported + 324 own): **0 hits**, not held. 70 units `unavailable` for the forced-end half: every home72 unit of both looks (reported only), because the play commit takes captures only for panel cells (FINDING 3) |
| **The choice-level guard** (`choices_guard: require`) | not applicable: the `d1` spec has no game-for-game guard group (no `import_off` / `determinism_guard`) |
| **Floor** (§4: ≥ 1,200 · 5 − 20 per arm) | 6,000 / 6,000 F games per arm; every look-2 unit counted on its first attempt (324 starts, 324 counted, 0 retries, 0 exhausted) |
| **Completeness** | `COMPLETE`; `reported_incomplete` empty |

## Dose (reported; `../../x35_breakers_stage_2026-10-17/dose/`)

N1-P 4.401 / 4.045 / 3.995 / 4.036 / 4.053 GPU-h (**20.53**; registered ≈ 20.5), C0 4.430 / 4.027 / 4.012 / 3.844 /
4.172 (20.49): mean gap 0.2 %. s3504 / s3505 1,418 / 1,410 steps/s (no first-hour HOLD).

## How it was read

- Spec [`../read_spec_look2.json`](../read_spec_look2.json), byte for byte (sha256 `f28654cc…`). State
  **`~/gen3ai_archive/x61p_n1p_read/look2`**. [`cells.json`](cells.json), [`inputs.json`](inputs.json),
  [`decision.json`](decision.json), [`void.json`](void.json), [`models.json`](models.json), [`code.json`](code.json)
  and [`OPERATOR_NOTES.md`](OPERATOR_NOTES.md) are the state's.
- **Playing code:** `de15dfc7b` (look 1's anchor-lane commit, tag `x61p-n1p-look1-sched-de15dfc7`), clean, from a
  fresh worktree `~/dev/gen3ai-reads/n1p_play_look2_de15dfc7` with its own Rust build (stamps E3), as AM-N1P-2
  registered before any game (`d78700253`). No repin. The record was written from main (`962c4ea44` + this change's
  docs), for the s3504 / s3505 K9(b) records and main's assemble checks.
- Look 1's six finals were IMPORTED (never re-played): 324 new Kakuna units (6,480 games: c0 / n1p s3504–s3505 × 81)
  and 4 × 3,500 new S pairs. CPU only, `HF_HUB_OFFLINE=1`, nice 19 on cores 5-7,13-15; Kakuna at `--concurrency 2` on
  the heavy slots' short lane, S in 40-min boxes.
- Wall: Kakuna 10-29 16:12 → 10-30 11:04 (18.9 h; 4–23 units/h with the box's contention). S 10-29 16:12 → 18:17,
  **paused** 18:17 → 10-30 03:19 on the coordinator's CPU-priority ruling (X59-S's full read), resumed from the ledger
  and done 05:51 ([`OPERATOR_NOTES.md`](OPERATOR_NOTES.md)).

## FINDINGS

1. **Compile canary UNCONFIRMED verdicts on the ON arm** (AM-N1P-2 FINDING 13). s3505 at u100 (R1, bf16:
   `flat_intent_head.hidden.weight` compiled 1.20e-01 vs eager 5.85e-03; reproduced on 1 of 2 slices; u10 / u200 PASS;
   dump `~/gen3ai_archive/n1p_s3505_canary_u100_unconfirmed_2026-10-29/`), and s3501 at u200, which look 1 read
   without a note. N1-P 2 / 15 scheduled checks, C0 0 / 15 (the X35 stage 4 / 95). The PREREG names no canary guard,
   so both seeds are read. Descriptive sensitivity (the rule's estimators, by hand, not a verdict): without s3505 F
   reads +0.81 [−1.19, +2.81] (still inside ±3), without s3501 +0.44 [−1.86, +2.73] (inside), without both +0.75
   [−1.56, +3.06] (UNDECIDED: the upper bound crosses +3 with 3 seeds per arm, as at look 1).
2. **The in-pool cost holds at −3.28 pp** with five seeds (look 1 −3.27), twice D1's −1.61 at the same share and below
   the registered 80 % guess. Kakuna's own home72 reading agrees in sign (−1.86, wide).
3. **G14's forced-end half could not be checked on home72**: the play commit captures panel cells only
   (`main.x35.kakuna` at `de15dfc7b`: `--capture-dir` only when the team set is a panel cell), so all 70 home72 units (42 look 1, 28 look 2)
   read `unavailable`. Peer-fault was read on them; the 740 other units are clean. home72 is reported only.
4. **The S lane was paused for 9 h by a CPU-priority ruling** and resumed from the ledger: every count a multiple of
   100, no batch lost or doubled (`play-many` appends one fsynced row per finished 100-pair batch; a killed batch
   writes nothing).
5. **The play commit predates `52ff8cb1f`** (AM-N1P-2 FINDING 14): its heavy-slot client reads a zombie waiter as
   alive. No effect was seen.
6. **Δ_H − Δ_F +3.59** (descriptive): the procedural corpus's gain, such as it is, sits on the panels that share its
   generator's sets, not on human-built teams (PREREG §7, §13 FINDING 5).
