# X61P / N1-P look 1, read as registered: **CONTINUE** (2026-10-28)

N1-P = C0 + `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` (12,000 procedural teams from the Smogon facade
alone), P\* + 20M at N1P_PIN `5f8f6cfa`, s3501–s3503. Control: C0 s3501–s3503 (production pin `f4b25238`),
**re-played** in this state. Rule: D1's, verbatim, on F alone (`main.reads` rule `d1`, arm `X61P`; [`../PREREG.md`](../PREREG.md)
§4–§9 and AM-N1P-1 §14). Record:
[`../../x35_breakers_stage_2026-10-17/reads/n1p_look1.json`](../../x35_breakers_stage_2026-10-17/reads/n1p_look1.json)
(it loads in `main.x35.batch.read_record`; `kit stage-check --arm n1p --seed 3504` now says ALLOWED).

## Result

| read | statistic | N1-P | C0 | N1-P − C0 | class |
|---|---|---|---|---|---|
| **F (primary)** | Kakuna ckpt 34, greedy vs greedy, `forum72`, 60 units = 1,200 games per seed final | 0.516 [0.500, 0.532] (1,857 / 3,600) | 0.508 [0.492, 0.525] (1,830 / 3,600) | **+0.75 pp [−1.56, +3.06]** (95 %, Newcombe; AM-5 φ̂ = 0.33, df 4 → not widened) | **UNDECIDED → CONTINUE** (Δ̂ > 0 at look 1) |
| **P (co-primary)** | S on `home72` vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 53.48 / 54.11 / 53.66 | 56.82 / 57.39 / 56.84 | **−3.27 pp [−4.20, −2.33]** (Welch 97.5 %, df 4.0) | **INFERIORITY-NOT-EXCLUDED** (the whole interval is below the −2.0 bar) |

- **F per seed:** +1.0, −0.25, +1.5 pp (2 of 3 positive: the look-1 sign rule holds; the interval does not clear 0).
- **Against the registered guess (PREREG §10):** Δ_F +1.5, 80 % [−1.5, +4.5] → read +0.75; Δ_P −1.5 [−3.0, 0.0] →
  read **−3.27, below the guess's 80 % interval**.
- **Guards** G-F / G-E / G-C: **not read** (F is not a GAIN; the A1 ruling (a)).
- **No GAIN, so no false-GAIN line yet.** If look 2 ends GAIN, it is read beside D1's two-look false-GAIN rate,
  3.5–7.4 % (PREREG §13 FINDING 1).

## REPORTED, never decided (PREREG §7; 7 units = 140 games per seed final and panel)

| panel | N1-P | C0 | Δ (95 %, Newcombe) | role |
|---|---|---|---|---|
| H `heldout72_hpfix` | 205 / 420 | 198 / 420 | +1.67 [−5.07, +8.38] | generator-adjacent (62 % of its full sets are in the corpus) |
| P72 | 208 / 420 | 196 / 420 | +2.86 [−3.89, +9.56] | the corpus's own population |
| home72 (vs Kakuna) | 232 / 420 | 236 / 420 | −0.95 [−7.63, +5.74] | the in-pool cost, against Kakuna |
| **U** (H, F, P72 equal weight) | 0.500 [0.477, 0.523] | 0.482 | +1.76 [−1.52, +5.03] (φ̂ 0.60, not widened) | goal (b)'s meter, **M4-hazarded** (§8): not a goal read |
| **U without H** (F + P72; the panel-validity ruling, 4f555e57) | 0.506 | 0.488 | +1.80 [−2.17, +5.77] (φ̂ 1.24, widened ×1.11) | by hand, `d1.strat_score` / `newcombe_strat` / `overdispersion` / `widen` |
| **Δ_H − Δ_F** | | | **+0.92 pp** | the "learned the generator" line: no concentration on H |

**M4 (PREREG §8):** U is inflated for any N1-trained model on H and P72, so U 0.500 is NOT read as goal (b) (its LB is
0.477 anyway). By hand, descriptive: N1-P − D1 on F (D1 look 2, 700 games at unit seeds 0–6) −1.27 pp [−5.29, +2.77].

## VOID guards (PREREG §9, AM-N1P-1 §14): every seed PASSES

| seed | 1 K9(b) at arm end (the module's gate, `void.json`) | 2 `[TEAM-MIX]` banner, held-out hits, pool copies | 3 pin (`pin_history`) / corpus-check | 4 pool-seed | 5 FATAL on a `corpus_<n>` team |
|---|---|---|---|---|---|
| 3501 | PASS (`k9b/rb_x35_n1p_s3501.json`) | launcher log lines 54–55: the exact banner; `0 hits over 5 panel(s) (registry sha ea3917c2f1e6)`; `0 corpus team(s) pack identically to a pool team` | `5f8f6cfa…` alone / PASS (12,000 teams, 120,000 soak battles) | PASS | none |
| 3502 | PASS | as s3501 | as s3501 | PASS | none |
| 3503 | PASS | as s3501 | as s3501 | PASS | none |

Evidence: `~/.claude/jobs/x35/rb_x35_n1p_s350X.{launcher.log,go.pin.txt,go.corpus.txt,pool_seed_check.txt}` and
each run's `metadata.json` / `launcher_child.full.log`. No `corpus_<n>` string appears in either log; every line
matching `FATAL` is the warn-mode KN(b) behaviour-policy mismatch or an armed guard's banner (K9(b) at end is PASS).

## Dose (reported; `../../x35_breakers_stage_2026-10-17/dose/`)

N1-P 4.401 / 4.045 / 3.995 GPU-h (12.44; registered ≈ 12.3), 1,388 / 1,413 / 1,433 steps/s (no first-hour HOLD).
C0 4.430 / 4.027 / 4.012 (12.47); mean gap 0.2 % (< the 10 % confound bar).

## How it was read

- Spec [`../read_spec_look1.json`](../read_spec_look1.json), byte for byte (sha256 `1266daad…`, the value look 2's
  import pins); state **`~/gen3ai_archive/x61p_n1p_read/look1`**, the path AM-N1P-1 registered. Copies sha-checked;
  `models/` only read. [`cells.json`](cells.json), [`inputs.json`](inputs.json), [`decision.json`](decision.json),
  [`void.json`](void.json), [`models.json`](models.json), [`code.json`](code.json) and [`repins.jsonl`](repins.jsonl)
  are the state's.
- **Playing code:** `46b9e572c` (tag `x61p-n1p-look1-play-46b9e572` on origin; clean tree), then, for the anchor lane
  from 2026-10-26 18:41, **`de15dfc7b`** (tag `x61p-n1p-look1-sched-de15dfc7`) by a `main.reads repin`: `46b9e572` +
  the reads module's `--concurrency K` and `repin` only; `main.reads.playpath` found 1,537 play-path files
  byte-identical (digest `9822a306…`). The S lane played on `46b9e572` throughout. CPU only, `HF_HUB_OFFLINE=1`,
  nice 19 on cores 5-7,13-15; Kakuna units on the heavy slots' short lane (K = 1, then K = 2), S in ≤ 40-min boxes.
- 486 / 486 Kakuna units counted (9,720 games), 6 × 3,500 S pairs. Wall: S 15:23 → 23:07 on 10-26; Kakuna 15:23 on
  10-26 → 08:54 on 10-28 (≈ 41.5 h; the registered estimate was ≈ 27 lane-h).
- **C0 was re-played, not imported:** the registered spec carries no `import_off` / `determinism_guard`, and editing
  it would move the sha look 2 pins. As PREREG §7's harness-drift line,
  [`harness_drift.py`](harness_drift.py) → [`harness_drift.json`](harness_drift.json) compares C0's re-played
  forum72 units 0–6 with D1's banked C0 units: **414 / 420 games identical**, wins 212 vs 208. By hand,
  descriptive: C0 here (60 units) − C0 at D1's read (7 units, 5 seeds) on F +1.98 pp [−2.07, +6.01].

## Hazards (FINDINGS)

1. **P reads a resolved in-pool cost of −3.27 pp**, twice D1's (−1.61 at the same 0.5 share) and below the
   registered 80 % interval. With F UNDECIDED the verdict is CONTINUE regardless; at look 2 a GAIN on F would be a
   TRADE, not an ADOPT-CANDIDATE, unless P's interval moves above −2.0.
2. **Kakuna play is near-deterministic, not exactly** (against X26c's premise, which AM-N1P-1 §3 cites): 6 of 420
   re-played C0 games differ from D1's banked rows, and the one uncounted unit (`n1p_s3503__p72__03`, a 1-turn
   "win" with `peer_clean: false`) matched its counted replay on 18 of 20 games. Look 2 imports look 1's rows, so its
   read is unaffected; a re-play would not reproduce counts exactly.
3. **The read spans two playing commits** (the repin above), against the brief's "one commit": the play path is
   byte-identical and the record lists the repin, but the record's `code` names only `46b9e572`.
4. **The repin was asked for by the reads-tooling worker "at the coordinator's request"**, adopted after an
   independent check that `46b9e572..de15dfc7` touches only `src/main/reads/` and `designs/tools/reads.md`.
5. **Heavy-slot contention set the read's length** (3–20 units/h; a quota-burn wave of ship gates on 10-27). K = 2
   gave 10.6 vs 7.7 units/h in its first hour.
6. **The pack regime:** both arms played on ONE engine: Kakuna's LEGACY pack (the play commit predates
   `b0441623e`, SOP H22), with our side's packer fixed (`d8d674b99` is in it). The primary's regime id is
   `00c8c0114bdf0836`, not D1's / D3's `09f297883701a35a`, so no banked D1 / D3 Kakuna row pools with this read. The
   6 drift games are on `sft053/054/058/081/105`, not the re-packed `sft431`, so they are play nondeterminism, not the
   packer fix. The trainee side's own pre-fix packing (N1-P 0.375 % of draws vs C0 0.125 %) is the 10-27 AUDIT's
   ≤ 0.2 pp bias.
7. **One unit lost its `end` event** (`c0_s3501__forum72__01`, its supervisor stopped for the repin while it played);
   the restarted supervisor counted it from its `summary.json` and did not replay it.
</content>
</invoke>
