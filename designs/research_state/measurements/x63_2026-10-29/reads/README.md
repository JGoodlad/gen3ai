# X63 Stage A — the read (2026-10-31): NOT PURSUED

**Verdict: NOT PURSUED** (rule `x63_kvk`, Stage A, PREREG §7 item 1; spec of record `../read_spec.json`, sha256
`61db0eac…`, AM-X63-SPEC §16). The long union continuation (X63-U s2601: X26@380M + 100M with 80 % of each side's
teams from the 17,876-team union `x63_union_v1`) did not clear the +1.0 pp screen against the two pool-only
continuations on forum72 against Kakuna. It claims nothing for the era (§7 item 1).

Record: [`stageA.json`](stageA.json) (`main.reads record`, written from main). Copies of the state's small files,
the pre-record checks and a by-hand recomputation: [`stageA_state/`](stageA_state/). State:
`~/gen3ai_archive/x63_read/stageA` (cells.json 3 MB, the unit cells and logs).

## The numbers

Each score is a plain proportion of counted games, a tie a non-win. All 900 units counted (18,000 games).

| model | forum72 (F), 4,000 games | home72, 2,000 games |
|---|---:|---:|
| X63-U s2601 (union 0.8) | 0.5123 (2,049) | 0.5835 (1,167) |
| X26c s2601 (pool) | 0.5213 (2,085) | 0.5855 (1,171) |
| X63-C s6302 (pool) | 0.5170 (2,068) | 0.6025 (1,205) |

- **Δ_F = −0.69 pp [−3.75, +2.37]** (95 %; z, because σ̂ ≤ the 1 pp floor; SE 1.56 pp = the binomial part 0.97 pp and
  the floored seed term). The screen needs Δ̂_F ≥ +1.0 pp; −0.69 is 1.69 pp short, far from rule 8's ε.
- The 72-team cluster bootstrap of Δ_F (REPORTED; 10,000 draws, seed 0): [−2.52, +1.14] pp, SD 0.93 pp.
- Δ_home72 (REPORTED at Stage A) = −1.05 pp [−4.62, +2.52].
- **σ̂_CC = 0** (X63-C's registered quantity, its PREREG §3): d = F(X63-C) − F(X26c) = −0.43 pp, and d² is below the
  two binomial variances (d² − p̂₁q̂₁/G₁ − p̂₂q̂₂/G₂ < 0), so the +100M seed spread on F is not resolved above
  binomial noise at 4,000 games a model. On home72 the same arithmetic gives σ̂ = 0.49 pp (d = +1.70 pp).
- **P(NOT PURSUED) at a true effect** (`power/power_x63_v2.log` A, the registered row nearest σ̂_CC, σ = 0):
  0.149 at +2 pp, 0.020 at +3 pp. The interval's UB (+2.37 pp) is below +3 pp.

## What it licenses

- **Stage B does not fire** (§7 items 1–2). X63-U s6302, trained speculatively under AM-X63-BSPEC (§17), is **"X63-U
  s6302: a speculative replicate, trained before Stage A's verdict, not read for any decision."** No s6302 game is
  played under X63.
- **AM-X63-N1L2 (§9):** N1-P look 2 read NULL (10-30), so X63 is unchanged and the verdict carries no n1 LOSS label.
- **AM-X63-CANARY (§15) / AM-X63-PARITY (§18):** no resume on s2601 (11 / 11 canaries pass, 0 restarts).
- **The price is not computed:** no GAIN, so neither FLOOR nor net applies (r = 1 stands for X62).
- What the read does NOT say: that breadth never helps. It says this lever (share 0.8 on this union, +100M from
  X26@380M under the production KL controller) gave no ≥ +1 pp F gain at one seed, with a true +3 pp unlikely.

## Checks (all PASS; `stageA_state/checks.json`)

- **VOID gate** (§8 record half): K9(b) PASS at the arm's end; every recorded pin 39eaa326; pool seed fork step
  380,000,195 (20 snapshots, 0 dropped). Operator half: the banner exact at 17:35:30, no FATAL / `does NOT inherit`,
  0 restarts, complete (last step 480,081,047; newest checkpoint 80,951 steps back).
- **One engine:** every unit on E1 (`port=7d932165…/90;env=7c532805…/125`, = `engine_at` of the pin); one regime id
  per team set (forum72 `5e1e14aaa845e97e`, the shared pin's id; home72 `df9df2aa9e535b5c`); peer code
  `9cc9caacde85f8ea`, policy mode `train`, our regime `greedy` on every unit.
- **Kakuna identity on that engine:** one counted unit (`x26c_s2601__forum72__1000`) re-played at the pin into a
  scratch state: 20 / 20 games equal by 5-tuple and by both players' choices.
- **Re-derived guards (G14, mode `count`):** 900 units checked, 0 hits; 300 home72 units `unavailable` (FINDING 1).
- **Choice-level guard:** not applicable (`x63_kvk` registers no game-for-game guard; `choices_guard` resolves to
  `require`).
- **By hand** (`by_hand.py` → `by_hand.json`): Δ, SE, LB, UB, σ̂, σ̂_CC and the class equal the rule's.
- Units: 900 / 900 counted, 0 exhausted; 4 attempts uncounted, each a Metamon long-tail reset (H5 / H21) voided in
  tree and re-played.

## FINDINGS

1. **home72 games are never captured at this pin**, so the re-derived forced-end check reads `unavailable` on all
   300 home72 units. `kakuna.unit_argv` passes `--capture-dir` to panel cells only, and `x63_kvk` refuses the one
   spec lever (`per_game`). Backing it: the in-tree forfeit-win guard (G6) is ARMED on 300 / 300 home72 units (and
   600 / 600 forum72), and it fired once (`x26c_s2601__home72__1026` attempt 1, a non-battle termination, voided and
   re-played); the `ws_frontend.log` witness ran on every unit. Orchestrator ruling (2026-10-30): option 1, "missing
   capture evidence ⇒ judged as before"; a Stage B amendment would state it before any home72 game. Moot now (no
   Stage B).
2. **The REPORTED groups (`traj`, `reported`, `pstar`) were not played** (AM-X63-SPEC FINDING 11: the lane interleaves
   groups by seed index, and `x63_kvk` refuses imported deciding groups). Their node files are fixed by sha256 in §16.
   They decide nothing; playing them now is an orchestrator call.
3. **σ̂_CC = 0 is a floor, not a measurement of zero spread**: two pool continuations differ by 0.43 pp, inside
   binomial noise (SE of a difference ≈ 1.12 pp). Stage A's power row is therefore the σ = 0 one; the design's prior
   (σ 1.25 pp) is neither confirmed nor refuted.
4. The pin's bootstrap smoke fails 5 tests beyond PREREG FINDING 9's 2, all from the hybrid pin's newer tests
   against `f4b25238`'s game paths (the legacy packer's happiness and Hidden Power tests; a missing
   `stamp_guard.js`). The engine and the build stamps are E1.
