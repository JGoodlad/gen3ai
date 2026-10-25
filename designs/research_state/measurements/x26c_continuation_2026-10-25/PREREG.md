# X26c — the X26 continuation (380M → 480M) against Kakuna: PRE-REGISTRATION (2026-10-25)

Written and shipped **before** any game of the 480M final was played (the run `models/rb_x26c_s2601` was at
≈ 447M when this was written, 01:56 PDT). Why the run exists: the orchestrator's 2026-10-24 ruling on FINDING 11
([`design_beat_metamon_path.md`](../../../endstate/design_beat_metamon_path.md) §9): the one lever with consistently
positive measured value is more training, so X26 @380M trains on to 480M at the production pin `f4b25238`, then is
re-read against Kakuna.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (X26's plateau checkpoint); H = `heldout72_hpfix`; F = `forum72`; U = the pooled unfamiliar score.

## 1. The run

`rb_x26c_s2601`, a FORK of `rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip` at the production pin
`f4b25238`, seed 2601, to 480,000,195 steps (`~/.claude/jobs/x35/status.txt`, X26C lines 2026-10-24 12:37). Its
K9(b) at the time of writing (`main.x35.batch.k9b_read`, 684 post-fork rows): PASS, judged max 6.3e-4 (ceiling
1e-3), p99 4.3e-5 (ceiling 1e-4).

## 2. The models

- **The 480M final** (`finals.py`): `final_model.zip` if its recorded step is ≥ 480,000,000, else the newest
  checkpoint whose step is ≥ 480,000,000; untouched for ≥ 5 min. Read only once one exists.
- **P\*** (`checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`, X51's file).
- **X26 @380M** (the fork point; `checkpoint_380000195_steps.zip`) for S2 only.
- Every model is a sha-checked COPY with its sidecars in `~/gen3ai_archive/x26c_read/`; `models/` is only read.

## 3. PRIMARY: Kakuna on the unfamiliar panels

- **Cell = D1 look 2's H cell** (`python -m main.x35.kakuna`): Metamon's Kakuna ckpt 34, greedy vs greedy, our side
  on CPU, the in-process Rust front end on a claimed 95xx port, the SYMMETRIC panel configs of `PANEL_CELLS`
  (sha-pinned), the counting rule `main.anchors.results.unit_counts`, 20-game role-balanced units with
  `--team-seed = --seed-base =` the unit seed.
- **One change from D1: unit seeds 0–19** (`plan --unit-seeds 0-19`, added for this read; D1 used 0–6, which stay
  a subset), so each final plays **400 games per panel, 1,200 in all** (D1: 140 per panel per seed final). At D1's
  140 per panel, goal (b)'s bar would need U ≈ 0.55.
- **P\* is RE-PLAYED in the same harness, same seeds**, not taken from X51: X51 played at the `63f97a57` checkout
  (config v158), and the 480M final (config v161) is played at this read's commit, so only a same-harness P\*
  makes the difference a model difference. X51's P\* (0.487 [0.459, 0.515]) is reported beside as a harness check
  (descriptive: no rule reads it).
- **The playing code** is a detached checkout of THIS registration's commit, bootstrapped, for every unit of both
  finals. P\*'s units may run before the 480M final exists (they carry no information about it).
- **Statistics** (`verdict.py`, shipped with this file): U = `main.x35.d1.strat_score` over H, F and P72 (equal
  weight, Kish n_eff, Wilson 95 %); **Δ = U(480M) − U(P\*)**, `main.x35.d1.newcombe_strat` (Newcombe, unpaired: the
  headline). Reported, no rule: each panel; the unit-paired difference (same panel and unit seed); the seeds 0–6
  subset beside D1 look 2's C0 (0.471) and D1 (0.492).

## 4. SECONDARY: S against the frozen R

- **S1 (registered): S of the 480M final in THE X26 DRIFT READ'S REGIME** (`s_read.py`, node `x26c_480m_pool`):
  `main.x35.readout.population_score` against `reference_population_production.json` (content `cef0960e…`, 7
  members), 500 mirrored pairs per member, greedy vs greedy, the eval team builder (no panel), CPU,
  `torch_threads` 1 (the drift read's regime id was `adfdbee824c9eb0a`). The 380M node is ALREADY measured there:
  **58.23 ± 0.55** (dbc49cc1), so it is not replayed. **S rising** iff S1(480M) − 58.23 has LB > 0 (se =
  √(se₄₈₀² + 0.55²)).
- **S2 (reported only): S on the home72 panel** (`--panel home72`, D1's co-primary P regime) at 380M
  (`x26_380m_home72`) and 480M (`x26c_480m_home72`). The brief named "home72 (the drift read's regime)"; the drift
  read is NOT on home72 (it used the eval team builder), so the registered S is S1, and S2 answers the home72 wording.
- Units: two engines, ≤ 40-min boxes, one heavy slot each (`mem_cap.sh --slot`), nice 19 on cores 5-7,13-15.

## 5. Predictions and decision rules (applied by `verdict.py`)

**Predictions** (from the measured slopes, stated before any 480M game):

| quantity | predicted | from |
|---|---|---|
| S1(480M) | ≈ 60.1 pp, i.e. ≈ +2 vs 380M (58.23) and ≈ +4 vs the trend's 280M | drift slope +2.05 ± 0.51 pp / 100M |
| U(480M) | ≈ 0.515 [0.49, 0.54] | +1.5 pp / 100M (161M → 280M: +3.0 over 119M), from P\*'s 0.487 |
| Δ = U(480M) − U(P\*) | ≈ +3 pp | the same slope over 200M |

**Rules:**

1. **VOID** iff the 480M run's K9(b) at its end (`main.x35.batch.k9b_read`) is not PASS, or fewer than 90 % of the
   60 units of EITHER final count (≥ 54). A VOID read states no verdict.
2. **Goal (b) is MET** iff U(480M)'s Wilson LB > 0.50.
3. **CLIMBING** (keep training) iff Δ's Newcombe LB > 0. **FLAT** otherwise (a negative Δ is FLAT too, reported
   with its sign).
4. **FLAT with S rising ⇒ revisit D3**, per the decision record's 2026-10-24 ruling ("the unfamiliar gap persisting
   with in-house strength rising"). FLAT with S not rising: reported as FLAT, nothing triggered.
5. Rule 8: a bound within 1e-9 of its bar does not clear it.

**Power, honestly:** at 1,200 games a final, U's Wilson half-width is ≈ 2.8 pp and Δ's ≈ 4.0 pp. If the
prediction is right (U ≈ 0.515, Δ ≈ +3), goal (b) is met with probability ≈ 0.17 and CLIMBING fires with
probability ≈ 0.3. The likely outcome even when the prediction holds is "goal (b) not met; FLAT". So **a FLAT here
is not evidence that training has stopped helping**; only Δ's interval says how much it can exclude (its UB).

## 6. Order of work, cost, hazards known in advance

1. Ship this file. 2. P\*'s 60 Kakuna units (short lane, `mem_cap.sh --max-s 600`) and S2 @380M (heavy slot)
   start at once. 3. On the 480M final: K9(b), then its 60 Kakuna units, then S1, then S2 @480M. 4. `verdict.py`.
- Cost ≈ 120 Kakuna units × ≈ 4 min of play (≈ 8 h of one lane) and 3 S nodes × ≈ 1.8 h of play; the heavy-slot
  queue has roughly doubled wall time on every read since X51.
- Not read: X26 @380M against Kakuna (the fork point's own standing), the reverse cell, home72 vs Kakuna.
