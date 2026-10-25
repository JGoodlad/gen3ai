# X26c — the X26 continuation (380M → 480M) against Kakuna: **goal (b) NOT met; FLAT** (2026-10-25)

Read as registered in [`PREREG.md`](PREREG.md) (shipped `e05c45eeb` before any 480M game). The run:
`rb_x26c_s2601`, X26 @380,000,195 forked at the production pin `f4b25238`, ended at **480,081,209**
(`final_model.zip`, sha256 `cba7a1f8…`). Abbreviations as in the prereg (U = the pooled unfamiliar score, P\* = X26
@280M, H = `heldout72_hpfix`, F = `forum72`).

## Answer

- **100M more steps moved nothing against Kakuna on unfamiliar teams.** U(480M) **0.4875 [0.459, 0.516]** vs P\*
  **0.4867 [0.459, 0.515]**, Δ **+0.08 pp [−3.90, +4.06]** (Newcombe; unit-paired +0.08 [−3.61, +3.78]). The
  prediction was ≈ 0.515 and Δ ≈ +3; the point estimate is ≈ 0.
- **Goal (b) NOT met** (LB 0.459 ≤ 0.50). **Trajectory FLAT** (Δ's LB < 0).
- **In-house S did not rise either.** S1 (the drift read's regime) **58.22 ± 0.55 at 480M vs 58.23 ± 0.55 at 380M**:
  −0.01 pp [−1.54, +1.52]. The drift slope's prediction (+2 pp) failed: the +2.05 pp / 100M of 290–380M did not
  continue. So "FLAT with S rising" does not hold, and **the D3-revisit rule does not fire**.
- Verdict not VOID: K9(b) at the run's end PASS (judged max 6.3e-4 < 1e-3, p99 4.9e-5 < 1e-4, 1,018 post-fork
  rows); 60 / 60 units counted for each final.

## The table (Kakuna ckpt 34, greedy vs greedy, symmetric cells, seeds 0–19, 20-game units)

| cell | 480M wins / n | P\* wins / n | 480M − P\* |
|---|---|---|---|
| H | 182 / 400 (0.455) | 199 / 400 (0.498) | −4.3 pp |
| F | 217 / 400 (0.543) | 212 / 400 (0.530) | +1.3 pp |
| P72 | 186 / 400 (0.465) | 173 / 400 (0.433) | +3.3 pp |
| **U** (equal weight, Kish n_eff ≈ 1,207) | **0.4875 [0.459, 0.516]** | **0.4867 [0.459, 0.515]** | **+0.08 [−3.90, +4.06]** |
| U, seeds 0–6 only (D1's units) | 0.462 [0.415, 0.510] | 0.457 [0.411, 0.504] | (D1 look 2: C0 0.471, D1 0.492) |

Per panel, no rule; each panel's difference is well inside its ≈ ±7 pp noise. One tie (480M, H).

## S against the frozen R (`reference_population_production.json`, content `cef0960e…`, 7 × 500 mirrored pairs)

| node | regime (ledger rows) | S (pp) | se | vs |
|---|---|---|---|---|
| **480M, eval team builder (S1, registered)** | `adfdbee824c9eb0a` (= the drift read's) | **58.22** | 0.55 | 380M 58.23 (drift read): −0.01 [−1.54, +1.52] |
| 380M, home72 panel (S2) | `d4de0fd7c16a2658` (= D1's P) | 57.61 | 0.56 | — |
| 480M, home72 panel (S2) | `d4de0fd7c16a2658` | 58.79 | 0.55 | +1.18 [−0.36, +2.72] vs 380M, reported |

Per member at 480M (S1): production 71.8, T\* 59.6, x26_080m 61.0, x26_140m 54.2, x26_210m 54.4, x26_270m 54.9,
P\* 51.7. Full blocks: [`read/s_results.json`](read/s_results.json).

## What it means

- **Plain continuation past 380M is not a lever for goal (b), on this evidence.** The interval still allows +4 pp
  per 100M, so this does not rule out a slow gain, but the point estimate is ≈ 0, and the in-house meter that
  justified the continuation (the drift slope) is flat over the same 100M. Both meters agree.
- The decision record's 2026-10-24 ruling names its own next step only for "the unfamiliar gap persisting **with
  in-house strength rising**"; in-house strength did not rise, so the registered branch is FLAT, nothing triggered.
  The next lever is the orchestrator's call (§9 of the beat-Metamon design).
- The home72-panel S rose +1.2 pp (not detected) while the eval-builder S did not move; neither change is
  detected, so the two regimes do not disagree on anything they can resolve.

## How it was read

- Every unit played from a detached checkout of the prereg commit (`/home/goodlad/.cache/gen3ai/x26c/pin_e05c45eeb`),
  CPU only, nice 19 on cores 5-7,13-15. Kakuna: `python -m main.x35.kakuna plan --teamsets unfamiliar --unit-seeds
  0-19` + `run`, short lane (`mem_cap.sh --max-s 600`), claimed 95xx ports; one state per final
  (`~/gen3ai_archive/x26c_read/kakuna_pstar`, `kakuna_480m`). S: [`s_read.py`](s_read.py), heavy slots, ≤ 40-min
  boxes. Verdict: [`verdict.py`](verdict.py) → [`read/verdict.json`](read/verdict.json). Units:
  [`read/kakuna_units.jsonl`](read/kakuna_units.jsonl); models: [`read/finals.json`](read/finals.json); K9(b):
  [`read/k9b_at_end.json`](read/k9b_at_end.json). Raw unit dirs and ledgers: `~/gen3ai_archive/x26c_read/`.
- Wall: P\*'s units 02:05–04:15, the 480M's 08:38–11:58; S 02:05–10:28. Games ≈ 2.5–3.5 min per 20-game unit.

## FINDINGS

1. **The harness reproduces X51 exactly.** P\* re-played at `e05c45eeb` gives the SAME wins on all 60 units as X51
   at `63f97a57` (199 / 212 / 173): Kakuna-cell play is deterministic per (model, unit seed) across those commits. So
   X51's P\* numbers ARE comparable to this read's, and a future read can reuse a banked P\* (checked by one unit).
2. **The final-selection rule raced the end chain.** The 480M checkpoint `checkpoint_480000085_steps.zip` was quiet
   while `final_model.zip` (08:32:58) was not, so the Kakuna chain twice started on the checkpoint (81k steps short)
   before the final was 5 min quiet. Both attempts were stopped by their process groups after ≤ 4 games of one unit,
   set aside (`kakuna_480m.void_ckpt480000085`, `…void2…`) and never read. Every counted unit and both S nodes played
   `final_model.zip` (sha `cba7a1f8…`).
3. **`s_read.py`'s default `--mem-cap` path was wrong** at the prereg commit (`parents[4]`); the first S launch
   failed before playing; relaunched with the flag typed. Fixed here (`parents[3]`).
4. **The per-node `regime_ids` in `s_results.json` are empty** (the ledger's cell objects carry no regime id); the
   regime was checked from the ledger rows instead (88 rows `adfdbee8…`, 188 rows `d4de0fd7…`).
5. **The brief's S regime was internally inconsistent** ("home72 (the drift read's regime)"): the drift read is on
   the eval team builder. Registered S1 there and reported S2 on home72 (prereg §4).
6. **The read was under-powered for the effect it predicted** (prereg §5: CLIMBING ≈ 0.3 if Δ = +3). It did not
   matter here: Δ̂ ≈ 0.
7. K9(b)'s judged max (6.3e-4) sits closer to its ceiling than C0's (2.3e-4); PASS. The run logged 157
   behaviour-violation lines against its separate 1e-4 warn bar (`--behaviour-check warn`).
8. Not read: X26 @380M against Kakuna, the reverse cell, home72 against Kakuna (prereg §6).
9. The coordinator relayed a canary agent's refusal ("fork-recipe resolution could not find --vf-coef in
   model_config.json") on an x26c checkpoint; neither loader here (`main.anchors`, `main.h2h play-many`) hit it.
