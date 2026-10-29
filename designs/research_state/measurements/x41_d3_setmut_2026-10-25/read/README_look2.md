# X41 / D3 look 2, the FINAL look, read as registered: **UNDECIDED** (2026-10-28)

D3 is C0 plus `--team-mix d3_setmut_v1@ALL:0.3 --team-mix-sides both`: 30 % of each side's teams come from 2,876 pool
teams, each with 1–2 mons' sets redrawn from Smogon usage. It is P\* + 20M at D3_PIN `9e289ba4`, seeds s3501–s3505.
- **Control:** C0 s3501–s3505, banked from D1's committed look-2 input.
- **Rule:** D1's, verbatim (`main.reads` rule `d1`, arm `X41`, look 2; [`../PREREG.md`](../PREREG.md) §3–§6).
- **Amendments:** AM-D3-ENG, AM-D3-L2 and its addendum items 1–11.
- **Record:** [`../../x35_breakers_stage_2026-10-17/reads/d3_look2.json`](../../x35_breakers_stage_2026-10-17/reads/d3_look2.json).
  It loads in `main.x35.batch.read_record`, and `final_class('d3')` reads UNDECIDED.

**Look 2 is TERMINAL** (addendum item 3): these rows are never pooled with further D3 seeds.

## Result

| read | statistic | D3 | C0 | D3 − C0 | class |
|---|---|---|---|---|---|
| **H (primary)** | Kakuna ckpt 34, greedy vs greedy, symmetric `heldout72_hpfix` / `forum72` / `p72`, equal weight, 700 games per panel per arm | 0.494 [0.473, 0.516] | 0.471 [0.450, 0.492] | **+2.33 pp [−0.69, +5.35]** (95 %, Newcombe; AM-5 φ̂ = 0.60, df 8 → not widened) | **UNDECIDED** (the LB is not > 0; sign rule 4 of 5 holds) |
| **P (co-primary)** | S on `home72` vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 57.08 / 56.74 / 56.59 / 55.89 / 56.44 | 56.78 / 57.39 / 56.86 / 57.38 / 56.58 | **−0.45 pp [−1.16, +0.25]** (Welch 97.5 %, df 7.8) | **NON-INFERIOR** (LB > −2.0) |
| H without `heldout72_hpfix` (reported; ruling `4f555e57`) | F + P72, same statistic | 0.497 | 0.481 | +1.64 pp [−2.06, +5.34] | (UNDECIDED; no rule reads it) |

- **The verdict is UNDECIDED, final.** H is not a GAIN, so the guards G-F / G-E / G-C were not played (`guards_not_read`;
  each is INCONCLUSIVE by registration). Set mutation is not adopted as a goal-(b) lever by this read.
- **The effect bound** (addendum item 2, M1). D3's effect on H lies within **[−0.69, +5.35] pp**.
  - A loss larger than 0.7 pp is excluded at 95 %.
  - The interval does not lie inside [−3.1, +3.1], so the "no effect beyond ≈ 3 pp" line does not apply.
  - NULL was unreachable at this size, as registered.
- **H per seed:** +2.1, −3.1, +4.3, +2.1, +6.2 pp. The two new seeds (s3504 +2.1, s3505 +6.2) both read positive.
- **H per panel (reported, no rule):**

  | panel | D3 | C0 | D3 − C0 |
  |---|---|---|---|
  | heldout72_hpfix | 342 / 700 | 316 / 700 | +3.7 [−1.5, +8.9] |
  | p72 | 346 / 700 | 331 / 700 | +2.1 [−3.1, +7.4] |
  | forum72 | 350 / 700 | 342 / 700 | +1.1 [−4.1, +6.4] |

  The registration predicted the gain spread over all three panels (|F − H| < 3 pp). Every panel reads positive, and
  |F − H| = 2.6 pp.
- **Against the registered guesses.**
  - H: the guess was +1.5 pp, 80 % [−1.5, +4.5]; the read is +2.33.
  - P: the guess was −0.8 [−2.5, +0.9]; the read is −0.45.
  - Classes at the final look: the registered UNDECIDED 0.67. The review's conditional P(GAIN) was 3–7 %.
- **Goal (b)** (LB > 0.50 on H's statistic): **not met** by D3 (0.494, LB 0.473).
- **Dose** (reported; `../../x35_breakers_stage_2026-10-17/dose/`): D3 20.68 GPU-h, C0 20.49 (each about 4.1 a seed;
  gap 1.0 %).

## Every game on ONE engine, E1

- **H, the guard and S all played on E1.** The H and guard groups stamp `port=7d932165…/90;env=7c532805…/125`, and S
  stamps `env=7c532805…/125`. The play pin was `c4a5c37b6` (tag `x41-d3-look2-play-c4a5c37b`; AM-D3-L2).
- **C0's banked rows (E1) imported** through `import_off`. The regimes are `09f297883701a35a` (H) and
  `d4de0fd7c16a2658` (S), and the engines were checked in both lanes.
- **The registered guard passed.** C0 s3501's 7 `heldout72_hpfix` units re-played game for game: `h_pooled` true,
  guard regime `a1c6cf53f17a355b`.

## The registered checks (AM-D3-L2 and its addendum), all PASS

| check | registered as | result |
|---|---|---|
| Kakuna-lane identity at the pin, before any game | AM-D3-L2 | 80 / 80 games identical choice by choice ([`look2_identity.json`](look2_identity.json)) |
| S-lane identity at the pin, before any Kakuna game | item 11 | C0 s3501 batch 0 vs `x26_pstar` / `tstar` equal on regime, seed, W/L/D and both outcome digests. Pool vs G1: 109/90/1, 121/79/0. home72 vs D1's P: 117/81/2, 119/78/3 ([`look2_s_identity_pool.json`](look2_s_identity_pool.json), [`look2_s_identity_home72.json`](look2_s_identity_home72.json)) |
| VOID gate | item 1 (B1) | d3_s3501–s3505 K9(b) PASS from the pin's own code ([`look2_void.json`](look2_void.json)) |
| guard, choice level | item 6 (m2) | 140 / 140 games identical: 5-tuple, START, both sides' choices, protocol bytes ([`look2_guard_choices.json`](look2_guard_choices.json)) |
| our loader | item 6 (m3) | 112 / 112 counted units `model_loader: bare` ([`look2_precheck.json`](look2_precheck.json)) |
| re-derived guards (G14: forced ends, disconnects, peer long-tail) | items 7, 10 | 112 / 112 counted units clean on both checks, none unavailable (main's `rederive.unit_check` at `715e0b482`; [`look2_precheck.json`](look2_precheck.json)) |
| external state | item 7 (m5) | see the table below |

**External state** ([`look2_external_snapshots.jsonl`](look2_external_snapshots.jsonl)), snapshotted at `start_s`,
`start_anchor` and `end`:

| what | result |
|---|---|
| Metamon checkout | `0a00a759`, no tracked change, the same at all three snapshots |
| `metamon` env | identical at all three |
| `gen3ai_torch28` env, conda list | identical at all three |
| `gen3ai_torch28` env, `pip freeze` digest | moved, but only through the editable install's line, which names main's HEAD (`ef05d165c` → `8f35c21b1` → `715e0b482`). Each digest is reproduced exactly from today's freeze with only that commit substituted. The play never imports it: the pin's `src/` is on `PYTHONPATH` |

## Descriptive rows (item 8, m6; no rule reads them)

- **D3 s3501–s3503, E1 (this read) vs look 1's E2 units, game for game**
  ([`look2_m6_e1_vs_e2.json`](look2_m6_e1_vs_e2.json)):
  - 9 of 1,260 games differ by 5-tuple (heldout72 2, forum72 3, p72 4), and 13 by our choice. That is the E1↔E3 rate
    of the determinism study (11 / 1,260).
  - Among the differing games, E2 won 6 and E1 won 4: a net −2 wins for E1 (−0.16 pp).
  - Regime ids are equal, and the START is equal on all 1,260.
- **Look 1's Δ̂ recomputed with its ON rows on E1:** +1.11 pp [−2.79, +5.00], CONTINUE. As recorded on E2 it was
  +1.27 [−2.63, +5.16]. Look 1's CONTINUE stands either way.

## How it was read

- **Spec** [`read_spec_look2.json`](read_spec_look2.json) (sha256 `f67c56d4…` = the state's `spec.json`).
  - The state is `~/gen3ai_archive/x41_d3_read/look2/`; the plan-only first state was moved aside as
    `look2.void_plan_k9b_1793122438`.
  - Copies were sha-checked, and `models/` was only read.
  - [`look2_cells.json`](look2_cells.json), [`look2_inputs.json`](look2_inputs.json) and
    [`look2_decision.json`](look2_decision.json) are the state's.
- **Playing and deciding code:** the pin `c4a5c37b6`, in a clean worktree `~/dev/gen3ai-reads/d3l2_pin` (`record` ran
  there, addendum item 1), with `HF_HUB_OFFLINE=1`. Its untracked `Cargo.lock` files are the legacy blobs.
- **Order** ([`look2_queue_start.py`](look2_queue_start.py), [`look2_queue_starter.jsonl`](look2_queue_starter.jsonl)):
  - the S lane ran 2026-10-27 14:56–20:26 PT, after H-CTX rc72b's anchor supervisor exited;
  - the Kakuna lane ran 2026-10-28 08:55–18:21 PT, at `--concurrency 1`, after N1-P look 1's last unit counted;
  - 95xx ports with held flocks; CPU only, nice 19 on cores 5-7,13-15.
- **Units:** 105 / 105 H units, 7 / 7 guard units and 5 × 3,500 S pairs were counted.

## FINDINGS

1. **Two units' first attempts were dropped whole for an H5 peer fault.** Kakuna's long-tail line appeared in
   `peer_challenger.log`, so the unit had `peer_clean = False` and was not counted. The two units were
   `d3_s3502__heldout72_hpfix__00` and `d3_s3502__p72__06`. Each re-ran at its seed and counted, and the re-derived
   guards read both re-runs clean. That is the pin's harness doing what `peer_clean` is for. 2 H5 faults in 114 unit
   plays is in line with the archive's rate.
2. **The `gen3ai_torch28` pip digest is not a stable fingerprint of the environment.** The editable install's line
   carries main's HEAD. A later snapshot script should drop that line before hashing.
3. **The read cost more lane time than the estimate.** The quiet-box estimate was about 16 h; the Kakuna lane took
   9.4 h of wall once it ran, at 2.5–4.5 min a unit when uncontended. The two deferrals (behind H-CTX, then N1-P)
   put about 33 h between registration and the first Kakuna unit.
4. **The guards were not played** (H not a GAIN), so D3 has no G-E / G-F / G-C read. The registration needs none on
   a non-GAIN.
