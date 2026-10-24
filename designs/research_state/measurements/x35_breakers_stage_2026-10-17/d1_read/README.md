# D1 (X38), read as registered: look 2 (FINAL) **UNDECIDED** (2026-10-24); look 1 CONTINUE (2026-10-23)

D1 = C0 + `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both` (1,981 forum teams), P\* + 20M at D1's pin `440b61c3`,
s3501–s3505 (each K9(b) PASS). Control C0 s3501–s3505 (the production pin `f4b25238`, each K9(b) PASS). Rule:
`main.x35.d1` (`python -m main.x35.readout decide-d1`). Records: [`../reads/d1_look2.json`](../reads/d1_look2.json)
(final), [`../reads/d1_look1.json`](../reads/d1_look1.json).

## Look 2 (final, 5 seeds per arm)

| read | statistic | D1 | C0 | D1 − C0 | class |
|---|---|---|---|---|---|
| **H (primary)** | Kakuna ckpt 34, symmetric unfamiliar cells H / F / P72, equal-weight mean, 700 games per panel per arm | 0.492 [0.471, 0.513] | 0.471 [0.450, 0.492] | **+2.10 pp [−0.92, +5.11]** (95 %, Newcombe; AM-5 φ̂ = 0.77, df 8 → not widened) | **UNDECIDED (final)**: not GAIN (LB < 0), not NULL (UB > +3) |
| **P (co-primary)** | S on the home72 panel vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 54.51 / 56.09 / 56.51 / 54.98 / 54.84 | 56.78 / 57.39 / 56.86 / 57.38 / 56.58 | **−1.61 pp [−2.90, −0.32]** (Welch 97.5 %, df 5.4) | INFERIORITY NOT EXCLUDED (LB < −2.0; the whole interval < 0) |
| pool S (reported, look 1's 3 seeds only) | S on the eval team builder, regime `adfdbee824c9eb0a` | 54.61 / 55.89 / 55.82 | 57.52 / 56.60 / 56.81 | −1.54 pp [−3.41, +0.34] | — |

- **H per seed:** +4.3, 0.0, +4.0, 0.0, +2.1 pp (3 of 5 positive: the look-2 sign rule holds; the interval does not
  clear 0). The two new D1 seeds added +0.0 and +2.1, pulling Δ̂ from +2.78 to +2.10.
- **H per panel (reported, no rule):** forum72 +4.0 [−1.2, +9.2] (370 vs 342 / 700), heldout72_hpfix +2.1 [−3.1, +7.3]
  (331 vs 316), p72 +0.1 [−5.1, +5.4] (332 vs 331). The point gain sits on D1's own population (forum); on the two
  populations it does not train on it is +2.1 and ≈ 0.
- **Verdict UNDECIDED** (`reasons`: H is not a GAIN). Guards G-F / G-E / G-C **not read** (H non-qualifying, the A1
  ruling as at look 1). G-K and dose not read (reported only).
- **Goal (b)** (LB > 0.50 on H's statistic): **not met** by D1 (0.492, LB 0.471).

## Look 1 (3 seeds): CONTINUE

H +2.78 pp [−1.12, +6.67] (D1 0.506 vs C0 0.479; per seed +4.3 / 0.0 / +4.0); P −1.30 pp [−4.54, +1.93]. Look 2's
rows for s3501–s3503 are exactly look 1's, reused.

## How it was read (CPU only, `nice 19` on cores 5-7,13-15, heavy slots via `mem_cap.sh`)

- [`d1s_read.py`](d1s_read.py): `../g1/g1_read.py` with only the finals, names and output changed (pool S, look 1).
- [`d1p_read.py`](d1p_read.py): the same with `--panel home72` and ONE request per look (`d1p_look1_b100`,
  `d1p_look2_b100` for s3504–s3505), one family and ledger (P); results [`p_home72_results.json`](p_home72_results.json),
  [`s_pool_results.json`](s_pool_results.json).
- H: `python -m main.x35.kakuna plan --teamsets heldout72_hpfix,forum72,p72` + `run`, one state per batch of finals
  (look 1: `h_a` C0 ×3 + D1 s3501, `h_b` D1 s3502, `h_c` D1 s3503; look 2: `h_d` C0 s3504 + D1 s3504–s3505, `h_e`
  C0 s3505), merged by [`assemble.py`](assemble.py) (`--look 2`) into [`read_look2_input.json`](read_look2_input.json)
  (look 1: [`read_look1_input.json`](read_look1_input.json)). 210 / 210 units counted. The local Metamon peer only,
  95xx ports claimed per unit (`kakuna.claim_port`), the halt marker clear; look 2's Kakuna units ran in the heavy
  slots' SHORT lane (`mem_cap.sh --max-s 600`).
- Ledgers and unit dirs: `~/gen3ai_archive/x35_d1_read/`. Finals are sha-checked copies; `models/` was only read.

## Hazards (FINDINGS)

1. **The co-primary is S on the home72 PANEL, not G1's pool S** (look 1's brief named the pool S); read as registered.
   The pool S is reported at look 1's 3 seeds only (not played for s3504–s3505).
2. **Heavy-slot contention** (both looks): another agent's 45-min chunks and P's own 20-min boxes held the slots. Look
   2's short-lane declaration kept the 3–5-min Kakuna units moving (≈ 10 / h) but the read still took ≈ 7 h wall.
3. **Port race (look 1)**: two supervisors bind-and-released 9540; look 2 claims each port with a held flock
   (`kakuna.claim_port`, tested). No look-2 unit failed.
4. **A void first `h_d` attempt (look 2):** the copied finals lacked their `model_config.json` sidecar, so every unit
   refused at once (rc 1, no game played); its supervisor was stopped by PID, the state moved aside as
   `h_d.void_nosidecar_20261024`, and `h_d` re-planned with the sidecars. Nothing of it is read.
5. **The playing code is this read's worktree (main `5e389ddac` + the port / short-lane change), not an X35 pin**,
   as at look 1 (`6c428453a`+); no commit between the two touches `src/main/anchors`, `src/rust_sim`, `src/main/h2h`,
   the observation or the model, and both P requests carry the same regime id.
6. G-K and dose were not read (reported-only items).
