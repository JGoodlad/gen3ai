# D1 (X38) look 1, read as registered: **CONTINUE** (2026-10-23)

D1 = C0 + `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both` (1,981 forum teams), P\* + 20M at D1's pin `440b61c3`,
s3501–s3503 (each K9(b) PASS). Control C0 s3501–s3503 (the production pin). Rule: `main.x35.d1`
(`python -m main.x35.readout decide-d1`). Record: [`../reads/d1_look1.json`](../reads/d1_look1.json).

| read | statistic | D1 | C0 | D1 − C0 | class |
|---|---|---|---|---|---|
| **H (primary)** | Kakuna ckpt 34, symmetric unfamiliar cells H / F / P72, equal-weight mean, 420 games per panel per arm | 0.506 [0.479, 0.534] | 0.479 [0.451, 0.506] | **+2.78 pp [−1.12, +6.67]** (95 %, Newcombe; AM-5 φ̂ = 0.33 → not widened) | **UNDECIDED → CONTINUE** (Δ̂ > 0) |
| **P (co-primary)** | S on the home72 panel vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 54.51 / 56.09 / 56.51 | 56.78 / 57.39 / 56.86 | **−1.30 pp [−4.54, +1.93]** (Welch 97.5 %) | INFERIORITY NOT EXCLUDED |
| pool S (reported, G1's regime) | S on the eval team builder, regime `adfdbee824c9eb0a` | 54.61 / 55.89 / 55.82 | 57.52 / 56.60 / 56.81 (G1) | −1.54 pp [−3.41, +0.34] | — |

- **H per seed:** +4.3, 0.0, +4.0 pp (2 of 3 positive: the sign rule holds; the interval does not clear 0).
- **H per panel (reported):** forum72 +4.0 [−2.7, +10.7], heldout72_hpfix +2.4 [−4.4, +9.1], p72 +1.9 [−4.8, +8.6].
  The point gain is largest on D1's own population but present on all three.
- **Guards G-F / G-E / G-C: not read** (H is non-qualifying; the orchestrator's A1 ruling). G-K (home72 / away,
  reported with no rule) and the dose descriptor were not read.
- **Goal (b)** (LB > 0.50 on this statistic): not met by D1 (LB 0.479).
- **Consequence:** look 2 = D1 and C0 at s3504–s3505 (5 seeds each side, final).

## How it was read (CPU only, `nice 19` on cores 5-7,13-15, heavy slots via `mem_cap.sh`)

- [`d1s_read.py`](d1s_read.py): `../g1/g1_read.py` with only the finals, names and output changed (pool S, reported).
- [`d1p_read.py`](d1p_read.py): the same with `--panel home72` and ONE request for the look (P); results
  [`p_home72_results.json`](p_home72_results.json), [`s_pool_results.json`](s_pool_results.json).
- H: `python -m main.x35.kakuna plan --teamsets heldout72_hpfix,forum72,p72` + `run`, one state per batch of finals
  (`h_a` C0 ×3 + D1 s3501; `h_b` D1 s3502; `h_c` D1 s3503), merged by [`assemble.py`](assemble.py) into
  [`read_look1_input.json`](read_look1_input.json). 126 / 126 units counted. The local Metamon peer only, own 95xx
  ports, the halt marker clear.
- Ledgers and unit dirs: `~/gen3ai_archive/x35_d1_read/`. Finals are sha-checked copies; `models/` was only read.

## Hazards (FINDINGS)

1. **The brief named "R1 on S vs C0" as D1's read; the registration's primary is H (Kakuna) and its co-primary P is
   S on the home72 PANEL, not G1's pool S.** Read as registered; the pool S is reported beside P.
2. **Heavy-slot starvation:** the 40-min S units held both slots and the 3-min Kakuna units got one turn per release.
   P was paused once (SIGTERM to its own process group mid-unit, losing at most one unfinished 100-pair batch) and
   resumed with a 20-min box. No counted row was lost.
3. **Two Kakuna attempts failed and were retried** (one a port collision between two supervisors on 9540,
   `choose_port`'s bind-and-release race); both counted on retry.
4. **The playing code is this read's worktree commit (main `6c428453a`+), not an X35 pin**, the same arrangement as G1.
5. G-K and dose were not read (reported-only items).
