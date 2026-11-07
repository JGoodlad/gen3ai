# X60-T / H-TEAMS look 1, read as registered: **CONTINUE** (2026-11-07)

X60-T = C0 + `--human-data teams:hr_spec_v1.r3@0.5` (2,295 human-built ladder teams, share 0.50 on both sides), P\*
+ 20M at X60T_PIN `7fe15726`, s3501–s3503. Control: C0 s3501–s3503 (production pin `f4b25238`), IMPORTED from X61P look 1's
`primary` group (§21.3). Rule `x60` ([`../PREREG.md`](../PREREG.md) §5, AM-X60T-1 §21). Record:
[`record_x60t_look1.json`](record_x60t_look1.json), the verbatim `main.reads record --out` output (sha256
`538700ee79b69778…`). The stage's copy is `../../x35_breakers_stage_2026-10-17/reads/x60t_look1.json`, re-recorded
from the same state by AM-X60T-2 (PREREG §23.1): every key but `created` is byte-identical to this file, plus the stage
identity `arm` / `look` / `delta_f`.
Run record (aborted s3501, resume #1, §13 / §15 per seed, the queue at the read's start): PREREG §22, committed before
`decide` ran.

## Result (`main.reads decide`, verbatim)

| read | statistic | X60-T | C0 | X60-T − C0 | class |
|---|---|---|---|---|---|
| **F (primary)** | Kakuna ckpt 34, greedy vs greedy, `forum72`, 60 units = 1,200 games per seed final | 0.5147 (1,853 / 3,600) | 0.5083 (1,830 / 3,600) | **+0.64 pp [−2.17, +3.45]** (95 %, §5's interval: V = binomial + 2 · max(1.0 pp, σ̂)² / n; σ̂² = −7.8e-5 → σ̂ 0; Welch not used, σ̂²'s LB95 −1.5e-4) | look-1 class NOT-RESOLVED → **CONTINUE** (Δ̂ > 0: "look 1: Δ̂ > 0, Δ_F goes to look 2") |

- **Per seed (X60-T / C0 F rate):** s3501 0.5017 / 0.5058 (−0.42 pp); s3502 0.5225 / 0.5208 (+0.17); s3503
  0.5200 / 0.4983 (+2.17). Arm SDs 1.14 / 1.15 pp.
- **Look 1 is futility only (§5).** Δ̂ > 0, so the read is not ended, and look 2 (8 v 8) decides.
- **Not read at look 1:** I_F, R72, H, P72 and home72 (§21.3 reconciliation). The decision shows them as
  `{"read": false, "reasons": ["…: absent"]}`. Guards: `guards_not_read` reads "Δ_F is not a GAIN". Equivalent
  (± 3 pp): false.
- **Against the registered guess (§16):** Δ_F +1.5 pp, 80 % [−1.5, +4.5]; the read is +0.64, inside it.

## Determinism guard (§4, §21.3): PASS

c0_s3501 forum72 unit seed 0, re-played at the play pin, against the unit imported from X61P look 1
(`~/gen3ai_archive/x61p_n1p_read/look1`, `c0_s3501__forum72__00`): tuple-equal, choices `equal`, counted.

## VOID (§13, §15): every run PASSES

| run | K9(b) at arm end (`k9b/…json`) | pin (`pin_history`) | dose vs `rb_x35_c0_s3501`, relative 1e-6 |
|---|---|---|---|
| x60t_s3501 | PASS (judged max 4.83e-4, p99 3.59e-5) | `7fe15726…` alone | PASS: dose_rate 2.0440550334496103e-08, lr_median 1.0047e-4, upd/step 2.0345e-4 |
| x60t_s3502 | PASS (2.18e-4, 3.14e-5) | `7fe15726…` alone | PASS, identical |
| x60t_s3503 | PASS (2.06e-4, 3.62e-5) | `7fe15726…` alone | PASS, identical |
| c0_s3501–s3503 | PASS | `f4b25238…` alone | PASS, identical |

The operator's checks that no module reads (PREREG §22.3), in every launch log of each seed, s3501's resumed one
included:
- the `[HUMAN-DATA]` stamp, with rows_sha `2e864572…` and hash `544c9bc00ea95030`;
- the `[TEAM-MIX] hr_spec_v1.r3@ALL (2295 teams …) share 0.50 on both sides` banner;
- the held-out panels line: 0 hits over 5 panels (registry `ea3917c2f1e6`), and 0 corpus teams pack identically to a
  pool team;
- no `corpus_<n>` line and no FATAL naming a corpus team;
- the T54 pool-seed check PASS.

s3501 was resumed once (§13, the T2 parity near-tie at 10:00:04, 875,048 steps re-trained). That is 1 of 3 ON seeds,
under §13's ≥ 3 bar for a without-resumed report.

## How it was read

- Spec [`../read_spec_look1.json`](../read_spec_look1.json), byte for byte (sha256 `0f5d9a5f…`); state
  **`~/gen3ai_archive/x60t_read/look1`**. [`cells.json`](cells.json), [`inputs.json`](inputs.json),
  [`decision.json`](decision.json), [`void.json`](void.json), [`models.json`](models.json), [`code.json`](code.json)
  and [`anchor_supervisor.jsonl`](anchor_supervisor.jsonl) are the state's.
- **Playing code `f564c333f`** (clean tree), checkout `~/dev/gen3ai-reads/x60t_play_de15dfc7`, with no repins. Lane
  `anchor`, concurrency 2, CPU only, nice 19 on cores 5-7,13-15. Wall 2026-11-06 09:38 → 2026-11-07 00:40.
- **181 Kakuna units played** (180 ON + the determinism unit = 3,620 games), all counted. C0's 180 units were imported.
- `assemble` / `decide` / `record` ran once each from main at `7880cd902` (regime `00c8c0114bdf0836`; 0 unplayed).

## Hazards (FINDINGS)

1. **RESOLVED by AM-X60T-2 (PREREG §23.1): the kit could not read this record.** Rule `x60` wrote no top-level `arm` /
   `look`, so `main.x35.batch.read_record` refused it and `kit stage-check --arm x60t --seed 3504` held look 2 (fail
   closed). `Rule.stage_keys` now writes the stage identity; the re-recorded `reads/x60t_look1.json` licenses X60Tb and
   the C0 top-up (stage-check ALLOWED, "x60t look 1 is CONTINUE").
2. **σ̂² came out negative (−7.8e-5),** so the 1.0 pp seed-SD floor sets the interval. Three finals a side cannot
   estimate the seed term (§5.1).
3. **C0 is a shared control (§11, m7)** with X61P and D3. A lucky-low C0 would lift all three.
