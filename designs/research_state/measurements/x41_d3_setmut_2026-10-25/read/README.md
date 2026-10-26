# X41 / D3 look 1, read as registered: **CONTINUE** (2026-10-26)

D3 = C0 + `--team-mix d3_setmut_v1@ALL:0.3 --team-mix-sides both` (2,876 pool teams, each with 1–2 mons' sets
redrawn from Smogon usage), P\* + 20M at D3_PIN `9e289ba4`, s3501–s3503. Control: C0 s3501–s3503 (production pin
`f4b25238`), its rows BANKED from D1's read. Rule: D1's, verbatim (`main.reads` rule `d1`, arm `X41`;
[`../PREREG.md`](../PREREG.md) §3–§6, registered at `9938d7b6`). Record:
[`../../x35_breakers_stage_2026-10-17/reads/d3_look1.json`](../../x35_breakers_stage_2026-10-17/reads/d3_look1.json)
(it loads in `main.x35.batch.read_record`; `kit stage-check --arm d3 --seed 3504` now says ALLOWED).

## Result

| read | statistic | D3 | C0 | D3 − C0 | class |
|---|---|---|---|---|---|
| **H (primary)** | Kakuna ckpt 34, greedy vs greedy, symmetric `heldout72_hpfix` / `forum72` / `p72`, equal weight, 420 games per panel per arm | 0.491 [0.464, 0.519] | 0.479 [0.451, 0.506] | **+1.27 pp [−2.63, +5.16]** (95 %, Newcombe; AM-5 φ̂ = 0.60, df 4 → not widened) | **UNDECIDED → CONTINUE** (Δ̂ > 0 at look 1) |
| **P (co-primary)** | S on `home72` vs the frozen R (7 × 500 mirrored pairs, regime `d4de0fd7c16a2658`) | 57.01 / 56.78 / 56.52 | 56.78 / 57.39 / 56.86 | **−0.24 pp [−1.10, +0.62]** (Welch 97.5 %, df 3.7) | **NON-INFERIOR** (LB > −2.0) |
| H without `heldout72_hpfix` (reported; the goal-(b) panel-validity ruling, 4f555e57) | F + P72, same statistic | 0.487 [0.453, 0.521] | 0.482 | +0.48 pp [−4.29, +5.24] | (UNDECIDED; no rule reads it) |

- **H per seed:** +2.4, −2.9, +4.3 pp (2 of 3 positive: the look-1 sign rule holds; the interval does not clear 0).
- **H per panel (reported, no rule):** heldout72_hpfix +2.9 [−3.9, +9.6] (210 vs 198 / 420), p72 +1.0 [−5.8, +7.7]
  (201 vs 197), forum72 +0.0 [−6.7, +6.7] (208 vs 208). The registration predicted the gain spread over all three
  panels (|F − H| < 3 pp); |F − H| reads 2.9 pp, with no panel detected.
- **Against the registered guess:** H +1.5 pp, 80 % [−1.5, +4.5] → read +1.27; P −0.8 [−2.5, +0.9] → read −0.24.
- **Guards** G-F / G-E / G-C: **not read** (H is not a GAIN; the A1 ruling (a)). G-K: not read (reported only).
- **Goal (b)** (LB > 0.50 on H's statistic): **not met** by D3 (0.491, LB 0.464).
- **Beside any GAIN at look 2:** D1's two-look rule has a false-GAIN rate of 3.5–7.4 % from the seed SD alone
  (5b84bca6 FINDING); a look-2 GAIN is read with that rate beside it.
- **Dose** (reported; `../../x35_breakers_stage_2026-10-17/dose/`): D3 4.413 / 4.069 / 4.059 GPU-h (12.54; the
  registered estimate was ≈ 12.1), C0 4.430 / 4.027 / 4.012 (12.47); mean gap 0.6 % (< the 10 % confound bar).

## VOID guards (PREREG §5): every seed PASSES

| seed | K9(b) at arm end | `[TEAM-MIX]` banner, held-out hits, pool copies | pin (`pin_history`) / corpus-check | pool-seed check | FATAL on a `d3m_*` team |
|---|---|---|---|---|---|
| 3501 | PASS (`k9b/rb_x35_d3mut_s3501.json`) | exact line; 0 hits over 5 panels; `0 corpus team(s) pack identically to a pool team` | `9e289ba4…` / PASS | PASS | none |
| 3502 | PASS | as s3501 | as s3501 | PASS | none |
| 3503 | PASS (shipped `3b381a32f`) | as s3501 | as s3501 | PASS | none |

The banners and checks are the launcher journal's (`~/.claude/jobs/x35/rb_x35_d3mut_s350*.launcher.log`,
`*.go.pin.txt`, `*.go.corpus.txt`, `*.pool_seed_check.txt`); the FATAL search covered those and each
`launcher_child.full.log`.

## C0's banked rows (PREREG §3 / §4)

- **H:** C0 s3501–s3503 come from D1's committed `d1_read/read_look2_input.json` (sha256 `8c4884aa…`), pooled only
  because the DETERMINISM GUARD PASSED. C0 s3501's 7 `heldout72_hpfix` units were re-played at this read's harness,
  and all 7 reproduce the banked units GAME FOR GAME (half, index, team, result, turns), first in a pre-run before any
  D3 unit (ordering) and again as the read's own `guard` group (the record's). The banked per-game rows are committed
  as [`c0_s3501_heldout72_banked_games.jsonl`](c0_s3501_heldout72_banked_games.jsonl), extracted from
  `~/gen3ai_archive/x35_d1_read/h_a/` (each row carries its source `games.jsonl`'s sha256).
- **P:** C0's S rows come from the same file, pooled only because this read's home72 regime equals the banked
  `d4de0fd7c16a2658`. The Kakuna group's id equals D1's `09f297883701a35a`. The golden test's archive half
  re-assembles D1's ledger and units to exactly those ids.

## How it was read

- Spec [`read_spec_look1.json`](read_spec_look1.json) (sha256 `4c51064f…`): `python -m main.reads plan` / `run --lane
  s` / `run --lane anchor` / `record`; state `~/gen3ai_archive/x41_d3_read/look1/` (copies sha-checked; `models/`
  only read). Finals by the settled rule with the X35 journal's `X35END … complete:` line as the done marker.
  [`cells.json`](cells.json), [`inputs.json`](inputs.json), [`decision.json`](decision.json) are the state's.
- Playing code: `2cc864f2f` = `aac96a565` + this unit's `d1`-rule commit (the same `rules.py` change that ships
  here), clean tree, `HF_HUB_OFFLINE=1`; Kakuna units on the heavy slots' short lane, S in ≤ 40-min boxes in one
  heavy slot; 95xx ports with the held flock. CPU only.
- 63 / 63 D3 units counted, 7 / 7 guard units, 3 × 3,500 S pairs; no unit failed.

## Hazards (FINDINGS)

1. **`main.reads` could not import a banked control for `d1`**, which D3's PREREG requires. This unit adds
   `import_off` + `determinism_guard` + `reported_without` to the `d1` rule (tests in `rules_test.py`). The
   cross-state `import` landed upstream meanwhile (`1bcedc7b3`), but it imports from another `main.reads` state;
   D1's read predates the module.
2. **main@`a260e9c3f` broke every Kakuna panel unit** (the panel cells' committed anchors configs carry no weight
   pins: `AnchorConfigError … hf_repo is not set`, rc 3, 0 games). The first state and guard pre-run are VOID and
   were moved aside (`*.void_a260_panel_cfg`), never read. The orchestrator ruled to play the whole read at the
   pre-a260 harness. The fix is with its author.
3. **The playing commit `2cc864f2f` is local**, not on origin. It is `aac96a565` + the `rules.py` /
   `rules_test.py` / spec diff that ships in this unit, so it is reproducible from main's history.
4. **Two supervisors on one state** (the guard pre-run): a state renamed aside took its in-state lock, and a
   `kill -0` watcher reported a live supervisor dead. The old supervisor played one pre-run unit into the
   re-planned state, at the new commit but without `HF_HUB_OFFLINE`. It was stopped by PGID. The pre-run only
   enforces ordering; the record's guard is the state's own group. Closed upstream by `1bcedc7b3` (path-keyed lock,
   `/proc` liveness).
5. **Units ran at nice 19 on cores 5-7,13-15**, the harness's fixed setting, not the brief's "normal priority". A
   contended box held Kakuna units at 9–31 min each (≈ 11.7 h for the lane).
6. **The K9(b) VOID check is outside the record.** The spec predates `1bcedc7b3`'s `void` parameter, so K9(b) was
   checked by hand (table above). Look 2's spec should register `void`.
