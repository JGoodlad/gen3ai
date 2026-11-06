# S′-HONEST stage 1 (2026-11-05): the builder's timing, exclusion and budget check, and its proofs

Stage 1 of [`design_sprime_honest.md`](../../../endstate/design_sprime_honest.md) (§8): the leak-honest builder
`r0s4_honest` measured against `r0s4_rolls` OUTCOME-BLIND (no referee, no game result, no outcome field is read), its
budget chosen by the coordinator's rule, and the identity of `r0s4_rolls` at the new engine stamp. No PREREG; nothing
registered.

All runs: this build's checkout, CPU only, `nice -n 19 taskset -c 5-7,13-15`, one thread, under `mem_cap.sh`, a
training run live on the other cores (rule 5b). P\* = `rb_x26_s2601` @280,000,170 (sha256 `814823c5…`), greedy.

| file | what it is |
|---|---|
| [`timing.py`](timing.py) | the timing and exclusion check: dlive's SCORED U states (H, F, P72), 100 a panel in the order sha256("sprime_honest_stage1:" + key), the live searcher's own `_search` for each preset; CPU per decision (its `cpu_s`, plus the shared validator child's CPU where measured), the exclusion rate, the two presets' agreement; `--presets`, `--honest-budget`, `--keys` |
| [`budget_curve.py`](budget_curve.py) | the budget curve: which states hit a budget limit at the base budget, and each larger budget's per-panel exclusion and cost (the builder is deterministic, so a larger budget changes only those states; the spot check re-runs six others) |
| [`noise_floor.py`](noise_floor.py) | `r0s4_rolls` against itself under another key on the first 100 states, the floor the presets' agreement is read against |
| [`rescore.sh`](rescore.sh) | S′-LIVE's registered identity re-score (600 decisions) re-run from this checkout at the new engine stamp |

| [`final_summary.json`](final_summary.json) | per panel at the chosen budget: both presets' exclusion rate and CPU (`r0s4_rolls` from the first run, `r0s4_honest` from the post-fix run merged with B3's re-runs), the agreement |
| [`budget_curve.json`](budget_curve.json), [`agreement.json`](agreement.json), [`rescore.json`](rescore.json) | the curve per setting; the agreement and its noise floor; the re-score's verdict |
| `timing_rows.jsonl.gz`, `timing_v2_rows.jsonl.gz`, `curve_b2_rows.jsonl.gz`, `curve_b3_rows.jsonl.gz`, `noise_rows.jsonl.gz` | every row (the first run of both presets; `r0s4_honest` after the legality fix; B2's and B3's re-runs, B3's with its six spot-check states; the noise floor) |

**The runs, in order.** (1) `timing.py`, both presets, 300 states (the builder before the legality fix). (2) The same
with `--presets r0s4_honest@10` after it (the budget's BASE: 300 steps, 600,000 chain feeds, 3 world attempts). (3)
`budget_curve.py select` (44 states limited) and `timing.py --honest-budget` at B2 (750, 1,500,000, 5) and B3 (1,800,
4,000,000, 8) on them, B3 also on six unlimited states (identical records: the prefix argument holds). (4)
`noise_floor.py` on the first 100 states. (5) `rescore.sh` at the new stamp.

**The budget curve** (honest exclusions; projected CPU-min a searched game, two projections, §8.4 of the design):

| setting | F | H | P72 | U | F CPU-min | U CPU-min |
|---|---|---|---|---|---|---|
| base | 19 % | 6 % | 3 % | 9.3 % | 7.1–8.9 | 5.0–7.6 |
| B2 | 12 % | 6 % | 1 % | 6.3 % | 10.5–13.1 | 6.8–10.3 |
| **B3 (chosen)** | **5 %** | **4 %** | **0 %** | **3 %** | **13.3–16.7** | **8.6–13.0** |

**At B3, per panel** (`r0s4_rolls` / `r0s4_honest`):

| panel | exclusions | CPU-s a decision (honest median, p90) | CPU-min a game | same pick | rolls vs rolls, another key |
|---|---|---|---|---|---|
| H | 0 % / 4 % | 6.9 / 12.2 (4.1, 7.6) | 4.4 / 7.8–10.9 | 56 % of 96 | 62 % of 34 |
| F | 6 % / 5 % | 7.7 / 20.8 (5.7, 58.5) | 5.0 / 13.3–16.7 | 71 % of 91 | 69 % of 35 |
| P72 | 1 % / 0 % | 4.5 / 7.2 (4.4, 11.0) | 2.9 / 4.6–9.9 | 57 % of 99 | 92 % of 25 |
| U | 2.3 % / 3.0 % | 6.4 / 13.4 (4.4, 21.0) | 4.1 / 8.6–13.0 | 61 % of 286 | 72 % of 94 (honest vs rolls on them: 67 %) |

**The identity of `r0s4_rolls`** at stamp `port=74eb0a3bc35bd465/95;env=87f06ac3ab0b911b/130`: PASS, 600 / 600 of
S′-LIVE's registered sample re-scored (forum72 217, heldout72 204, p72 179), 0 mismatches, 4,637 CPU-s.

**Limits (FINDINGs).**
- `r0s4_rolls` was timed in run (1) and `r0s4_honest` in runs (2)–(3), under different co-tenants of the other slot;
  in run (1), where both ran in one process, their playouts cost 2.7 and 2.5 CPU-s on the states both searched.
- The Showdown validator child's CPU is in `r0s4_honest`'s figures from run (2) on (0.1–0.4 CPU-s a decision) and in no
  `r0s4_rolls` figure (S′-LIVE's records omit it too).
- The projection scales S′-LIVE's per-game decision count (38.4) and CPU (6.2 CPU-min); dlive's states (mean turn 19.2)
  are a sample of kept raw decisions, not S′-LIVE's every-decision-from-turn-10 mix.
- The noise floor's per-panel samples are small (25–35 states).
