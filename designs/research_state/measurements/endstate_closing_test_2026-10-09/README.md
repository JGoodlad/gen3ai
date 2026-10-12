# The END-STATE closing test: the registered rule, coded before any seed (2026-10-09)

Registration: [`designs/endstate/design_endstate_closing_test.md`](../../../endstate/design_endstate_closing_test.md)
§3; ledger `2026-10-09 · REGISTRATION · THE END-STATE CLOSING TEST`. The rule was coded before any seed; the
cross's read is in § Results (2026-10-12).

| file | what |
|---|---|
| `closing_rule.py` | `decide(h)`: the owner's rule on the 8 × 8 cross (X5's `cross_stat`; the 90 % interval on 14 df; PASS iff upper end > 0 and lower end > −2.0 pp; rule 8 against PASS); `self-check` (the rule on the static screen's look-3 matrix: FAIL on both clauses, interval [−3.60, −1.04], matching that README to 1e-9); `simulate` (the design's operating characteristics and the priced, NOT registered, 4-seed harm look) |
| `simulate.json` | `closing_rule.py simulate`'s output (20,000 reps, seed 20261009) |

```bash
cd <checkout>/src
python ../designs/research_state/measurements/endstate_closing_test_2026-10-09/closing_rule.py self-check   # "OK"
python ../designs/research_state/measurements/endstate_closing_test_2026-10-09/closing_rule.py simulate > ../designs/research_state/measurements/endstate_closing_test_2026-10-09/simulate.json
```

Identity facts measured at P_prod `c0f528b4` (CPU, `static_recovery_2026-10-09/graph_sha.py '{}'`): production graph
`421c6b98ce7937f4`, state `749c56159ab028f4`, outputs `51c02c6c6342a045`, 1 graph, 1,937,942 params; K9 golden blobs
`9ef44772…` (json) / `9f13350d…` (buffer).

## Results (2026-10-12): the cross, read by the coded rule

The registered 8 × 8 cross (§3.1), played at P_end `62085750` (tag `closing-test-pend2`) from a detached checkout of
that commit: rows `rb_es_end2_s2001`–`s2008` (`--arch endstate`, trained at `62085750`), columns `rb_es_prod_s2001`–`s2008`
(`--arch production`, trained at `95d014fa`), each run's 15M `final_model.zip`; 64 cells × 1,000 mirrored pairs (batches
of 500), both sides greedy, schedule seed 0, `--oracle-reveal-mode off`, purpose `ab`, family `es_closing_strength_steps`,
request `es_closing_steps`, label `es_closing`, the archive's eval COUNT ledger (`models/_ledger`). Played 2026-10-12
08:27–09:19 PDT; 64 / 64 cells at 1,000 completed pairs, 0 aborted games. `closing_rule.py decide` on the matrix
(rows = end state, columns = production, pp) returned, verbatim (`cross/verdict.json`):

```json
{
 "outcome": "PASS",
 "clause_a_includes_zero_or_above": true,
 "clause_b_lower_above_minus_2": true,
 "interval90_pp": [
  -0.7231002436962337,
  1.9152877436962366
 ],
 "t_0.95": 1.761310135774891,
 "delta_hat_pp": 0.5960937500000014,
 "se_pp": 0.7489845013103577,
 "df": 14,
 "s2_rows": 2.6678201729910684,
 "s2_cols": 1.8200020926339333,
 "t_sup": 0.7958692722708254,
 "near_boundary": [],
 "stat": {
  "n": 8,
  "h_pp": [
   [
    49.675000000000004,
    51.74999999999999,
    51.0,
    49.675000000000004,
    48.5,
    54.50000000000001,
    50.0,
    50.349999999999994
   ],
   [
    52.449999999999996,
    52.525,
    51.37500000000001,
    52.325,
    49.875,
    54.974999999999994,
    52.625,
    52.75
   ],
   [
    49.875,
    48.275,
    48.75,
    47.875,
    45.65,
    51.775000000000006,
    47.925000000000004,
    48.35
   ],
   [
    52.300000000000004,
    52.55,
    50.625,
    50.949999999999996,
    49.025,
    52.5,
    49.8,
    53.525
   ],
   [
    51.775000000000006,
    51.349999999999994,
    51.7,
    49.6,
    48.625,
    52.300000000000004,
    49.925000000000004,
    51.1
   ],
   [
    51.225,
    51.925,
    52.5,
    50.525,
    49.0,
    53.75,
    52.675000000000004,
    51.37500000000001
   ],
   [
    47.375,
    47.75,
    47.25,
    47.85,
    46.375,
    49.35,
    46.425,
    49.05
   ],
   [
    51.675000000000004,
    52.2,
    52.275000000000006,
    47.5,
    49.6,
    54.75,
    52.175000000000004,
    53.1
   ]
  ],
  "delta_hat_pp": 0.5960937500000014,
  "row_means_pp": [
   50.68125,
   52.3625,
   48.559375,
   51.409375,
   50.796875,
   51.621875,
   47.678125,
   51.659375000000004
  ],
  "col_means_pp": [
   50.79375,
   51.040625,
   50.684375,
   49.5375,
   48.33125,
   52.987500000000004,
   50.19375,
   51.2
  ],
  "s2_rows": 2.6678201729910684,
  "s2_cols": 1.8200020926339333,
  "v_hat": 0.5609777832031252,
  "se_pp": 0.7489845013103577,
  "df": 14,
  "t": 5.468863164503183,
  "margin_pp": 3.5,
  "ci95_pp": [
   -1.0103182378672086,
   2.2025057378672113
  ],
  "upper95_one_sided_pp": 1.9152877436962366
 },
 "n_cells_read": 64
}
```

| file (`cross/`) | what |
|---|---|
| `plan_cross.py` → `cells_plan.json`, `shas.json` | the sixteen runs' preconditions (final present, one `pin_history` entry at the arm's pin, ≥ 15M steps, equal `init_num_threads`) and the 64 [player, opponent] cells; row / column sha256 |
| `drive_cross.sh` | the driver as run (pin check, family register, `main.h2h play-many`, the read) |
| `read_cross.py` | `main.h2h.cross.look_cells` + `matrix_from_cells`, then `closing_rule.decide` → `h.json`, `cells.json`, `verdict.json` |
| `h.json`, `cells.json`, `verdict.json` | the matrix, the per-cell W/L/D / pair counts, the rule's output |
| `run.log`, `drive.out` | the play's log (family registration, every cell and batch line) and the driver's status lines |
| `panel_bots.py` → `panel_bots.json` | §3.8's secondary panel (REPORTED, never gated), added 2026-10-12 with the ledger entry: each run's last in-loop eval (8 training bots, random excluded) + sentinel monotonicity; end state 90.83 vs production 89.62 pp (+1.21), no HARM note (threshold −4 pp). `python panel_bots.py` from this folder |
