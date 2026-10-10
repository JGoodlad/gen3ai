# The END-STATE closing test: the registered rule, coded before any seed (2026-10-09)

Registration: [`designs/endstate/design_endstate_closing_test.md`](../../../endstate/design_endstate_closing_test.md)
§3; ledger `2026-10-09 · REGISTRATION · THE END-STATE CLOSING TEST`. Nothing here has read a closing-test cell.

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
