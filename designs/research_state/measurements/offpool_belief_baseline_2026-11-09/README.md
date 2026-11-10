# The off-pool belief baseline (2026-11-09): REPORTED, no verdict

The first read of `python -m main.offpool_belief` ([`../../../tools/offpool_belief.md`](../../../tools/offpool_belief.md)):
each leg of the opponent belief against its own Smogon prior, on the in-pool reference panel and the three held-out
panels, for P\*, the pool-only control C0 (5 finals), N1-P (5 finals) and X60-T (6 finals). The ledger entry of
2026-11-09 (*OFF-POOL BELIEF BASELINE*) is the record; the design it feeds is
[`../../../endstate/design_offpool_belief.md`](../../../endstate/design_offpool_belief.md).

| file | what |
|---|---|
| `table.md` | the `table` step's output: per panel and metric, belief, prior, belief − prior, the 0.25 floor − prior, arm − C0, each with its team-clustered interval |
| `table.json` | the same numbers, with every checkpoint's score file digest, the bank digests and the `obs_gap` rows |

The banks (144 games a panel, P\* greedy on both seats, seed 20261109) and the 17 score files are outside the
checkout, in `~/gen3ai_archive/offpool_belief_2026-11-09/` (`bank/`, `scores/`): a bank states a held-out panel's
teams in the clear and is a scoring input only. This folder is a frozen record, not a tool: re-run the read with
the tool's three steps.
