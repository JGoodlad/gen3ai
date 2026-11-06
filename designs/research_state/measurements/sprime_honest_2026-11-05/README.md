# S′-HONEST — design inputs (2026-11-05): no read, no PREREG

Five OUTCOME-BLIND measurements on S′-LIVE's state (`~/gen3ai_archive/sprime_live_play_2026-11-01/`), taken to design
the leak-honest replication S′-LIVE's GO-LEAKY licenses
([`design_sprime_honest.md`](../../../endstate/design_sprime_honest.md)). None reads a game result: every script
reads `search_decisions.jsonl`, the captures' protocol for our own side, the START teams, or replays kept logs. The
replication's build stopped at its unit's 2-day bound (that doc §6), so this folder holds no PREREG. A PREREG written
here later is a new registration, never a re-read of these files.

Run at `fcfe2575e` (main's release build), CPU only, `nice -n 19 taskset -c 5-7,13-15`, one thread. The 9
`.partial.<epoch>` cells a resume set aside are skipped everywhere. Each script's docstring is its command.

| file | what it measures | headline |
|---|---|---|
| [`decision_records.py`](decision_records.py) → [`decision_records.json`](decision_records.json) | the template's leak block on all 103,616 search-path decisions; the opponent move tokens absent from the true set on the 6,615 kept logs; the search's CPU split | a rewritten (never-seen-executed) opponent choice in 34.9 % of decisions, a verbatim one in 31.1 %: all `hiddenpower` (poke-env's untyped token) or `struggle`. Search CPU 258 CPU-h: worlds 138, exclusions 49, playouts 66 |
| [`exposure.py`](exposure.py) → [`exposure.json`](exposure.json) | which persisted dice were live at each decision (our side's own stream) | a non-Rest sleep, confusion or pending Future Sight / Doom Desire at 6.9 % of decisions (16.4 % of games), sleep almost all of it; Quick Claw, Encore, Disable, traps, locks, Uproar 0 |
| [`speed_pin.py`](speed_pin.py) → [`speed_pin.json`](speed_pin.json) | the info audit's counterfactual flips (`main.info_audit.checks.seed_reuse`) on 400 sampled decisions | a seen mon's Speed pinned a point tighter by the true seed in 23.5 % of decisions (5.4 % of pairs); the control passed 400 / 400 |
| [`segment_match.py`](segment_match.py) → [`segment_match.json`](segment_match.json) | per segment of 40 kept prefixes (950 segments), the share of 64 fresh seeds that reproduce our side's lines, true teams, damage uncontrolled | 0.51 (0.90 without a damage line, 0.42 with); 1.6 / ≈ 21 tries a segment; 0.15 ms a try; a whole prefix on one fresh seed ≈ 10⁻²⁸ |
| [`world_cost.py`](world_cost.py) → [`world_cost.json`](world_cost.json) | a timing run of today's `r0s4_rolls` builder on 16 kept decisions | 51–417 screening replays (median 140), 11.8 ms each, 1.5 CPU-s a decision |

**Limits (FINDINGs of the inputs).**
- `exposure.py` aligns a decision to the n-th non-wait request of our stream and approximates a lock or a trap from its
  start line. A counter whose onset our side never saw (a mon that slept before it was ever seen) is counted from
  its switch line's status.
- `segment_match.py` keeps the TRUE dice for every earlier segment, so a counter rolled earlier is the truth's there,
  and its cost to a fresh-dice matcher (D4's backtracking) is not in these rates. It measures the truth's team, not
  a drawn world's.
- `speed_pin.py` and `world_cost.py` sample 400 and 16 decisions with a fixed seed (`default_rng(20261105)`).
