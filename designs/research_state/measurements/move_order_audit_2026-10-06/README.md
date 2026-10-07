# Move-order audit, 2026-10-06 (`gen3_move_legality_by_id_v1`)

The question: does the production model ever act on, score, or gate the WRONG move because our active's
moves exist in two orders (per-mon slots SORTED BY `Move.id` string; the request block and actions 6–9 in
REQUEST order)? CPU only; inputs are the Lane S turn bank (`../m5_laneS/bank_v1`, 580 real eval battles
from trained policies, inputs only) re-played through the Rust core (`core_events --trackers --obs`).

| file | what it does |
|---|---|
| `real_audit.py` | Every answered decision of both viewers: (A) obs request block == the `|request|` JSON, by num and legality; (B) action 6+k's choice token names request move k; (C) the viewer's next `|move|` line == the chosen move; (D) per-mon slot order vs request order, and whether a POSITIONAL application of request legality to the sorted slots differs from the identity one. Run from the worktree root with `PYTHONPATH=src`, optional arg = number of battles. |
| `real_audit_full.out` | Its output on all 580 battles. |
| `mimic_probe.py` | Two constructed battles (Mimic copying Body Slam; Ditto's Transform): the request block still resolves every choosable move to exactly one sorted slot. |
| `gain_probe.py` | "Source 2" of the original report: permuting the request block moves the op's per-move pointer cells only through the per-slot learned `out_gain` (pre-gain cells: max deviation 0.0). `ALL_LEGAL=1` sets every legality bit. |
| `ckpt_gain.py` | Reads a checkpoint's per-request-slot `out_gain` (trained X5 arms: the slot-0 gains exceed slot 3's, e.g. pko 1.365 vs 1.146). |

## Results (all 580 battles, 42,465 decisions)

- (A) 37,358 move-bearing rows: 0 id mismatches, 0 legality mismatches against the request.
- (B) 142,598 legal move actions: 0 tokens naming a different move than request slot k.
- (C) 27,361 played moves (4,587 turns with no `|move|` line, 117 caller moves skipped): 0 executed ≠ chosen.
- Request ids with no unique sorted slot: 0 (Mimic and Transform resolve too, `mimic_probe.py`).
- (D) per-mon order == request order on 3,717 of 37,358 rows. Rows carrying any illegal move: 3,595
  (1,659 one, 633 two, 1,303 three). On 2,542 of them (6.8 % of all move-bearing rows, 71 % of the
  illegal-move rows) the positional write put a choosable move's legality on a different move; on every
  one of those a choosable move read "illegal".

⚠️ The bank is LOSS-ENRICHED by design and holds both viewers (scripted bots' sides too), so 6.8 % is a
rate in these states, not in a training rollout.
