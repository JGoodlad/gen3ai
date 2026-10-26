# P\* vs Metamon's `PastaMittens` ensemble: the record (2026-10-25)

The rules are in [`PREREG.md`](PREREG.md), committed before any counted game. This file records how its
preconditions were met, and the result: **Δ-U −0.25 pp [−4.23, +3.73], NOT DETECTED. Kakuna stays the goal's
opponent.**

## Precondition 1: the two downloaded weights (owner's OK, 2026-10-25)

Plain Hugging Face hub downloads (`huggingface_hub.hf_hub_download`, repo `jakegrigsby/metamon`, revision
`ac8e88e7da8dc2a2b85176d88f81fd8854b1445b`, the snapshot our cache already held) into Metamon's own cache
(`/home/goodlad/dev/metamon/cache/pretrained_models`; not `models/`, not `data/`). The hub stores each file under its
sha256, and both blob names equal the sha256 we computed.

| file | bytes | sha256 |
|---|---:|---|
| `kakuna/ckpts/policy_weights/policy_epoch_30.pt` | 571,581,627 (as predicted) | `0d66a65ea74f1b02a6d3cf540d22a6ae14d77f9bd2d30e568062569415e9adc5` |
| `alakazam/ckpts/policy_weights/policy_epoch_8.pt` | 304,154,601 (inside the 0.2–0.35 GB estimate) | `47a98516df7b346118fa824f9137e6aaaa1428d5a7150af01bf606c1ea935944` |

## Preconditions 2 and 3: the agent entry and the router's regime instrument

`designs/ops/anchors.json` gains `PastaMittens` (preset `dt4_k34_midjudge_switchk28`, sha256 `1b0d180e…0a9b`), and
`metamon_side.py` gains the router instrument (SOP § "An ENSEMBLE agent"). **The first live smoke caught a false
alarm:** a state compare on Python's `random` moved on 1 of 23 router calls (and 1 of 46, 1 of 45 on a rerun). The
moving RNG was always Python's, never torch's or NumPy's. Poke-env's event-loop thread shares that state and draws
while the router computes. The check is now per thread: no Python draw from the router's own thread, and torch and
NumPy states byte-equal. On the 4-game smoke against `bot:heuristic` (`away`), it read **1.0000 on 95 / 95 calls**
and `regime_verified_decisions = true`. A router call takes 0.22–0.23 s at the median and 0.28 s at p90. Peak memory
is 3.9 GB.

## Precondition 4: the X51 identity unit

`ident_home72_00` (P\* vs Kakuna ckpt 34, home72, seed 0, X51's own config) was played at the read's commit
`d32ffcee4`. It replayed X51's `pstar_home72_00` **game for game: 20 / 20**, with the same winner and turn count.
So Δ-home reads against X51 directly, and the fallback control `kakctl_home72` was not played.

## The result (2026-10-26): Kakuna stays the goal's opponent

**Δ-U −0.25 pp [−4.23, +3.73], NOT DETECTED**: `PastaMittens` is not detectably stronger than Kakuna ckpt 34 for us.

| panel | n | W–L–T | P\* vs PastaMittens [Wilson 95 %] | vs 0.50 | P\* vs Kakuna (X51) | Δ (Newcombe) | class | paired: mean [normal 95 %], ens-only / Kakuna-only wins |
|---|---:|---|---|---|---|---|---|---|
| home72 | 400 | 220–180–0 | **0.550 [0.501, 0.598]** | BETTER | 0.598 | −4.75 pp [−11.53, +2.10] | NOT DETECTED | −0.048 [−0.110, +0.015]; 73 / 92 |
| H `heldout72_hpfix` | 400 | 195–205–0 | 0.488 [0.439, 0.536] | n.d. | 0.497 | −1.00 [−7.89, +5.90] | NOT DETECTED | −0.010 [−0.070, +0.050]; 72 / 76 |
| F `forum72` | 400 | 208–192–0 | 0.520 [0.471, 0.569] | n.d. | 0.530 | −1.00 [−7.88, +5.89] | NOT DETECTED | −0.010 [−0.074, +0.054]; 82 / 86 |
| P72 | 400 | 178–220–2 | 0.445 [0.397, 0.494] | WORSE | 0.433 | +1.25 [−5.60, +8.09] | NOT DETECTED | +0.013 [−0.045, +0.070]; 71 / 66 |
| **U** (H, F, P72 equal weight; Kish n_eff 1,205) | — | — | **0.484 [0.456, 0.512]** | n.d. | 0.487 [0.459, 0.515] | **−0.25 [−4.23, +3.73]** | **NOT DETECTED** | — |

- **Against the prediction:** the registered guesses were E-home 0.56, E-U 0.46 and Δ-U −2.5 pp. The reads are
  0.550, 0.484 and −0.25 pp. The ensemble is, if anything, closer to Kakuna than predicted. The panel order stayed
  F > H > P72.
- **The regime held on every decision:** a router no-RNG rate of 1.0000 over **80,893 router calls** in 160 peer
  halves (none moved an RNG), `sample == [False]` everywhere, and one regime stamp per cell (the PREREG's id
  `5c1ea828e951126b` fields, server `ws_frontend[in-process]@d32ffcee4`). A router call took a median 0.24 s (per
  half: 0.18–0.45 s; p90 0.33 s).
- **Counting:** all 80 units counted on their first try (`skipped_units` empty), the capture check included. One unit,
  `pm_p72_14`, ran past the short lane's 600 s and was re-run once as a long job, as registered. Its partial first
  attempt is not read.
- Per role (we challenge / they challenge): home72 0.585 / 0.515, H 0.485 / 0.490, F 0.485 / 0.555, P72 0.450 / 0.440.
  The 250-turn forfeit fired 2 / 1 / 2 / 0 times. There were two ties, both on P72. The median game ran 31 / 38 / 37 / 40 turns.
- PROVISIONAL: one checkpoint (P\*), one preset. The ensemble's whole-game cost and Kakuna's twin comparison come
  from the same P\* file and the same seeds.

## Cost

78 ensemble-unit runs. The total from start to finish, including waits for the shared CPU queue, was 13.5 h, at
nice 19 on the agent cores, in the short lane. Units took 300–1,050 s each, mostly queue wait. The read was paused
twice for the coordinator (each time between units, never mid-unit), then resumed.

## FINDINGS

1. **A plain RNG-state compare misreads an ensemble** (README § Preconditions 2 and 3). It flagged 1 router call in
   ~23, because poke-env's thread shares Python's `random` state. The instrument now checks Python draws per thread.
2. `rb_x26c` reached 480M with U 0.4875 against Kakuna (`x26c_continuation_2026-10-25/`), level with P\*. P\* remains
   the right "best checkpoint" for this read.
3. One short-lane overrun (`pm_p72_14`) was handled by the registered long-job re-run. The read was paused twice by
   the coordinator for CPU priority, with no effect on any counted unit.
4. Not read: whether the ensemble is stronger on Metamon's own `competitive` set, or on its own Showdown pin. The
   prereg did not register those.

## Files

`PREREG.md` (the rules, committed `62bb07c00` before any game), `cells.json` (every cell, `analysis` with the
per-panel and U blocks and the identity verdict, every unit's counting verdict), `rows.jsonl` (every counted row
+ `read_cell`, `unit`, `unit_seed`, `their_team`), `scripts/` (`mmq.py` the queue, `make_configs.py`, `aggregate.py`).
Not committed: the unit dirs, captures, logs and `units.jsonl` in `/home/goodlad/dev/gen3ai-reads/mm_strongest_2026-10-25/`.
