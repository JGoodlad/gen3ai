# S-U gate test — READ: **GO** (2026-10-28)

Registered in [`PREREG.md`](PREREG.md) (`ef05d165c`; pin `835c6b4b2`; spec [`read_spec.json`](read_spec.json), rule
`deviation_sign`). The question comes from the S-U review's "gate test"
([`../su_headroom_2026-10-27/REVIEW.md`](../su_headroom_2026-10-27/REVIEW.md), `7aafa527`). It asks: on unfamiliar teams,
when the registered search S (`r0s4@10`) overrules P\*'s top move, is it right more often than wrong under S's own model
of the opponent (the self-referee: 24 truth playouts, P\* on both sides, common dice)?

**Verdict:** `GO: pooled-U sign accuracy 0.635 [0.578, 0.687] (n 346; GO iff LB > 0.5)`. The record is COMPLETE, the
re-derived guards found no hits over 40 units, and the engine is `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`
(the registered one). [`record_summary.json`](record_summary.json) is the record without its inputs. The state is
`~/gen3ai_archive/su_gate_2026-10-27/`: the cells, the kept input logs, `gate/selected.jsonl` (sha256 `cd28bd5d…`),
`gate/scored.jsonl` (`975f4a2d…`) and `record.json`.

**The GIGO tooth** ([`bank_check.json`](bank_check.json), run at the pin after the PREREG): the gate's scorer
re-scored the review's bank and reproduced every printed digit: sign accuracy 0.5132 [0.4468, 0.5801] / 0.5798
[0.5184, 0.6447]; IMP +0.0001 / +0.0193 with their intervals; deviation 0.4019 / 0.4145.

## Result

| | home72 | H (`heldout72_hpfix`) | F (`forum72`) | P72 | pooled U |
|---|---|---|---|---|---|
| states (games) | 400 (200) | 360 (180) | 400 (200) | 400 (200) | 1,160 |
| excluded (no consistent world) | 39 (9.8 %) | 5 (1.4 %) | 43 (10.7 %) | 6 (1.5 %) | |
| deviation rate (of searched) | 0.418 [0.362, 0.473] | 0.414 [0.357, 0.468] | 0.381 [0.332, 0.437] | 0.419 [0.369, 0.469] | |
| countable deviations | 134 | 125 | 97 | 124 | 346 |
| **sign accuracy** | **0.537 [0.458, 0.617]** | 0.640 [0.550, 0.727] | 0.660 [0.563, 0.755] | 0.605 [0.517, 0.694] | **0.635 [0.578, 0.687]** |
| IMP (T(pick) − T(top), per searched state) | +0.009 [−0.003, +0.022] | +0.020 [+0.006, +0.035] | +0.013 [+0.002, +0.024] | +0.010 [−0.002, +0.024] | |
| search CPU-s a state (median / mean) | 5.1 / 6.5 | 4.1 / 4.7 | 5.7 / 7.3 | 4.2 / 4.9 | |
| referee CPU-s a deviation (mean) | 12.5 | 10.4 | 12.1 | 12.3 | |
| Speed ±1 flips: states with a pinned mon | 0.148 | 0.217 | 0.188 | 0.220 | |
| leak: states with rewritten past moves | 144 | 124 | 161 | 155 | |

REPORTED only: **U − home72 +0.098 [−0.002, +0.194]**, sign +. Home72's 0.537 is consistent with the review's banked
0.513 at P\*.

**Cost.** Raw play was 41 unit attempts, 2.1 CPU-h. Scoring was ≈ 4.7 CPU-h single-threaded (median 7.8 s a state),
plus the superseded first chunks (FINDING G-4). That is ≈ 8–10 CPU-h in all, under the 10–15 estimated.

## Reading

- **GO licenses the next stage only, never a gain.** The referee shares S's model of the opponent, so it is S's best
  case. E1-0's direction is that gains shrink against a real bot. Routing (precommit memo §2.7) is V-RANK's live
  branches next, then the amended S-U if V-RANK does not turn it OFF.
- **Unlike its pool-team self-play states (B2), S's deviations on U beat chance under its own referee**, on every U
  panel separately (each LB ≥ 0.517). The deviation RATE is the same on U as on home72 (≈ 0.40), so the review's
  "noise dose" worry, more deviations on U, is not seen.
- **The U − home72 difference is +0.10 with an interval touching 0.** It is the mechanism H-MID would predict, but it
  is sign-only and leak-confounded (review M1). Leak exposure is HIGHER on U: Speed ±1 pins 0.19–0.22 against 0.15.
  Exclusions are LOWER on H and P72 (1.4–1.5 % against home72's 9.8 %), the opposite of the PREREG's guess. Nothing
  here attributes the U − home72 gap to V.
- **Predictions** (PREREG §8): pooled U ≈ 0.52 (0.45–0.60), P(GO) ≈ 0.25. Both MISSED high. Home72 ≈ 0.51 held.

## FINDINGs

- **G-1.** The gate's engine (main's at `835c6b4b2`) is not S-U's registered engine (`bdc5d2be…`, before
  `995c04a8a`). An S-U amendment that wants this gate as "the read's engine" must register this one.
- **G-2.** The optional second keying of S was not run.
- **G-3.** One unit, `pstar__heldout72_hpfix__106`, was EXHAUSTED: Kakuna's deterministic turn-250 long-tail fault
  (SOP H5) hit at 9 / 10 games of a half, and again on the retry. H holds 9 seeds (360 states). The rule draws from
  counted cells only, so this does not VOID the read.
- **G-4.** The first scoring chunks ran ≈ 150 s of wall a state: PyTorch's default thread pool oversubscribed the 6
  shared agent CPUs (playout CPU 4–5× its wall). Scoring was restarted with `OMP_NUM_THREADS=1`, at ≈ 8 s a state.
  The 15 states scored first were kept aside (`gate/scored_default_threads_superseded.jsonl`). Every one re-scored
  bit-identically (Q̂, the pick, the referee outcomes), and the read uses the single-threaded file throughout.
  Any offline S run on the agent cores should set one thread.
- **G-5 (a defect of the build, fixed).** `deviation_sign` returned its verdict at the top level. `main.reads record`
  reads `decision.verdict` for a non-X35 rule, so the pipeline's record refused ("the decision carries no verdict")
  after `decide` had written the same GO. The fix wraps the verdict and adds a record-path test. The record was
  written from the fixed checkout; the decision's numbers are unchanged.
