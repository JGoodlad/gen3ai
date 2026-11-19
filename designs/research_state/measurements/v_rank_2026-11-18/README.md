# V-RANK — REGISTERED 2026-11-18; LOOK 1 UNDER WAY (stage 0 played, blind)

**Question.** Does the honest one-ply search's value rank the moves of ONE position worse when P\* pilots an
unfamiliar team than when it pilots a home72 team, judged by REAL games against Kakuna? It decides whether MF-4's
leaf-value step is funded ([`PREREG.md`](PREREG.md) §1).

| file | what |
|---|---|
| [`PREREG.md`](PREREG.md) | the registration: the estimand, the classes, the two looks, the VOIDs, the power, the cost, the launch block (§8) |
| [`read_spec.json`](read_spec.json) | look 1's `main.reads` spec (rule `v_rank`; groups `u1`, `home1`; pin `c933afd5a`; sha256 `d7342bc9…dc4f`, the `spec_sha256` look 2's imports name) |
| [`stage0_params.json`](stage0_params.json) | the spec's decision params plus the engine and the preset's name, for `v_rank stage0 --params` (the outcome-free teeth) |

**State of the read.** Look 1's 30 raw units are played and counted; stage 0 (the first 5 blocks) is scored, branched and sized
([`record/stage0.json`](record/stage0.json), outcome-free: 200 final blocks, the cap does not bind). The full look-1 score and branches are running; no
outcome has been read. A `record/` file appears with look 1's decision.

**No result yet.** Any number from the build smoke is a pipeline check, never evidence.
