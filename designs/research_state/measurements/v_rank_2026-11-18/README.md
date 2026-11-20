# V-RANK — REGISTERED 2026-11-18; LOOK 1 PLAYED 2026-11-19: CONTINUE

**Question.** Does the honest one-ply search's value rank the moves of ONE position worse when P\* pilots an
unfamiliar team than when it pilots a home72 team, judged by REAL games against Kakuna? It decides whether MF-4's
leaf-value step is funded ([`PREREG.md`](PREREG.md) §1).

| file | what |
|---|---|
| [`PREREG.md`](PREREG.md) | the registration: the estimand, the classes, the two looks, the VOIDs, the power, the cost, the launch block (§8) |
| [`read_spec.json`](read_spec.json) | look 1's `main.reads` spec (rule `v_rank`; groups `u1`, `home1`; pin `c933afd5a`; sha256 `d7342bc9…dc4f`, the `spec_sha256` look 2's imports name) |
| [`stage0_params.json`](stage0_params.json) | the spec's decision params plus the engine and the preset's name, for `v_rank stage0 --params` (the outcome-free teeth) |

**State of the read.** Look 1 (100 blocks) is played, decided and recorded: **CONTINUE** (RANK-NULL-ON-U not reached): β_U +0.390 [+0.181, +0.575],
β_home +0.354 [+0.180, +0.516], Δβ +0.036 [−0.232, +0.277] ([`record/`](record/README.md); the ledger entry of 2026-11-19). Stage 0 sized the final at
200 blocks (the cap does not bind). Look 2 (the second spec, PREREG §7) is not yet written; MF-4 is unchanged until it reads.

**Result so far.** Look 1 only (above). Any number from the build smoke is a pipeline check, never evidence.
