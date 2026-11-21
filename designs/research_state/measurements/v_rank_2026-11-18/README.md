# V-RANK — READ 2026-11-20: RANK-HOLDS-ON-U (registered 2026-11-18; look 1 CONTINUE, look 2 final)

**Question.** Does the honest one-ply search's value rank the moves of ONE position worse when P\* pilots an
unfamiliar team than when it pilots a home72 team, judged by REAL games against Kakuna? It decides whether MF-4's
leaf-value step is funded ([`PREREG.md`](PREREG.md) §1).

| file | what |
|---|---|
| [`PREREG.md`](PREREG.md) | the registration: the estimand, the classes, the two looks, the VOIDs, the power, the cost, the launch block (§8) |
| [`read_spec.json`](read_spec.json) | look 1's `main.reads` spec (rule `v_rank`; groups `u1`, `home1`; pin `c933afd5a`; sha256 `d7342bc9…dc4f`, the `spec_sha256` look 2's imports name) |
| [`read_spec_look2.json`](read_spec_look2.json) | look 2's spec, exactly as PREREG §7 registers it (committed at `ec17bd0d8`, before any look-2 game): `u1` / `home1` imported from look 1's state by `spec_sha256`, fresh `u2` (unit seeds 2005–2009) and `home2` (2115–2129), `look: 2`, each artefact list extended by `selected2.jsonl` / `scored2` / `branches2`; sha256 `0c540aa9…af18` |
| [`stage0_params.json`](stage0_params.json) | the spec's decision params plus the engine and the preset's name, for `v_rank stage0 --params` (the outcome-free teeth) |

**State of the read.** Played to its decision. Look 1 (100 blocks, 2026-11-19) read CONTINUE, an interim that computed no class but NULL.
**Look 2, the final (200 blocks, 2026-11-20), reads RANK-HOLDS-ON-U:** β_U +0.315 [+0.153, +0.449], β_home +0.333 [+0.210, +0.447], Δβ −0.018 [−0.215, +0.167]
on 1,124 U / 1,114 home positions, team-clustered ([`record/`](record/README.md); the ledger entry of 2026-11-20). LB(Δβ) clears the −0.25
bar by 0.035 and LB(β_U) is above 0; COMPLETE, no VOID, no boundary flag. The PREREG carries one dated ERRATA (2026-11-19: §8 step 5's
`record --out` takes a file, not a directory; no rule change).

**What it licenses** (PREREG §1). MF-4's leaf-value step is NOT FUNDED on this evidence. HOLDS excludes a deficit of 0.25 or more on
unfamiliar teams; it does not show the slopes equal, and it says nothing about whether search pays (the contender slope is near zero on both
sets). Any number from the build smoke is a pipeline check, never evidence.
