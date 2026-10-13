# The END STATE's adoption as production (2026-10-12) — the evidence

Ledger: 2026-10-12 BUILT (MAJOR) "THE END STATE IS PRODUCTION". After the closing test's PASS
(`../endstate_closing_test_2026-10-09/cross/verdict.json`), `designs/production_config.json` took the `--arch endstate`
overlay. These files show that what is ADOPTED (HEAD's end state) is the TESTED build (`62085750`, tag
`closing-test-pend2`) modulo one math-equal change, and that `--arch production` now resolves to it.

| file | what |
|---|---|
| `shas2.py` → `shas_pend2.out`, `shas_head.out` | the registered `graph_sha.py` method (CPU, seed 0, 8 seeded rows, one thread) at `62085750` and at HEAD `f45d0c27` (pre-adoption tree), production and the end-state overlay: production unchanged (`421c6b98ce7937f4` / `749c56159ab028f4` / `51c02c6c6342a045`); end state `42ee361b11fa5a8d` → `a1e771af47523746`, state `df71da47741a8a0f` equal, outputs `5adfaa6c40a7a71e` → `fb2134d54039c341` (the closed-form exact KO, `5eef9934`) |
| `outdiff.py` → `outdiff_head.json`, `outdiff_pend2_control.json` | the SAME end-state weights (built at `62085750`, loaded strictly at HEAD) on 136 rows: max \|Δ\| ≤ 3.8e-6 absolute, ≤ 1.3e-6 relative on every output; greedy action 136 / 136; the control (`62085750` reloading its own weights) bit-equal |
| `resolved_prod.py`, `cmp.py` → `resolved_tested_pend2.json`, `resolved_production_head.json`, `resolved_tested_vs_production.out` | the closing test's end-state argv resolved by the parser + `resolve_config` at `62085750` (`--arch endstate`) vs the same argv with `--arch production` at the adopted tree: 204 keys, differences only the provenance tags (`arch`, `arch_source`, `recipe_source`) and three new runtime knobs at their off defaults |
| `checkargs_production.out` | `python -m main.checkargs --argv "--arch production --steps 15000000 --seed 2001 --device cuda"` on the adopted tree: ARCH ✓, RECIPE ✓ (28 knobs), launches ✓ |
| `dryrun_production.out` | `python -m main.launcher --dry-run --arch production --steps 15000000 --seed 2001 --device cuda --run-name rb_adoption_dryrun`: ✓ DRY RUN |
| `smoke.py` → `smoke_lines.out` | the CPU `--arch production --debug --steps 10000` smoke on the adopted tree (scratch models dir; the trainer module name assembled at runtime, standing rule 3): exit 0, "Training complete" |
| `golden_obs_core_check.out` | `python -m agents.training.golden_obs_core --check` on the adopted tree: 991 / 991 |

The per-file classification of the `62085750..HEAD` diff (114 non-test files) is the ledger entry's §1.
