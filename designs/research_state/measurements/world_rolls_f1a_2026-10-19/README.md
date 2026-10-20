# F1 tier A: the honest world builder on dice-marginalised damage rolls (2026-10-19): measure, build, re-measure

F1 of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §3.3.1 / §7: the honest builder replays
each world's prefix on the TRUE battle seed, so a world must reproduce every printed damage number with the truth's own
damage roll. Tier A (built in this unit: `pokesim::rolls`, `WorldSampler(dice="rolls")`) lets each damage roll's VALUE
be any roll that prints what the searched side saw, the dice stream kept aligned, and weights each world by Π k / 16.
World BUILDS only: no playout value, no model, no outcome, no study arm re-run. The numbers and their reading are in the
design doc (§3.3.1, §3.5, §7 F1).

| file | what |
|---|---|
| `screen_shadow.py` | STEP 1 (measure first): today's builder (`dice="seed"`), every gate replay SHADOWED by the same replay under tier A's match control; counts the candidates today rejects that tier A accepts, and checks that nothing today accepts is rejected |
| `rebuild_dice.py` | STEP 5: the builder rebuilt with `--dice seed` and `--dice rolls` on the same states: exclusions, fact match rates, the seen mons' exact HP, the posterior's width, the weights, cost |
| `analyze.py` | the read of both (`--write result_f1a.json`) |
| `result_f1a.json` | the read |

**Sample.** The states are a seeded random prefix of the stage-0 population (40: e2_end2 / e2_prod / the maturity
LEAN ids, the F2 read's `populations`) plus a seeded sample of 20 states the STORED F rows excluded. Step 1 covers the
40 random states and 3 of the 20 excluded ones: it was stopped there (orchestrator, 2026-10-19: one heavy job at a time
beside the X34 plateau check), because shadowing a failing 200-step chain twice costs about 5 minutes a state on the
contended cores and step 5 reads the excluded stratum in both modes anyway. Step 5 covers all 60 in both modes.

**Binary.** Both steps ran the self-check `core_events` built from this unit's Rust (frozen copy, sha256 prefix
`daa7fb9a22823420`, via `POKESIM_CORE_EVENTS_BIN`) and the worktree's Python at `5f39ff4d` plus this change.

**Raw rows** (not committed): `~/gen3ai_archive/world_rolls_f1a_2026-10-19/{step1/shadow_0_of_1.jsonl,
step5/seed.jsonl, step5/rolls.jsonl}`.

Rerun (light CPU, one job at a time; root rule 5b):

```bash
OUT=~/gen3ai_archive/world_rolls_f1a_2026-10-19
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export CUDA_VISIBLE_DEVICES="" POKESIM_EMISSION_SELFCHECK=1
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 $PY -I screen_shadow.py --n-random 40 --n-excluded 20 --out $OUT/step1
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 $PY -I rebuild_dice.py --dice rolls --out $OUT/step5
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 $PY -I rebuild_dice.py --dice seed --out $OUT/step5
$PY -I analyze.py --out-dir $OUT --write result_f1a.json
```

**Inputs** (read only): the stored F rows and banks of `~/gen3ai_archive/hidden_info_maturity/` (LEAN ids only) and
`~/gen3ai_archive/hidden_info_e2/{end2,prod}/`, through the F2 read's `rebuild_worlds.populations` / `stored_rows`.
