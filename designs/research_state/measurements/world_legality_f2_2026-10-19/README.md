# F2: the honest world builder made LEGAL (2026-10-19): rebuild read

The F2 fix of [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §7 (the build:
`src/main/value_search/world_legality.py`) and its descriptive read. It covers world BUILDS only: no playout, no model,
no outcome, and no study arm re-run. The numbers and their reading are in the design doc (§6.1, §7 F2).

| file | what |
|---|---|
| `rebuild_worlds.py` | rebuilds F (`WorldSampler(reveal="off")`, J 8, `max_tries` 60, the studies' own call) on the stage-0 population's states, in a SEEDED order so every done prefix is a uniform random sample. `--mode on` uses the shipped rule; `--mode off` injects an always-legal rule (the pre-F2 draw) and runs only on the states `on` touched, plus a control sample |
| `analyze.py` | stored F worlds (the studies' pins) against `on` and `off`: illegal worlds, exclusions, defects, the identity control, and the species / move marginals |
| `result_f2.json` | the read |

**Raw rows** (teams per state; not committed): `~/gen3ai_archive/world_legality_f2_2026-10-19/{on,off}_*_of_2.jsonl`,
built at `f99d75cb` plus this change.

**Sample.** The rebuild crawled on the contended agent cores (about 2.5 states per minute over two shards). The ON pass
stopped after 316 of the 1,520 states: a uniform random prefix of the seeded order (e2_end2 138, e2_prod 105, mat_lean
73). The OFF pass covered every touched state of the ON rows (113 in the prefix) plus 20 controls. The rest is
resumable: rerun `--mode on`, then `--mode off`, then `analyze.py`.

Rerun (light CPU; root rule 5b):

```bash
OUT=~/gen3ai_archive/world_legality_f2_2026-10-19
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 6 $PY -I rebuild_worlds.py --mode on --shard 0 --of 2 --out $OUT
#   (shard 1 likewise; then --mode off --control 20 per shard; then:)
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 $PY -I analyze.py --out-dir $OUT
```

**Inputs** (read only): the stored F rows and banks of `~/gen3ai_archive/hidden_info_maturity/` (LEAN ids only; the
plateau bank is not read) and `~/gen3ai_archive/hidden_info_e2/{end2,prod}/`. The maturity F file was being appended
by the running plateau step: torn lines are skipped, and nothing is written there.
