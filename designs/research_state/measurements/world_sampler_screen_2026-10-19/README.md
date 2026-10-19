# World-sampler screen, stage 0 (2026-10-19): zero playouts

Stage 0 of the first experiment in [`design_world_sampler.md`](../../../endstate/design_world_sampler.md) §6. The
numbers and their reading live there (§6.1). This folder holds the scripts and their outputs.

**What it is.** A POST-HOC, DESCRIPTIVE screen of how much weight each candidate world sampler puts on the opponent's
TRUE hidden team. It runs on banks whose own studies have already been read:
- the maturity battery's interim bank, LEAN ids only;
- E2's end2 and prod banks.

It reads only those studies' INPUTS:
- each state's revealed opponent information;
- the true opponent team;
- the stored F / FN world records;
- the network's belief rows.

It never reads a playout outcome or an IMP. Nothing here is under `data/`, and nothing reaches a network or a run.

| file | what |
|---|---|
| `screen_samplers.py` | the screen. Samplers: F's rule (a); joint Smogon teammates (b); a team-level corpus model (c) on `w_forum_v1`, its STRICT variant and the `hl_05_26` ladder corpus; the network's belief (d) at X26's four LEAN anchors. Meters: species hit, exact never-seen-set mass, log-likelihood, recall@h, the seen mons' move exactness (per species set priors too), the exact-world mass q_SM. Its GIGO check compares the re-implemented F / FN rules with what the stored worlds held |
| `result_screen.json` | the summary: pooled, per bank, per opponent stratum, per k bin; battle-clustered bootstrap |
| `result_screen.rows.jsonl.gz` | one row per state |
| `validate_worlds.py` | F2: Showdown's `TeamValidator('gen3ou')` on F's stored world teams, with the true teams as the control |
| `result_validate.json` | its counts, error kinds and examples |

Rerun (light CPU, one process at a time; root rule 5b):

```bash
F=designs/research_state/measurements/world_sampler_screen_2026-10-19
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 $PY -I $F/screen_samplers.py      # ≈ 7–11 wall-min
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 $PY -I $F/validate_worlds.py      # ≈ 3 wall-min (Node)
```

**Inputs** (read only):
- `~/gen3ai_archive/hidden_info_maturity/` (bank, `worlds/shared/F`, `worlds/<anchor>/FN`, `fn_belief/`);
- `~/gen3ai_archive/hidden_info_e2/{end2,prod}/` (bank, `worlds/F`);
- `utils.team_corpora` (`w_forum_v1` ABC) and `utils.ladder_corpus` (full);
- the Smogon priors through `agents.gen3_data`;
- the validator through the MAIN checkout's built `deps/pokemon-showdown`.

The maturity F file was being appended by the running plateau step. Only the LEAN ids were read (all built before
2026-10-19's read); torn lines are skipped; nothing is written there. The PLATEAU bank is not read: its registered read
is pending.

**Fixed before the first run:** ALPHA 0.05 (b's smoothing), the corpus backoff λ = n / (n + 1), NEAR 4 (the strict
filter), N_MC 4,000, the bootstrap seed 20261019. Nothing was tuned on these banks; the over-confident backoff (§6.1
point 4 of the design) is reported, not refit.
