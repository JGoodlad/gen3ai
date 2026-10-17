# X26 vs Metamon — the external-anchor read, 2026-10-17

**Question (owner, 2026-10-17):** where is X26, our production baseline, against Metamon? Have we beaten
Metamon's best model on our teams?

**Answer.** On OUR teams (home72), X26 beats `metamon:Kakuna`, Metamon's best public policy:
**0.600 [0.537, 0.660]** with Kakuna greedy, and **0.600 [0.537, 0.660]** with Kakuna at T = 1.0, its
strongest temperature by the 2026-09-28 read. n = 240 each. Both lower bounds clear 0.50, so the near-term
goal's bar is **MET at this checkpoint**. Against the same cells N0 (`ai_v14_01_base`, the 2026-09-28 read)
scored 0.460 and 0.380, so X26 is **+0.140 [+0.046, +0.230]** and **+0.220 [+0.127, +0.308]** ahead of N0.
Both gaps are DETECTED.

Three limits on that answer:

- **On Metamon's OWN teams, X26 vs Kakuna is NOT DETECTED as better.** It reads 0.570 [0.472, 0.663] at
  n = 100. The goal is defined on our teams; the away cell is a descriptor.
- **Against `SmallRL`, X26 is level with the previous lineage's best.** That is K2, `ai_v14_07_g0p_k2`,
  the G0′ plateau. Away: +0.020 [−0.058, +0.094]. Home: +0.030 [−0.040, +0.093]. Both NOT DETECTED.
  So the gain over N0 against Kakuna does not show up against the weaker anchor once the comparison is
  K2 rather than N0. K2 was never read against Kakuna, so **we do not know whether K2 had already
  closed the Kakuna gap**.
- **PROVISIONAL (owner rule, 2026-10-11):** X26 is a live run that has not plateaued (161M of a 500M
  cap). This is one checkpoint of one seed, read once.

## The cells

Every cell uses the same settings. Our side is `models/rb_x26_s2601/checkpoints/checkpoint_161000169_steps.zip`,
**step 161,000,169**, the run's latest periodic checkpoint at the read's start. It was copied to the
read's state dir, sha256 `bc6050d1…2992` (identical to the run's file), with the run's `model_config.json`
(`arch_signature gen3_mon_tied_gain_v1`). Our side plays greedy (`--temperature 0`) on CPU, `loader=bare`,
on `our_transport=rust_core_slot`, with the trainer's 250-turn forfeit. The server is the in-process Rust
front end (`ws_frontend[in-process]@63f97a57+rust:cargo:src/rust_sim`, Showdown pin `e0551883f`, port
9571). Metamon is checkout `0a00a759`, on CPU (`CUDA_VISIBLE_DEVICES=""` in the peer env and inside
`metamon_side.py`), `OMP_NUM_THREADS=1`, Kakuna with windowed attention (H20). Halves are
role-balanced, 20-game units with `--team-seed = --seed-base = unit seed`. A unit counts iff
`status == OK`, `n == 20`, `regime_verified_decisions == true` and `team_source_asymmetry == false`. Every
counted unit has `peer_clean == true`; no H17 occurred.

| cell | opponent | regime (ours / theirs) | team set | n | W–L–T | **X26 win rate [Wilson 95%]** | vs 0.50 | their argmax rate |
|---|---|---|---|---:|---|---|---|---|
| `K_greedy_h72` | Kakuna ckpt 34 | greedy / greedy | home72 | 240 | 144–94–2 | **0.600 [0.537, 0.660]** | **BETTER** | 1.0000 |
| `K_t10_h72` | Kakuna ckpt 34 | greedy / sample T = 1.0 (`regime_matched = false`) | home72 | 240 | 144–96–0 | **0.600 [0.537, 0.660]** | **BETTER** | 0.864–0.945 |
| `K_greedy_away` | Kakuna ckpt 34 | greedy / greedy | away (Metamon's 20 `competitive`) | 100 | 57–43–0 | 0.570 [0.472, 0.663] | NOT DETECTED | 1.0000 |
| `S_greedy_away` | SmallRL ckpt 40 | greedy / greedy | away | 200 | 129–71–0 | **0.645 [0.577, 0.708]** | BETTER | 1.0000 |
| `S_greedy_home` | SmallRL ckpt 40 | greedy / greedy | home (the 719-team pool) | 200 | 157–43–0 | **0.785 [0.723, 0.836]** | BETTER | 1.0000 |
| `S_t1_away` | SmallRL ckpt 40 | t1 / t1 (tier B distribution check) | away | 100 | 73–27–0 | 0.730 [0.636, 0.807] | BETTER | 0.579–0.679 |

The two Kakuna home72 cells use paired seeds 0–11. Per role, our challenge vs theirs: K greedy
0.558 / 0.642; K T1 0.608 / 0.592; S away 0.630 / 0.660; S home 0.770 / 0.800. The 250-turn forfeit
fired 0–1 times per cell (3 in 1,080 games). Median game length was 29–40 turns. Kakuna saw 59 distinct
home72 teams on our side; SmallRL home saw 97 pool teams.

## Compared with history (Newcombe, independent samples; same opponent, regime and team set)

| cell | earlier model of ours | its read | X26 − it [95%] | verdict |
|---|---|---|---|---|
| K greedy · home72 | N0 `ai_v14_01_base` @75,005,952 | 0.460 (92/200) | **+0.140 [+0.046, +0.230]** | X26 BETTER |
| K T = 1.0 · home72 | N0 | 0.380 (76/200) | **+0.220 [+0.127, +0.308]** | X26 BETTER |
| S greedy · away | K2 `ai_v14_07_g0p_k2` @91,127,808 (G0′) | 0.625 (375/600) | +0.020 [−0.058, +0.094] | NOT DETECTED |
| | C_fix `ai_v14_06_lbat_ctrl_fix` | 0.658 (395/600) | −0.013 [−0.091, +0.061] | NOT DETECTED |
| | `ai_v12_02_winprob_critic` @75,005,952 | 0.650 (65/100) | −0.005 [−0.115, +0.111] | NOT DETECTED |
| S greedy · home | K2 | 0.755 (453/600) | +0.030 [−0.040, +0.093] | NOT DETECTED |
| | C_fix | 0.717 (430/600) | +0.068 [−0.003, +0.132] | NOT DETECTED |
| | `ai_v12_02` | 0.520 (52/100) | +0.265 [+0.151, +0.375] | X26 BETTER |
| S t1 · away | `ai_v12_02` | 0.560 (56/100) | +0.170 [+0.037, +0.295] | X26 BETTER |

Sources: `kakuna_goal_read_2026-09-28/` (N0), `learner_battery_2026-09-26/read/battery_read.json`
`ga_levels` (K2, C_fix; 600 per team set), `metamon_matched_regime_2026-09-14/` (`ai_v12_02`).
**The comparisons are not like-for-like on everything.** The N0 read ran torch 2.5.1 (`gen3ai_stable`)
and the `dcddac0b` `sim_bridge`; this one ran torch 2.8 and HEAD `63f97a57`'s bridge. The team files,
opponent checkpoints, regimes and protocol are the same. The pool `home` set was 719 teams then and now.
Pre-Rustboro runs are history, not comparators (root `CLAUDE.md` § ERAS); they are listed here only
for the SOP's continuity question.

**Kakuna's strongest temperature is still unresolved.** Here greedy and T = 1.0 read identically
(Δ 0.000 [−0.087, +0.087], NOT DETECTED). The 2026-09-28 ranking (T = 1.0 strongest) was a
point-estimate order, and this read does not reproduce it. The goal is met at both settings, so the
choice does not change the verdict.

## Cost (sizing)

One lane on root `CLAUDE.md` rule 5b's three physical cores (`nice -n 19 taskset -c 5-7,13-15`, under
`mem_cap.sh 12`), next to X26 on the GPU and the P0.1 `hidden_info_e2` runner. Load1 mean was 7.2
(max 10.1). Kakuna took **8.8 s/game** (9.4 on away). SmallRL took **7.2–7.9 s/game**, slower than the
SOP's 3.0, because our in-process side now dominates: one torch thread on the end-state network. Each
lane uses about 2 cores (ours + the peer), so **≈ 2.5 h wall ≈ 5 CPU-h for 1,080 games.** CPU is
enough for Metamon at this size; no GPU window is needed.

**Planned vs run.** The plan was 400 games per Kakuna home72 cell. The read was stopped at **240 each
(paired seeds 0–11)** at the brief's ~3 h time box, between units, with the decision already resolved.
The in-flight unit `K_greedy_h72_12` was killed by its process group and is not counted.

## FINDINGS

1. 🚨 **`--teamset home` (the pool source) depends on the CWD, and it failed as an unnamed `IndexError`.**
   The first queue ran `main.anchors` with `cwd = src/`. `utils/team_loader/loader.py:9`
   (`base_dir="data/teams"`) and `:49` (`os.path.join("data", rel_file_path)`) resolve against the CWD,
   so `TeamLoader()` printed a warning and returned **0 teams**. `main/anchors/session.py:118` (the pool branch)
   passed that to `Gen3Teambuilder([])`, whose error path `utils/teambuilder.py:95` indexes
   `validations[0]` on an empty list, so the read died as `our_side_error: IndexError: list index out
   of range`. All 5 `S_greedy_home` units failed this way, twice each, in about 10 s, with no games. The
   cell was **re-run from the checkout root** (the runner's `cwd` is now `TREE`). The 10 re-run units
   are the cell above. The summaries record it (`ours 0 teams / theirs 719`, `team_source_asymmetry
   true`). **No counted unit is affected:** the Kakuna home72 and every away cell use the `dir` source
   with an absolute path, and every counted unit has `status OK` and asymmetry false. The `dir` source
   already refuses an empty set by name (`no_teams`); the pool source did not. A proposed fix and a
   regression test are in the agent's report; they are not shipped here. The training pool's own
   `TeamLoader()` callers share the CWD dependence.
2. **SmallRL is no longer a 5-minute cell.** At 7.2–7.9 s/game on the agent cores, a 100-game tier-B
   cell is ~13 min. SOP §2's cost table is updated.
3. Our side's own argmax rate is not stamped in a greedy cell (it is 1 by construction). Kakuna's T = 1.0
   argmax rate is 0.864–0.945, in line with 2026-09-28's 0.847–0.928.

## Files

- `rows.jsonl`: every game row of every counted unit (the tool's full row plus `x26_cell`, `unit`,
  `unit_seed`), 1,080 rows.
- `cells.json`: pooled per cell, the Wilson interval, integrity counters, and the history Newcombe
  deltas.
- `scripts/xq.py`: the detached, resumable, one-lane queue (`run | status | one <unit>`).
  `scripts/aggregate.py`: pools the done units. Raw unit dirs, logs and `units.jsonl` live outside the
  repo, in `/home/goodlad/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/`, together with the copied
  checkpoint.
