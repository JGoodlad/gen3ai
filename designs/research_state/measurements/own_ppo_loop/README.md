# Owning the PPO loop — the equivalence reads (2026-10-01)

Design: [`designs/endstate/design_own_ppo_loop.md`](../../../endstate/design_own_ppo_loop.md) §2.1, §2.3, §4.
All runs CPU, `gen3ai_torch28` (torch 2.8.0), sb3 / sb3-contrib 2.8.0, seed 42, `--debug`, run dirs
under the worktree's own (gitignored, since deleted) `models/`; the scratch copies of the scripts ran
from `~/.cache/gen3ai/tmp/`.

## 1. Stock-vs-stock CONTROLS (code at `ef1db2be`, before stage 1)

| core | argv beyond `--debug --seed 42` | result |
|---|---|---|
| Python (`DummyVecEnv`, rust bridge) | `--steps 6000` | **NOT reproducible**: same 106 tags and step grid, but 66 of 103 non-wall-clock series differ from the FIRST dump (`belief/hptype_n_slots` 1945 vs 2095 — different games); `policy.pth` differs |
| Rust env core | `--steps 4000 --arch production --env-core rust --n-envs 4 --n-steps 256 --batch-size 256` | **IDENTICAL**: 342 / 342 non-wall series (tag, step, value; NaN = NaN), 16 updates; `final_model.zip` `policy.pth` sha256 `b695e42be3c03f3f…` both runs |

Wall-clock tags excluded: `time/fps`, `*_ms`, `*_ms_per_host_step`, `rust_env/trainee_decisions_per_s`
(`tb_compare2.py`).

## 2. E3 — the real-run A/B: upstream loop (`GEN3AI_PPO_LOOP=sb3_reference`) vs the OWNED loop

`ab.sh`, at stage 1's commit; each reference run printed its `[PPO LOOP] … UPSTREAM` line, each owned
run did not.

| core | result | reading |
|---|---|---|
| **Rust** | 342 / 342 non-wall series identical; `policy.pth` `b695e42be3c03f3f…` in BOTH arms — and equal to the pre-stage-1 control's | **IDENTITY** (the declared bar) |
| Python | same 106 tags, identical step grid on every tag; values diverge from the first dump, `policy.pth` differs | exactly the stock-vs-stock control's behaviour (the env is not reproducible) — identity on this core rests on E1 + E2 |

## 3. E1 / E2 (in the routine and `sim` tiers)

* `src/agents/training/own_ppo_loop_test.py` (E1, routine, ~20 s): exact hook trace / buffers / dumps
  / params / resume vs upstream; mutation-checked — moving the dump after the update fails 5 tests,
  dropping the truncation bootstrap 4, renaming a collect local 2.
* `src/agents/training/own_ppo_loop_parity_test.py` (E2, `sim`, ~27 s): the same recorded games replayed
  through the Python core under both loops, every buffer field byte-exact; mutation-checked — firing
  `on_step` after `rollout_buffer.add` fails it on `win_target` / `win_mask`.

## 4. The eval-dump KL skip (design §2.1)

`kl_skip_scan.py <run dirs>` (read-only, each run's OWN event files; `LRCHECK=1` adds the LR read):

| run | controller | updates | `train/*` dumped at an eval step | eval cycles |
|---|---|---|---|---|
| `ai_v14_01_base` (N0) | live | 739 | 37 (5.0 %) | 37 |
| `ai_v12_02_winprob_critic` | live | 741 | 37 (5.0 %) | 37 |
| `ai_v12_11_ladder_ctrl10M` / `_19_ladder_lambda09` / `_23_ladder_rollout` | live | 100 / 100 / 98 | 5 / 5 / 5 | 5 each |
| `ai_v14_02_lbat_ctrl` / `_03_lbat_e5` / `_05_lbat_l95` / `_06_lbat_ctrl_fix` / `_08_g0p_k3`, `ai_v13_12_plateau`, `ai_v13_22/23_popr1_*` | frozen (`--fork-lr-freeze`) | 80–81 | 4 each | 4 each |

N0: LR unchanged after 37 / 37 skipped updates, and after 695 / 701 normal ones.
