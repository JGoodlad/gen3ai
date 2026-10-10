# P_end after the four-move-closure amendment: preconditions A–D, the end-state identity, the argv (2026-10-10)

**Status: MEASURED (CPU). P_end = `b132b099`.** The owner's 2026-10-10 decision ("Do A"): the END-STATE arm of the
closing test (`designs/endstate/design_endstate_closing_test.md`) gains `--move-set-closure on`
(`gen3_move_set_closure_v1`, v154) before any end-state seed has run. Production stays at P_prod = `95d014fa`
(`rb_es_prod_s2001` banked, `s2002` running at the time). P_end moves from `95d014fa` to the commit that lands the
overlay, `b132b099`, which also carries the infra commits that landed on `main` since `95d014fa`.

## In plain language

The end-state arm now also uses the "four moves seen means no other move" fact. To keep the comparison fair, the
production network at the new end-state commit must be EXACTLY the one the production seeds are training at
`95d014fa`. It is: the same compiled graph, the same weights at init, the same outputs, the same learner golden and the
same observations. The only difference in the production config is one new key, `move_set_closure`, whose production
value is OFF. Every file that changed since `95d014fa` is listed below with the reason it cannot change a production
run. None is a `data/` file or a Rust file.

## (A) Production's extractor is byte-identical (`shas.py` → `shas.out`)

`graph_sha.py` (`../../static_recovery_2026-10-09/`), seed 0, 8 seeded rows, one thread, run with this checkout's
`src` first (the `src` path is printed in each row).

| config | graph lines | graph sha256[:16] | state_dict sha256[:16] | outputs sha256[:16] | params |
|---|---|---|---|---|---|
| production at P_end | 20,148 | **`421c6b98ce7937f4`** | **`749c56159ab028f4`** | **`51c02c6c6342a045`** | 1,937,942 |
| P_prod's record (§2.1) | 20,148 | `421c6b98ce7937f4` | `749c56159ab028f4` | `51c02c6c6342a045` | 1,937,942 |

EQUAL. One dynamo graph (no break).

## (B) The K9 learner golden

- `learner_golden.json` blob **`9ef44772a3f76fcfa4de7e59590aec7be6b05812`**, `learner_golden_buffer.npz` blob
  **`9f13350daec82f90f37cf557b776646c316e6415`**: EQUAL to P_prod's.
- `agents/training/learner_golden_test.py`: **6 passed** at P_end (production reproduces the golden).

## (C) The production mirror and the obs golden (as AMENDED: identical modulo new keys that are OFF in production)

- `designs/production_config.json`: blob `49cd523f38ee6fe30ff512454b1eeaed5feda162` (checkargs tag
  `production_config@49cd523f38ee`); P_prod's is `5ad40492fe21c3b5c46694604a37939329e468b9`. The diff is ONE added key:
  `"move_set_closure": "off"` (`a9d6fd79`, v154). Production's value is OFF, so it builds nothing and changes no
  expression: (A) and (B) are the proof.
- `golden_obs_fixture.json` blob **`09733f8cef9a42be6b7eb03e8f327ab20153e953`**: EQUAL. `python -m
  agents.training.golden_obs_core --check`: **✓ the core reproduces all 991 committed hashes** (this checkout's own
  Rust build).

## (D) Every changed file since `95d014fa`, classified

`git diff --name-only 95d014fa b132b099 -- src/ data/ ':!*_test.py'` (the tests are listed after the table). Classes:
**(a)** gated on a flag production keeps OFF; **(a′)** the RECORD / parser surface of such a flag (a new
`model_config.json` field or argv flag whose production value is OFF; it changes what a save WRITES, never what
production builds or computes, proven by (A) + (B)); **(b)** docs / tooling / processes the trainer child never
imports; **(b′)** the LAUNCHER parent process (it runs from the launching checkout, not the pin, so both arms' later
launches run the same launcher whatever the pins; the child imports only `main.launcher.ipc`, unchanged);
**(s)** on the training path but STORAGE only; **(c)** on production's training path (voids identity).

| file | commit(s) | class | why it cannot change a production run |
|---|---|---|---|
| `src/agents/model/extractor_build.py` | `a9d6fd79` | (a) | validates the mode and refuses `on` without the belief family; builds nothing |
| `src/agents/model/extractor_forward.py` | `a9d6fd79` | (a) | `if self.move_set_closure == "on"`; the `off` expression kept verbatim ((A)'s graph text equal) |
| `src/agents/model/hypothesis_tokens.py` | `a9d6fd79` | (a) | `slot_move_presence(..., graph=False)` default keeps the detached expressions verbatim; the op roster (in production's forward) calls the default ((A) equal) |
| `src/agents/model/hypothesis_set.py` | `a9d6fd79` | (a) | the `MOVE_SET_CLOSURE_MODES` constant |
| `src/agents/model/flag_registry.py` | `a9d6fd79` | (a′) | the registry row, default `off` |
| `src/agents/model/model_version/{compat,constants,construct,fields,migrations}.py` | `a9d6fd79` | (a′) | config 153 → 154; the field `move_set_closure` (default and migration `off`), string-compared in `check_compatible`. A production save writes `"config_version": 154, "move_set_closure": "off"`; a P_prod checkpoint (v153) migrates to `off` on load at P_end (the cross, F-ES-4) |
| `src/agents/model/snapshot.py` | `a9d6fd79` | (a′) | `current_model_version` / `arch_toggles_from_model` record the field (`off` for production) |
| `src/main/train/config.py` | `a9d6fd79` | (a′) | `_resolve("move_set_closure", "off")` |
| `src/main/train/parser/clean_world.py` | `a9d6fd79` | (a′) | the `--move-set-closure` argparse entry (default None → `off`) |
| `src/main/train/arch_arms.py` | `b132b099` | (a) | the `endstate` overlay gains `("move_set_closure", "on")`; read only under `--arch endstate` |
| `src/agents/training/eval_collect.py` | `52c2cdd9` | (s) | see below: on-disk STORAGE of the persisted eval snapshot only |
| `src/main/train/parser/eval_subprocess.py` | `52c2cdd9` | (b) | `--keep-eval-snapshots` HELP text only |
| `src/main/prober/groom.py` | `52c2cdd9` | (b) | the offline groom CLI (counts a hard-linked snapshot as shared); not imported by the trainer |
| `src/agents/training/CLAUDE.md`, `src/main/prober/CLAUDE.md`, `src/main/prober/web/CLAUDE.md`, `src/main/launcher/CLAUDE.md`, `src/agents/model/CLAUDE.md`, `src/agents/gen3_data/CLAUDE.md`, `src/agents/observation/CLAUDE.md` | `52c2cdd9`, `52598e84`, `59c53d73` and the 2026-10-10 leaf-cut docs commits (`59eff547` … `63211d0f`) | (b) | docs |
| `src/rust_sim/CLAUDE.md`, `src/rust_sim/README.md` | the leaf-cut docs commit `6900a964` | (b) | Markdown only; NO Rust source, `Cargo.*` or build file under `src/rust_sim/` or `src/rust_env/` changed. (D)'s "any change under `src/rust_sim/`" is read as a change to what is compiled: a `.md` file is not |
| `src/main/launcher/run.py` | `52598e84`, `59c53d73` | (b′) | `--restart-interval-hours` default 3 → 6 (the closing-test argv TYPES 6, so the value is the same either way); the submodule preflight (refuses a launch whose `deps/pokemon-showdown` is unusable, before anything exists) |
| `src/main/launcher/dry_run.py`, `src/main/launcher/submodule_gate.py` | `59c53d73` | (b′) | the same preflight in `--dry-run` and its verdict |
| `src/main/launcher/worktree.py` | `59c53d73` | (b′) | `showdown_link_source()` names the same directory the pin's `deps/pokemon-showdown` link already pointed at (the launching checkout's submodule) |
| `src/utils/showdown_deps.py` | `59c53d73` | (b′) | the preflight's probe; imported only by the launcher (`worktree.py`, `submodule_gate.py`) |
| `src/main/play.py`, `src/main/anchors/{server,peers}.py`, `src/main/anchors/peer_scripts/metamon_side.py`, `src/utils/bridge/ws_frontend.py` | `86f31b88` | (b) | live play / anchor reads / the websocket front end: comment and refusal-text rewording for port 8001; not on the training path |
| `data/` | — | — | **EMPTY** |
| `src/rust_sim/`, `src/rust_env/` code | — | — | **EMPTY** (only the two `.md` files above) |

Tests changed in the window (no run imports a test): `ctor_kwarg_snapshot_test.py`, `flag_requires_test.py`,
`move_set_closure_test.py`, `eval_snapshot_dedup_test.py`, `anchors/{cli,peers,server}_test.py`,
`launcher/{dry_run,pin_commit,pinned_argv,restart_interval_default,submodule_gate}_test.py`, `play_test.py`,
`utils/{run_archive,showdown_deps}_test.py`, `utils/team_loader/committed_team_bytes_test.py`, and P_end's own
`arch_arms_test.py` / `move_set_closure_test.py`.

**The eval-snapshot dedup (`52c2cdd9`) changes STORAGE, never behaviour or the eval read.** `persist_eval_snapshot`
runs AFTER the cycle's games have been played and its results recorded (`eval_callback._persist_snapshot`;
`selfplay_callback` after `record_eval_selection`); the games read the scratch snapshot (`pending["snapshot"]`), which
is unchanged. The persisted `eval_traces/step_<N>/snapshot.zip` becomes a HARD LINK to `checkpoints/
checkpoint_<N>_steps.zip` only when that file exists on the same filesystem, has the same size AND the same sha256 as
the scratch snapshot; otherwise it is a copy, as before. Either way its BYTES are the same (sha-verified), so every
reader (the prober, `main.ops.eval_trace_gen`, the retention prune) sees an identical file. The manifest gains a
`snapshot_storage` block (mode, note, sha256). The only runtime cost is two sha256 passes over ~62 MB per eval cycle
when cadences coincide (sub-second; speed is reported, never decides). The closing test's cross reads
`final_model.zip`, never an eval snapshot. `eval_snapshot_dedup_test.py`: 17 passed.

## The end-state identity at P_end (CPU, same `graph_sha.py`)

| config | graph lines | graph sha256[:16] | state_dict sha256[:16] | outputs sha256[:16] | params |
|---|---|---|---|---|---|
| **`--arch endstate` at P_end (closure ON)** | 35,662 | **`42ee361b11fa5a8d`** | **`df71da47741a8a0f`** | **`5adfaa6c40a7a71e`** | **2,031,460** |
| control: the `95d014fa` overlay (closure off) in this tree | 34,898 | `55c7f5c009d4595b` | `df71da47741a8a0f` | `c78ca89766000c71` | 2,031,460 |

The control reproduces `95d014fa`'s recorded arm (`endstate_facts_2026-10-09/README.md` addendum) hash for hash, so
the only difference in the end-state graph is the closure: +764 graph lines (the fixed-mass presence over every slot,
through the graph), the same parameters (the closure builds nothing; the state sha is unchanged), different outputs (by
design: every bench / hidden slot's reinjection weights change).

**K9(b) tie-margin recorder** (`k9_tie_margins.py` → `k9.out`; every zero-init projection planted, 64 fixture rows):
no undeclared discrete op in production, the closure-off arm or the closure-on arm; excluded share at 2e-4: 0.0625 /
0.0781 / 0.0781 (DESCRIPTIVE).

**The CPU `--debug` smoke of the new arm** (`debug_shape_smoke_integration_test.py::...[endstate]`, under
`mem_cap.sh 24`): **PASSED**, 1,010 s, peak 3.39 GB; the run's `model_config.json` records every overlay key including
`move_set_closure: "on"` (the test asserts it). `smoke.out`; recorded in `designs/ops/slow_tier_status.json`. CPU,
eager, no compile: the CUDA / compiled layer is E's job.

## (E) at the new P_end

The GPU check that met E at `95d014fa` (`rb_es_gpucheck_end`) does not carry over: the closure adds a graph path (the
fixed-mass presence through the move head's graph, `torch.where` over σ(a + τ)) that R1's CUDA compile parity has not
seen. Per the brief, **the first end-state seed's own startup gates ARE precondition E**: R1 compile parity (forward +
backward), the K6 freeze, K9(b) on update 1 and the update-10 canary (compiled == eager). If any refuses, the chain
HOLDS (the training session's follower already holds on a non-DONE ending) and no data counts.

## The argv

```bash
cd /home/goodlad/dev/gen3ai && PYTHONPATH=src python -m main.launcher --restart-interval-hours 6 --pin-commit b132b099 \
  --arch endstate --steps 15000000 --seed 2001 \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda --run-name rb_es_end_s2001
```

Seeds 2002–2008: `--seed 200N --run-name rb_es_end_s200N`. Validation at P_end: `checkargs.out` (12 flags, 0
unrecognized; `✓ every ARCH-surface key matches the production mirror + the arm 'endstate'`, tag
`endstate@production_config@49cd523f38ee+overlay@2b359668`; `✓ every RECIPE knob matches recipe.fresh (28 knobs)`;
`✓ this command still launches`) and `dry_run.out` (FRESH, pin `b132b09952ba…`, torch 2.8.0+cu126, restarts every 6.0 h,
desktop GPU ✓, disk ✓ 9.53 GiB required, showdown deps ✓, `✓ DRY RUN — this command would launch`; nothing created).

## Files

- `shas.py` / `shas.out`: (A) and the end-state identity at P_end (run: `python shas.py <checkout root>`);
  `shas_pre_commit.out` is the same read on the working tree before the commit (identical).
- `k9_tie_margins.py` / `k9.out`: the K9(b) recorder.
- `checkargs.out`, `dry_run.out`: the argv's validation at P_end.
- `smoke.out`: the CPU `--debug` smoke of the new arm.
