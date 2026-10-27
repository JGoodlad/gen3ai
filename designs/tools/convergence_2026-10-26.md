# Ops and read tooling: one copy per concept (2026-10-26)

**Status:** stage 0 (the safe refactors) LANDED with this doc; stage 1's pid-liveness part LANDED 2026-10-27 (its
other items are still staged); stages 2-4 are STAGED, each with the condition that lets it start. Owner grant:
maintain the active tooling, converge where it helps, deep interfaces, aligned with the end state. Keep this doc
current as stages land (it is in an always-current tree); delete a stage's section once it lands and its row in §1
says CONVERGED.

In three days five tool families grew side by side: `src/main/x35/` (kit, kakuna, batch, readout, d1, ge),
`src/main/reads/` (the one read module, `reads.md`), `src/main/ops/arm_chain/` (the chain supervisor),
the launch / watch shell (`scripts/ops/watch_run.sh`, `_common.sh`, and `launch_x35.sh`, which lives in a
measurement folder: `designs/research_state/measurements/x35_breakers_stage_2026-10-17/`), `src/main/anchors/`, and
the box layer (`utils.procfs`, `utils.procgroup`, `utils.heavy_slots`, `utils.mem_cap`). Each re-grew the same
small helpers. A copy that drifts is a GIGO bug that no single tool's tests can see (two probes disagreeing about
whether a port is free; one liveness rule reading a zombie as alive and another as dead).

## 1. The overlap map

CANONICAL is the copy every tool should import. The concept is converged when only the canonical copy (plus a pinned
literal where an interpreter boundary forces one) is left.

| concept | copies (file: symbol) | canonical | status |
|---|---|---|---|
| reserved ports 8000 / 8001 | `src/utils/bridge/ws_frontend.py`, `src/main/play.py`, `src/main/anchors/server.py`, `src/main/anchors/peer_scripts/metamon_side.py` (each a literal dict) | `utils.ports.RESERVED_PORTS` | CONVERGED; Metamon's peer keeps a literal (it runs under Metamon's interpreter and never imports this tree), pinned equal by `src/utils/ports_test.py` |
| "can a server bind this port" | `kakuna.choose_port` / `claim_port` (0.0.0.0, no `SO_REUSEADDR`), `anchors.server._free`, `team_strength.cli._port_free` (127.0.0.1, `SO_REUSEADDR`); the frozen measurement queues `mmq.py` / `hrrq.py` | `utils.ports.bindable` / `first_bindable` / `claim` (host and reuse are the caller's) | CONVERGED (in-repo); the measurement scripts are records, never edited |
| a port CLAIM (flock held for a unit's life) | `kakuna.claim_port` only; the standalone anchors `pick_port` and the team-strength lane bind-and-release with NO claim | `utils.ports.claim` | stage 2: every server-starting tool claims |
| non-blocking exclusive flock on an open file | `kakuna.claim_port` / `cmd_run` (`supervisor.lock`), `reads.runner._lock`, `reads.supervisors.acquire`; fd-level `heavy_slots._try`, `gate_lock._try`; `arm_chain.chain.run`, `arm_chain.ship._lock` | `utils.filelock.try_exclusive` | CONVERGED for kakuna and reads; the fd-level and arm_chain copies are stage 1 / 2 |
| ONE supervisor per state path | `reads.supervisors.Hold` (path lock outside the state + `/proc` scan + dir identity), used by `main.reads run` and `kakuna run`; the legacy in-state locks (`<state>/supervisor.lock`, `<state>/<lane>.lock`); `arm_chain.chain` (a flock on its state file) | `main.reads.supervisors` | kakuna and reads already share it (N1-P); the in-state locks are deleted in stage 2; arm_chain keeps its own until stage 3 |
| pid liveness | `procfs.pid_alive` (one `/proc/<pid>/stat` read; Z / X = gone); the remaining copies: `_common.sh` `pid_alive` and `launch_x35.sh` (a verbatim copy of `_common.sh`'s), and `agents.training.learner_benchmark` (`os.path.exists("/proc/<pid>")`: a zombie reads ALIVE, so its exit wait runs to the deadline) | `utils.procfs.pid_alive`; the shell twin `_common.sh` | CONVERGED for the Python sites (2026-10-27: `atomic_io`, `eval_ledger.store`, both `compile_cache*`, `launcher.worktree`, `heavy_slots._alive` read `/proc`; `utils.liveness_fakes.assert_pid_liveness` is the one contract each site's test runs: a zombie is gone, a live pid is alive when `kill` says ESRCH or EPERM). Left: the shell twins (stage 1) and the benchmark's wait |
| process IDENTITY (pid + start ticks) | `arm_chain.procwatch.ProcRef` / `ref_of` / `alive` / `signal_if`; `procfs.Proc.key`; `gpu_lease` (`start_ticks`); `launcher.gpu_window.protocol.process_identity`; `launcher.worktree._proc_starttime` (its own `/proc` parse); `_common.sh` / `launch_x35.sh` `pid_start` | `ProcRef` and its three functions, promoted into `utils.procfs` | stage 3 (arm_chain is the owner; move after its cut-over settles) |
| live training runs | `procfs.live_training_runs` (by trainer argv; `kit.live_runs` delegates); `arm_chain.live` (by launcher argv + trainer descendants, reports the unresolved) | `arm_chain.live`'s rule, promoted to `utils.procfs` | stage 3 |
| sha256 of a file | `kakuna._sha`, `reads.models.sha256_file`, `x35.gc._file_sha`, `x35.refpop.sha256_file`, `x35.batch.sha256_of`, `arm_chain.canary._sha256`, `peer_scripts/weight_pins.sha256_file`, and ~15 outside the tool layer (`agents.training`: weights_link, lineage, baselines, cycle_ledger, untaught_meter, policy_drift; `main`: h2h.play, belief_roles, probe_battery, policy_spectrum, human_data, x50_r1, hidden_info_study) | `utils.hashing.sha256_file` | CONVERGED for kakuna, `reads.models`, gc; refpop / batch / canary in stage 1; `weight_pins` stays (Metamon's interpreter); the rest is stage 4 |
| JSONL event append | `kakuna._jl` (no `default=`), `reads.runner._jl` (`default=str`, mkdir) | a `utils.records.append_jsonl` | stage 2 |
| atomic JSON write | `utils.atomic_io.atomic_write` (unique tmp, sweep); `reads.runner._atomic` (a fixed `.part`: two writers collide); `reads.record.write`; `launcher.gpu_window.protocol.write_json_atomic`; `live.halt._atomic_write`; `heavy_slots._write`; `gpu_lease._write_record`; `gate_lock_status.write_record` | `utils.atomic_io.atomic_write` | stage 2 (temp names change) |
| regime ids | `eval_ledger.schema.regime_id` (S lane), `reads.assemble.anchor_regime_id` / `group_regime_id` (anchor lane); the same digest recipe (sha256 of sorted JSON, 16 hex) | both stay: two id SPACES, each owned once | no duplicate; the shared 16-hex digest may become one helper in stage 4 |
| interval statistics | Wilson: `anchors.results.wilson`, `x35.d1.wilson_pn`, `agents.training.stats.wilson_ci`, `policy_drift_cond.wilson`, `prober.replay.wilson_ci`, `ops.plateau_signal.wilson`; Newcombe: `anchors.results.newcombe`, `reads.rules.newcombe_diff_ci`, `best_response_gap.newcombe_diff_ci`, `x35.d1.newcombe_strat`; Welch: `x35.readout`, `probe_battery.report`, `human_data.x59s_read` | `main.anchors.results` (`wilson`, `newcombe`) and `x35.readout.welch_interval` until one stats module exists | stage 4: verdict-steering numbers; each move needs a bitwise grid-equivalence test |
| agent cores `5-7,13-15` | `kakuna.CORES`, `reads.runner.CORES`, `team_strength.kakuna_lane.CORES`, `rebuild_main_rust.AGENT_CPUS`, `ship_gate.AGENT_CPUS` (a tuple), `arm_chain` (`spec` / `cli` defaults) | `utils.ship_gate.AGENT_CPUS`, with a cpuset-string helper | stage 2 |
| an agent child's env (CPU only, this checkout's `src`) | `reads.runner.child_env`, `kakuna.unit_env`, `arm_chain.runner.cpu_env` | `arm_chain.runner.cpu_env`, promoted | stage 3 |
| `git` subprocess wrapper | `_git` in `push_guard`, `compile_cache_shared`, `x35.batch` (two), `live.halt`, `launcher.pinned_argv`, `arm_chain.{window,ship,testing_fakes}`, `reads.playpath`; `utils.git` | `utils.git` (grown a `run(repo, *args)`) | stage 4 |
| worktree, commit, rebase, push | the `/gen3ai-ship` skill (by hand), `arm_chain.ship.Shipper` (programmatic, with the push guard and the worktree guard), `launcher.worktree` (run worktrees, a different job) | one `utils.ship` module that the skill and the Shipper both call | stage 3 |
| read checkouts | made by hand under `~/dev/gen3ai-reads/<name>_<sha>`; `reads.playpath` / `repin` decide when one may move | `main.reads` | stage 2: a `reads checkout` verb |

**Near the size cap** (`src/file_size_gate_test.py`: over 2,000 lines fails; 1,000-2,000 is reported): no tool file
is near 2,000. Two sit in the reported band, both on the chain's path: `src/main/x35/batch.py` (1,073) and
`src/main/x35/batch_test.py` (1,200). Split them in stage 1 (pins / gates / records). Next largest:
`anchors/cli.py` 916, `anchors/cli_test.py` 911, `reads/rules.py` 867, `ws_frontend.py` 865, `x35/kit.py` 813.

## 2. Target structure: the deep modules and their interfaces

Each module is stdlib-only unless it says otherwise, because the play path, the gate path and the chain all import it.

| module | interface | who imports it |
|---|---|---|
| `utils.procfs` | `read_stat`, `pid_alive`, `snapshot` / `ancestors` / `descendants`, `cmdline`, `/proc/locks` `holders`; stage 3 adds `ProcRef`, `ref_of`, `alive(ref)`, `wait_gone`, `signal_if` and the launcher-argv live-run rule | everything that asks "is it alive" or "who is it"; the shell side sources `_common.sh` |
| `utils.filelock` | `try_exclusive(path, mode) -> file or None` | ports, reads, kakuna; later heavy_slots, gate_lock and arm_chain |
| `utils.ports` | `RESERVED_PORTS`, `bindable(port, host=, reuse_addr=)`, `first_bindable(lo, hi, ...)`, `claim(lo, hi, lock_dir, ...)` | anything that starts or dials a server |
| `utils.hashing` | `sha256_file(path)` | anything that pins a file by content |
| `utils.records` (stage 2) | `append_jsonl(path, obj, *, stamp=True)`, `write_json(path, doc)` over `atomic_io` | reads, kakuna, the chain's status records |
| `main.reads` | THE read scheduler: the spec, the anchor and S lanes, `supervisors` (THE one-supervisor-per-path module), `playpath` / `repin`, the record | read drivers; `kakuna` shrinks to the unit player it calls |
| `main.ops.arm_chain` | THE launch-and-end supervisor for training arms | the orchestrator; it imports the utils above and keeps its own launch / end policy |
| `utils.ship` (stage 3) | `ship(files, message) -> sha`: worktree, commit, rebase, `main.ledger_index --write`, `push_guard`, push | `/gen3ai-ship` and `arm_chain.ship` |

**Delete when its stage lands:** the in-state `supervisor.lock` and `<lane>.lock` (stage 2, once no supervisor of
code older than `de15dfc7` can be live); `kakuna run` as a queue of its own (stage 2: `main.reads` runs the anchor
lane; `kakuna` keeps `plan`, `run_unit`, `smoke` and the manifest checks); `launch_x35.sh`'s copy of `pid_start`
(stage 1: it sources `_common.sh`, and the script moves to `scripts/ops/`); `kakuna.choose_port` (nothing calls it but
its own test); every stage-1 copy in §1.

## 3. Stage 0: what landed with this doc (pure refactors)

Behaviour is identical by construction, and proven: `src/utils/ports_test.py` keeps each replaced function VERBATIM
and checks, with a recording fake socket, that the new call makes the same syscalls in the same order with the same
result over every busy / claimed pattern of a 4-port range. The test fails on revert (a flipped host or reuse flag
fails 64 cases).

- `utils.ports` (new): kakuna's `choose_port` / `claim_port` (host `0.0.0.0`, no reuse, as before:
  `kakuna.PORT_HOST`), `anchors.server` (`_free` deleted, `RESERVED_PORTS` imported), `team_strength.cli`
  (`_port_free` deleted), `main.play` and `ws_frontend` (`RESERVED_PORTS` imported).
- `utils.filelock` (new): kakuna's claim and in-state `supervisor.lock`, `reads.runner._lock`,
  `reads.supervisors.acquire` (`"a+"`, so a refused caller still reads the holder's record).
  `supervisors_test` adds the in-state refusal: nothing is played or written, and the path lock is released.
- `utils.hashing` (new): `kakuna._sha`, `reads.models.sha256_file`, `x35.gc._file_sha` deleted; their callers
  and tests import `utils.hashing.sha256_file`.

**Not touched, on purpose:** everything the chain runs tonight (`src/main/ops/arm_chain/`, `src/main/x35/kit.py`,
`src/main/x35/batch.py`, `src/main/x35/refpop.py` and kit's other module-level imports, `launch_x35.sh`, `_common.sh`,
`utils.procfs`, `utils.gpu_lock`, `utils.gpu_lease`), and the box layer every heavy job runs through
(`utils.heavy_slots`, `utils.mem_cap`, `utils.gate_lock`). arm_chain's 222 tests pass unchanged.

## 4. The staged plan (not done)

Each stage is its own unit, with its own equivalence or regression tests. Before stage 1 and stage 3, check that no
chain item is mid-flight: the chain runs main's `kit` / `batch` from the main checkout.

1. **After the arm_chain cut-over has run one item cleanly.** `refpop.sha256_file`, `batch.sha256_of`,
   `canary._sha256` onto `utils.hashing`. `launch_x35.sh` sources `_common.sh` and moves to `scripts/ops/`
   (`arm_chain.launch` and its tests name the path). (The Python pid-liveness copies landed 2026-10-27; §1.) The fd-level `_try` in `heavy_slots` / `gate_lock` onto
   `utils.filelock` (an fd form). Split `src/main/x35/batch.py` and its test.
2. **When no read pinned before `de15dfc7` is live.** Delete the in-state supervisor locks and `kakuna run`'s queue
   (the anchor lane is `main.reads`). Every server-starting tool takes a `utils.ports.claim` in ONE shared lock
   dir: anchors' `pick_port` and the team-strength lane change from bind-and-release to claim, and kakuna's 0.0.0.0
   probe meets the others' 127.0.0.1, so this is a behaviour change. `utils.records` (`append_jsonl`, atomic JSON)
   replaces the `_jl` / `_atomic` copies; `reads.runner._atomic`'s fixed `.part` is a latent two-writer collision.
   One agent-cores constant. A `main.reads checkout` verb makes the pinned read checkouts.
3. **arm_chain idle between items, with the owner of the chain consulted.** Promote `ProcRef` and the
   launcher-argv live-run rule into `utils.procfs`, re-pointing `gpu_lease`, `gpu_window.protocol` and
   `launcher.worktree` at it. One `utils.ship` behind `/gen3ai-ship` and `arm_chain.ship`. One child-env helper.
4. **Repo-wide, low urgency.** The ~15 other sha256 copies; one `utils.git.run`; one interval-statistics module
   (Wilson / Newcombe / Welch), moved only with a bitwise equivalence test over a grid of (k, n) inputs, because
   these numbers steer verdicts.

## 5. Hazards

- **The play-path digest moved.** `src/main/x35/kakuna.py`, `src/main/anchors/server.py`, `src/utils/bridge/ws_frontend.py`
  and `src/main/reads/supervisors.py`
  are on the anchor lane's play path (`main.reads.playpath`), and stage 0 adds three modules to its import
  closure. A read pinned before stage 0 can no longer `repin` past it. The live N1-P read (pins `46b9e572` /
  `de15dfc7`) could already not repin to main, because the Rust sim fixes (`8dd36e9db`, `995c04a8a`) changed its
  play path first. Batch future play-path refactors so the digest moves rarely.
- **`launch_x35.sh` is live tooling inside a measurement folder.** Measurement folders read as frozen records, so an
  agent could leave a fix out of it or "restore" it.
- **The port probes disagree by design today.** Kakuna probes `0.0.0.0` without `SO_REUSEADDR`; anchors and the
  team-strength lane probe `127.0.0.1` with it, and they take no claim. A standalone `main.anchors` can pick a port
  a Kakuna unit has claimed but not yet bound. Stage 2 closes this.
