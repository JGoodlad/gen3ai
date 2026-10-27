# `main.reads` — one module for registered reads

Always current. Owns `src/main/reads/`. A **registered read** measures trained models against a rule written
before any counted game (a PREREG): an arm against its control, a final against a comparator, a model against an
external anchor. Before this module (2026-10-25) every read re-implemented the same seven pieces as one-off scripts
under `designs/research_state/measurements/<read>/`. The same bugs recurred (§8). New reads use this module. The old
folders stay as FROZEN records: their scripts are never edited or re-run as tools.

## 1. The interface

A read is one **spec** file (`read_spec.json`, committed with the PREREG) and one **state** directory. The state
lives outside the repo and outside `models/`, for example `~/gen3ai_archive/<read>/`.

```bash
python -m main.reads check-spec --spec read_spec.json              # a pin must name an existing commit (below)
python -m main.reads check-spec --spec read_spec.json --pin-pending   # review a spec BEFORE its pin is decided
python -m main.reads plan     --spec read_spec.json --state DIR    # copy settled models; plan units
python -m main.reads run      --state DIR --lane anchor --hours 20 # detached, resumable; one per lane
python -m main.reads run      --state DIR --lane anchor --hours 20 --concurrency 2   # K units at once (§3)
python -m main.reads run      --state DIR --lane s      --hours 20
python -m main.reads pid      --state DIR --lane s [--wait 180]    # the lane's LIVE supervisor pid, as it recorded itself (§3)
python -m main.reads status   --state DIR                          # .supervisor_pids (the pid to watch), .holds, progress
python -m main.reads hold     --state DIR --lane s --by WHO --why TEXT      # no new unit; those in flight finish; exit 0 (§3)
python -m main.reads resume   --state DIR --lane s --hours 20 [--by WHO]    # = run (after printing what is left); a HELD lane is released first
python -m main.reads repin    --state DIR --to COMMIT --by WHO --why TEXT   # another playing commit, if the play path is identical (§3)
python -m main.reads record   --state DIR [--out <stage>/reads/<arm>_look<k>.json]
```

`record` runs `assemble` (writes `cells.json`) and then `decide` (runs the registered VOID gate, §5, then writes
`inputs.json` and `decision.json`). Both are also verbs on their own. Exit 0 means success. Exit 2 is a typed refusal,
named on stderr. Exit 3 means R moved under an S lane. In Python, `main.reads.api` exposes the same verbs as functions.

**The spec** (`main.reads.spec`, schema `gen3_read_spec_v1`). Every field is validated, and an unknown field is a
refusal.

| field | holds |
|---|---|
| `models` | `{label: {arm, seed, ref, load?}}`. `ref` is `{"kind": "pinned", path, sha256}` (a frozen file) or `{"kind": "final", run, min_step, quiet_s, done_marker?}` (§2). `load` holds model-LOAD options every unit of the label passes (`spec.LOAD_OPTIONS`: `belief_prior_mix` λ ∈ (0, 1], `main.anchors --our-belief-prior-mix`), so two labels may play one file two ways |
| `anchor` | `{group: {opponent, teamsets, unit_seeds, models, import?}}`: the external-anchor cells (§3). `opponent` is one of `metamon:Kakuna`, `SmallRL`, `PastaMittens`, `LargeRL`, `LargeIL` (`kakuna.OPPONENTS`) |
| `s` | `{group: {refpop, refpop_content_sha256, panel, pairs, batch_pairs, family, models, regime_id?, import?}}`: S against a frozen R. **One eval-ledger family per group**, because a family holds one regime |
| `…import` | `{state, spec_sha256, group, models}`: these models' cells come from group `group` of an EARLIER read's state, never re-played here (§3) |
| `decision` | `{rule, params, allow_partial_at_stop?}`: a rule from `main.reads.rules.RULES`, with its registered parameters checked at load. `allow_partial_at_stop: true` registers that the PREREG accepts a reduced read at the registered stop (§6); it is refused for `x35_r1` and `d1` (their records are consumed by `main.x35.batch`, which does not read the class) and for a spec with no anchor group |
| `pin` | optional. The commit the playing checkout must be at, with a clean tracked tree. It must EXIST before the spec (the registration flow below) |

Example. This is x26c's read written as a spec; `golden_test.x26c_spec` parses it.

```json
{"schema": "gen3_read_spec_v1", "read": "x26c_continuation_2026-10-25", "pin": "e05c45eeb",
 "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": {"kind": "pinned",
              "path": "models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", "sha256": "8148…"}},
            "x26c_480m": {"arm": "x26c_480m", "seed": 2601, "ref": {"kind": "final",
              "run": "models/rb_x26c_s2601", "min_step": 480000000, "quiet_s": 300}}},
 "anchor": {"unfamiliar": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix", "forum72", "p72"],
                           "unit_seeds": "0-19", "models": ["pstar", "x26c_480m"]}},
 "s": {"pool": {"refpop": "designs/…/reference_population_production.json", "refpop_content_sha256": "cef0960e…",
                "panel": null, "pairs": 500, "batch_pairs": 100, "family": "x26c_s_pool_2026-10-25",
                "models": ["x26c_480m"], "regime_id": "adfdbee824c9eb0a"}},
 "decision": {"rule": "anchor_trajectory", "params": {"on": "x26c_480m", "off": "pstar",
   "anchor_group": "unfamiliar", "goal_bar": 0.5, "min_unit_frac": 0.9, "subset_below": 7,
   "s1": {"group": "pool", "label": "x26c_480m"}, "s_baseline": {"s_pp": 58.23, "se_pp": 0.55, "source": "…"},
   "k9b": "designs/…/k9b_at_end.json"}}}
```

How paths resolve: an absolute path is used as written; `models/…` resolves under the main checkout's run archive
(`utils.paths.main_models_dir`); anything else is repo-relative.

**The registration flow (one commit; `main.reads.speccheck`).** A spec cannot pin "the commit that adds this spec" (a
commit cannot name its own hash), and the belief-prior-mix read landed its PREREG and specs in two commits because it
pinned the PREREG's own commit. The pin is the commit that ALREADY lands the code the read plays, an ancestor of the
registration commit, and the PREREG, the spec(s) and any docs are not on the play path (§3), so they land after it
without changing what plays:

1. Land the code the read plays. Its commit is `P`.
2. Write the PREREG and the spec WITHOUT `pin`; review with `check-spec --pin-pending`. It validates everything else,
   prints `pin: PENDING`, the checkout's HEAD as the candidate and `sha256_final: false` (adding the pin changes the
   spec's sha256, which a PREREG or a later look's `import` may quote), and refuses a spec that already carries a pin.
3. Add `"pin": "<P>"`. `check-spec` now refuses a pin that names no commit of the repo and reports
   `play_path_differs_at_head` (0 = the code a reviewer reads at HEAD plays exactly as it does at `P`; otherwise the
   files).
4. Commit PREREG + spec(s) together, then play from a clean detached worktree at `P`
   (`git worktree add --detach <dir> <P>`; `plan --spec <the committed spec>` there). The spec file is read by path, so it
   need not exist at `P`. A verb added after `P` is not in that checkout: run the operator verbs (`status`, `hold`,
   `record`) from the newer one.

## 2. Which file is read (`main.reads.models`)

- **The settled rule.** A `final` ref reads `<run>/final_model.zip` and nothing else. It is settled when four
  things hold: the file exists, its recorded step is at least `min_step`, nothing has written it for `quiet_s`
  seconds (minimum 60), and the `done_marker` line is present if the spec names one. Until then the model is
  WAITING, and its units are not even planned.
- **No checkpoint fallback.** The x26c rule fell back to "the newest quiet checkpoint ≥ 480M". Twice, the Kakuna
  chain started on a checkpoint 81k steps short while the end chain was still writing the final (x26c FINDING 2).
  A read that wants a checkpoint names it as a `pinned` ref, with its sha256.
- **The copy.** Each settled model is copied into `<state>/ckpts/<label>/`. The checks are:
  - the copy must hash to the source, and the source to a pinned sha256;
  - the copy carries its `.json` sidecar and the run's `model_config.json`;
  - a missing `model_config.json` refuses at copy time (D1 look 2's void `h_d`, d1_read FINDING 4);
  - every unit re-hashes the copy before it plays.

## 3. Lanes, state, resume (`main.reads.runner`, `main.reads.layout`)

**The anchor lane** plays `main.x35.kakuna` units, through its own functions (`plan_units`, `run_unit`, `is_done`,
`MAX_ATTEMPTS`). A group's `opponent` rides every unit (`--opponent`). Kakuna and SmallRL play the cell's committed
config, their pins filled from the registry (ff54265e). An agent the committed config does not name (PastaMittens,
LargeRL, LargeIL) gets a DERIVED config, written once at plan into `<kakuna state>/configs/`: the committed config plus
that one agent, exactly the registry's entry. Each unit re-checks the derived file's sha256 and that it is still the
committed config plus that agent (`kakuna.check_derived_config`). The peer is offline on the pinned revision
(`HF_HUB_OFFLINE`, a260e9c3), and the registry's weights are the banked ones, so the regime id is unchanged by them.

- 20-game units on the heavy slots' short lane (`mem_cap.sh --max-s 600`). A unit the short lane kills past its
  bound (exit 88) and that is not counted is played ONCE more in the same attempt as a normal heavy-slot job
  (`mem_cap.sh 12 timeout 2700`, `kakuna.run_unit`, logged `short_lane_overrun`), as FR-3's queue did. A unit that has
  overrun goes straight to the long job on every later attempt, so a unit whose long retry also failed is not queued
  short again; `MAX_ATTEMPTS` (2) still bounds the attempts;
- 95xx ports claimed with a held flock;
- each unit preflighted: the T28 halt marker, the committed team manifest or panel cell, and the model bytes.

There is one kakuna state per (group, model), `<state>/anchor/<group>/<label>/`, planned when that model
settles. Unit seeds are the outer loop, so a stopped queue stays balanced by seed.

**Concurrency** (`--concurrency K`, anchor lane only, default 1; `main.reads.anchor_pool`; N1-P look 1 projected about
60 h for 486 units one at a time). SCHEDULING ONLY: the lane keeps up to K units in flight, and each is
`kakuna.run_unit` unchanged (its own short-lane `mem_cap` job, its own 95xx port from the held-flock claim, its own cell
dir and log), started on a thread that only waits on the child. Nothing on the play path (`main.x35.kakuna`,
`main.anchors`, the peers, model loading) knows about K.

- **The cap.** K is clamped to the box's heavy-slot count (`utils.heavy_slots.slot_count()`; `$GEN3AI_HEAVY_SLOTS`;
  2 by default, read at the supervisor's start) with a logged line. The supervisor log's `start` event records
  `concurrency` (effective) and `concurrency_requested`. `--lane s` with K above 1 is refused.
- **Order and results.** The queue order is unchanged (unit seeds outermost); the first K of it start together, and a
  unit is never in flight twice. Counting, the `MAX_ATTEMPTS` retry and the exit-88 long-lane fallback are the unit's
  own attempt and the supervisor loop's, as at K = 1. Results do not depend on K: the same units, seeds and per-unit
  records, only the order of completion differs (`anchor_pool_test` pins a byte-identical `cells.json` and unit
  records between K = 1 and K = 3).
- **Stops.** The registered stop starts no unit past it and waits for those in flight. A unit that raises (a preflight
  refusal, say) lets its siblings finish, then ends the lane with that refusal; the lane's claims are held until its last
  unit ended. SIGINT leaves the units running in their own sessions, as at K = 1.
- **Cost.** All K units share the agent cores (`5-7,13-15`, rule 5b), and each takes a heavy slot while a training run is
  live or the GPU is leased, so beside the S lane's slot a K above the free slots only queues FIFO. **UNVERIFIED:** the
  throughput gain over K = 1 is not measured; read units/h from `units.jsonl` (`start` to `end` `wall_s`) at K = 1 and
  K = 2 before relying on it.

**The S lane** plays `main.h2h play-many` cells:

- `pairs` mirrored pairs per cell, recorded in batches of `batch_pairs`, into the group's own ledger and family
  (`<state>/s/<group>/ledger`);
- one request per node, `<group>_<label>_b<batch_pairs>`;
- a unit is up to `--workers` engines in ONE heavy slot (`mem_cap.sh --slot`), time-boxed to `--unit-s` (default
  2,400 s), at nice 19 on cores 5-7,13-15;
- R is re-verified before every unit (every member's sha256 and `content_sha256`). If R moved, the lane stops
  with exit 3 (the drift pre-read's hazard 1);
- the box ends at the registered stop at the latest (the slot wait does not extend it); an engine that exits
  non-zero on its own is reported even when another hits the box; a unit that ends OK with no new pair is a
  named refusal, never a loop; a cell counts only at the spec's `pairs` (>= 500 for `x35_r1` / `d1`).

**Both lanes:**

- ONE supervisor per lane per state PATH (`main.reads.supervisors`; the D3 reader's duplicate, 2026-10-26: a state
  renamed aside took its in-state lock with it, and a second supervisor started on the state re-planned at the same
  path). A supervisor is refused when another holds the path lock (a flock on
  `~/.cache/gen3ai/reads_locks/<sha256(canonical path, lane)>.lock`, OUTSIDE the state; `$GEN3AI_READS_LOCK_DIR`
  overrides) or when a `/proc` scan finds a live `main.reads run|resume` whose `--state`, resolved against its own
  cwd, names the same path (this also sees a supervisor of older code, which took only the in-state lock, still
  taken too). A running supervisor stops, writing nothing, when its path no longer names the dir it started on (device
  and inode). `plan` refuses a FRESH state (no `spec.json`) that a live supervisor names. `status` lists the live
  supervisor pids per lane (the `/proc` scan; `supervisor_pids` is the supervisor's own record, below). The standalone queue `python -m main.x35.kakuna run --state S` takes the same claims through
  `supervisors.Hold` (lane `kakuna`: the path lock, the `/proc` scan, the dir-identity stop) beside its in-state
  `supervisor.lock`. A `kakuna run` aimed at a state INSIDE a reads state (`<state>/anchor/<group>/<label>/`) is not
  seen by the reads supervisors, and the reverse (§10);
- a registered stop (`--hours` or `--stop-at-epoch`, never open-ended);
- a counted unit or a recorded batch is never replayed;
- every child gets THIS checkout's `src` on `PYTHONPATH`;
- the playing checkout must stay at the commit `plan` wrote to `code.json` (or one a `repin` accepted, below), and a
  dirty tree at the same `git diff HEAD` digest. This addresses "the playing code is not
  the pin" (d1_read 5, sprt 1, g1 2, x47 F-X47L1-4), which before was only ever checked by hand.

**Repin** (`python -m main.reads repin --state S --to COMMIT --by WHO --why TEXT`; `main.reads.repin`,
`main.reads.playpath`). The pin guard refuses every commit but `code.json`'s, which also kept a scheduling fix to the
supervisor itself (`--concurrency`) from ever reaching a long read. `repin` accepts `COMMIT` as another playing commit of
the read IF AND ONLY IF the tracked **play path** is byte-identical between `code.json`'s commit and `COMMIT` (read from
git objects, not the working tree); otherwise it exits 2 naming the files that differ, and writes nothing.

- **The play path** (`playpath.play_files`, a deliberate over-approximation: a false "differs" refuses, a false "same"
  would be silent): the static import closure of `main.anchors`, `main.x35.kakuna`, `utils.bridge.ws_frontend`,
  `utils.mem_cap` and `utils.heavy_slots` (every `import` anywhere in a module, lazy ones included), plus
  `src/main/anchors/`, the front end, `src/agents/{model,inference,observation,gen3_data}/`, the Rust sources and build
  inputs that make `sim_bridge` (`src/rust_sim/src/`, `Cargo.*`, `build*.rs`), `data/`, the `deps/` pointers,
  `mem_cap.sh`, `anchors.json`, the panel registry and the committed cell configs and home72 teams. Tests and `*.md`
  are never on it. Within `main/reads/` only `__init__.py` and `supervisors.py` are ON it (`kakuna` reaches them), so editing
  either blocks a repin; every other module (`runner`, `api`, `cli`, `repin`, `playpath`, `anchor_pool`, `lanectl`,
  `speccheck`, …) is OFF it (`playpath_test` pins this). NOT covered, because git does not hold them: the Metamon cache and weights,
  panel team dirs outside the repo, the model copies, the toolchain. Each unit verifies those at play (team digests,
  weight pins, the model sha256).
- **What it writes.** One line per repin in `<state>/repins.jsonl`, APPEND-ONLY (`from_commit`, `to_commit`,
  `playpath_digest` and file count, `by`, `why`, `time`, the checkout `root`). `code.json` is left exactly as `plan`
  wrote it, so a supervisor of older code, still on the old checkout, keeps working through a repin (the S lane stays up
  while the anchor lane moves). A supervisor of this code accepts `code.json`'s commit or any repinned one, with the
  same clean-tree and diff-digest checks; a damaged `repins.jsonl` stops it rather than being skipped. Every repin
  compares with `code.json`'s commit, never with an earlier repin.
- **Edges.** A spec that registers a `pin` refuses (a repin never moves a registration). A repeat of an accepted commit
  is a no-op (`already: true`). `--by` and `--why` are required. `status` lists the repins, `record` carries
  `repins` (§6), and an import's record carries the earlier state's.
- **To move a lane to a newer checkout**: stop ONLY that lane's supervisor (`hold` it when `status` shows a
  `supervisor_pids` record for it; otherwise by PID; the other lane may keep running), check out `COMMIT` in the lane's
  playing worktree, `repin` from there, `resume` the lane from that checkout.

Run each lane detached, one process per lane, and wait on it through `/proc/<pid>` only, never `kill -0` (inside a
sandboxed agent shell `kill -0` read a live supervisor as dead, acf2193f). The pid to watch is the supervisor's own,
never `$!`:

```bash
setsid nohup python -m main.reads run --state DIR --lane s --hours 20 > DIR/s.log 2>&1 &
PID=$(python -m main.reads pid --state DIR --lane s --wait 180)             # NOT $!: that is the setsid wrapper's pid
timeout 3300 bash -c "while [ -d /proc/$PID ]; do sleep 30; done"           # one bounded watcher; re-arm on wake
python -m main.reads status --state DIR                                      # .supervisor_pids and .supervisors (the /proc scan)
```

**The supervisor's own pid** (`main.reads.lanectl`; belief-prior-mix FINDING 2). `setsid nohup … &` leaves `$!` naming the
`setsid` wrapper, which exits within seconds (a process-group leader makes `setsid` fork), so a watcher armed on it watched
nothing. `run` writes ITS OWN pid and start ticks (`/proc/<pid>/stat` field 22) to `<state>/<lane>/supervisor.pid` once it
holds the lane, and removes the file when it ends. `status` prints it as `supervisor_pids.<lane>`
(`{pid, start_ticks, alive, stale, stop_epoch, started}`; `alive` = that pid is running with exactly those start ticks, so a
reused pid is not taken for it, and a killed supervisor's leftover file reads `stale`); `pid --wait S` prints the live
pid, waiting at most S s (≤ 600) for a supervisor that is still importing.

**Hold and resume** (`main.reads.lanectl`; belief-prior-mix FINDING 5: a hold meant killing the supervisor by pid by hand).
`hold --state S --lane L --by WHO --why TEXT` writes `<state>/<lane>/hold.json` and appends to `<state>/holds.jsonl`
(APPEND-ONLY, with the release). The lane's supervisor looks for the file at the top of every loop turn (every 30 s while
it waits for a model to settle): it starts NO new unit, waits for the units in flight (all K of them) and counts them,
logs `held` and `held_stopped` to `<lane>_supervisor.jsonl`, and exits 0. A held lane is refused by `run` until released.
`resume` (= `run`, which already printed what is left) first RELEASES a held lane: it needs `--by` then, and refuses while
a supervisor of the lane is still alive (it is still draining; a second one would be refused by the path lock anyway).

- **A supervisor of code older than this cannot be held**: it writes no pid file and never reads the hold. `hold` lists the
  live supervisors as `honoring` (a valid pid file) and `not_honoring`, and warns on the latter, so a lane is never
  reported held when it is not. Such a supervisor is stopped by pid, once, and restarted from a checkout with these
  verbs (`repin` accepts that checkout when the play path is identical).
- A repeat `hold` is a no-op (`already: true`) that keeps the first `by` / `why`. A damaged `hold.json` is a refusal,
  never a free lane. A hold on a lane no supervisor is running only keeps `run` from starting one.
- The S lane notices a hold between units, not inside one (a unit is time-boxed to `--unit-s`).

**Cross-state import** (`main.reads.imports`; N1-P FINDING 6: look 2's spec re-played every look-1 final, about 60 %
of its CPU). A group's `import` names an earlier state, the sha256 of its `spec.json` (registered in this spec), its
group and the labels taken from it. Those labels are never planned or played here. `plan` refuses (exit 2) unless the
earlier spec hashes to the registered sha256 and its group is the same cell: the same opponent, with team sets and unit
seeds that include this group's (anchor); the same R content, panel, `pairs`, `batch_pairs` and registered regime (S);
each label with the same arm, seed and load options. `assemble` refuses unless the earlier state PLAYED the bytes this
state resolved (both `models.json` sha256s), on the ENGINE this state played on (§4). Imported anchor units then pool with this state's own only under one
regime id per team set (else the group is INCONCLUSIVE, as within one state); an imported S node is read from the
earlier ledger and family and must carry the group's regime. The record's `imports` names each import (state, spec
sha256, rows and their digest, the earlier playing commit; for S the family and ledger digest). `spec_sha256` is the
sha256 of the earlier spec FILE's bytes (`plan` copies it byte for byte into `spec.json`), so a later look registers it
as soon as the earlier spec is committed, with no need to wait for the earlier state; only the earlier STATE path must
be agreed in advance. Pin it with a test that hashes the committed earlier spec (`rules_test`, X61P look 2), since
editing the earlier spec moves it.

## 4. Assemble: the regime id is ALWAYS recorded (`main.reads.assemble`)

Five S drivers wrote `regime_ids: [""]` (x26c FINDING 4, g1 FINDING 3), and the regime was then checked by hand.
Here every group carries an id, or it is not conclusive.

**An S group's id** is the eval-ledger read's own (`LedgerRead.regime_id`), recorded with the read's digest. A group
is INCONCLUSIVE if any of these holds:

- it registers a `regime_id` and the ledger holds another;
- it names a `panel` and the rows' team set is a different one;
- a request has no row.

`s_node` is `readout.population_score` plus the per-member SE and se(S) = √Σse²/K.

**An anchor unit's id** is `anchor_regime_id`: a digest over the cell identity that `summary.json` stamps:

- the opponent and its version;
- both regimes;
- the transport and the server;
- the format and the forfeit limit;
- both team-content digests;
- the declared team cell;
- a Metamon peer's verified weights (`their_weights` / `our_weights`: hub revision + `weights_sha256`), included only when they
  differ from `assemble.LEGACY_WEIGHTS` (the one commit every pre-pin read played), so banked ids are unchanged and a
  re-read of the same weights keeps the id (`EXTERNAL_ANCHORS_SOP.md` § Weight pins);
- the ENGINE (`cell.engine_stamp`), included only when it is none of the banked engines (below);
- a Metamon peer's TEAM PACK (`peer_pack_changed` / `our_peer_pack_changed`), included only when True: the team set
  plays differently than under the legacy poke-env pack (below, "The Kakuna team pack").

Each (group, team set) must share ONE id. The group's id is `group_regime_id` over its team sets. The playing commit
and file paths are recorded per unit but are not part of the id; the engine the commit played on is. A unit played on bytes other than the resolved
model's is a refusal, never a pooled row. A model-load option changes the id through `our_regime` (a
`belief_prior_mix` unit is stamped `greedy+belief_prior_mix=<λ>`), so it never pools with the unchanged model. **One group
may hold models with DIFFERENT loads** (`anchor_group(..., loads=)`, from each `models.<label>.load`); the id stays
correct per (model, load) because a mixed group is read three ways: the counted units of one (load, team set) share one id;
across ALL its models a team set shares one BASE id (`base_regime_id` of each unit row: the id with the load's stamp
stripped, so the opponent, teams, transport and the rest are the same for every model); and a unit whose model declares a
load it did not stamp is not conclusive. The group id is then `mixed_group_regime_id` over the base ids and each load's ids;
the group also carries `regime_ids_by_load` and `loads`. A group whose models share one load (or none) is read as always,
with the id it always had. `prior_mix_ab` still wants one group per arm (its control is banked). Every unit
row also carries its `games` (`[half, index, our_team, result, turns]` from `games.jsonl`), which is what a
game-for-game determinism guard compares.

### The engine (`main.reads.engine`; REVIEW F F-M2, 2026-10-26)

Kakuna units and S cells play on our in-process Rust engine. An ENGINE is the build-stamp refusal key
(`utils.rust_env.stamp`):
- **A unit's:** `port=<src>/<n>;env=<src>/<n>`, for `sim_bridge` and the env core that our `rust_core_slot` side loads.
  A summary of this code stamps it at play.
- **An S row's:** `env=<src>/<n>`, from `compute.core_stamp`.

Every unit row and group carries `engine`. A group whose counted units, or an S family whose rows, span two engines is
INCONCLUSIVE.

**The LEGACY rule.** A banked summary stamps no engine, so its id is unchanged, re-checked for every unit of D3 look 1
and the belief-prior-mix read. Its engine is `LEGACY_ENGINE` of its playing commit, a FROZEN table reconstructed from
git (`engine_at`: the stamp's listing at that commit, plus the untracked `Cargo.lock` blob `LEGACY_LOCK_BLOBS` names).
- The reconstruction reproduces the core stamps the ledger RECORDED at 6c428453, 1ed71c9dd and 46b9e572
  (`engine_test`).
- A new read on a banked engine keeps that engine's id; any other engine enters the id.
- Ids cannot tell two banked engines apart, so every import compares the engines themselves: the cross-state import,
  `import_off` (anchor and S), `prior_mix_ab`'s bank, the determinism guard (it must play on the bank's engine) and
  `x35_r1`'s imported S.
- An import across engines REFUSES, and so does one from an unknown engine. A bank that names no engine per row is
  read by its registered `engines` or `LEGACY_BANK_ENGINE` (by sha256).
- `repin` names the engine at both commits and refuses if they differ. The play path holds every tracked file the
  stamp hashes (`playpath_test`).

| engine | port / env | banked reads (playing commit) |
|---|---|---|
| E0 | `05b106e2…/84` / `dfd24aec…/119` | X51 and x37a (`63f97a575`), `cb0f1769c`, `acb99527`. X51's `k28v34_home72` rows: `b9c758841`, E0B `e64b127b…` / `b92ea7ad…` |
| **E1** | `7d932165…/90` / `7c532805…/125` | x26c (`e05c45eeb`); D1 H (`6c428453a`, `5e389ddac`) and P; X52 (`f4b25238d`); SPRT (`eb99ab362`) and G1 (`3f3a2ed41`) S; human-only vs P\* and metamon-strongest (`d32ffcee4`, `4b0ab220e`); PastaMittens (`62bb07c00`); belief prior mix (`4ee655648`; smoke `033fcbcd2`). **The X35 stage's training engine** (`f4b25238`, D3_PIN `9e289ba4`) |
| E2 | `806f97bd…/90` / `9ecb4249…/125` | D3 look 1 (anchor `2cc864f2f`, S `1ed71c9dd`) |
| E3 | `0d3d7629…/91` / `393631b8…/126` | N1-P look 1 (`46b9e572c`; repin `de15dfc7b`): one engine throughout |
| main | `67a272e7…/93` / `d79c3a74…/128` | 2026-10-26 HEAD, after the full-parity batches; no banked read |

**Mixed-engine comparisons in shipped records.** Each was guarded game for game. No verdict is shown to be affected.
1. **D3 look 1** (CONTINUE): ON was played on E2 against C0 banked from D1 on E1 (H, and P). The E1 → E2 Rust diff is
   `f9978586b` (lock-in and recharge counters) and `d8d674b99` (encoder input fixes). The guard reproduced 140 / 140
   games, which bounds divergence at < 2.1 % of games (95 %, three over 140) against Δ̂ = +1.27 pp.
2. **Belief prior mix** (NOT DETECTED, both arms): the arms were played on E1 against X51's banked λ = 0 on E0. The
   guard reproduced 80 / 80 games, and x26c FINDING 1 re-played P\* on E1 with X51's wins on all 60 units.

3. **Metamon strongest** (PastaMittens; Δ-U −0.25 pp, NOT DETECTED) and **human-only vs P\*** (FR-1 / FR-3; Δ-K
   +0.14 to +0.30): both were played on E1 and read Δ against X51's Kakuna rows on E0. The identity unit reproduced
   20 / 20 games. The human-only Δs are far beyond any engine bias.

Re-decided by this code, items 1 and 2 refuse at `inputs`; item 3's scripts predate `main.reads`. The recorded
verdicts stand as recorded.

### The Kakuna team pack (`gen3_peer_showdown_pack_v1`; `EXTERNAL_ANCHORS_SOP.md` H22, 2026-10-27)

🚨 **A regime split beside the engine's, on forum72 (goal (b)'s primary panel), away and the pool.** Before it the
Metamon peer packed its team files with upstream poke-env, which is not Showdown's reading of a paste. On these sets
Kakuna played other sets than the files say:

| team set | the file | Kakuna played (legacy pack) | Kakuna plays now |
|---|---|---|---|
| **forum72** (and `tsr_k_forum72`) | `sft431` Gyarados, `Happiness: 0`, Frustration | Frustration at base power **1** (happiness dropped to 255) | base power 102 |
| away (`competitive`, `tsr_k_away`) | `team_23` Gyarados, Hidden Power Flying, no IV line | IVs 31/31/31/30/30/30 | the validator's 30/30/30/30/30/31 |
| the pool (`gen3ai_pool`) | 4 teams: 2 Frustration sets (an explicit `Happiness: 0`, or none, which Showdown's import reads as 0 for Frustration) and 2 `Hidden Power: <Type>` lines on mons without the move | Frustration at 1; all-31 IVs | 102; the type's IVs |

Unchanged (the legacy pack played every file as Showdown does): home72, p72, heldout72_hpfix, rc72b, the x37a and
`tsr_k_*` sets except the two above, `tsr_ref24`.

The peer now packs every file through Showdown at startup and refuses a file the pack oracle disputes. Its report's
`peer_pack_changed` is True iff any file of the set it may draw plays differently from the legacy pack, and
`anchor_regime_id` adds it to the id only then (`assemble.peer_pack_identity`). So an unchanged set keeps its banked id
and pools as before. A changed set gets a new id: old forum72 / away / pool Kakuna reads stay valid as their own regime
and are never pooled with new ones. Machine-checked:
- within one state, two ids on a (group, team set) are INCONCLUSIVE, as always;
- a cross-state import whose counted units differ from this state's in `peer_pack` on a shared team set REFUSES
  (`imports.check_peer_pack`, `imports_test`);
- `import_off` refuses on the regime id it registers.

**D3 look 2** imports C0's banked rows, so its pin must predate this change, as its E1 requirement already implies;
a pin carrying it refuses that import on forum72. The live reads at the change (N1-P look 1 at `de15dfc7b`, H-CTX
rc72b at `905c29f5b`, X59-S at `fca58b24d`) run from their own pinned worktrees and are untouched.

**Reads in flight or planned:**
- **D3 look 2** (the orchestrator's ruling, 2026-10-26): EVERY game on **E1**, the engine both arms trained on.
  - D3 ON's look-1 units are re-played on E1, and the look-2 units are played on E1.
  - C0's banked rows (E1) then import cleanly.
  - No commit carries both E1's Rust and `import_off` (it landed at `2cc864f2`, already on E2). So the read needs a
    pin whose engine is E1 (`engine_at(<pin>) == E1`, checked before the spec registers it).
- **X59-S games / X60 / X61P look 2:** one engine per read, named at registration. An import from N1-P look 1 needs E3.
- **The human-only / PastaMittens reads:** each was played on one commit, all on E1.

## 5. Decide: the registered rules (`main.reads.rules`)

A rule is two pure halves: `inputs(cells, params, spec)` produces the rule's input document, and `decide(doc,
params)` produces the verdict as recorded. A decision can therefore be re-checked from `inputs.json` alone.

| rule | is | from |
|---|---|---|
| `x35_r1` | X35 R1 / R2 / guards / verdict on S per seed final (SPRT, LRF, A1 …); `import_off` takes a banked control from another read's committed `results.json`, sha-checked | `main.x35.readout.read` |
| `d1` | D1's H (the arm's primary Kakuna panels, `d1.arm_panels`: D1's / D3's three, X61P's `forum72` alone at 60 units; widened by AM-5's φ̂) and P (S on home72); an optional `reported_group` (an anchor group disjoint from the primary, same models) adds per-panel rows, U and Δ_H − Δ_F to `reported.reported_panels`, which no rule reads; `reported_without` re-computes H's statistic with named panels dropped (`reported.without`); `import_off` pools a BANKED OFF side from another read's committed d1 input (sha-checked, regimes matched, refused if the OFF arm also plays), its Kakuna rows only when `determinism_guard`'s group re-plays committed banked units game for game (D3) | `main.x35.d1.read_d1` |
| `anchor_trajectory` | x26c PREREG §5: the goal bar on the ON score's Wilson LB; CLIMBING / FLAT on the Newcombe-of-strata Δ; S1 RISING against a banked node; VOID on K9(b) or under-counted units | ported from x26c `verdict.py` |
| `prior_mix_ab` | the belief-prior-mix read (`measurements/belief_prior_mix_2026-10-26/PREREG.md` §5): each λ arm (its own anchor group, one regime) against a BANKED λ = 0 control (another read's committed per-game rows, sha-checked, never re-played; `banked_units`), unit-paired on the same seeds; VOID unless a guard group re-playing banked units at λ = 0 reproduces them GAME FOR GAME; PRIOR-HELPS / -AT-A-COST / -HURTS / NOT DETECTED; D0f's DiD and H3 kill rule on a λ = 1 arm | new (2026-10-26) |
| `score_bands` | the H-CTX read (`measurements/hctx_rc72_2026-10-26/PREREG.md` §5): ONE model's pooled score on ONE panel placed in two registered bands (`>= kill_at` KILL, `<= support_at` SUPPORT, else UNDECIDED), compared as exact fractions (no rounding error sits on a boundary); VOID on `prior_mix_ab`'s game-for-game guard against banked rows or under-counted units; Newcombe against the banked familiar cell and the banked unfamiliar mean, descriptive; superseded for its read by `premium_recovery` before any game | new (2026-10-26) |
| `premium_recovery` | the H-CTX read as amended (`measurements/hctx_rc72_2026-10-26/PREREG.md` Amendment 1, the engine-identity rule): a panel (rc72) and its familiar (home72) and unfamiliar (F) controls played on ONE engine in one read, interleaved; KILL / SUPPORT on the share of the premium (familiar − unfamiliar) the panel recovers, by exact fractions; INCONCLUSIVE unless the premium's unit-bootstrap lower bound clears 0; VOID on two engines or an unknown one; absolute bars against a banked read only as a secondary, cross-engine line | new (2026-10-26) |

**The estimators are not re-derived.** Each is the one function a read already used, re-exported under a name:

- `welch_interval` and `classify` (readout);
- `strat_score`, `newcombe_strat`, `overdispersion`, `widen` and `wilson_pn` (d1);
- `wilson` and `newcombe` at z = 1.959963985 (anchors);
- `newcombe_diff_ci` at z = 1.96 (`best_response_gap`);
- rule 8 as `clears`.

Migrated here from the scripts, which had no home in `src/`:

- `cell_se` and `s_se`;
- `paired_unit_diff`;
- `guards_not_read`, the orchestrator's A1 ruling (a): guards gate only a QUALIFYING primary, so an unplayed guard on
  a non-qualifying primary is recorded INCONCLUSIVE with the read's reason.

A new rule is a new `Rule` entry with a test. It never replaces an existing rule's numbers.

**The VOID gate** (`rules.void_gate`; be60f24f FINDING 3). An X35-stage rule (`x35_r1`, `d1`) may register
`params.void = {k9b_dir, runs: {label: run dir}}`, every ON model named; `x35_r1` on a batch-2 arm (SPRT, LRF) MUST
(the spec is refused without it). `decide`, and so `record`, runs it FIRST and writes `void.json`: each run's AM-6
K9(b) record (`<k9b_dir>/<run>.json`) must read PASS, and on a batch-2 arm its V-S1 / V-S2 (SPRT) or LR-F KL guard
(`main.x35.batch2_void.read_void`) too. Any other verdict refuses the decision and names the seed: a VOID seed enters no
read (SPRT's `assemble.py` STOP). The record carries the gate, and a record of a batch-2 arm without one is refused.
The gate reads K9(b) records and the batch-2 guards only. A read whose PREREG lists other VOID guards (N1-P's §9: the
startup banner, the pin and corpus, the pool seed, a corpus-team crash) registers K9(b) here and keeps the rest as
named operator checks in its PREREG amendment (AM-N1P-1, FINDING 11).

**Reported-only groups** (`Rule.reported`; N1-P FINDING 7). A group a rule carries for description and no verdict
reads (`d1`'s `reported_group`) never blocks the record of the decided primary. When it is short (unplayed units, no
regime id yet), the decision's `reported.reported_panels` carries `status: INCOMPLETE` and its reasons, and the record
lists it under `reported_incomplete`. A short PRIMARY group still refuses the record.

## 6. The record (`main.reads.record`, schema `gen3_read_record_v1`)

The record holds:

- `read`, `rule`, `spec_sha256`, `prereg`, `verdict` and `created`;
- `models` (each with the sha256 played);
- `regimes` (`{anchor: {group: id}, s: {group: id}}`);
- `ledgers` (each S group's family, digest and row count);
- `code` (the playing commit) and `repins` (each other playing commit a `repin` accepted, §3; `[]` when none);
- `cells_inconclusive` (the decided groups only);
- `reported_incomplete` (`{"<lane>.<group>": [reasons]}`: reported-only groups that are short, §5);
- `completeness` (`{"class": "COMPLETE"}`, or `INCOMPLETE_BY_STOP`, below; absent on a record older than the field);
- `void` (the VOID gate as run, §5, or null);
- `imports` (each group's cross-state import, §3);
- the rule's `inputs` and `decision`.

**A reduced read at a registered stop** (`decision.allow_partial_at_stop: true`; belief-prior-mix FINDING 8: `record` needed
every planned unit played even when the rule's `min_unit_frac` would decide). When a decided ANCHOR group has units
unplayed and the spec registered the option, `record` writes the counted units with `completeness.class` =
`INCOMPLETE_BY_STOP`, `opt_in: "decision.allow_partial_at_stop"`, `groups` (`planned`, `played`, `unplayed`, `counted` per
group) and `stops` (the lane's `stop_at` event: its time, `left`, the registered `stop_epoch`); `summary_line` names the
class. It does so only if each such lane ENDED at its registered stop (`lanectl.stop_evidence`: no supervisor of it alive
and the last event of `<lane>_supervisor.jsonl` is `stop_at`, which a supervisor writes only once its `--hours` /
`--stop-at-epoch` has passed); a lane never run, held, killed, refused or done is refused, naming its last event. Only the
unplayed units are exempted: a wrong regime, an S node short of its `pairs` or a failed VOID gate still refuse, and the
RULE's own floor still decides (`min_unit_frac` makes the verdict VOID). Without the option `record` refuses as always.
Units that exhausted their attempts are unplayed too; the record cannot tell them from units the stop cut off.

The record is refused if any regime id is missing (a reported-only group listed INCOMPLETE excepted), a model was
never resolved, a ledger digest or the playing commit is missing, `cells_inconclusive` is not empty (a verdict decided
on cells assemble could not count: a wrong regime or panel, unplayed units unless the reduced read above, a cell short of the spec's `pairs`), the
VOID gate holds a non-PASS run (or a batch-2 arm's record carries none), or the verdict is not the decision's. An X35-stage rule (`x35_r1`, `d1`) also carries the
decision's keys at the top level. When written to `<stage>/reads/<arm>_look<k>.json`, the file therefore loads in
`main.x35.batch.read_record` and `final_class` unchanged. A ledger entry cites the record by its path and
`record.summary_line`.

## 7. Golden reproductions (`src/main/reads/golden_test.py`)

Three shipped reads, re-decided by the module, reproduce their recorded verdicts and intervals EXACTLY, by float
equality.

| read | recorded | committed half (always runs) | archive half (owner's box) |
|---|---|---|---|
| SPRT look 1 | NULL, Δ −0.25 pp [−1.65, +1.16] | the input → `x35_r1`: `r1`, `r2`, `guards`, `reasons` and verdict equal `reads/sprt_look1.json` | re-assembled from `~/gen3ai_archive/x35_sprt_read/ledger`; C0 from `x35_g1/ledger` equals the imported `g1/results.json`; regime `adfdbee824c9eb0a`; the record loads in `batch.read_record` |
| D1 look 2 | UNDECIDED, H +2.10 pp [−0.92, +5.11] | the input → `d1`: `h`, `p`, `guards` and verdict equal `reads/d1_look2.json` | H from the five kakuna states `h_a`…`h_e`, P from `p/ledger`, both equal the committed input; regimes: Kakuna `09f297883701a35a`, home72 `d4de0fd7c16a2658` |
| x26c | goal (b) not met; FLAT; Δ +0.08 pp [−3.90, +4.06]; S1 −0.01 [−1.54, +1.52] | the committed unit rows + S1 → `anchor_trajectory`: the `kakuna` block and every decision value equal `verdict.json` | the two kakuna states reproduce the committed `kakuna_units.jsonl` row for row; the three S nodes equal `s_results.json` (S, se, CI, per-member); regimes recorded: Kakuna `09f297883701a35a`, pool `adfdbee824c9eb0a`, home72 `d4de0fd7c16a2658` |

D1's and x26c's Kakuna units carry the SAME group regime id. They were played at different commits on the same
committed configs, on the SAME engine (E1, §4). This matches x26c FINDING 1.

The archive half reads `~/gen3ai_archive` read-only (`$GEN3AI_READS_ARCHIVE` overrides). Where it is absent, it
skips and names why.

## 8. The inventory it replaces, and the bug classes it closes

| read (frozen) | scripts | pieces it hand-rolled |
|---|---|---|
| X26 drift pre-read | `x35_breakers_stage_2026-10-17/x26_drift_preread/preread.py` | S cells on slots, copy_checked, se(S), ledger read |
| G1, D1 (P, pool S), SPRT | `g1/g1_read.py`, `d1_read/d1{p,s}_read.py`, `sprt_read/s_read.py` | near-verbatim copies of the pre-read machinery |
| D1, SPRT assembly | `d1_read/assemble.py`, `sprt_read/assemble.py` | merging kakuna states and requests, the A1 guard ruling, imported C0 |
| x26c | `x26c_continuation_2026-10-25/{finals,s_read,verdict}.py` | the final-selection rule, S cells, the trajectory rule |
| X47 look 1 | `x47_look1/{play_cells,assemble_read}.py` | per-cell play-many, population score |
| Kakuna-era queues | `x26_metamon_anchor/xq.py`, `x51…/x51q.py`, `x52…/x52q.py`, `x37a…/xq.py`, `heldout*/hq*.py`, `k2…/kq2.py` + their `aggregate.py` | ports, units, Wilson / Newcombe tables |
| Metamon strongest, human-only FR (RUNNING 10-25; untouched) | `metamon_strongest_2026-10-25/scripts/{mmq,aggregate}.py`, `human_only_vs_pstar_2026-10-25/scripts/{hrrq,aggregate}.py` | copies of `claim_port`, other opponents' units, per-read classes |

| bug class (where recorded) | recurs | closed by |
|---|---|---|
| the final-selection race (x26c F2) | three readiness rules in use | the settled rule, no checkpoint fallback (§2) |
| hand-counted `parents[N]` (x26c F3; xq.py `parents[5]`) | copies at different depths | `utils.paths` only; no script paths |
| empty `regime_ids` (x26c F4, g1 F3; 5 copies) | yes | regimes always recorded; a record without one refuses (§4, §6) |
| port race (d1_read F3; x51 F6) | `claim_port` re-copied in mmq / hrrq | kakuna's held-flock claim, reused |
| copied finals without `model_config.json` (d1_read F4; refpop F-R4) | yes | refused at copy time (§2) |
| R moving under a read (drift hazard 1) | every S driver | re-verified before each unit, exit 3 (§3) |
| playing code ≠ pin (d1 5, sprt 1, g1 2, x47 L1-4, hr 1, x51 9) | the most common | `code.json` + `pin` (§3) |
| heavy-slot starvation (d1_read F2; drift 2-3; x51 F5) | yes | short lane for Kakuna units; time-boxed S units in one slot each |
| rule-8 and statistic copies (≈10 scripts' `EPS`; x26c's `clears`, hand-copied baselines) | yes | one `clears`; baselines in the spec with their source |
| a missing panel argument to `strat_score` (x26c `verdict.py`, correct only because its panels were d1's default) | latent | panels always passed (rules_test caught it) |
| two supervisors on one read (D3 reader 10-26: the state renamed aside took its in-state lock) | first live use | the path lock outside the state, the `/proc` scan, the dir-identity stop, `plan`'s refusal (§3) |
| a later look re-plays the earlier look's finals (N1-P F6) | every multi-look read | the sha-checked, regime-matched cross-state import (§3) |
| a short reported-only group blocks the primary's record (N1-P F7) | X61P's spec | reported groups recorded INCOMPLETE (§5) |
| a VOID seed read (SPRT's STOP, absent from `x35_r1`; be60f24f F3) | latent | the registered VOID gate before `decide` (§5) |
| a short-lane overrun (exit 88) left uncounted (FR-3's queue re-ran it once by hand; the module did not) | yes | the once-only long retry (§3) |
| two standalone `kakuna run` queues on one state | latent (the same lock-inside-the-state flaw) | the path lock and `/proc` scan, shared with the lanes (§3) |
| the anchor lane plays one unit at a time (N1-P look 1: 486 units, about 60 h projected) | any long Kakuna read | bounded `--concurrency K`, scheduling only (§3) |
| a scheduling fix cannot reach a running read (the pin guard refuses any other commit) | any long read | `repin`, only when the play path is byte-identical (§3) |
| a watcher armed on `$!` of `setsid nohup … &` (the wrapper's pid; belief-prior-mix F2) | any detached lane | the supervisor's own `supervisor.pid`, `status`, `pid` (§3) |
| a hold meant killing the supervisor by pid (belief-prior-mix F5) | any read that waits on another | `hold` / `resume` (§3) |
| a read cut at its registered stop could not be recorded (belief-prior-mix F8) | any long read | the opt-in `INCOMPLETE_BY_STOP` record (§6) |
| the pin had to exist before the spec, so PREREG and specs took two commits (belief-prior-mix F8) | any pinned read | the one-commit registration flow, `check-spec --pin-pending` (§1) |
| a load option needed its own anchor group (belief-prior-mix F3) | `anchor_trajectory` / `d1` with an option on one side | a regime id per (model, load) in one group (§4) |

## 9. What was migrated, wrapped, left

- **Wrapped, numbers unchanged:**
  - `main.x35.readout` (R1 / R2 / guards / verdict);
  - `main.x35.d1` (H / P / φ̂, `refpop_members`);
  - `main.x35.kakuna` (units, ports, preflight, counting);
  - `main.anchors.results` (`wilson`, `newcombe`, `unit_counts`).

  `main.x35.batch.read_record` loads the module's X35-stage records as they are.
- **Migrated:**
  - from the per-read scripts: the S-cell runner, `copy_checked`, the per-cell and per-final SE, the trajectory rule,
    the unit-paired difference, the A1 ruling and the final-selection rule;
  - from `d1.p_input` / `kakuna.d1h_input`: their equivalents over assembled cells (`rules.p_input`,
    `rules.h_input`).
- **Left:**
  - every frozen measurement folder;
  - the reads RUNNING on 2026-10-25 (Metamon strongest, the human-only FR-1 / FR-3 reads, the X59-S build, D3's
    arm-end). They finish on their own scripts.

## 10. Not built (each a FINDING of the build)

- **Reported-only extras** (SPRT's mechanism block, dose from launcher summaries) stay per-read. A rule's `params`
  can carry them, but only the VOID gate reads run dirs.
- **A repin's reach.** `repin` proves the TRACKED play path identical (§3). It cannot see what git does not hold (the
  Metamon cache, panel team dirs outside the repo), and its closure is static (a module named only in a string is not
  followed; the prefixes over-approximate for that). A change in the box's untracked inputs is the units' own preflight's.
- **Nested supervisors.** A `kakuna run` on `<state>/anchor/<group>/<label>/` and a `main.reads run --lane anchor` on
  `<state>` can both play that kakuna state: the lanes call `run_unit` without taking the sub-state's
  `supervisor.lock`, and neither path scan looks at a parent or child path.
- **The `d1` VOID gate is K9(b) only.** X61P's specs register it (AM-N1P-1); the rest of its PREREG §9 (banner, pin and
  corpus, pool seed, corpus-team crash) is checked by the operator from the kit's launch records and the launcher log,
  which lives outside the run dir. A module guard for them would read `pin_history` and `snapshots/pool_seed.json`.
- **The always-run D1 golden re-decides only.** `rules.h_input` / `p_input` are compared with the shipped
  input only in the archive half (D1's per-unit rows are not committed); off the owner's box that half skips.
- **The tests stub play.** The verbs are tested with a stubbed unit runner and stubbed heavy slots (`runner_test`,
  `pipeline_test`, `imports_test`); the play-many and anchor argv are checked against the real parsers, and each
  derived opponent config against a real `main.anchors --dry-run`. The live reads (D3 look 1, N1-P, from 2026-10-26)
  are the end-to-end runs; their first uses found the live-use classes in §8's later rows.
- **The belief-prior-mix read's rough edges** (2026-10-26, `measurements/belief_prior_mix_2026-10-26/README.md` FINDINGS 2, 3, 5,
  8) are closed by `pid` / `hold` / `resume` (§3), the one-commit registration flow (§1), mixed loads in one group (§4) and
  the `INCOMPLETE_BY_STOP` record (§6). What they leave: a supervisor of code that predates `hold` cannot be held (§3);
  a reduced read is the anchor lane's alone, because an S node short of its `pairs` has no S, and `x35_r1` / `d1` refuse
  the option; a lane stopped by `hold` is not a registered stop, so it cannot be recorded partial until it runs on to one.
