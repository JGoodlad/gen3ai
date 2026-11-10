# `main.reads` — one module for registered reads

Always current. Owns `src/main/reads/`. A **registered read** measures trained models against a rule written
before any counted game (a PREREG): an arm against its control, a final against a comparator, a model against an
external anchor. Before this module (2026-10-25) every read re-implemented the same seven pieces as one-off scripts
under `designs/research_state/measurements/<read>/`. The same bugs recurred (§8). New reads use this module. The old
folders stay as FROZEN records: their scripts are never edited or re-run as tools.

## 1. The interface

A read is one **spec** file (`read_spec.json`, committed with the PREREG, or with a light screen's one-page
`SCREEN.md`, which the spec's `prereg` then names: `designs/research_state/README.md` § Registration tiers; every
guard and rule below applies to a screen unchanged) and one **state** directory. The state
lives outside the repo and outside `models/`, for example `~/gen3ai_archive/<read>/`.

```bash
python -m main.reads check-spec --spec read_spec.json              # a pin must name an existing commit (below)
python -m main.reads check-spec --spec read_spec.json --pin-pending   # review a spec BEFORE its pin is decided
python -m main.reads plan     --spec read_spec.json --state DIR    # copy settled models; plan units
python -m main.reads run      --state DIR --lane anchor --hours 20 # detached, resumable; one per lane
python -m main.reads run      --state DIR --lane anchor --hours 20 --concurrency 2   # K units at once (§3)
python -m main.reads run      --state DIR --lane anchor --hours 20 --max-units 40   # start 40 units, finish, stop (§3)
python -m main.reads run      --state DIR --lane s      --hours 20
python -m main.reads pid      --state DIR --lane s [--wait 180]    # the lane's LIVE supervisor pid, as it recorded itself (§3)
python -m main.reads status   --state DIR                          # .supervisor_pids (the pid to watch), .holds, progress, .exhausted
python -m main.reads hold     --state DIR --lane s --by WHO --why TEXT      # no new unit; those in flight finish; exit 0 (§3)
python -m main.reads resume   --state DIR --lane s --hours 20 [--by WHO]    # = run (after printing what is left); a HELD lane is released first
python -m main.reads repin    --state DIR --to COMMIT --by WHO --why TEXT   # another playing commit, if the play path is identical (§3)
python -m main.reads release-guard-hold --state DIR --by WHO --why TEXT   # the orchestrator's ruling on a HELD decide (§4)
python -m main.reads record   --state DIR [--out <stage>/reads/<arm>_look<k>.json]
python -m main.reads search-cost --state DIR | --cells DIR...     # a searched read's cost and G-PREFIX (§3)
```

`record` runs `assemble` (writes `cells.json`) and then `decide` (runs the registered VOID gate, §5, then writes
`inputs.json` and `decision.json`). Both are also verbs on their own. Exit 0 means success. Exit 2 is a typed refusal,
named on stderr. Exit 3 means R moved under an S lane. In Python, `main.reads.api` exposes the same verbs as functions.

**The spec** (`main.reads.spec`, schema `gen3_read_spec_v1`). Every field is validated, and an unknown field is a
refusal.

| field | holds |
|---|---|
| `models` | `{label: {arm, seed, ref, load?}}`. `ref` is `{"kind": "pinned", path, sha256}` (a frozen file) or `{"kind": "final", run, min_step, quiet_s, done_marker?}` (§2). `load` holds model-LOAD options every unit of the label passes (`spec.LOAD_OPTIONS`: `belief_prior_mix` λ ∈ (0, 1], `main.anchors --our-belief-prior-mix`; `search` = a registered search `<preset>@<turn>`, `main.anchors --our-search`, e.g. S-U's `r0s4@10`; `input_log` = a turn window `FROM-TO` whose decisions' honest input logs a RAW label's units keep, `main.anchors --keep-input-logs`, the S-U gate test's states: no play change, no stamp, never with `search`), so two labels may play one file two ways |
| `anchor` | `{group: {opponent, teamsets, unit_seeds, models, per_game?, games?, import?}}`: the external-anchor cells (§3). `opponent` is one of `metamon:Kakuna`, `SmallRL`, `PastaMittens`, `LargeRL`, `LargeIL` (`kakuna.OPPONENTS`). `per_game: true` plays each seed's 20-game series as 20 ONE-GAME units (§3, "Per-game units"); a group holding a `search` model must be `per_game`, import nothing, and hold exactly one RAW TWIN per searched label (the same `ref`, the same other load options: `spec.search_twins`). `games` (`per_game` only) = `{"<unit seed>": [game, ...]}` for EVERY unit seed: the group plans, counts and imports only those games (X65's G-BANK) |
| `s` | `{group: {refpop, refpop_content_sha256, panel, pairs, batch_pairs, family, models, regime_id?, import?}}`: S against a frozen R. **One eval-ledger family per group**, because a family holds one regime |
| `…import` | `{state, spec_sha256, group, models}`: these models' cells come from group `group` of an EARLIER read's state, never re-played here (§3) |
| `decision` | `{rule, params, allow_partial_at_stop?}`: a rule from `main.reads.rules.RULES`, with its registered parameters checked at load. `allow_partial_at_stop: true` registers that the PREREG accepts a reduced read at the registered stop (§6); it is refused for `x35_r1`, `d1` and `n4_screen` (their records are consumed by `main.x35.batch`, which does not read the class) and for a spec with no anchor group |
| `pin` | optional. The commit the playing checkout must be at, with a clean tracked tree. It must EXIST before the spec (the registration flow below) |
| `rederived_guards` | optional, `count` or `report`: how assemble acts on the anchor guards it re-derives (§4). Absent: `report` for a spec registered before 2026-10-27's change (`rederive.REGISTERED_BEFORE`, the sha256 of every spec on file then), else `count`; `check-spec` prints the resolved value |
| `choices_guard` | optional, `require` (default) or `five_tuple`: a game-for-game guard whose expected capture is missing is INCONCLUSIVE, unless the spec registers that its banks lack captures (§4) |

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
   A value that cannot exist yet (X65: the students' zips and sha256s before its pin amendment, an earlier look's
   final spec sha256, a VOID gate's `void.pin` before the training pin exists) is written as the literal `"PENDING"`
   (`spec.PENDING`): `--pin-pending` lists every such field as
   `pending`, and every other load (`check-spec` without the flag, `plan`, an import of the spec) refuses the spec, so it
   never plays. The amendment that fills them is the registration's.
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

**Per-game units and searched models** (`anchor.<group>.per_game`; S-U, `designs/endstate/design_search_u_headroom.md`
§6 B3). A searched game is 25-75 min of wall (an ESTIMATE until stage 0), so no 20-game searched unit fits a lane.
- **The unit.** A `per_game` group plans each (model, team set, seed) series as 20 units
  `<label>__<set>__<seed>__g<game>`, each `main.anchors --games 20 --deal-game <game>` with `--capture-dir`: the one
  game the whole series plays at that index (its role, both sides' team draws, its seed). It counts 1 game
  (`results.unit_counts(games=1)`); a stopped read resumes at game granularity.
- **The queue.** Within a seed a `per_game` group plays DEAL by deal: game g on every team set by every model (a
  searched game right after its raw twin's) before game g + 1. So S-U's stage 0, "the first 5 deals of unit seed 0 on
  each panel", is the queue's first 40 units: `run --max-units 40`, then `search-cost`.
- **A searched unit** (`models.<label>.load.search`) is never a short-lane job: it plays as a long heavy-slot job
  bounded by `timeout` (`kakuna.UNIT_SEARCH_S` + 600 s), with its own watchdog at `UNIT_SEARCH_S` (10 h: a searched game's max ≈ 87–204 CPU-min, wall 1–2.5× that
  under contention; a shorter bound would drop the longest games, a length-correlated drop). One that
  runs past 45 min holds its slot past the slots' chunk policy: a game cannot be split (FINDING of the build). Raw
  per-game units keep the short lane.
- **G-PREFIX** (`main.reads.prefix`). The raw and searched games of one deal must be byte-identical (both sides'
  protocol, `|t:|` removed, and the `START` document) through the search's first turn (`|turn|10` inclusive). The
  supervisor compares each pair as soon as both units count; a mismatch HOLDS the lane (`by: G-PREFIX`, the deal and the
  first differing line in `why`) for a root cause, and `assemble` keeps the group inconclusive while any pair diverges,
  so no record is written (the read is VOID unless it is resolved).
- **A subset of a seed's games** (`anchor.<group>.games`; X65's G-BANK re-plays one banked deal a panel and seed):
  `plan` writes only the registered (seed, game) units, `assemble` counts the cell's `planned` and `unplayed` over them
  and refuses a row of any other, and an import of such a group needs every one of them played there.
- **`--max-units N`** (anchor lane): the supervisor starts at most N units in its life, lets them finish, logs
  `max_units` and exits 0. It is not a registered stop (`INCOMPLETE_BY_STOP` needs one).
- **`search-cost`** (`main.reads.search_cost`): per searched game and pooled, the CPU-s and wall-s of a searched decision
  and of an excluded one (median, mean, p10, p90), the share of our decisions from the search's first turn with >= 2
  legal actions, the exclusions (no consistent world), the safety net's cuts (`net_cut`), the deviation rate, each
  game's search CPU-min INCLUDING its exclusions (`search_cpu_min`; S′-HONEST's exclusions were 29 % of it, the cost
  read's FINDING SP-2) split into `searched_cpu_min` and `excluded_cpu_min` (and into world building and playouts, from
  the decision log's `world_cpu_s` / `playout_cpu_s`) and wall-min, THE TIMER REPLAY (`timer`: every game's decisions
  through Showdown's turn timer under each policy of `main.reads.timer_model`, per panel and pooled U), each searched unit's
  WHOLE-process CPU (the `end` event's `cpu_s` in the kakuna state's `units.jsonl`: the unit's child reaped by
  `os.wait4` with every descendant, so concurrent units never mix; the conservative figure), G-PREFIX per group, and
  S-U's re-size input (the median searched game against 50 CPU-min), REPORTED. Its `--cells` form reads `main.anchors
  --out` dirs (a smoke).

Two rules read a searched group (§5). `search_headroom` pairs each deal's two counted games, and a G-PREFIX mismatch,
two engines or a search record other than the registered preset VOIDs it. `searched_play` (U only) owns its G-PREFIX
mismatches (`Rule.owns_g_prefix`: `assemble` does not mark the group inconclusive for them): it classifies each with
AM-DL-PEER's classifier (`main.reads.peer_divergence`), drops a `peer_first` deal and VOIDs on any other; the lane
still HOLDS on the mismatch, so release it once it is classified: `python -m main.reads.peer_divergence prefix --state
DIR --group G` prints each mismatch's class and divergence turn, read-only and outcome-blind.

**The live identity guard** (`main.value_search.live_identity`). Every searched-path record carries its decision's
`log_sha256` / `frame_sha256` (the digests a kept raw decision carries), and a hashed 1-in-16 sample by key
(`search_side.rescore_sampled`) also carries the honest `log`, `mask` and `tokens`. A decision an envelope preset's
SAFETY NET cut (`why: "deadline"`: π played; the search ran whole and its pick is kept in the record's `net` block) is
compared as its search returned it (`search_side.search_view`). `live_identity rescore` re-runs
S′ offline on that sample in a registered order (sha256(key) first across every cell, so any first N is uniform over panels and seeds; panel, unit seed, game as tie-breaks) and compares the
searched / excluded status, the pick, Q̂, log π and the root value exactly, and for a leak-honest preset (`r0s4_honest`, `r0s4_honest_inc`) also `search_key` and the `honest` build record (`HONEST_FIELDS`; the incremental builder's `inc` block with its carry digests); `searched_play` reads its JSON and VOIDs a leak-honest read whose re-score did not compare them.
For `r0s4_honest_inc` a decision is a function of I through its battle's CARRY, so `--carry` says how a re-scored
state gets it: `rebuild` (the default) runs each state on an EMPTY carry store, so the builder rebuilds the battle's
earlier checkpoints from I alone and `honest.inc.carry_in` checks the live carry was that function of I (it costs the
battle's world building up to the decision); `seeded` runs from the carry the record kept (`inc_carry_in`), refused
unless it hashes to `carry_in` (the last step only). The spec registers the mode as `identity.carry` (required for an
incremental preset, refused for any other) and `searched_play` VOIDs a re-score run in another. `--jobs N` re-scores N
states at once, one worker process and scorer each (the rows are the same). The JSON records the re-scoring checkout
(`inputs.code`: commit, dirty), and a PINNED leak-honest read VOIDs unless it is clean, plays as the pin
(`main.reads.playpath`) and holds the pin's bytes of the re-score's own off-path code (`live_identity.py`,
`sign_gate.py`). A KILLED attempt's decision log (a `.partial` dir, or the last attempt of a unit that never counted)
may end inside a record: that torn, newline-less last line is dropped and named (`torn_tails`); any other unreadable
line refuses.
`live_identity check` matches live records to a banked offline read's rows (`selected.jsonl` / `scored.jsonl`, the
same key AND digests) and compares them the same way. `live_identity bank-tooth` re-runs S′ at this checkout on a
hashed 1-in-64 sample of a bank's selected states (the first 80 it scored, by key hash) and requires every non-timing
field of the bank's `search` block exactly: the pin is the S′ the bank scored.

**Concurrency** (`--concurrency K`, anchor lane only, default 1; `main.reads.anchor_pool`; N1-P look 1 projected about
60 h for 486 units one at a time). SCHEDULING ONLY: the lane keeps up to K units in flight, and each is
`kakuna.run_unit` unchanged (its own short-lane `mem_cap` job, its own 95xx port from the held-flock claim, its own cell
dir and log), started on a thread that only waits on the child. Nothing on the play path (`main.x35.kakuna`,
`main.anchors`, the peers, model loading) knows about K.

- **The cap.** K is clamped to the box's heavy-slot count (`utils.heavy_slots.slot_count()`; `$GEN3AI_HEAVY_SLOTS`;
  4 under admission by memory, 2 in count mode, `designs/ops/testing.md`; read at the supervisor's start) with a logged line. The supervisor log's `start` event records
  `concurrency` (effective) and `concurrency_requested`. `--lane s` with K above 1 is refused.
- **Order and results.** The queue order is unchanged (unit seeds outermost); the first K of it start together, and a
  unit is never in flight twice. Counting, the `MAX_ATTEMPTS` retry and the exit-88 long-lane fallback are the unit's
  own attempt and the supervisor loop's, as at K = 1. Results do not depend on K: the same units, seeds and per-unit
  records, only the order of completion differs (`anchor_pool_test` pins a byte-identical `cells.json` and unit
  records between K = 1 and K = 3).
- **Stops.** The registered stop starts no unit past it and waits for those in flight. A unit that raises (a preflight
  refusal, say) lets its siblings finish, then ends the lane with that refusal; the lane's claims are held until its last
  unit ended. SIGINT leaves the units running in their own sessions, as at K = 1.
- **Cost.** All K units share the agent cores (`python -m utils.core_split agent-cpus`, rule 5b), and each takes a heavy slot while a training run is
  live or the GPU is leased, so beside the S lane's slot a K above the free slots only queues FIFO. A read the owner
  gives the WHOLE box (no run live, no lease) names its cores in the supervisor's environment
  (`$GEN3AI_ANCHOR_CORES`, e.g. `0-15`, with `$GEN3AI_HEAVY_SLOTS` raised to K); a unit launched while the slots are
  wanted sets that shape aside (`kakuna.unit_shape`): it runs on the agent cores and queues for the box's own slot
  admission, so a run that starts mid-read meets only the units in flight and the jobs the box's rule admits; the operator still holds the
  lane and drains it (`hold`) before such a run starts. **UNVERIFIED:** the
  throughput gain over K = 1 is not measured; read units/h from `units.jsonl` (`start` to `end` `wall_s`) at K = 1 and
  K = 2 before relying on it.

**The S lane** plays `main.h2h play-many` cells:

- `pairs` mirrored pairs per cell, recorded in batches of `batch_pairs`, into the group's own ledger and family
  (`<state>/s/<group>/ledger`);
- one request per node, `<group>_<label>_b<batch_pairs>`;
- a unit is up to `--workers` engines in ONE heavy slot (`mem_cap.sh --slot`), time-boxed to `--unit-s` (default
  2,400 s), at nice 19 on the agent cores (`utils.core_split`);
- an engine is one label plus some of its pending members (`runner.s_engines`). The members are striped over the
  free engines (`left[w::k]`), unless a stripe would hold more architectures, the label's counted, than one
  `main.h2h` engine serves (`main.h2h.arch.MAX_GROUPS`, 2). Then the members go on engines BY ARCHITECTURE: each
  engine is the label plus members of one architecture, each group striped over its share of the free engines; a
  group left without one waits for the next unit. The architecture key (`runner.arch_keys`) is every architecture
  toggle in the copy's `model_config.json` (`trunk_layers`, `trunk_ffn_dim`, `trunk_ffn_by_kind` among them), a
  missing one read as its default, computed in one short child so the supervisor never imports torch. N4 needs
  this: a width-grown toy final, the toy members and `pstar_x26` are three architectures. X48 worked around it with a
  driver (its PREREG NOTE 5). A read on at most two architectures keeps the stripe exactly, checked on every
  archived state. Which process plays a cell never enters its row: X48 NOTE 5 (a), and for this code
  `designs/research_state/measurements/n4_capacity_screen_2026-10-27/s_engines_identity.py`. Equal keys that still
  differ in a state-dict signature are refused by the engine's own preflight (`CellArchMismatch`, before any game);
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
  `supervisor.lock`;
- **NESTED supervisors** (N1-P / D3 look 1, 2026-10-26). The anchor lane plays its units on the kakuna states
  `<state>/anchor/<group>/<label>/`, so a `kakuna run` on one of them plays the SAME units while sharing no path with the
  lane. Three claims (`supervisors.py`): a `kakuna` queue is refused when a live anchor-lane supervisor names a state that
  CONTAINS its own (the `/proc` scan, which also sees older code) or when an ancestor's anchor path lock is held; an anchor
  lane is refused when a live `kakuna run` names a state INSIDE its own; and the lane holds every kakuna state it plays (its
  `kakuna`-lane path lock, holder record `via: main.reads anchor lane`, and its in-state `supervisor.lock`, the two a
  queue of any age honors) from its first loop turn, or from when a model settles, to its end. A conflict is a typed refusal
  (exit 2) naming the state and the pid. The S lane plays no kakuna unit and is never nested. `status` lists
  `nested_kakuna` (queues inside the state);
- a registered stop (`--hours` or `--stop-at-epoch`, never open-ended);
- a counted unit or a recorded batch is never replayed. A unit gets `MAX_ATTEMPTS` (2) attempts per supervisor life; one
  that spends them uncounted is **EXHAUSTED** (`main.reads.attempts`, from the kakuna state's `units.jsonl`: one `end`
  event per child that did not count, a short-lane overrun netted out; an interrupted attempt is not a failure). `status`
  prints each label's `exhausted`, `assemble` each anchor group's `exhausted_units` and `unplayed_exhausted_units`, and the
  supervisor's `stop_at` / `done` events list `not_counted`. An exhausted unit is a defect of the play; a unit the stop never reached
  is a defect of the schedule, and the record keeps them apart (§6);
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
  are never on it. Within `main/reads/` only `__init__.py`, `supervisors.py` and `timer_model.py` (the searched player's safety-net
  clock) are ON it (`kakuna` reaches them), so editing any blocks a repin (the nested-supervisor claims and the package docstring edited both: a read pinned before them
  repins only to a commit that leaves them as they were); every other module (`runner`, `api`, `cli`, `repin`, `playpath`, `anchor_pool`, `lanectl`,
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

- **A supervisor of code older than this cannot be held**: it writes no pid file and never reads the hold, and no change on
  main reaches a running process. `hold` lists the live supervisors as `honoring` (a valid pid file) and `not_honoring`,
  and warns on the latter, and `status` lists them as `not_honoring_hold`, so a lane is never reported held when it is
  not. Such a supervisor is stopped by pid, once, and restarted from a checkout with these verbs (`repin` accepts that
  checkout when the play path is identical), or left to reach its registered stop.
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

A group takes ONE import, never transitive (the earlier state's own units only). Cells banked in TWO earlier states
(X60-T's C0: s3501-s3503 in X61P look 1, s3504-s3505 in look 2) come in as two groups, each with its one import and every
check above, and the RULE pools them (`rules_x60.pool_anchor` / `pool_s`) only when they are one cell: one engine, one
regime id per team set, one peer code and policy mode, one Kakuna pack, each label in one group; else that meter is
INCONCLUSIVE. Read-only on the archive, those two groups pool to exactly X61P look 2's C0 forum72 cells (E3,
`2118812c27885de3`).

## 4. Assemble: the regime id is ALWAYS recorded (`main.reads.assemble`)

Five S drivers wrote `regime_ids: [""]` (x26c FINDING 4, g1 FINDING 3), and the regime was then checked by hand. Here
every group carries an id, or it is not conclusive.

**An S group's id** is the eval-ledger read's own (`LedgerRead.regime_id`), recorded with the read's digest. A group
is INCONCLUSIVE if any of these holds:

- it registers a `regime_id` and the ledger holds another;
- it names a `panel` and the rows' team set is a different one;
- it names a panel main does not register (one registered only at its read's PLAY PIN: X60-T's `r72`, whose team files
  are local and whose ids are reversible, so it never enters main's registry) and registers no `regime_id`
  (`assemble.panel_team_set`). With one, the id pins the team set (it digests it), and the rows must be on ONE team set;
- a request has no row.

`s_node` is `readout.population_score` plus the per-member SE and se(S) = √Σse²/K.

**An anchor unit's id** is `anchor_regime_id`, a digest over the cell identity that `summary.json` stamps. What enters
it, what does not, and the guards that police the rest are `EXTERNAL_ANCHORS_SOP.md` § 0, the one home. What the module
does with it:

- Each (group, team set) must share ONE id, else the group is INCONCLUSIVE. The group's id is `group_regime_id` over its
  team sets.
- The playing commit and file paths are recorded per unit and are not part of the id; the engine the commit played on is.
  A unit played on bytes other than the resolved model's is a refusal, never a pooled row.
- Every unit row carries its `games` (`[half, index, our_team, result, turns]` from `games.jsonl`), its `cell_dir`, its
  `engine`, its team-pack state (`peer_pack`) and its sides (`peer_code`, `policy_mode`).
- **One group may hold models with DIFFERENT loads** (`anchor_group(..., loads=)`, from each `models.<label>.load`). A
  load option changes the id through `our_regime` (a `belief_prior_mix` unit is stamped `greedy+belief_prior_mix=<λ>`),
  so it never pools with the unchanged model, and a mixed group is read three ways: the counted units of one (load, team
  set) share one id; across ALL its models a team set shares one BASE id (`base_regime_id` of each unit row: the id with
  the load's stamp stripped, so the opponent, teams, transport and the rest are the same for every model); and a unit
  whose model declares a load it did not stamp is not conclusive. The group id is then `mixed_group_regime_id` over the
  base ids and each load's ids, and the group also carries `regime_ids_by_load` and `loads`. A group whose models share
  one load (or none) is read as always, with the id it always had. `prior_mix_ab` still wants one group per arm (its
  control is banked).

**The game-for-game guard** (`rules.guard_games`: `determinism_guard`, `prior_mix_ab`'s and `score_bands`' guards, and
X59-S's repeat unit) compares each game's 5-tuple AND its choices: `assemble.game_choices` is a digest of each player's
`CHOOSE` sequence in the unit's capture, the two players kept apart so an order-only game (the same choices racing to the
server) stays equal. A guard unit passes only if both are equal (`games_equal`; the row also carries `tuple_equal`,
`choices` and `choices_differing`). The banked side's capture is its rows' one `source_unit`, read under
`$GEN3AI_ARCHIVE_DIR` (default `~/gen3ai_archive`) as `<read state>/cells/<unit>`, and only when that cell holds exactly
the banked games. When the bank names no single source, `choices` is `unavailable` with its reason and the 5-tuple
decides alone (X51's rows name no capture source, so the X59-S identity unit records `unavailable`). Once the bank
NAMES its source, it fails closed (anchor-guards review H2): the source missing under the archive (moved, pruned,
another box), the re-played unit's capture or cell dir missing, or a source holding other games makes `choices`
`inconclusive`, `games_equal` false, and the rule names it (`INCONCLUSIVE: ...` in the guard's reasons; `d1`'s
guard lists the units as `inconclusive`), so one decision never passes or fails by the environment. A spec with
`choices_guard: five_tuple` (banks known to lack captures) turns that into `unavailable`, recorded per unit;
`assemble` writes the policy into `cells.json` as `guards`. The 5-tuple alone misses about a quarter of the
divergences (11 vs 15 below), those that change neither the result nor the turn count.

**The anchor guards re-derived** (`main.reads.rederive`; anchor-guards review H1). The forfeit-win guard (SOP G6)
and the H21 peer-fault check (G2) run inside the playing tree, so a unit a pre-guard pin plays carries neither,
and a registered read keeps playing at its pin. Assemble re-derives both for every played unit, its own and
imported, from what every pin writes, and records the result on the unit row as `rederived` (`forced_end`,
`peer_fault`: `clean` / `hit` / `unavailable`, the `hits`, the `why`):

- **forced end**: a `FORCELOSE <slot>` in a game's capture is a hit unless it is OUR turn-limit cap (the row booked
  a `loss` stamped `hit_forfeit_limit` at `turns >= forfeit_turn_limit > 0`, our side not a Metamon peer). The
  capture holds no cause, so our own drop AT the cap reads as the cap, and a FORCELOSE after the result (it decides
  nothing) reads as a hit: fail closed;
- **peer fault**: the long-tail line in any played half's `peer_*.log`;
- **evidence missing** (capture off, a pruned log): `unavailable`, and the unit is judged as before.

The spec's `rederived_guards` decides what a hit does. `count`: the unit does not count (`rederived_uncounted`,
its reason named). `report` (every read registered before the change: N1-P looks 1 and 2, H-CTX rc72 / rc72b, X59-S, D3,
S-U and its gate test, the belief-prior-mix reads): counting never changes mid-read; a hit on a COUNTED unit is a
FINDING that HOLDS
`decide` (and so `record`; typed refusal, exit 2) until the orchestrator rules with `release-guard-hold`, which
writes `<state>/rederived_release.json` naming the digest of exactly those hits, so a later hit holds again.
The record carries the check as `rederived_guards` (§6).

### The engine (`main.reads.engine`; REVIEW F F-M2, 2026-10-26)

Kakuna units and S cells play on our in-process Rust engine. An ENGINE is the build-stamp refusal key
(`utils.rust_env.stamp`): a unit's is `port=<src>/<n>;env=<src>/<n>` (`sim_bridge`, and the env core that our
`rust_core_slot` side loads; a summary of this code stamps it at play), an S row's is `env=<src>/<n>` (from
`compute.core_stamp`). Every unit row and group carries `engine`. A group whose counted units, or an S family whose rows,
span two engines is INCONCLUSIVE.

**The LEGACY rule, in code** (the SOP's § 0.1). A banked summary stamps no engine, so its id is unchanged, re-checked for
every unit of D3 look 1 and the belief-prior-mix read. Its engine is `LEGACY_ENGINE` of its playing commit, a FROZEN
table reconstructed from git (`engine_at`: the stamp's listing at that commit, plus the `Cargo.lock` blob
`LEGACY_LOCK_BLOBS` names for a commit that predates the locks being tracked, 2026-11-09; a later commit's own blob
otherwise); the reconstruction reproduces the core stamps the ledger RECORDED at 6c428453, 1ed71c9dd and
46b9e572 (`engine_test`). `python -m main.reads.engine` is the command line: `--check-inputs` (a checkout stamps its
commit's engine), `--engine --tree DIR [--expect KEY]` (print / judge any tree's engine), `--install-locks --tree DIR`
(an old pin; `designs/launcher/pinning_and_worktrees.md`, "Building an old pin"). A bank that names no engine per row is read by its registered `engines` or `LEGACY_BANK_ENGINE`
(by sha256).
- Ids cannot tell two banked engines apart, so every import compares the engines themselves: the cross-state import,
  `import_off` (anchor and S), `prior_mix_ab`'s bank, the determinism guard (it must play on the bank's engine) and
  `x35_r1`'s imported S. An import across engines REFUSES, and so does one from an unknown engine.
- `repin` names the engine at both commits and refuses if they differ. The play path holds every tracked file the stamp
  hashes (`playpath_test`).

| engine | port / env | banked reads (playing commit) |
|---|---|---|
| E0 | `05b106e2…/84` / `dfd24aec…/119` | X51 and x37a (`63f97a575`), `cb0f1769c`, `acb99527`. X51's `k28v34_home72` rows: `b9c758841`, E0B `e64b127b…` / `b92ea7ad…` |
| **E1** | `7d932165…/90` / `7c532805…/125` | x26c (`e05c45eeb`); D1 H (`6c428453a`, `5e389ddac`) and P; X52 (`f4b25238d`); SPRT (`eb99ab362`) and G1 (`3f3a2ed41`) S; human-only vs P\* and metamon-strongest (`d32ffcee4`, `4b0ab220e`); PastaMittens (`62bb07c00`); belief prior mix (`4ee655648`; smoke `033fcbcd2`). D3 look 2 (`c4a5c37b6`, the hybrid pin). **The X35 stage's training engine** (`f4b25238`, D3_PIN `9e289ba4`) |
| E2 | `806f97bd…/90` / `9ecb4249…/125` | D3 look 1 (anchor `2cc864f2f`, S `1ed71c9dd`) |
| E3 | `0d3d7629…/91` / `393631b8…/126` | N1-P look 1 (`46b9e572c`; repin `de15dfc7b`) and N1-P look 2 (`de15dfc7b`, AM-N1P-2; look 1 imported): one engine throughout. X60-T (`f564c333f`, AM-X60T-1: `de15dfc7b`'s play path but the panel registry; C0's X61P cells import) |
| E4 | `e00a212a…/94` / `3d0d5c9c…/129` | the S-U gate test (`835c6b4b2`), its re-gate (offline, scored on the gate's states) the deviation-only live read (`c01ae8ee5`; raw lane, scorer and branches) and S′-LIVE (pin `0c76b521e`, both arms, all 5,400 units and the identity re-score; 2026-11-01 to 11-05): one engine throughout. A checkout of a commit BEFORE the locks were tracked stamps it only once `cargo` wrote them or `python -m main.reads.engine --install-locks` copied them (the stamp hashes `src/rust_env/Cargo.lock`; dlive DL-4); from the commit that tracks them every checkout does |
| E6 | `f0fe71f6…/95` / `c0b7c649…/130` | S′-HONEST (pin `e46e3ddf8`, both arms, all 10,200 units and both identity re-scores; 2026-11-06 to 11-09): one engine throughout. The stamp the fresh-dice matcher and its carried trajectory added (opt-in; `r0s4_rolls` re-scores S′-LIVE byte-identically on it) |
| unbanked | `67a272e7…/93` / `d79c3a74…/128` | main after the 2026-10-26 full-parity batches (BF4, `2a877361a`), X59-S's registered engine. Any engine not listed is unbanked and enters the id; `main.anchors --dry-run` prints a checkout's |

**Mixed-engine comparisons in shipped records.** Each was guarded game for game, and no verdict is shown to be affected.
Re-decided by this code, items 1 and 2 refuse at `inputs`; item 3's scripts predate `main.reads`. The recorded verdicts
stand as recorded.
1. **D3 look 1** (CONTINUE): ON on E2 against C0 banked from D1 on E1 (H, and P); the E1 → E2 Rust diff is `f9978586b`
   (lock-in and recharge counters) and `d8d674b99` (encoder input fixes). The guard reproduced 140 / 140 games
   (divergence < 2.1 % at 95 %, against Δ̂ = +1.27 pp); the rate measured later (1 / 420 on that panel, 11 / 1,260 across
   three) makes 140 / 140 likely (P ≈ 0.72): low exposure, not luck. On forum72 the newer engine shifted +4 wins in 420
   (+0.95 pp).
2. **Belief prior mix** (NOT DETECTED, both arms): the arms were played on E1 against X51's banked λ = 0 on E0. The guard
   reproduced 80 / 80 games per arm. x26c FINDING 1 re-played P\* on E1, and all 1,200 of its games equal X51's
   (measured game for game, 2026-10-27).
3. **Metamon strongest** (PastaMittens; Δ-U −0.25 pp, NOT DETECTED) and **human-only vs P\*** (FR-1 / FR-3; Δ-K +0.14 to
   +0.30): both were played on E1 and read Δ against X51's Kakuna rows on E0. The identity unit reproduced 20 / 20 games.
   The human-only Δs are far beyond any engine bias.

**OUR side's pack regime** (SOP H24; not in any id). A play commit before `d8d674b99` packs forum72's `sft431` with
Frustration at base power 1.
- **Pre-fix:** every E0 and E1 read in the table, and D3 look 2's hybrid `c4a5c37b6`.
- **Post-fix:** D3 look 1's ON arm (E2), N1-P looks 1 and 2 and X60-T (E3), H-CTX and X59-S.
- **Our side on `sft431`:** 0 games in D1 and D3 look 1, 2 of 1,920 F games in N1-P look 1, and 6 of 400 per model in
  the x26c read (`measurements/x35_pack_regime_2026-10-27/`).

A registered read names its play pin, and so its regime. Its arms never split across it on forum72.

**Replay determinism, measured** (`measurements/kakuna_determinism_2026-10-27/`):
- **On one engine, a counted Kakuna unit replays exactly.** Across every replayed unit in the archive whose two plays sit
  on the same side of an observation change, 0 of 2,100 games differ (95 % upper bound 0.14 %). Kakuna's choices never differ first between two whole units; threads are pinned to 1. E0 → E1 crosses no observation change: 1,360 of 1,360 games are equal.
- **Across `d8d674b99`** (`gen3_volatile_silent_ends_v1`, inside E1 → E2), our own greedy policy sees a different input
  after a Destiny Bond or an Endure (11 of the 12 protocol-changing divergences). E1 vs E3: 11 of 1,260 games differ by
  5-tuple (forum72 6 / 420, p72 4 / 420, heldout72_hpfix 1 / 420) and 15 by choice. Each one is OUR choice, made from a
  byte-identical protocol.
- **A LONE one-game replay is not always game G of its 20-game unit** (the deviation-only live read's DL-7, E4,
  2026-11-01). In 9 of 2,100 forced branches Kakuna chose differently from the raw game on a byte-identical view: 8 at
  its first choice, 1 at turn 4. Both attempts agreed every time, so it is a fresh process against a long one, not
  noise. It clusters by game; 0 of 135 null replays were affected. Root cause unknown. A read that replays single
  deals must class such divergences (`designs/research_state/measurements/dlive_read_2026-10-28/scripts/peer_divergence.py`: peer first on an identical view)
  before its prefix tooth VOIDs.
- **So the rate belongs to the engine pair and the panel, not to Kakuna**, and a guard proves exactness only on one
  engine. The one same-engine divergence is the long-tail fault (SOP H5 · H17 · H21): its unit is `peer_clean = False`
  and is dropped whole, because the fault also corrupts the game after the peer's forfeit.

### The Kakuna team pack (`gen3_peer_showdown_pack_v1`; the SOP's H22)

A regime split beside the engine's, on forum72 (goal (b)'s primary panel), away and the pool. Which sets changed and how
Kakuna's play changed is `EXTERNAL_ANCHORS_SOP.md` H22 (§ 3); here, what the module does with it. The peer's `peer_pack_changed` enters
the id only when True (`assemble.peer_pack_identity`), so an unchanged set keeps its banked id and pools as before, while
old forum72 / away / pool Kakuna reads stay valid as their own regime and are never pooled with new ones. Machine-checked:
- within one state, two ids on a (group, team set) are INCONCLUSIVE, as always;
- a cross-state import whose counted units differ from this state's in `peer_pack` on a shared team set REFUSES
  (`imports.check_peer_pack`, `imports_test`);
- `import_off` refuses on the regime id it registers.

A read that imports banked forum72 / away / pool Kakuna rows (D3 look 2's C0) must play at a pin that predates
`b0441623e`; a pin carrying the change refuses that import on forum72. The reads live at the change (N1-P look 1 at
`de15dfc7b`, H-CTX rc72b at `905c29f5b`, X59-S at `fca58b24d`) run from their own pinned worktrees and are untouched.

**Reads that cross a regime on purpose:**
- **D3 look 2** (orchestrator ruling, 2026-10-26): EVERY game on **E1**: D3 ON's look-1 units are re-played on E1, the
  look-2 units are played on E1, and C0's banked rows (E1) import. No commit carried both E1's Rust and `import_off`, so
  the play pin is a HYBRID, `c4a5c37b6` (tag `x41-d3-look2-play-c4a5c37b`; PREREG AM-D3-L2): `b0441623e^`'s read harness
  with every GAME path (`src/rust_sim/`, `src/rust_env/`, `src/agents/{model,observation,inference}/`,
  `src/utils/bridge/`, the team packer and loader) at the training pin `f4b25238`'s bytes. Restoring only the Rust is not
  enough: the Python play path also changed what our side plays between those commits (the op's consumed-item rule, the
  eval-mode load, the packer), and the engine key cannot see it (SOP § 0.1). Checked: `engine_at` and the built
  artifacts' stamps equal E1; the play path differs from the bank's commits only in harness files; four banked E1 units,
  three of them where E3 diverges from E1, re-played identically, 80 / 80 games choice by choice. **READ 2026-10-28:**
  UNDECIDED, final, every group on E1 (record `x35_breakers_stage_2026-10-17/reads/d3_look2.json`).
- **The X35-frame SHARED play pin, FROZEN** (REVIEW_2 M1 option (C), `strategy_review_2026-10-26/REVIEW_2.md`): ONE pin
  for every read of the X35-frame arms (all trained at `f4b25238`): PR-1′, X62, X63 and X48's C0 replay. `86cfdf7ef`, tag
  `x35-shared-play-86cfdf7e` (on origin, not on main). It is D3's method on the newest harness: main `f6bef5d68` with
  the same game paths restored to `f4b25238`'s tree objects (`src/rust_sim/`, `src/rust_env/`,
  `src/agents/{model,observation,inference}/`, `src/utils/team_loader/`, the packer, `teambuilder.py`, `gen3_utils.py`).
  `src/utils/bridge/` stays main's: against `f4b25238` it differs only in the Kakuna peer's Showdown pack scripts, the
  front end (the validator's re-pack, the forced-end record, the input log), `team_validator` / `validate_team.js` and
  `sim_bridge_bin`. The front end's re-pack changes no set OUR side plays (0 field differences over home72, forum72,
  p72, heldout72_hpfix and away). **Its regime:**

  | component | at this pin |
  |---|---|
  | engine | **E1** (`engine_at`, and the built stamps) |
  | Kakuna's team pack | post-H22: `peer_pack_changed` True on forum72 / away / pool, False on home72 / p72 / heldout72_hpfix |
  | peer code / policy mode | `9cc9caacde85f8ea` (not legacy, so in every id) / `train` (legacy) |
  | OUR side's pack | the legacy pack (H24): forum72 `sft431` plays Frustration at base power 1, as the arms trained |
  | ids it gave | p72 `0e9cde5bb0396d64`, forum72 `5e1e14aaa845e97e`: each the banked E1 id plus exactly these stamps |

  - Its ids differ from every banked id, so no banked E1 row imports into a read at it (G9, G14). Its C0 bank is the
    one every X35-frame read shares; disjoint unit seeds per read (REVIEW_2 m2).
  - **Checked** (`measurements/x35_shared_pin_2026-10-27/`): three builds of the same restore (on main `48eeb0b70`,
    `eb39f1b77`, then `f6bef5d68`), the play path identical in every game file. 220 / 220 captured game comparisons
    were equal by 5-tuple and by both players' choices. A further 20 / 20 home72 games were equal by 5-tuple against an
    E1 bank with no capture. The references were banked E1 units on p72, heldout72_hpfix and forum72 (each one where
    E3 diverges from E1) and the same units played at the D3 pin. 218 of the 220 were also equal in protocol bytes.
    The other 2 are Kakuna's `sft431` games at unit seed 16 (games 1 and 11), and their bytes differ only in Kakuna's
    own request: `frustration1` → `frustration102`.
  - **FROZEN:** a later harness change reaches a read only through a new, named pin. Not in it: `29b90161f`'s H3 (an
    unstorable forced end reads blind; play path). Its H1 / H2 run at assemble, off the play path, so a deciding checkout
    at main applies them. `--our-belief-prior-mix` cannot load (`f4b25238` has no `belief_prior_mix`).
  - **Its X63 successor** `65b998daffffaa1669b0b4ef840874e23edc12d3` (local branch `x63-play-pin`; X63 PREREG §5):
    `86cfdf7ef` plus ONE commit adding only rule `x63_kvk` (`rules_x63.py`, its test, the registration in `rules.py`,
    the rule set in `rules_test.py`), because the frozen pin's spec parser refuses a rule it does not hold. Its play
    path equals `86cfdf7ef`'s (`playpath.compare`: 0 differences, digest `afedc505f0e2…` over 1,715 files), so its
    regime is the frozen pin's. Its `package_doc_test` fails by construction: naming the module in
    `main/reads/__init__.py`, which is on the play path, would break that identity. **READ 2026-10-31:** X63 Stage A,
    NOT PURSUED (spec of record `x63_2026-10-29/read_spec.json`, AM-X63-SPEC; record `x63_2026-10-29/reads/stageA.json`):
    900 units on E1, forum72 id `5e1e14aaa845e97e` (= the frozen pin's), a counted unit re-played 20 / 20 by choice.
    Like N1-P's play commit it captures panel cells only, so G14's forced-end half reads `unavailable` on home72, where
    the in-tree forfeit-win guard is armed on every unit. Its anchor lane queues every group by seed index, so a
    REPORTED group sharing a state with a deciding group plays interleaved with it (AM-X63-SPEC FINDING 11).
- **X59-S games / X60:** one engine per read, named at registration. **X60-T** plays at `f564c333f` (tag
  `x60t-play-f564c333`, AM-X60T-1) = `de15dfc7b` + rule `x60`, its `void.pin` / `void.dose` CHECK (the pin's `void_gate`
  refuses to READ them: assemble / decide / record run from main) and R72 in the pin's panel registry only (local
  absolute paths: not portable off this box); E3, so C0's X61P cells import. **X61P look 2** (READ 2026-10-30, NULL) played
  at look 1's commit `de15dfc7b`, so its import from look 1 is on E3 with Kakuna's legacy pack and the legacy sides
  (PREREG AM-N1P-2); its record was written from main. The play commit writes no capture for home72 units, so G14's
  forced-end check reads `unavailable` on those 70 reported-only units (both looks).
- **The human-only / PastaMittens reads:** each was played on one commit, all on E1.

### Foul Play's played team (`gen3_frontend_validated_pack_v1`; the SOP's H22)

The front end plays the validator's re-pack, as a real server does. Our side and Kakuna play the same sets; Foul Play does not
(its packer writes no Hidden Power IVs: 11 of 72 pool files and `away`'s `team_23`, one of the only two `away` files Foul Play can start; the SOP's H22 and H23 have the numbers).
`FoulPlayPeer.read_report` stamps `frontend_pack_changed` (`main.anchors.frontend_pack`), `assemble.peer_pack_identity` adds it
to the id only when True, `unit_row` carries it as `peer_pack = "frontend_changed"` and `imports.check_peer_pack` refuses an
import across it. A record that cannot be computed voids the regime verification. `frontend_pack_integration_test.py` pins the
rule and, on a box with Foul Play, the measurement.

### The sides: the peer's code and our load mode (`main.reads.side_identity`; 2026-10-27)

Two inputs changed play without moving any id: the Metamon peer's own Python, and the mode our policy was loaded in.
What enters the id is the SOP's § 0.1; G14 is its guard. Here, what the module does with them.
- **`peer_code`** (`our_peer_code` for a Metamon our-side) is the 16-hex digest of the record the peer reports at startup
  (`peer_scripts/peer_code.py`). It covers our peer scripts (the bridge pack scripts only beside `team_pack.py`), the
  Metamon checkout's HEAD and any tracked change or untracked file in its `metamon/` package, the interpreter, and every
  installed distribution with its VCS commit. The harness refuses a missing record, and one whose scripts are not its own.
- **`policy_mode`** is `eval` / `train` / `mixed`, read back from the loaded modules. `nn.MultiheadAttention` runs another
  kernel in each mode (`141f36f7f`: logits within 4.8e-7).
- **Halves that differ** in either void the unit (`runner.check_side_halves`).

**The legacy values.** Each is reconstructed and FROZEN:

| what | value | banked commits |
|---|---|---|
| peer code `f230353c40e43488` | `metamon_side` e17d4869 | X51, x37a, X52, D1, x26c, PastaMittens, human-only (`4b0ab220e`) |
| peer code `64c220c05cca7382` | `metamon_side` 9f8520a1 | metamon strongest / human-only (`d32ffcee4`), belief prior mix, D3 look 1 |
| peer code `b817d3216c418f75` | `metamon_side` 9a550a80 + `weight_pins` | N1-P look 1 |
| policy mode | `train` | every banked commit but N1-P look 1's two, which are `eval` (`LEGACY_COMMIT_MODE`) |

- Each peer digest also holds Metamon at `0a00a759`, clean, and the `metamon` env as it stands (`LEGACY_PEER_ENV`, 104
  packages, Python 3.10.21).
- `0a00a759` is the `opponent_commit` of all 1,281 archived Metamon units. The env's last conda change and every
  site-packages file date from 2026-09-14, before X51.
- The checkout's dirty state at the banked reads is UNVERIFIED (it was clean on 2026-10-27).
- The three script versions differ only by instruments and refusals (the ensemble router's no-RNG check, the preset pin,
  the weight pins), so they are ONE legacy peer regime: a unit row says `their=legacy`.
- `side_identity_test` re-derives the tables from git.

**What it does.**
- A stamp enters the id only when it is not legacy.
- Every unit row and group carries `peer_code` and `policy_mode`: stamped, else from the banked commit's tables, else
  unknown (`None`). A group spanning two values is INCONCLUSIVE.
- These refuse across either side, or from an unknown one: the cross-state import (`imports.check_sides`), `import_off`,
  the determinism guard and `prior_mix_ab`'s bank (`rules._same_sides`). A bank names its sides by its rows' commits,
  its registered `sides`, or `LEGACY_BANK_SIDE`.
- Every read stamped by this code gets a new id, because its peer scripts are not a banked version (`b0441623e` and this
  change). Main's engine is unbanked already.

**Checked.**
- Banked ids are unchanged: all 1,423 archived unit summaries give the same `anchor_regime_id` before and after (2026-10-27).
- Pinned reads are untouched: they play and read at their own pins (D3 look 2's hybrid `c4a5c37b6`, N1-P look 2, X59-S,
  H-CTX).

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
| `search_headroom` | S-U (`design_search_u_headroom.md` §3, §4.4; `rules_search_headroom`): ONE `per_game` group, a searched label and its raw twin, each deal a pair only when both games counted (a one-sided deal is excluded and counted); the unit (panel, seed) statistic is the mean paired difference (tie ½); I = G_U − G_home72 and G_U by a percentile bootstrap over units within each panel (`default_rng(seed)`, U panels drawn before home72); Primary A (U-HEADROOM / U-LESS / NO-U-PREMIUM, futility-only at phase 1) and Primary B (DOES-NOT-CLOSE-B, U-GAIN and GOAL-B-REACHED leaky) by exact fractions with `rule8_eps`; VOID on G-PREFIX, two or unknown engines, other bytes, a searched unit not stamped and played as `expected_search`'s preset, a raw unit with a search stamp; the discordance is b + c only | new (2026-10-27) |
| `deviation_sign` | the S-U gate test (`measurements/su_gate_2026-10-27/`; `rules_deviation_sign`): ONE raw label keeping input logs (`load.input_log`) on home72 and U; two ARTEFACTS read by path with their sha256 (`main.value_search.sign_gate select` / `score`: the hashed draw of 2 states a game, S and the self-referee on each); SIGN ACCURACY of S's deviations, pooled U equal-weighted, a game-clustered bootstrap; GO iff LB > `bar` (within `rule8_eps`: STOP, flagged); VOID on two, unknown or unregistered engines, other bytes or a search stamp, a state from an uncounted cell or with a wrong draw hash, > `per_game` states a game, a selected state unscored | new (2026-10-27) |
| `searched_play` | the S′ live-play read (`design_sprime_live_play.md`; `rules_searched_play`): ONE `per_game` group on the `u_panels` only, a searched label and its raw twin; a deal is a pair when both games counted and it is not a PEER DROP (a G-PREFIX mismatch AM-DL-PEER's classifier calls `peer_first`: PRE-treatment, excluded); every other finished unpaired deal is POST-treatment and IMPUTED at its worst case (GO: searched missing −y_raw, raw missing y_s − 1, neither −1; NULL: the opposite); G_U by `search_headroom`'s within-panel unit bootstrap (a fresh `default_rng(seed)` per estimate: paired, GO-imputed, NULL-imputed); GO iff the GO-imputed LB > 0 (spelled GO-HONEST for a leak-honest preset, `dice` `fresh` or `fresh_inc`, else GO-LEAKY), NULL iff not and the NULL-imputed UB < `null_margin_pp`, else STOP, exact fractions with `rule8_eps` (`boundary`, `go_below_margin` flagged); DECIDED flags: `goal_b` (GOAL-B-REACHED iff GO and the searched arm's pooled-U score LB > 0.50 with every post-treatment deal's missing searched game a loss) and, with the optional `max_excluded_frac` `{panel: fraction}`, `excess_exclusions` (any panel's share of excluded search-path decisions over its bound, near counting as over; it never VOIDs); with the optional `timer` `{reserve_s, overhead_s, max_cut_frac}`, `timer_safe` (TIMER-SAFE iff every counted searched unit's decision log, replayed through Showdown's turn timer with the bank-aware deadline min(bank, 150) − reserve − overhead and the cut step running on (`main.reads.timer_model`), forfeits 0 games AND cuts ≤ `max_cut_frac` of the searched decisions pooled over U, exact, near never safe; a preset carrying the live safety net must register the net it played; it never VOIDs and never moves the class), the safety net's recorded cuts counted beside the exclusions and the prototype's policy table REPORTED; REPORTED `distill` (X65-F2's fields on every counted searched record, and units with no capture); VOID on any other G-PREFIX class, two / unknown / unregistered engines, other bytes, the search config, a raw unit with a search stamp, post-treatment unpaired above `max_post_unpaired_frac` or pre-treatment above `max_pre_unpaired_frac` of a panel's planned deals, and the live identity guard (`live_identity rescore`'s JSON: missing, FAIL, short of `min_rescored`, not the cells' registered-order sample, a leak-honest re-score that did not compare `HONEST_FIELDS`, or an incremental one not in the registered `identity.carry` mode) | new (2026-11-01); GO-HONEST, `goal_b`, `max_excluded_frac`, `distill` (2026-11-06, S′-HONEST); `r0s4_honest_inc` and `identity.carry` (2026-11-06, S′H-14); `timer` / `timer_safe` and the net's cuts (2026-11-09, the cost read's R3 / R5) |
| `distill_play` | X65 ExIt-0 (`measurements/x65_exit0_2026-11-05/PREREG.md` §1, §4–§6, §10; `rules_distill_play`; X66 ExIt-0H reads with it unchanged, `measurements/x66_exit0h_2026-11-09/PREREG.md` §6): three distilled P\* students (`students`, each its read panel's only student) against P\* raw (`raw`), lone one-game units paired by deal; `groups` names each panel's anchor groups (played or imported), `seeds` each look's unit seeds (look 2 reads the cumulative pool). A side is counted / exhausted (spent here, or imported and not counted) / pending / missing; a pending or missing side is UNFINISHED. D = y(student) − y(raw), exact in halves; Ĝ_X = the mean of D over the panel's deals, Ĝ = the mean of the panels; three estimates (paired, GO- and NULL-imputed: a drop's missing student 0 / raw 1 for GO, 1 / 0 for NULL); SE = the sample SD of B exact replicates (a fresh `default_rng(seed)` per estimate, panels in order, seeds ascending as one pool, `integers(0, n, (B, n))`, deals whole). In order: VOID on the delivery gate's file (`delivery`: missing, schema `gen3_x65_delivery_v1` and key set, `B`, `seed`, each student's `zip_sha256`, `bank_manifest_sha256`, `delivered` / `boundary` against its own LB), NOT-DELIVERED when it says so (no game read), VOID on an unfinished deal, drops above `max_drop_frac` of a panel's deals, a unit off `engine` or its pinned sha256, not greedy, results disagreeing, a panel spanning regime ids / peer codes / policy modes, < 2 seeds, a counted pair whose two captures' START documents differ in the format, seed, either packed team or our core side (`start_match`: not one deal), or G-BANK (`control: banked`: the `gbank` group's replays of game `(seed − origin) mod 20` compared whole-game with the banked units, START + both sides without `\|t:\|`; fewer than `min_identical` of `n` VOIDs the banked control, an unfinished replay VOIDs); then GO iff LB(z_go[look]) > 0, NULL iff UB(z_null) < `null_bar_pp`, else CONTINUE (look 1) / UNDECIDED (look 2), exact fractions with `rule8_eps` (`boundary`). Checked by `measurements/x65_exit0_2026-11-05/by_hand.py` (own loops; `rules_distill_play_test` holds them equal). NOT-DELIVERED is decided on an unplayed state, which `record` refuses (its units are unplayed): `decide` writes it | new (2026-11-05) |
| `deviation_live` | the deviation-only live read (`measurements/dlive_read_2026-10-28/`; `rules_deviation_live`): ONE raw label keeping input logs in a `u_group` (the U panels) and a `home_group` (home72); three ARTEFACTS read by path with their sha256 (`main.value_search.live_branch select` / `score` / `plan`: the gate's hashed draw, S′ and T in the registered order to each panel's `quota` of S′ deviations, and the branch state of `main.anchors --our-force` per-game units, `main.x35.kakuna run`); the mean of D = y(S′'s forced branch) − y(raw game), pooled U equal-weighted, a game-clustered bootstrap; GO iff LB > `bar` (within `rule8_eps`: STOP, flagged); VOID on the raw arm's engines, bytes, regime or search stamp, the draw, the quota prefix, and the branch teeth (G-BRANCH in any attempt, the capture prefix, a null control that changes its game, a forced action not the scored pick, the branch engines, the preset, the plan, drops above `max_drop_frac`). PEER DROPS (the PREREG's AM-DL-PEER, 2026-10-31): when the read's classifier wrote `<branch_state>/peer_drops.jsonl` (sha256 recorded), a G-BRANCH failure whose every failing attempt it calls a peer drop (Kakuna chose differently first on an identical view; for a branch, before the branch turn) no longer VOIDs, and a null that changed its game Kakuna-first is excluded as dropped; it reports peer drops per panel, branches beside nulls | new (2026-10-28); peer drops 2026-10-31 |
| `premium_recovery` | the H-CTX read as amended (`measurements/hctx_rc72_2026-10-26/PREREG.md` Amendment 1, the engine-identity rule): a panel (rc72) and its familiar (home72) and unfamiliar (F) controls played on ONE engine in one read, interleaved; KILL / SUPPORT on the share of the premium (familiar − unfamiliar) the panel recovers, by exact fractions; INCONCLUSIVE unless the premium's unit-bootstrap lower bound clears 0; VOID on two engines or an unknown one; absolute bars against a banked read only as a secondary, cross-engine line | new (2026-10-26) |
| `x59s_floor` | X59-S's floor pilot as first registered (every arm's mean U rate < `floor_bar` fires the in-house fallback); superseded by `x59s_gate` before it was decided, kept so the pilot state's spec loads; VOID 6 repeat | new (2026-10-25) |
| `x59s_gate` | X59-S's per-contrast floor gate (Amendment 5 A5.1) on the imported pilot and repeat groups: a contrast is UNINFORMATIVE iff both arms' pilot means are <= the better floor reference + `band`; releases booleans only; VOID 6 | new (2026-10-25) |
| `x59s` | X59-S's full read: `x59s_read.read` / `inhouse` / `verdict` on the Kakuna cells and the read's sha-recorded artefacts (three contrasts with A5.2's widened interval, the gate's booleans, yardsticks); VOID 3, 5, 6, 7 or a second engine | new (2026-10-25) |
| `x59s_restriction` | X59-S's Delta_restriction = S-P0 - S-P on U with A5.2's interval (Amendment 6 A6.1), the closure's guard; the full read takes its record as an optional artefact | new (2026-10-26) |
| `n4_screen` | the one-look capacity screen (N4 PREREG §5 with its review's M2 / M4 / M5; X48 PREREG §5, §8; `rules_n4_screen`): S = Welch two-sided `confidence` (90 %) on the finals' S, every cell at least the S group's `pairs` (X48 2,000, N4 2,000 by AM-N4-FIXES M3); K = the `d1` estimators on the anchor group (forum72) at `z`, φ̂-widened, φ̂ printed. In order: INCONCLUSIVE (a missing input, an S cell short, K beyond one uncounted unit per arm (`d1.min_games`), unplayed units, a meter on two or unknown engines) → MIXED (`mixed_veto`: GO-S with UB_K < 0 or GO-K with UB_S < 0; classed UNDECIDED, never GO) → GO (`GO (both)` / `(in-house)` / `(unfamiliar)`) → NO-GO (NO-GO-S and not GO-K, M5's text from `nogo_scope` (+ its `qualifier`), `LR-confounded` when the ON arm's mean `snapshot_history` LR < 0.9× the OFF arm's; a GO is `LR-elevated` above 1.1×) → UNDECIDED. Exact fractions, `rule8_eps`; `near_scope` `bound` (X48) or `meter` (N4: a near bound reads its meter UNDECIDED). `params.on` / `off` are label lists. REPORTED: R1 at look 1, per-member Δ_S, the `slope` group (reported-only), `pin_drift`, σ̂, the S power at σ̂ (seeded). An X35-stage record (top-level `arm` = `batch.READ_LABEL` of the ON arm, `look` 1; no top-level `r1`). Read so far: X48 look 1 and N4 look 1 (`reads/toy_n4ffn_look1.json`), both NO-GO | new (2026-10-27) |
| `x63_kvk` | X63's k-v-k decision on forum72 against Kakuna (`measurements/x63_2026-10-29/PREREG.md` §6–§7; `rules_x63`): `params.on` / `off` label lists (k_on and k_off may differ; `stage` A = 1 v 2, B = 2 v 2, C = 3 v 3, checked at load), a `primary` and a `home` anchor group (one panel each). Each model a PLAIN proportion (a tie a non-win); Δ = mean_on − mean_off, B = Σ p̂q̂/G / k² per side, σ̂² = the mean over the sides with ≥ 2 models of (sample variance − mean p̂q̂/G) (at Stage A the OFF side's alone, = X63-C's σ̂_CC), SE² = B + max(`sigma_floor`, σ̂)² (1/k_on + 1/k_off), q = 1.959964 iff σ̂ is clearly ≤ the floor, else Student-t 0.975 on Σ (k − 1). Stage A: CONTINUE iff Δ̂ ≥ `screen`, else NOT PURSUED, σ̂_CC and (`stage_a_power`) the registered P(NOT PURSUED) row printed; B / C: INCONCLUSIVE (σ̂ > `inconclusive_sigma`) > GAIN (LB > 0) > RETIRE (UB < `kill_bar`) > UNDECIDED, B says whether C fires (UNDECIDED and Δ̂ ≥ `stage_c_bar`); a GAIN is FLOOR-checked (the ON models' pooled `home` Wilson LB > 0.50) and priced (net = Δ_home + r·Δ_F): FLOOR fails ⇒ OVER-RATE, r null ⇒ UNPRICED (r* printed), WITHIN ⇒ ADOPT-CANDIDATE, EXCEEDED ⇒ OVER-RATE, else TRADE. Exact fractions, `rule8_eps` (a near statistic takes the side that licenses less). A deciding group not conclusive is INCONCLUSIVE with its reasons. REPORTED: Δ_home at every stage, the team-cluster bootstrap of Δ_F (our team per game, seeded), the `reported` groups; `home` is reported-only at Stage A. Optional `void` (the VOID gate below) | new (2026-10-29) |
| `x60` | X60-T (its PREREG §5-§11, §15 and AM-X60T-1, `x60t_h_teams_2026-11-06`; `rules_x60`). Δ_F on forum72 at 60 units a final: `x60_redesign_2026-10-26/power.py`'s `x60_interval` / `classes` / `rule_x60` PORTED (binomial + 2·max(1 pp, σ̂)²/n, Welch when σ̂²'s one-sided 95 % LB > (1 pp)²; GAIN / LOSS / NO-USEFUL-GAIN / NOT-RESOLVED with power.py's 1e-9; each arm's pooled Σwins/Σgames, which is power.py's mean at equal games; `rules_x60_test` holds the two equal on fixed and simulated draws); `look` 1 (ON and C0 s3501-s3503) futility only (Δ̂ ≤ 0 stops, else CONTINUE), `look` 2 (s3501-s3508) recomputes look 1 on s3501-s3503; the SEEDS are constants (REVIEW I M3: another set refuses at load, a run dir must end `_s<seed>`, a run recording another seed is INCONCLUSIVE); the Welch guard within 1e-12 of the floor takes the wider interval; the floor 20·units·finals − 20 per arm. At look 2 `guards` and `void` are required, `void.pin` per arm, every final gated. Look 2 also: Δ_P (D1's Welch 97.5 %, LB > −2.0), I_F (Welch 95 % GAIN; the guard its one-sided 90 % LB), R72 (DELIVERED iff LB > 0), goal (a) (ON's home72 Wilson LB > 0.50), G-F / G-E / G-C from a sha-pinned artefact on a GAIN only (`readout`'s arithmetic; G-C's production side holds C0's seeds 3501-3508 since AM-X60T-1, so an 8-pair record fits and a five-pair one does not); since AM-X60T-3 each guard is INCONCLUSIVE unless its inputs are provably the read's finals by the state's sha256s (G-F: the 8 × 8 ON × OFF matrix in seed order, each cell ≥ 500 pairs and OK, `h_pp` re-derived from the pentanomials; G-E: `ge_x60t` / `ge_c0` whose targets are the seed-3501 finals; G-C: every pair's `checkpoint_sha256`), the artefact and the re-decision spec built by `python -m main.reads.x60_guards` (`gf-cells`, `build`, `redecide-spec`: every cell imported, the determinism unit re-played, every final pinned), T − D1 (H + F, 5 v 5, reported-only groups), and `reported.pooled`: Δ_H / Δ_P72 / Δ_home72 per team set over units 0-6 of every final, pooled across the reported groups in code (one-cell checks; REPORTED only). §11's verdict is the pure `verdict`: ADOPT-CANDIDATE / TRADE / KAKUNA-SPECIFIC / LOSS / NO-TRANSFER / IN-HOUSE-ONLY / NOT-DELIVERED / NO-USEFUL-GAIN (look 1) / NOT-RESOLVED / CONTINUE / INCONCLUSIVE. Each meter may pool several groups that are ONE cell (§3, two-state imports). Optional `determinism_guard` (a re-played unit against the imported one, choices included; a miss is INCONCLUSIVE). Optional `void`. A stage record (`Rule.stage_keys`, AM-X60T-2): top-level `arm` X60T, `look`, `delta_f` `{final, class}` (final at a look-1 stop or a classed look 2), so `kit stage-check` licenses look 2 and `ge.trigger` the `ge_x60t` reader. Read so far: look 1 CONTINUE (`reads/x60t_look1.json`) | new (2026-11-06) |

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

**The VOID gate** (`rules.void_gate`; be60f24f FINDING 3). An X35-stage rule (`x35_r1`, `d1`, `n4_screen`), `x63_kvk` and `x60` may register
`params.void = {k9b_dir, runs: {label: run dir}}`, every ON model named; `x35_r1` on a batch-2 arm (SPRT, LRF) and `n4_screen` always MUST
(the spec is refused without it). `decide`, and so `record`, runs it FIRST and writes `void.json`: each run's AM-6
K9(b) record (`<k9b_dir>/<run>.json`) must read PASS, and on a batch-2 arm its V-S1 / V-S2 (SPRT) or LR-F KL guard
(`main.x35.batch2_void.read_void`) too. Any other verdict refuses the decision and names the seed: a VOID seed enters no
read (SPRT's `assemble.py` STOP). The record carries the gate, and a record of a batch-2 arm without one is refused.
Three further guards read the RUN DIR's own artefacts and are OPT-IN, each a key of `params.void` (a spec that names none
is gated exactly as before, its gate rows unchanged):

| key | holds | the gate checks, per gated run |
|---|---|---|
| `pin` | a commit sha (7 to 40 hex digits; `PENDING` under `--pin-pending` only), or `{arm: sha}` (each run against its arm's pin; X60-T gates ON, C0 and D1 at their own) | every git hash the run recorded (`metadata.json`'s `git_hash` and each `pin_history` span) is that commit; a run with no `pin_history` span fails (N1-P PREREG §9 guard 3) |
| `pool_seed` | `{fork_step}` | `snapshots/pool_seed.json` is the T54 schema-2 record, its `fork_step` is the registered one, no seeded snapshot is later than it and every dropped step is later (guard 4's record half) |
| `dose` | `{reference, rel_tol}` | `main.dose`'s `lr_median`, `updates_per_env_step` and `dose_rate` each equal the reference run's (the control) to `rel_tol`; a field either run cannot state fails (X60-T PREREG §13.3: a resume that changed the dose) |

The row carries `pin` / `pool_seed` / `dose` (`{pass, why, ...}`), `pass` is the AND of every guard registered, and
`void_failures` names the failing guard beside the K9(b) text. What stays an operator check, as AM-N1P-1 words it: the
startup banner and the held-out hit lines (guard 2; the launcher log is outside the run dir) and the pool-seed check's
startup-line half; a corpus-team crash (guard 5) is the settled rule's (a run that stopped has no final). The same holds
for X60-T's `[HUMAN-DATA] lever=… rowset=… rows_sha=…` stamp "in every launch": the line goes to the launcher log only.
The run dir holds the RECORD it prints (`metadata.json` `lineage.human_data`, the first launch, immutable;
`cli_args._human_data`, the last process's), so a run-dir check could show the first and last launches but never a
resume in between; no `void` key reads it yet.

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
- `rederived_guards` (the re-derived anchor guards, §4: mode, units checked, those `unavailable`, the hits, `held`, the release; absent on a record older than the field; a held record never released is refused);
- `imports` (each group's cross-state import, §3);
- the rule's `inputs` and `decision`.

**A reduced read at a registered stop** (`decision.allow_partial_at_stop: true`; belief-prior-mix FINDING 8: `record` needed
every planned unit played even when the rule's `min_unit_frac` would decide). When a decided ANCHOR group has units
unplayed and the spec registered the option, `record` writes the counted units with `completeness.class` =
`INCOMPLETE_BY_STOP`, `opt_in: "decision.allow_partial_at_stop"`, `groups` (`planned`, `played`, `unplayed`, `counted` per
group, and `unplayed` split in two: `unplayed_exhausted`, with the units named in `unplayed_exhausted_units`, whose retries
were spent with no summary, and `stop_cut`, which the stop never reached; `check` refuses a split that does not add up) and
`stops` (the lane's `stop_at` event: its time, `left`, `not_counted`, the registered `stop_epoch`); `summary_line` names the
class and any exhausted units. It does so only if each such lane ENDED at its registered stop (`lanectl.stop_evidence`: no supervisor of it alive
and the last event of `<lane>_supervisor.jsonl` is `stop_at`, which a supervisor writes only once its `--hours` /
`--stop-at-epoch` has passed); a lane never run, held, killed, refused or done is refused, naming its last event. Only the
unplayed units are exempted: a wrong regime, an S node short of its `pairs` or a failed VOID gate still refuse, and the
RULE's own floor still decides (`min_unit_frac` makes the verdict VOID). Without the option `record` refuses as always.
An exhausted unit that left an uncounted summary is a played, skipped unit (the rule's floor reads it), not an unplayed one.
A lane that finished (`done`) with exhausted units is not at a stop, and the refusal names them.

The record is refused if any regime id is missing (a reported-only group listed INCOMPLETE excepted), a model was
never resolved, a ledger digest or the playing commit is missing, `cells_inconclusive` is not empty (a verdict decided
on cells assemble could not count: a wrong regime or panel, unplayed units unless the reduced read above, a cell short of the spec's `pairs`), the
VOID gate holds a non-PASS run (or a batch-2 arm's record carries none), or the verdict is not the decision's. An X35-stage rule (`x35_r1`, `d1`, `n4_screen`) also carries the
decision's keys at the top level. A rule whose decision keeps its own shape but gates a stage (`Rule.stage_keys`: `x60`)
carries only its stage identity there (`arm` = the ON arm's `batch.READ_LABEL`, `look`, and its primary's `{final,
class}` under `batch.PRIMARY_KEY`, `delta_f` for X60-T), derived from the record's `models` and `decision`, which `check`
re-derives. When written to `<stage>/reads/<arm>_look<k>.json`, either file therefore loads in
`main.x35.batch.read_record`, `final_class` and `ge.primary_class` unchanged; `read_record` runs `record.check` on a
`gen3_read_record_v1` file first, so a hand-edited stage key refuses. A ledger entry cites the record by its path and
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
| two supervisors on one read (D3 reader 10-26: the state renamed aside took its in-state lock; a `kakuna run` on a state inside the read's) | first live use | the path lock outside the state, the `/proc` scan, the dir-identity stop, `plan`'s refusal, the lane's claims on the kakuna states it plays (§3) |
| a unit whose retries were spent recorded as cut off by the stop (INCOMPLETE_BY_STOP) | any reduced read | the attempts read back from `units.jsonl`; `exhausted` in `status`, `assemble` and the record (§3, §6) |
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

- **S-U's phase 2 and its REPORTED extras** (`search_headroom`, §5). The rule reads ONE `per_game` group, whose team
  sets share its unit seeds, and a searched group imports nothing, so phase 2 (home72 seeds 20–59 only) has no spec
  shape yet. The level-at-10 subgroup, the world-quality covariate, the raw arm against X51 and H-MID's Δbias re-read
  need the captures, `search_decisions.jsonl` or X51's rows, which the cells do not carry; the decision lists them
  under `not_reported`.

- **Reported-only extras** (SPRT's mechanism block, dose from launcher summaries) stay per-read. A rule's `params`
  can carry them, but only the VOID gate and `n4_screen`'s mean LR (`metadata.json`) read run dirs. `n4_screen` lists
  what it cannot compute (ρ, approx KL, G-F, the experts' divergence, home72, K's power) under `not_reported`.
- **A repin's reach.** `repin` proves the TRACKED play path identical (§3). It cannot see what git does not hold (the
  Metamon cache, panel team dirs outside the repo), and its closure is static (a module named only in a string is not
  followed; the prefixes over-approximate for that). A change in the box's untracked inputs is the units' own preflight's.
- **The VOID gate's run-dir guards are opt-in** (§5): X61P's committed specs register K9(b) alone (AM-N1P-1; editing a
  registered spec moves its sha256), so their `pin`, pool-seed and banner checks stay the operator's. A new spec registers `pin` /
  `pool_seed`; the launcher log's banner lines are outside the run dir and stay manual.
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
