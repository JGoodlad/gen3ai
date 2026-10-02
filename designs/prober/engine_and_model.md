# The prober's torch boundary and its trace-discovery layer

Owned by this tree (always current, per the root `CLAUDE.md`). The rules an agent needs before
touching either — the engine/app seam, the module map, the per-battle resolution ladder — stay in
`src/main/prober/CLAUDE.md`; this file is the detail those two modules hold.

## `model.py` — `ProbeModel`, the only place torch is called

- **`model.py`** — `ProbeModel`: the torch boundary. `ProbeModel.load(ckpt)` does
  raw `MaskablePPO.load` (no env, no `ModelVersion` check — matching the legacy
  CLI) and resolves `ObsOffsets` once from `enc.get_layout()`. `action_dist` /
  `logit_grad` are the only forward/backward passes (`belief` adds one when a
  belief-on checkpoint is loaded — see below). **`ProbeModel.belief(obs, mask)`**
  runs one clean forward and reads the belief head's stash
  (`features_extractor.last_belief_logits["species"]` + `last_opp_believed_mask`) →
  `(species_logits[6,n_species], believed_mask[6])`, or `None` when the checkpoint
  has no belief head; the engine decodes/matches it (the OPP-TEAM belief, below). Three **non-torch decode helpers** also live here (they need the encoder,
  so the model is the natural home): `describe_global` (weather/spikes/screens + a **pending-Wish**
  `wish_our`/`wish_opp` flag decoded from the `gen3_wish_wired_v1` reactive scalars — the floating heal,
  surfaced on the FIELD line as `💧wish: our/opp`); `describe_team` —
  decodes each mon block's **held item + moveset** via `pokemon_encoder.describe_vector` over BOTH
  team blocks (`OFFSET_OUR_TEAM`/`OFFSET_OPP_TEAM`), surfacing the **opponent's item + revealed
  moves the moment they appear** (unrevealed item → `ITM-UNKN`, skipped); and `describe_turn_outcome`
  — decodes the **most-recent TurnDelta** (the history block's LAST slot) for each side's **crit**
  (`OFFSET_*_CRIT`), **couldn't-move reason** (`*_cant`), **boost change** (`*_boost_delta` →
  `atk+1`), and **move order** (`move_order` → who went first), via
  `turn_delta_encoder.describe_vector`. The engine overlays `describe_team` on the summary's
  our-only teams block, and reads `describe_turn_outcome` from the NEXT decision's obs (turn T's
  events land in decision T+1). (Hidden Power's specific TYPE — all 16 share move-num 237 — is
  recovered in `observation/moves.py::describe_vector` from the move's type channel, so a decoded
  moveset shows `hiddenpower(fire)` for our own / a revealed HP; an opp's un-revealed HP stays bare.)
- **`discovery.py`** — pure filesystem. `build_trace_tree(path)` accepts a run
  dir, an `eval_traces` dir, or a single `*_summary.json`, and groups
  step → opponent → battle by **parsing path strings only** (never opens the
  JSON/npz — lazy-loaded on selection, so opening a 1000+-battle run is instant).
  `_FNAME_RE` matches BOTH `<outcome>_<idx>` and the work-stealing eval's
  shard-namespaced `<outcome>_s<shard>_<idx>` (the shard folds into `index` so two
  shards' same-idx traces stay distinct; un-sharded `loss_001` is unchanged). Without
  this the whole prober was blind to every sharded-eval run (outcome parsed as `?`).
  🚨 **`<outcome>` is `win` | `loss` | `draw`, and the alternation is BUILT from
  `agents.training.trace_result.OUTCOMES`** rather than retyped here — same producer↔consumer
  pair, same failure mode. **`draw` joined the vocabulary on 2026-09-07**
  (`gen3_trace_result_v2`); before it, a 250-turn TIMEOUT was written as an ordinary `loss_*`
  and a true TIE matched neither quota branch and was **dropped without a file**. Verified over
  the whole archive: 145,173 traces, every one `win_*` or `loss_*`, `meta.result` never anything
  but `WIN`/`LOSS`. **A tree with no draws is not a tree that had none** — see *THE RESULT
  VOCABULARY* below.
  It also reads each cycle's `eval_manifest.json` (model identity). The model to
  re-run a trace through is chosen **per battle** by `resolve_model_for_step`.
  Each battle's `*_summary.json` / `*_states.npz` has a sibling
  `*_replay.html` (`write_battle_record`) — a browser-watchable Showdown replay
  the prober ignores but a human can open directly. Bridge-eval traces add a
  fourth sibling, `*_reconstruction.json` — the battle's full-information
  replay/re-roll record (`utils/bridge/reconstruction.py`) — consumed by the
  `falsify` / `lookahead` / `replay_counterfactual` re-roll probes (and the
  privileged opp-team belief view).

## Rust-eval CORE TRACES — expanded on read (`core_trace.py`)

The Rust eval path (`--env-core rust`, M5 Lane H) has no poke-env battle, so
`agents.training.rust_eval.traces.write_core_trace` persists a **core trace**
(`meta.trace_source.schema == "gen3_core_trace_v1"`, `is_core_trace`): a META-ONLY
`*_summary.json` (no `teams`, no `invocations`), the two sides' `gen3_core_event_v1` records
(`<prefix>.p1.jsonl.gz` / `.p2.jsonl.gz`), the `*_reconstruction.json` and a `*_states.npz` in the
Python recorder's keys (legal LOG-PROBS in `logits`, illegal = −1e9).

**THE LOADER — every `*_summary.json` read under `src/` goes through `core_trace`** (F-LH-5, closed
2026-09-30). A direct `json.load` of a core trace's summary reads `meta` only, and a reader that then
walks `summary.get("invocations", [])` reports ZERO decisions without raising. Three entry points:

| entry point | returns | for |
|---|---|---|
| `load_summary(path, *, run_dir=None)` | the full summary; a core trace EXPANDED (below) | every reader of decisions / teams |
| `load_summary_meta(path)` | the STORED `meta` (a core trace stores the full recorder meta + `trace_source`); never expands | outcome / turn-count / count readers |
| `refuse_core_trace(path, *, reader, why)` | the stored summary of a Python trace; RAISES `CoreTraceUnsupported` on a core trace | a reader that needs what a core trace does not carry (the recorded `win_probs` head) |

The static gate `src/trace_summary_reader_gate_test.py` (unmarked, ~1 s, EMPTY allowlist) fails any
module other than `core_trace.py` that opens a `*_summary.json` for reading. It tracks the path
through names, `os.path.join` / `Path` / `glob`, for / list / dict comprehensions (`.values()` /
`.items()`), parameters named `summary_path` / `summ_path` / `spath` / `smf` / …, `.summary_path`
attributes, same-module helper calls (a carried argument taints the parameter; a function that
RETURNS a carried path taints its call), and stops at a stripped or replaced suffix (the siblings). A path handed in from another module
under an unrelated name is its blind spot. Opt out with `GEN3AI_SKIP_SUMMARY_READER_GATE=1`.

**Which reader does what on a core trace** (behaviour pinned by `core_trace_readers_test.py`):

| reader | entry point | on a core trace |
|---|---|---|
| `ProbeSession._summary` / `_meta` (every prober view), `forensics`, `rust_eval.parity.compare_traces` (both sides), `probe_replay`, `mechanic_usage_baseline`, `audit_states` (mask fallback), `search_dividend.search_decision_benchmark` / `ab_racing`, `rust_sim/harness/better_line_bench` / `gen_search_golden` | `load_summary` | expanded — reads every decision (`ab_racing` re-raises a `CoreTraceError` rather than log it as a skip) |
| `ops.conditioning_meters.extract_cycle` | `load_summary` | expanded under `--v-column values`; REFUSED under the `win_probs` column (NaN head) |
| `critic_gate._trace_turns` (G7), `ops.quota_match.classify_on_disk`, `harvest_meter._load_tail` / `control_battles` | `load_summary_meta` | stored meta; `harvest_meter`'s `recorded_phi_T` is `None` (not NaN) on a NaN head |
| `cf_audit.build_frame`, `harvest.build_candidates` | `refuse_core_trace` | REFUSED — both sample by the recorded win-prob head; the refusal escapes their counted-skip `except` |
| `scaffolding_gauge.collect_slices` (and every meter over it: `critic_gate`, `ops.critic_readouts`, `ops.perbot_*`, `ops.negskill_null`) | `is_core_trace` on the NaN head | REFUSED — contestedness / the V-vs-P(win) gauge have no P(win); the gauge's old refusal blamed `--win-prob-mode none` |

**The expansion** (`core_trace.expand` / `load_summary`; `ProbeSession._summary` calls it, as does
`forensics.build_decision_table`):

1. `replay_battle(record, impl="rust")`;
2. **CROSS-CHECK** — the replayed trainee-side protocol must equal the stored record's `text` lines,
   `|t:|` lines and blanks dropped on both sides. **The record is the authority**: a difference
   raises `CoreTraceMismatch` naming the first differing line;
3. the replayed chunks go through `obs_materializer`'s transport-less replay player (the mirror of
   `EvalRLPlayer.choose_move`: the stall check first, an all-zero mask is no decision), and at each
   decision a real `BattleRecorder.record(battle, actions[i], softmax(masked logits[i]), mask,
   {"obs", "logits", "value"})` runs — the same torch softmax the live player computes — then
   `finalize` + `to_summary`. The per-turn `outcome.reward` is scored with the run's
   `RewardConfig.from_dict(<run>/model_config.json)` (the eval worker's rule; absent ⇒ the default);
4. **REFUSE** unless the decision count equals `meta.invocations` and the `states.npz` rows, each
   decision's legal mask equals `action_mask[i]`, and the replayed result equals `meta.result`;
5. the recomputed `meta` is replaced by the STORED one (it keeps `trace_source`).

The trainee's stall forfeit mirrors the live rule on both sides: the decision at `turn >=` the
threshold is a forfeit, not a row (`rust_env::episode::stall_forfeit_due`, the executor's row
filter, `_handle_stall` here). The threshold is `trace_source.turn_limit` when the writer records
it, else `StallConfig().threshold` (production runs the core at exactly that). An opponent forfeit
just ends the protocol. Expansions are cached **in memory** by path + input mtimes (≤256); the run
dir is never written.

**Pinned equal to the live recorder.** `core_trace_integration_test.py` plays each core game again
LIVE on the rust bridge (same seed and teams, p1 an `EvalRLPlayer`-shaped scripted player
recording with `BattleRecorder`, p2 replaying the stored tokens, written by `write_battle_record`):
the expanded summary equals that summary field for field (bar `meta.battle_id` /
`meta.trace_source`), including a stall-forfeit game, and the live encoder's obs equal the core's
stored obs row for row.

**What a core trace does NOT carry:**

| absent | what reads it | how it degrades |
|---|---|---|
| `*_replay.html` | `_protocol_lines` / `_protocol_for` (`turns`' timeline, `loops`, `analyze`'s raw protocol) | `core_trace.protocol_log` stands in — `battle._build_replay_events()` of the expansion's battle, exactly what `save_replay` would have rendered |
| `win_probs` (NaN), `move_logits`, `spread_belief` (and `value_dist`, which no new trace carries) | the win-prob / belief-trajectory views | read "unavailable" exactly as on a head-off run; `analyze` re-runs the model on the stored obs |
| `belief` / `opp_intent` per invocation | `opp_intent` text, belief panels on the summary | absent, as on a head-off run |

**Cost:** one rust replay subprocess + one poke-env feed per battle, measured 0.03–0.7 s on the
fixture games (first call pays the imports). A `scan` over N core traces pays N of these once per
process.

**The room tag.** A core trace's `battle_tag` is its `battle_id` (`core-<step>-<opp>-g<k>`), which is
not a poke-env room name (segment 1 must be the format). `obs_materializer._next_tag` now PREFIXES
such a tag — before, every `falsify` / `lookahead` / `better_line` / `replay_counterfactual` on a
core trace replayed ZERO decisions and read "replay desync".
