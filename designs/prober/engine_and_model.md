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
  has no belief head; the engine decodes/matches it (the OPP-TEAM belief, below). **On load it silences the
  policy's `ObservationDebugger`** (a `--log-level periodic` checkpoint prints a
  "DEEP TRACE" banner on every forward — pure noise that would corrupt the
  output). Three **non-torch decode helpers** also live here (they need the encoder,
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
