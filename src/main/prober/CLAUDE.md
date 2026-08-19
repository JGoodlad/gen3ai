# CLAUDE.md — `src/main/prober/` (forensic-replay inspector)

The **prober**: browse the `eval_traces` a training run writes and inspect *why the policy chose
what it did* at any saved decision point.

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober <models_dir | run_dir>        # the browser front end -> :6008
python3 -m main.prober.query <cmd> ...               # the JSON CLI, for agents and scripts
```

**TWO surfaces over one engine, and that is the whole design.** `engine.py` + `session.py` are the
analysis; `web/` renders it for a human and `query.py` prints it for an agent. Neither is a layer
on the other.

**⚠ THE TEXTUAL TUI IS GONE** (`app.py`, `prober.tcss`, `review.py` — deleted, ~4,400 lines). It was
a *third* renderer over the same engine, which meant every new signal had to be drawn twice for a
single reader — the v67 α/β read was, days before this. `python -m main.prober` now starts the web
app, and its two TUI-only flags (`--ckpt`, `--inv`) print what replaced them instead of failing.
Dropped with it, each on evidence rather than taste:

| dropped | why |
|---|---|
| **Flow** (box-art dataflow) | `model.g5d.io` already draws the architecture, interactively, with the measured-dependence overlay |
| **Team** / **Board** panels | subsets of `/battle`'s board; the field line (weather/hazards/screens) survives on `/analyze` |
| **Review mode** (flags + notes) | 5 of 84 runs, unused since June, and the only thing that would have made the web read-write. Notes exported to `<run>/review_notes.md` first |
| **refine rounds** (axis A) | needs `--damage-refine-rounds`, which the production config does not run |

Everything else was ported: the per-decision **`analyze`** view and the counterfactual tier
(`lookahead` / `better_line` / `replay_counterfactual`, as password-gated background jobs off
`/analyze`). See `web/CLAUDE.md`.

**Reading a game turn by turn** is `battle_turns()` (below) — model-free, so it opens instantly.
`query turns` prints the whole game as JSON; `/battle` renders it as a phone-readable replay. The
plain-text battle log is `engine.timeline_entry_text`, and the vocabulary it draws on
(`engine.CANT_PHRASE` / `NO_EFFECT_TEXT` / `surprise_phrase`) lives in the ENGINE precisely so that
a reason one surface learns cannot go missing on another.

## Engine / app split (the important seam)

The analysis is a **pure, framework-agnostic engine** (`engine.py` + `model.py`); the web front end
(`web/`), the JSON CLI (`query.py`) and the one-shot `probe_replay.py` are all thin callers. This is
the single source of truth — change the analysis once, every surface follows. It is also why
retiring the TUI cost no analysis: the deleted 4,400 lines were rendering, not reasoning.

- **`engine.py`** — `analyze_invocation(model, summary, npz, inv_index) →
  InvocationAnalysis` (a tree of frozen dataclasses: `ActionRow`, `MatchupView`,
  `InterventionSweep`, `Saliency`, …). No printing, no Textual, no file IO beyond
  the passed-in arrays. Split into `_faithfulness` / `_matchups` /
  `_intervention_sweep` / `_saliency`. **Every torch call goes through the
  injected `model`**, so the whole engine is unit-tested with a `FakeProbeModel`
  (no torch) — see `engine_test.py`.
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
  It also reads each cycle's `eval_manifest.json` (model identity). The model to
  re-run a trace through is chosen **per battle** by `resolve_model_for_step`.
  Each battle's `*_summary.json` / `*_states.npz` has a sibling
  `*_replay.html` (`write_battle_record`) — a browser-watchable Showdown replay
  the prober ignores but a human can open directly. Bridge-eval traces add a
  fourth sibling, `*_reconstruction.json` — the battle's full-information
  replay/re-roll record (`utils/bridge/reconstruction.py`) — consumed by the
  `falsify` / `lookahead` / `replay_counterfactual` re-roll probes (and the
  privileged opp-team belief view).
- **`web/`** — the browser front end (FastAPI + Jinja2/HTMX over `ProbeSession`). It is
  **first-class for the GPU obs**: the learned belief/op signals tagged `🔷 GPU` render PRIMARY
  and the decoded CPU obs regions they subsume tagged `📋 CPU-obs` render dimmed, because the
  operator's physics supersede the older type-effectiveness decode and a reader must be able to see
  which is which. **Beliefs** is the model's world-model vs ground truth; **Threats** leads with the
  DamageOperator. See `web/CLAUDE.md`, and *Beliefs / Threats (GPU-first observability)* below for
  what those panels MEAN.

## ⚠ Architecture drift — a model-loading probe only works on the CURRENT generation

**MEASURED 2026-08-13 over every run in `models/`: 79 runs carry a checkpoint, and 0 of them load
under current code.** Not one archived run is even at the current obs dim — the closest is 2667
against the code's 2669 (the v65 deadline clock's +2). What they were trained on: `2992` ×42 ·
`3409` ×8 · `3457` ×8 · `3469` ×6 · `2667` ×5 · `3390` ×3 · `3391` ×3 · `2889` ×3 · `2925` ×1.

This is **by design, not a bug** — the root `CLAUDE.md` says *"checkpoint compatibility is not a
concern"* and `ARCH_SIGNATURE` exists to reject stale checkpoints. But it decides how to read this
whole tool:

| tier | works on | why |
|---|---|---|
| **model-free** — `scan` · `triage` · `turns` · `awareness` · `loops` · `overview` · `find` (bar `disagree`) · `falsify` · `falsify_scan` · `calibration` · `decision_table` | **every run, forever** | reads the trace on disk; no checkpoint |
| **model-loading** — `analyze` · `probe` · `lookahead` · `better_line` · `replay_counterfactual` · `history_saliency` · `find disagree` | **only a run at the CURRENT arch** | re-runs the policy under today's code |

So the durable surface is the model-free one, and it is not a coincidence that the web front end was
built there first. A model-loading view is worth having for the run you are *currently training* and
stops working the day the obs layout moves.

**The failure is now a DIAGNOSIS** (`ProbeModel.load` → `ArchDriftError`, `model.py`). Three walls
were measured, each of which used to surface as a raw error from inside SB3:

1. a DELETED flag still baked into the zip's `features_extractor_kwargs` → `TypeError: unexpected
   keyword argument 'spread_belief_nature_marginalize'`. **Recovered**: unknown kwargs are dropped
   (`_accepted_extractor_kwargs` introspects the live `__init__` signature) and *which* ones is
   reported — a dropped flag means the rebuilt extractor is not the one that played.
   ⚠ **Recovery rate on today's archive is ZERO** — every run carrying a deleted flag also differs
   in obs dim, so the drop alone never rescues one. It is unit-tested, not archive-proven, and it
   exists for the *next* pure-flag deletion (v66 was exactly that shape).
2. a value the code now VALIDATES → `move_candidate_floor=0.0`, legal when trained, rejected since
   the v65 legality guard. **Not** recovered: relaxing a correctness guard to probe would answer
   with a different model than the one under the microscope.
3. weight SHAPES that no longer fit → `mat1 and mat2 shapes cannot be multiplied (12x380 and
   386x256)`. Not recoverable in principle.

The error names the saved-vs-current obs dim, the saved-vs-current `arch_signature`, the dropped
flags, the underlying cause, **the exact `git checkout <hash>`** to re-probe from (read from the run
`metadata.json` — the sidecar search walks up THREE levels, because an eval snapshot sits at
`<run>/eval_traces/step_<N>/snapshot.zip` and a two-level search silently lost the hash on every one
of them), and which model-free views still work on that run. `peek_checkpoint` reads the arch
fingerprint from the zip's JSON `data` member in **~5 ms** — that is what makes diagnosing cheap
enough to do before the load is even attempted. `analyze`'s `model_resolution.dropped_kwargs`
carries the drop to a surface. Tests: `model_test.py`.

## Per-battle model resolution (exact → nearest → most recent)

A trace was generated by the eval snapshot at *its* step, so the prober resolves
and loads the model **per selected battle**, not once at startup
(`discovery.resolve_model_for_step(tree, step, override, tier)` → `ModelChoice`):

1. **exact** — the retained `eval_traces/step_<N>/snapshot.zip` (written when
   training ran with `--keep-eval-snapshots`); bit-exact, faithfulness ≈ 100%.
2. **nearest** — the persisted `checkpoint_<N>_steps.zip` with the smallest
   `|Δstep|` (exact weights at a nearby step). `discovery.list_checkpoints` searches BOTH the
   current `<run>/checkpoints/` and the legacy `<run>/` root (deduping a copy-backported step to
   the `checkpoints/` path), so the ladder finds checkpoints under either layout.
3. **most recent** — `best_model` / latest.

`--ckpt` forces an override. The badge shows the active tier + the trace's
`git_hash`/`arch_signature` from the manifest, so any faithfulness drift is
explained. `m` cycles the preference (`auto` → `nearest` → `recent`) to view a
trace under a different model; `R` reloads. Loaded models are cached by path
(`_model_cache`), so revisiting a step is instant. Even for runs that predate the
manifest (no snapshots), the ladder picks the **nearest checkpoint** — strictly
better than always using `best_model`.

## What one decision's analysis CONTAINS (the `/analyze` panels)

Everything below renders purely from one `InvocationAnalysis` — this section is about what the
FIELDS mean, which is why it survived the TUI that used to draw them. Where it still describes a
terminal (glyphs, colour ramps, fixed-width columns), read that as the SEMANTIC it encodes: `/analyze`
renders the same distinction in HTML, and `web/CLAUDE.md` says how. Panel-by-panel field map, kept
current with the renderer, lives there.

The lead panel is **Summary** — the decision dashboard for walking "funky turns". Its header is
three groups — SITUATION (matchup + FIELD + THREAT), DECISION (CHOSE), OUTCOME
(RESULT + REWARD + CRITIC): line 1 the matchup, each active as **species + colour-graded HP bar**
(`_hp_bar`) + bundled **status/volatiles** in `[...]` (e.g. `[TOX(5)|SUB]`) + **boosts** in
`{...}` magenta (e.g. `{atk:-1 spa:+6}`) + held **item** as `@item` (incl. the **opponent's once
revealed** — Choice items highlighted) + outcome; then **FIELD** (weather/hazards/screens/turn +
a `💧wish: our/opp` tag when a Wish is floating — `gen3_wish_wired_v1`, the ~50% end-of-turn heal —
the same `_field_text` the Board shows) · **THREAT** (STACKED, so the Summary is self-sufficient —
line 1 incoming P(KO)·outspeed·worst-on-team·opp-recovery, line 2 the incoming type-**effectiveness**
`worst N× · revealed X%` folded in from Matchups; P(KO) reds with danger in BOTH places —
`gradient_color(1 − pko)`) · **EXPECT** (the LAST line of SITUATION, only on an
`--opp-intent-coef>0` run) the v67 `α` head's ranked NAMED options for what the OPPONENT was about to
do — their believed moves plus `SWITCH`, with `β`'s named switch-in appended *only* when `α` actually
leads with the switch. It sits in SITUATION, not DECISION, because it is part of the position as the
model read it; and it is the ONE line that distinguishes a turn the model played AROUND a threat from
one where it never saw the move coming — the board, the log and the critic's numbers are identical in
both (`a.opp_intent`, `engine.build_opp_intent`, model-free) · **CHOSE** chosen+confidence [+ a
`⚠ now prefers X` on disagree] ·
**RESULT** what actually happened — an **ordered, one-line-per-action battle log**
(`engine.build_result_timeline`, a pure list-of-dicts attached to `outcome["timeline"]`, so the CLI
`analyze` JSON carries it too). The recorder stores each side's action + that mon's OWN net HP change,
which renders as nonsense ("we icebeam (−72%)") because a mon's HP loss is dealt by the OPPONENT's
move; the timeline **RE-ATTRIBUTES** each loss to the move that caused it and pairs it with the
target's before→after HP (`before = after + damage`, `after` from `next_board`): `opp hiddenpower did
72% (tyranitar 100% → 28%) ⚡CRIT` / `we icebeam did 100% (salamence 100% → faint)` / `opp sends in
metagross`. Lines read **top-to-bottom in execution order** (a voluntary switch resolves first, else
the TurnDelta `move_order`, which folds from the **real event-log sequence** — `TurnView._compute_move_order`
reads the order of `|move|` events, not a speed heuristic — so **no `«1st»` tag**). When BOTH sides moved
but `move_order` wasn't recorded (a no-state / model-free decision), the engine first tries to READ
the order off the turn's raw protocol slice — `engine.move_order_from_protocol`, since the sim emits
`|move|` lines in EXECUTION order, which is the fact rather than an inference from it — and only
flags `order_certain=False` (neutral `·` bullets + a *(move order not recorded)* note) when even
that cannot settle it. **This is the common path, not an edge case:** `move_order` is decoded from
the OBS, so it exists only on the model-loading route — measured over five runs, ZERO decisions
carry one in a model-free `battle_turns` read, and the note fired on **56.3%** of decisions with a
timeline (389 decisions, ai_v9_17_tdaux_lam3). With the protocol read it is **3.6%**. Sides are
identified by matching the actor's nickname to the two active species, never by assuming we are
`p1`; a mirror match, a real nickname, or a switch-only turn all return `None` and keep the honest
"not recorded" rather than a coin-flip. A RECORDED `move_order` still wins — it folds from the event
log the analysis was built on — and a voluntary switch outranks both, because that is mechanics. One line per move / switch / forced replacement /
standalone faint, each carrying a **`⚡CRIT`** tag, a **`→ atk+1`** boost (Meteor Mash / Intimidate),
an applied **status** (`opp thunderwave → milotic PAR`), or a **"couldn't move (asleep/…)"** note
(crit + boost + cant + move_order + **effectiveness** decoded from the NEXT decision's TurnDelta via
`describe_turn_outcome`; Hidden Power's BP-0 placeholder is special-cased so its hits aren't dropped).
**A move that did NOTHING visible is explained, never left blank** (`engine._no_effect_reason`): an
attack/status move blocked by a type immunity reads `— no effect (immune)` (Seismic Toss vs Ghost), a
move that **missed** reads `— missed` (Hypnosis), a connected-but-fizzled move `— no effect`; a hazard /
heal / boost move (whose effect is legitimately invisible in the outcome) is left bare. **miss/fail is
a RECORDED fact** — the `gen3_move_outcome_v1` TurnDelta block encodes each side's `[hit, miss, fail]`,
decoded by `describe_turn_outcome` as `our_move_outcome`/`opp_move_outcome` (so it distinguishes a true
miss from a hit-that-did-nothing); only on a model-free / pre-`v1` trace does it fall back to inferring
a miss from the move's accuracy. When the OPPONENT
voluntarily switched, the recorded hp_delta can't price the hit on the switch-IN (it compares the mon
that left), so the attack shows the **resulting HP** instead (`we rockslide → celebi (now 11%)`) — the
attack is never dropped. The shared renderer is `app._append_timeline_entry` / `_append_happened`
(Summary + Review card) ·
**AFTER** the RESOLVED
board at the start of the next decision — mirrors the matchup line via `_append_summary_active`, so it
carries the same **species + HP bar + `[status]` + `{boosts}` + `@item`** (a freshly applied PAR/SLP or
a boost shows here, from `a.next_board` = `build_board` of inv+1) so before (matchup line) → after
reads at a glance — a switch/faint shows the new mon ·
**REWARD** the env's reward (total + per-component breakdown) · **CRITIC** V·ΔV·**TD-surprise** (always paired
with a plain-language gloss — "worse than the critic expected" — via `_append_surprise`/
`_surprise_phrase`, so the ML term is self-explaining) · **WIN-PROB** (last, only on a
`--win-prob-mode != none` run) the win-prob head's calibrated **P(win)** + **ΔP(win)** to the next
decision — the interpretable [0,1] complement to CRITIC's shaped-return V ("how much this move moved
the win odds"); greener = better odds, red at low P(win) (it is `None`/absent on a non-win-prob run).
Below sit **three** side-by-side panels (packed at the left): **MOVES** (a DataTable — each move's
type-effectiveness `×mult` fused with its policy prob, ranked; a **non-damaging move** (Spikes/Toxic/
Protect — `not gen3_data.moves.is_damaging`) renders **`—`** instead of its multiplier, because the
obs computes a phantom `×mult` for every request slot and a "2.00×" on Spikes is a misleading artefact,
not a signal — the applicability rides `MatchupView.applicable`, so the `analyze` JSON CLI carries it
too) and two **custom-rendered Static
panels** (NOT DataTables, so a mon's **moveset spans the full width** below it as `⮡ m1 · m2 · …`):
**SWITCHES** (each target's prob · **hp** colour-bar · **status/volatiles** · **risk-in** =
`incoming.per_slot_pko`, with the held **item inlined into the name** as `(leftovers)` lowercase)
and **OPP TEAM** — the opponent's **WHOLE team** (privileged, from the `reconstruction.json`) with a
**revealed/unseen icon** on every **mon**, **held item**, and **move**: `✓` (green) = seen on field this
battle · `○` (dim) = known from the team but not yet revealed in-game (active mon keeps `▶`; HP bar +
status show only for revealed mons). Built by `engine.build_opp_full_team` → `OppFullTeamView` (merging
the truth `opp_team_details` with the board's revealed moves — typed-HP-aware via `_norm_move`), rendered
by `app._opp_full_team_text`, and on the `analyze` JSON too. Falls back to the **revealed-only** panel
(`_team_panel_text`) when there's no privileged team (websocket/older traces). The model's **belief**
about the still-hidden mons used to be appended here — it now lives ONLY in the dedicated **Beliefs
section** (key `b`, `_render_beliefs`). Two belief forms, best-available wins:
- **Privileged truth + matched guess** (`a.belief_truth`, `engine.build_belief_truth` → `BeliefTruthView`,
  `app._append_belief_truth`) when the trace has a **`reconstruction.json`** sibling (bridge-eval referee
  data): shows the opponent's **FULL** team — revealed mons listed, then each STILL-HIDDEN mon with the
  model's species guess **slot-matched** to it, sorted **best-match-first** (by the true species' rank).
  A 3-way marker scores each: `✓` top-1 right · `≈` the true mon IS in the belief but not top-1 (a
  near-miss) · `✗` not in the top-k at all; the true species is highlighted in the guess list, with its
  rank `(#k)` when not top-1 + a `n_correct/n_hidden` header. The believed slots are
  anonymous, so they're **Hungarian-assigned** to the true hidden mons by min `-log P(true species | slot)`
  — **the SAME species-CE cost the training aux loss matches on** (`instrumented_ppo._belief_aux_loss`), so
  the correspondence is how the model itself aligns the slots (`scipy.optimize.linear_sum_assignment`). The
  privileged team is loaded by `app._load_opp_team` (the `reconstruction.json` sibling → `team_details`, file
  IO kept OUT of the pure engine) and threaded into `analyze_invocation(opp_team=…)`.
- **Anonymous belief** (`a.belief`, `BeliefView`, `app._append_belief`) as the fallback (no reconstruction
  record / websocket trace): the per-unrevealed-slot top-k `species NN%` guesses without a true-mon match.

### The SPECIES-CLAUSE reading (`a.exclusive_belief`) — what it is and what it is NOT

`BeliefHead` publishes one **independent** softmax per hidden slot, so nothing in its
parameterization can express *"at most one of you is Salamence"*. Measured on gen-15, three hidden
slots read P(Salamence) = 0.39 / 0.60 / 0.39 at one decision — an expected count of 1.38 on a team
the species clause caps at 1. `engine.build_exclusive_belief` (over the pure operator
`agents.inference.species_exclusivity`) applies that constraint at READ time and publishes an
`ExclusiveBeliefView` beside the raw one: the adjusted per-slot rows, a **point team hypothesis**
(the greedy no-duplicates assignment — most likely team consistent with the clause), and the raw
belief's incoherence headline.

> ⚠️ **The model's belief is `a.belief`, the raw marginals. `a.exclusive_belief` is a reading aid.**
> Both are always rendered; showing only the adjusted view would substitute the prober's arithmetic
> for the model's actual state, which is the same class of dishonesty the whole tool exists to
> avoid. The panel says so in its own copy, and `app_test.py` pins that it does.

**It hangs off the RE-COMPUTED branch only, never the summary fallback** — the summary's `belief`
block carries the top-3 per slot, whose rows do not sum to 1, so running the operator on them would
answer a different question while looking identical. That is also why `battle_turns` / `/battle`
(model-free by construction) do **not** carry it.

**Two DIFFERENT defects, kept apart** (both on the view, neither folded into the other):

| field | defect | measured on gen-15 (3000 decisions, `tmp/species_exclusivity_measure.py`) |
|---|---|---|
| `max_expected_count` / `illegal_mass` | the DISTRIBUTION is jointly illegal (E[count] > 1) | 2.5% → 3.5% of decisions; peak 1.70 |
| `duplicate_top1` | the DISPLAY is illegal — two hidden slots NAME the same mon | 6.5% → 14.2% of decisions (@2M → @6M), 29.9% at 4 hidden slots |
| `revealed_leak_max` | mass on an already-revealed species — flatly wrong under any reading | **clean**: max 3.2e-4, i.e. the `SPECIES_CLAUSE_LOGIT` floor |

The duplicate-top-1 case is the common one and the whole reason this view exists: two slots can share
an expected count of 0.74 — which no clause forbids — while the panel still reads *"both hidden mons
are Salamence"*.

⚠️ **The clean leak reading is a property of `--species-prior-fusion`, not of the publication path.**
That flag floors a revealed species at `SPECIES_CLAUSE_LOGIT` (~1e-6) inside the fused prior. Nothing
downstream masks: `belief_decode.decode_species_belief` and `engine.belief_view_from_logits` both
softmax the FULL species vocab. So on a fusion-OFF run the leak is unbounded and untested — do not
carry "the belief never leaks onto revealed mons" as a general fact.

The belief itself is **re-computed from the loaded model** each analysis (`ProbeModel.belief` → the belief
head's per-slot species logits + believed mask → `engine.belief_view_from_logits`; one clean forward, since
the intervention-sweep/saliency passes clobber the extractor's stash), so it works for **any belief-on
checkpoint** — including runs whose recorder predates the summary's per-decision `belief` block. `engine.build_belief`
reads that summary block as a **model-free fallback** (available even without a captured `.npz`). Both
`belief`, `exclusive_belief` and `belief_truth` ride the `analyze` JSON output (`asdict` —
`ExclusiveBeliefView.coherent` is a stored FIELD rather than a property for exactly that reason).
`None`/absent on a belief-off run (then
only the revealed mons show). **Mon names are
blue** (`_MON_COLOR`); **disabled slots** (a fainted mon / an illegal switch / a no-PP move) render
**grey** (`_DISABLED_GREY`), NOT the red of a low value — so "dead/unavailable" reads differently
from "alive but low HP = real danger". Hidden Power shows its **type** (`hiddenpower(fire)`) — for a
revealed HP from the obs type channel, and **for OUR OWN un-revealed HP from the reconstruction
record** (`engine.build_our_hp_types` → `_retype_hp`, threaded as `analyze_invocation(our_hp_types=…)`,
loaded by `app._load_our_hp_types` / `session._our_hp_types`): Showdown's request carries only the bare
`hiddenpower` id (the type is IV-derived), so without this our own mons showed an untyped HP until they
used it. **OUR side only** — an opponent's un-revealed HP MUST stay bare (no leak), and the retype is a
no-op on websocket/older traces with no `reconstruction.json`.

### Beliefs / Threats (GPU-first observability)

The **Beliefs** section (`b`, open by default) is the model's **world-model vs ground truth** — built to
be first-class for the learned GPU obs (every datum tagged `🔷 GPU`; `_render_beliefs` + the
`_beliefs_*_text` helpers). Six self-hiding sub-panels (each blank when its belief leg is off; a fully-off
checkpoint shows one "belief heads not enabled" note):
- **species clause — the coherent reading** (`a.exclusive_belief`) — the point team hypothesis, plus the
  hidden slots where the raw top-1 and the clause-consistent read DISAGREE. Silent (one line) when the raw
  belief was already coherent. See *The SPECIES-CLAUSE reading* above for what it is not.
- **species belief vs TRUE team** — reuses `_append_belief_truth` (privileged, Hungarian-matched, `✓/≈/✗`)
  or the anonymous `_append_belief` fallback (no `reconstruction.json`).
- **move belief** (✓ revealed · ≈ believed unseen) — the revealed opp's still-unseen moves (`a.move_belief`,
  MOVED here from the old Matchups panel — it's a belief, not a threat).
- **believed SPREAD vs true** (`a.spread_belief`, `engine.build_spread_belief` → `SpreadBeliefView`) — per
  REVEALED opp mon the DamageOperator's believed DERIVED stats `[atk,def,spa,spd,spe]` (🔷) next to the
  TRUE derived stats (📋, computed from `reconstruction.team_details()` base+IV+EV+nature via the gen3 L100
  formula `_derived_stat`, which uses the mon's REAL IV — `gen3_data.priors.gen3_stat` hardcodes IV31) + the
  Smogon usage prior. Match is by **species** (exact — a revealed mon's species is known + unique), NOT
  Hungarian. A wrong spread is an otherwise-invisible damage root-cause (the op consumes these stats), so
  this surfaces e.g. "believes Metagross Atk 385 vs true 305 → over-prices its hits". `mean_abs_err` is the
  headline. Believed-only (no `true`/prior-from-truth) on a websocket trace.
- **refinement TRAJECTORY (axis B, across-battle)** — `engine.build_belief_trajectory` (model-FREE, from the
  on-disk per-decision `belief` blocks + the privileged team): a top-1 species-confidence sparkline + `✓/✗`
  correctness dots across the battle's decisions (confidence still shown without truth; the `►` marks the
  decision being viewed). Correctness mirrors `build_belief_truth`'s precision — per decision the true HIDDEN
  set is `opp_team` minus the species revealed by then (decoded model-free from the inv board), matched with
  **one-time consumption**, so guessing an already-revealed species or two slots naming the same hidden mon
  can't double-count. When the trace npz carries the captured `move_logits` / `spread_belief` arrays (new
  runs), it ALSO draws the opp-active **move-belief entropy** (`Hmv`, should decay) + believed opp-active
  **Atk** (`bAtk`) sparklines — the move/spread analog, decoded WITHOUT re-running. The "watch the belief
  sharpen as reveals accumulate" view.
- **value-dist × belief cross-read** — does critic bimodality co-occur with low belief confidence?

The **Threats** section (`6`, was *Matchups*) is reordered **GPU-first** (`_render_matchups`, still the
method name): the `🔷` DamageOperator physics (outgoing per-move · incoming worst-hit per defender · opp
the discrete per-move incoming matrix with per-pivot safe-switch) render PRIMARY into `#threats-gpu`;
the `📋` CPU obs decodes (the per-move type-multiplier table + the `their_matchups` effectiveness + the
usage-prior `incoming P(KO)`) render below into `#matchups-table`/`#matchups-threat`, **dim** when the op is
present (it subsumes them) and **full-styled** when there's no op (the demote only applies when the op is
present — the graceful-degradation contract). The **Flow** diagram flags the learned belief/physics phases
(`BeliefSlots`/`BeliefHead`/`MoveBelief`/`SpreadBelief`/`DamageOperator`) with a `🔷 GPU-computed` callout
(`_FLOW_GPU_PHASES`). New `analyze`-JSON fields (`asdict`): `spread_belief` (+ the existing
`belief`/`belief_truth`/`move_belief`/`damage_op`); `None` when the head is off.

**Switch-in OUTGOING panel (`a.switch_in_outgoing`, `engine.build_switch_in_outgoing` → `SwitchInOutgoingView`,
rendered in `_render_matchups` right after the op's "our damage (out)").** On a FORCED SWITCH the op's outgoing
block is all-zero (it prices the fainted active only), so the model picks a switch-in from INCOMING threat alone
with no estimate of what each candidate would then DO to the opp active. This **prober-only, CPU-computed (📋)**
panel fills that view: per ALIVE bench candidate → its best BP-damaging move (self-KO Explosion/Selfdestruct
excluded) vs the opp active → `low–high %HP · →KO · ×mult · P(outspeed)`, from the **privileged true spreads**
(`reconstruction.team_details(our_side)`, threaded as `analyze_invocation(our_team_details=…)` mirroring
`opp_team_details`; `SessionBackend._our_team_details` / `app._load_our_team_details`). Reuses
`observation.incoming_damage.{gen3_damage_max,p_ko,p_outspeed,type_is_physical}` + `gen3_mechanics.effective_multiplier_by_types`.
Gated to `phase == "forced_switch"`; `None` off a forced switch or without a `reconstruction.json`. NO model change —
the model still lacks switch-in outgoing damage (the op's outgoing is active-only; the symmetric "_outgoing_matrix
transpose" is a separate, un-built arch follow-up). Pair it with the per-mon INCOMING block: "what hits me on the
way in" vs "what I'd then do".

**Capture (axis B beyond species).** `RLPlayer._move_belief_active_row` (the opp-active move posterior,
`[n_moves]`) and `_spread_belief` (the opp-active believed-spread row `[5]`) stash into the trace, and
`BattleRecorder.states_arrays` writes them as `move_logits`/`spread_belief` npz arrays — **OMITTED when the
head is off**, NaN for a captured-but-headless row (parallel to `value_dist`). `build_belief_trajectory`
READS them (the `Hmv`/`bAtk` sparklines above) so move/spread trajectories decode on future runs WITHOUT
re-running the model; absent on older traces (then species-only). The capture is opp-active-row (not the
full `[6,5]`) so the trajectory needs no separate active-index array — the per-decision spread PANEL still
re-runs the model for the full `[6,5]`-vs-truth view.

The remaining helpers
`_col` / `_mon_label` / `_moves_line` / `_team_panel_text` build the panels (the last shared by OPP
TEAM + both Team tables). The **Team** section (`2`, collapsed) is the full per-mon detail — our team
and the opp team **side-by-side (2-column)** — every mon's **moveset** (ours complete; opp's
revealed-only, from `describe_team`) + hp · status · item.
It composes
existing `InvocationAnalysis` fields only (no new obs/engine analysis): `actions` (probs),
`matchups` (effectiveness), `incoming` (the P(KO) belief + `per_slot_pko`), `value`
(critic), `board` (hp/status/item — status+volatiles bundled by the recorder's
`_mon_display_status`; **items** from the summary teams block for our side, overlaid per-turn
by `ProbeModel.describe_team_items` which adds the opp's revealed items + reflects consumption),
`outcome` (result + events + reward), `field`. The pairing relies on the fixed obs action layout —
the *i*-th `switch:` action is team slot *i* is `per_slot_pko[i]` (verified: `active_pko` ==
`per_slot_pko[active_slot]`) — so it pairs BEFORE sorting by prob. Shared render helpers keep it
DRY — `_chosen_prob` / `self._td_residual` + `_append_surprise` (also used by Review + Outcome),
`_append_happened` (the what-happened line, shared with the Review card), `_hp_bar` / `_status_cell` /
`_item_cell` / `_side_attr_map` / `_append_summary_active`. The remaining sections
render the same data unfused: **Board** (each side's active species/hp/status/boosts +
benched **status** now shown too — `_parse_bench` splits the `species(hp%,STATUS)`
recorder format so the status no longer mangles the hp cell +
revealed bench + our moveset from `engine.build_board`, model-free; plus a **field**
line — weather/spikes/screens/turn decoded from the obs global block via
`ProbeModel.describe_global`, so it needs captured state), **Faithfulness**
(recorded vs re-run probs), **Beliefs** + **Threats** (the GPU-first observability sections — see
*Beliefs / Threats (GPU-first observability)* above; Threats keeps the `📋` CPU decodes — the per-move
type-multiplier table (OUR Hidden Power renders TYPED as `hiddenpower(grass)` in the `×mult` table AND
the "our damage (op)" line — `engine._matchups`→`_display_hp` normalizes the recorder's typed id
[`hiddenpowergrass`; distinct-num traces, `gen3_typed_hidden_power_ids_v1`] OR a bare own `hiddenpower`
via `our_hp_types` [legacy], while the opponent's bare HP stays untyped — no leak), the `their_matchups`
**incoming eff** `worst N×`/`revealed XX%`/`BLANK`, and the
`incoming_damage` **incoming P(KO)** `active NN%`·`outspd NN%`·`worst-on-team NN%`·opp-recovery — DIM below
the `🔷` DamageOperator physics when the op is present),
**Intervention**, **Saliency** (two heads: `π` policy-logit blocks AND `V` critic
value-gradient blocks, each incl. `their_matchups(144)` and `incoming_damage(33)`, so
you can see whether the **value** head — where OHKO tail-blindness lives — actually
reads the belief block vs the rest), **Flow** (`_render_flow`, default-open) — a **`Static`
box-art DATAFLOW diagram** (not a Tree — a Tree can only nest down-and-right, so it can't DRAW a
fork or sit the two heads side-by-side) that SHOWS the model instead of asking the reader to imagine
it. A single left **rail** (`│`, dim cyan) is the forward spine; it flows DOWN into three stage **bands**
via a `▼ BAND → <what it produces>` header (e.g. `▼ ENCODE → role tokens, self-attended`) — `ENCODE` ·
`BELIEF` · `⑂ FORK` (`_flow_pipeline_lines` buckets via `_FLOW_BAND`, forward order preserved within a
band) — drawn from `ProbeModel.architecture()` (introspected LIVE, so flag-gated phases + dims reflect
THIS checkpoint, currently config v32). Per-phase glyph/colour encodes the category (no text tier-tag —
the glyph/colour IS the tag, see the legend): active required `① …` numbered bold (CLSPool `⑂` cyan = the
fork, `ProjectionAssembler` `◆`), active optional `① …` numbered bold GREEN, inactive optional dim `· …`,
**side readout** (`BeliefHead`/`WinProbHead`/`ValueDistHead` — stashed, does NOT feed pi/vf) `└┄▷ …  ✗→heads` yellow. A
LIVE **attention** layer (`PokemonEncoder` move self-attn, `TeamTransformer`, `CLSPool` cross-attn pools,
`HiddenOppBeliefPool`) gets a magenta **`⊛`** marker in a fixed column + `SELF-ATTENTION`/`CROSS-ATTEND` in
its role (the `attn` flag rides each `architecture()` phase) — so the reader sees WHERE the network
attends at a glance. Roles render **full width-aware** (`role_w` from the live panel width via
`_flow_width`, so descriptions aren't needlessly cut) and are live-interpolated (`DamageOperator` gains
`+ outgoing` under `damage_outgoing`, `MoveBelief` `+ prior-fusion`, `BeliefHead` `+ latent`). The
belief + physics rows carry `· T0 RESOLVE, PRE-transformer` / `· T1 REASON, PRE-transformer, once`
UNCONDITIONALLY and are drawn AHEAD of `TeamTransformer` (`gen3_tiered_pipeline_v1`: one placement,
no flag). The
**CLSPool fork** then SPLITS the rail
(`├──┐` + two `▼`) into two **side-by-side lanes** (`_flow_combine_lanes` zips them with a full-height
gutter, `_pad`-ing each left line to `LANE_W`): `π POLICY` (cyan, `chose <move> (<prob>%)`) and `V VALUE`
(magenta, `V(s) <real> · norm <z>`), each opening with `↳ <pi_combined|vf_combined> → proj(<dim>)`
(tying the head to the architecture) then its obs blocks **sorted most-read-first** with a smooth
eighth-block `gradient_color` bar (`LANE_BAR`-wide) + abbreviated block + within-head share %, **dominant
bold**, <8% greyed. **The bar sorts AND sizes by `SaliencyBlock.mean_abs` (|∂out/∂obs| PER OBS DIM,
size-normalized), NOT `total_abs` (the block's summed gradient)** — so a big region can't dominate the
lane by sheer dim-count (the ~1590-dim turn-history block summed to 100% in BOTH heads as a pure
block-SIZE artifact; per dim its real reliance is far lower, the recent-turn signal concentrated in a
few dims). The Saliency TABLE (§8) still shows BOTH `|grad|/dim` and the `sum`. Putting the lanes
side-by-side sits π's and V's bars on the SAME row for read-across ("`move_mults` 100% for π vs 18% for
V"). Below `MIN_TWO_LANE` cols (`_flow_width`) the lanes STACK
vertically instead of clipping. A category legend closes the panel.
(A human-facing companion to the precise Saliency table; SENSITIVITY, not proof of causal use — same
caveat. `None`-head / model-free / no-state → a graceful hint line, never a crash.)
`ProbeModel.architecture()` is the torch-boundary single source (a future `query` subcommand can
call it too); the per-head attribution composes existing
`InvocationAnalysis.saliency`/`value_saliency` only — no new engine work, and
it inherits the Saliency obs-mismatch guard, and
**Outcome** — the last surfaces the critic's `V(s)` (recorded — with its **PopArt-normalized**
companion `(norm …)` when the run used `--use-popart`: the critic's own [-1,1]-ish learning
scale, `(V − μ)/σ` from the loaded model's `PopArtNormalizer`, vs the de-normalized real-return
V — · re-run · ΔV → next ·
**TD δ** = `r + γV(s′) − V(s)`, the critic-surprise residual, in parity with the CLI's
overview/analyze `td_residual`; γ from the run's `metadata.json`) + the win-prob head's
**P(win)** + ΔP (when present),
whether the loaded model still picks the recorded action (agrees / DISAGREES → X),
the per-step **reward breakdown** (`total` + components), **events**, and the **raw Showdown
protocol log for this decision's turn** — the `|move|`/`|-damage|`/`|-crit|`/`|-miss|`/`|-immune|`
lines parsed from the `*_replay.html` sibling (`engine.parse_protocol_log` + `protocol_for_turn`,
file IO in `app._load_protocol` / `session._protocol_for`, lightly tinted by event kind), so the
exact mechanics the summary collapses (a miss, the per-hit damage, a switch-in) are visible
in-prober without opening the browser replay. Empty when the trace has no `replay.html`. The `analyze`
JSON CLI carries the same slice as a `protocol` list.

Per-invocation **flags** (`engine.summary_flags`, model-free): `switch`,
`uncertain` (top recorded prob < `UNCERTAIN_THRESHOLD`=0.34 — a genuine tossup),
`faint` (a faint in this turn's events), **`opp-switch`** (the OPPONENT voluntarily
pivoted this turn — `engine.opp_voluntary_switch`, glyph `⇄`), **`cure-skipped`**
(glyph `☣` — see below); plus `disagree`
(added per-analysis when the loaded model's argmax ≠ chosen). The list shows
`?`/`✗`/`⇄`/`☣` glyphs; `n`/`N` jump to the **discrete** flags (faint/switch/opp-switch/cure-skipped —
`uncertain` is the norm for a low-confidence policy, so it's a glyph, not a jump
target). `f` cycles a battle-outcome filter (all → loss → win), rebuilding the tree.

**The "heal ≠ cure" trap (`cure-skipped`).** Recover / Soft-Boiled / Wish restore **HP and nothing
else** — in Gen 3 only Refresh (self) and Heal Bell / Aromatherapy (team) CLEAR a status, and Rest
"cures" by *inflicting* sleep. A move list shows all of them side by side, so a Toxic that keeps
escalating through a heal loop reads as a sim bug when it is correct mechanics plus a policy choice.
Three pure engine helpers make that legible: `is_status_cure(move_id)` (data-driven off the facade's
`curesSelfStatus`/`curesTeamStatus` — never a hardcoded id list), `has_curable_status(status)` (splits
the recorder's bundled `"TOX(2)|TAUNT"` — a volatile is not curable), and **`self_cure_options(inv)`**
→ the cures that were **legal AND would have done something** (a Taunted cure and a cure with nothing
to cure are both non-options). The flag fires when a cure was on the table and we did something else;
`InvocationAnalysis.cure_options` carries the labels, so the page and the `analyze` JSON read ONE
engine output rather than each deciding what counts as a real option. `query find <battle> cure-skipped` lists them.

**The "computed-vs ≠ resolved-vs" guard (`opp-switch`).** When the opponent voluntarily switches, our
move RESOLVES against the switch-IN, not the active we computed damage against — and the net result
(e.g. Earthquake → immune) sits right next to a damage table computed vs the *pre-switch* active, which
is easy to misread as "the model attacked the switch-in." So `analyze_invocation` carries
`opp_switched_to` (the pivoted-in species), surfaced THREE ways: the `opp-switch` flag/glyph (markable +
jumpable), a `⇄ opp→<species>` marker in the always-visible battle header, and a one-line callout at the
top of the **Threats** panel (`⇄ opp pivoted <active>→<switch-in> — damage below is vs <active>
(pre-switch); your move RESOLVED vs <switch-in>`). The battle header ALSO shows the opponent AGENT +
its eval play **regime** for self-play sentinels — `opp: sentinel_0 [greedy]` vs `[stochastic@T]` (read
from the run `metadata.json` `cli_args` `eval_sentinel_greedy`/`self_play_temp` by `_read_eval_opp_regime`)
— so a self-play loss reads correctly: `greedy` = best-vs-best (the mirror genuinely out-decided us),
`stochastic@T` = the opp was sampling its distribution (some "great play" is the temperature handout).

**Manual review mode is RETIRED** (`review.py`, `<run>/review_notes.json`). It let you flag a
decision *funky* and append timestamped notes while stepping through a battle — and it was the only
thing that would have forced the web front end to become read-write, with an auth story for
anonymous writes on a box that trains. The usage said it was not worth that: **5 of 84 runs, 12
annotated decisions, none flagged, nothing since mid-June**. Every note was exported to
`<run>/review_notes.md` before the code was deleted, so the content outlives the feature.

The EXPECTED → DID → HAPPENED story it framed was never review-specific — it is what `/analyze`'s
decision + outcome + critic panels show for any decision, now with α/β as the "expected" half.

## Agent API & JSON CLI (`session.py`, `query.py`)

`ProbeSession` is a framework-agnostic facade so **agents/scripts** can probe a
model without a UI — all methods return JSON-serializable dicts and model
loading uses the same exact→nearest→recent ladder (cached per process). A
`battle_id` is the trace's `*_summary.json` path **or** a short
`step_<N>/<Opponent>/<outcome>_<idx>` id.

- `loops(outcome=, opponent=, step=, max_battles=, near_zero_frac=0.01, top=12)` — **model-free
  BAIT-LOOP scan**: *the opponent voluntarily pivots a mon our attack cannot touch, and we fire
  anyway — repeatedly.* Detection lives in `main/prober/loops.py` (pure, no torch, no session,
  unit-tested on hand-written protocol lines); this method is the run-level fold. **It reads the
  raw Showdown PROTOCOL from each battle's `*_replay.html`, never the rendered timeline** — the
  timeline's `— no effect` deliberately collapses an immunity, a full-paralysis `cant` and an
  unpriced small hit into one phrase, so a detector built on it would count all three (verified on
  the calibration battle: its T54 `we surf — no effect` is a `|cant|…|par` and its T40 `rapidspin
  — no effect` is a real 1% resisted hit).
  Definitions, fixed in `loops.py` so every surface means the same thing: a **voluntary pivot** is
  a `|switch|` with no faint earlier in the turn block and no `|drag|` (turn-0 leads excluded); we
  **moved into** it if we then used a move, after the arrival, TARGETING that side (a self-targeting
  Recover/Protect is not a bait and never enters the denominator); a **whiff** is `immune` /
  `fail` (a `-fail` with no external `[from]` cause) / `near_zero` (≤ `near_zero_frac` of the
  target's HP) — a **MISS is counted separately and is never a whiff**, because taxing dice would
  make the metric partly a luck reading; a **loop** is one `(move, arrival)` pair whiffing ≥2× in a
  battle (symmetric over the battle); a **re-click** is the 2nd..Nth click of such a pair (ordered)
  — the sharpest signal, since an immunity is deterministic and fully observable once seen.
  ⚠️ **Sides come from the recorded board, never from `p1`** (`identify_our_side`: the side whose
  protocol active agrees more often with the trace's `our.species`) — eval seats the trainee on
  either side, and a mirror match makes species names useless as a tell. An undecidable battle is
  SKIPPED with a reason and counted in `coverage`, never silently judged.
  Per-decision joins (model-free, from the summary + npz): chosen-probability on whiff decisions,
  ΔV and ΔP(win) bucketed `loop_step` / `other_bait` / `other` (the third bucket is the point — a
  loop-turn ΔV means nothing without the ordinary turn from the SAME battles), and the α/β readout
  on the same pivots split first-time / repeat / loop-step. **β's slot is graded STRUCTURALLY**
  (obs slot *k* = the *k*-th REVEALED opp mon, so the true slot is the arrival's index in the
  reveal order as of that turn) — never against β's printed species, which is an unsupervised
  posterior decode that names an off-team mon on 73% of pivots; grading by it grades the head
  against itself.
  Three headline rates on three DIFFERENT denominators on purpose (`whiff_rate_per_pivot` /
  `whiff_rate_per_decision` / `loop_battle_rate`), each shipping `{n, d, rate}`, because the two
  registered CONFOUNDS are conditioned for rather than mentioned: loop rate rises with game LENGTH
  and concentrates in WINNING positions (gen-15: 23.1% loop-battle rate in wins vs 7.0% in
  losses), so `by_outcome` is always reported and the comparison is win-arm to win-arm. A `mirror`
  block runs the same detector with the sides swapped — a CONTROL (it measures the opponent),
  not a target. `--opponent` is an **fnmatch pattern**, so `sentinel_*` reads the self-play
  sentinels as ONE population (an exact name still matches exactly); the gen-15 baseline was
  measured there. Those baselines live in `loops.LOOP_BASELINES` and ride the result as
  `baseline`, so the CLI and any future view quote ONE reference point.
  Measured on gen-15 (`ai_v9_18_gen15_v8rewards_0818`, 843 sentinel battles, ~2 s): 16.5% of 4923
  moved-into pivots whiff · 117 loop battles · 264 re-clicks · median chosen-prob on loop steps
  **0.963** · loop-step median ΔV −4.31 / ΔP(win) −0.096 · β slot 52.0% first-time → 65.9% repeat
  → 82.1% on loop steps · α SWITCH 76.3% on loop steps. **Both heads are right at the moment the
  wrong move is fired at p≈0.96** — the gap is actuation, and the injection probe proved no channel
  exists. The pre-registered gen-16 bars are in `designs/research_state/bait_loop_hunt.md`; this
  method is that hunt's instrument. CLI: `query loops <run_dir> --opponent 'sentinel_*'`.
- `triage(step=, opponent=)` — **rank the failure LEVERS across a whole run**
  (model-free; the natural first call when the question is "what do we fix next").
  Categorizes every loss's single worst-ΔV turning point into a fixed taxonomy
  (`engine.LOSS_TAXONOMY` — the one place to extend), then ranks the categories by
  `est_recoverable_winrate_pct` = mean over the fixed-**bot** opponents of
  `loss_rate(opp) × category_share(opp)` (an upper bound: assumes fixing the lever
  flips that loss). Each category carries the **lever** it implicates (obs / reward /
  policy / critic-capacity / upstream / measurement), a blurb, `by_opponent`, and
  worst-turn `examples`. The taxonomy splits the deaths by the signal that names the
  lever: belief **under-read** a healthy death = OBS (`surprise_ohko`); belief
  **fired** + a pivot existed but the mon died = REWARD/POLICY (`ignored_threat_death`,
  the under-switch target); no pivot left = UPSTREAM (`doomed_already`); already
  fainted = a forced replacement, look one turn back (`post_faint_replacement`). The
  no-death value craters split on **whether the model rated itself WINNING** right before
  the cliff (`engine._was_winning`): WINNING then craters = `critic_blindspot` (a confident-wrong
  THROW — CRITIC CAPACITY / a missing obs feature); already behind = `positional_grind`
  (upstream/material — never ahead to throw). **The winning signal is the CALIBRATED win-prob head
  `P(win) ≥ wp_even` (default 0.5)**, NOT the sign of V — V is a shaped/discounted RETURN with a
  structural **negative offset** (a measured self-mirror 50/50 reads V≈−6.5; PopArt μ≈−3.6), so the
  old `V>0` test systematically OVER-counted grinds (mislabeled even/favored positions as "already
  behind"). It falls back to `V > v_even` (default 0) only when no win-prob was recorded; pass
  `--v-even` = the checkpoint's self-mirror V / PopArt μ to re-center a head-less run. The result
  carries a `winning_split` block (`wp_even`/`v_even`/`wp_coverage`) + a caveat naming the signal.
  Reads the
  true per-opponent win-rates from `eval_results.jsonl` (falls back to ranking by raw
  loss volume, announced in the metric + a caveat, when absent). Carries explicit
  `caveats` (loss-weighted sampling; one-cause-per-loss; bot-only rating weight; the winning-split signal).
- `probe(target, step=, opponent=, which=, max_decisions=)` — **representation
  probe**: fit a cross-validated LINEAR probe on the model's INTERNAL activations
  (`which='vf'` value-head / `'pi'` policy-head post-projection features, via
  `ProbeModel.features`) to recover a derived quantity, and compare it to a
  baseline probe on the raw obs/belief feature we ALREADY provide. The decisive
  "is this info already in the representation, or should we hand it over" test: a
  linear probe recovering X ⇒ the model computed X (a new feature is redundant); a
  probe that can't ⇒ an extraction gap (a real obs lever — "let it learn" hit this
  small net's capacity wall for X). Targets (`engine`-free label/group logic in
  `session._PROBE_TARGETS`): **`is_faster`** (true base-speed order vs the provided
  `active_outspeed`; contested = close speeds where Leftovers/Sandstorm-timing
  inference matters), **`damage_taken`** (HP fraction lost this turn vs `active_exp`;
  contested = the `active_pko` 0.1–0.9 coinflip band where a p50/p90 spread would
  help), **`faint_soon`** (imminent faint vs `active_pko`; grouped by whether the
  belief flagged it), and the **opponent-anticipation family** — **`opp_switches`**
  (will the opp voluntarily switch this turn) + **`opp_status_move`** (if the opp uses a
  move, status vs attacking — its INTENT) + **`big_hit_incoming`** — the pre-registered
  Gate-0 falsifier for an opponent-action / world-model head: rep AUC ≫0.5 ⇒ the trunk
  already models the opponent ⇒ such a head is REDUNDANT. (Measured @53M: opp_switches
  0.89/0.90, opp_status_move 0.82/0.87, big_hit 0.75/0.78, faint_soon 0.86 — opponent
  modelling comprehensively present; the head was FALSIFIED before building.) Every
  result splits **overall vs by-group** (the easy-vs-hard
  contrast is the signal) and reports the representation probe AND the provided-feature
  baseline. The probe stats (`engine.fit_probe`) are pure numpy — standardized
  ridge/logistic, k-fold OUT-OF-FOLD predictions, **auto-tuned l2** over a grid
  (essential at d≈512: a fixed weak penalty overfits to a negative OOF R²). One
  checkpoint load per call (step → one model). Measured @70M: `is_faster` rep
  AUC 0.94 on contested vs the provided feature's 0.75 (the model already infers
  speed — not a feature gap).
- `run_summary()` — **orient** (model-free): steps, per-step model identity
  (git/arch/snapshot-available), opponents with win/loss tallies, persisted
  checkpoints, and γ. The natural first call.
- `battles(outcome=, opponent=, step=)` — list/filter battles (each carries an
  `id` + `short_id`).
- `scan(outcome=, opponent=, step=, metric=, limit=)` — **cross-battle, model-free
  turning-point triage**: for every matching battle, its single worst decision
  (`metric="value_drop"`, the most negative ΔV(s→s'), default; or
  `"td_residual"`, the most negative critic surprise), ranked globally. Each row is
  `{id, short_id, opponent, step, outcome, turns,`
  **`knew_by_turn, lead_time, blind_loss, awareness_text`** (the battle's "did it KNOW?" verdict
  beside the decision that lost it — a crater the model never saw coming is a missed signal; the
  same crater with 20 turns of warning is a position it could not convert)`, worst:{inv, turn, chosen,
  our_active, opp_active, delta_v, td_residual, reward_total, events, flags,
  incoming_active_pko, incoming_max_pko, incoming_active_outspeed}}`. The
  `incoming_*` fields decode the incoming-damage / OHKO **belief** the obs HELD at
  that cliff (model-free, from the saved obs — see `decode_incoming_belief`): the
  decisive A/B for "did the feature fill the obs gap" — a high `incoming_active_pko`
  at a value cliff means the OHKO WAS in the obs (any remaining error is downstream
  policy/critic usage); a low one where our active then faints means the belief is
  mis-calibrated (an encoder gap). The one-call form of "list losses → overview each
  → rank by the biggest drop" — the usual first move of a loss sweep. No model loaded.
- `switch_vs_info(step=, opponent=, outcome=, max_battles=)` — **model-free behavioural
  probe** of OUR policy: voluntary switch-rate bucketed by how many opponent mons we'd
  revealed (the information level), a correlation, and the double-switch rate (switch the
  turn AFTER the opp switched; consecutive own switches). Tests "do we switch more when we
  know less" (negative correlation ⇒ information-sensitive). Measured @53M: corr −0.025
  (information-BLIND), 57% reactive switch-after-opp-switch. Caveat: revealed-count
  correlates with game progress — a confound to control for.
- `battle_turns(battle_id)` — **model-free TURN-BY-TURN replay**: the same trace read as a GAME
  rather than as a ranked table. Decisions grouped by **game turn** (a turn is not a decision — a
  faint puts the `move_selection` and the `forced_switch` it caused on the same turn), each row
  carrying the **board** it was made on (`our`/`opp`: species · `hp` + numeric `hp_pct` · status ·
  boosts · item · moves · revealed bench), what was **chosen** + its recorded probability, the
  ordered **`timeline`** of what then happened (`engine.build_result_timeline` entries, each with a
  `text` field rendered by `engine.timeline_entry_text` so no surface re-derives the sentence),
  `order_certain` (false ⇒ both sides moved, `move_order` wasn't recorded, AND the turn's protocol
  slice could not settle it either — so top-to-bottom is NOT the real sequence; say so, never
  guess. The protocol read takes this from 56.3% of decisions to 3.6% — see the timeline section), **`opp_intent`** (the v67 `α`/`β` read — `alpha`
  ranked NAMED options + `SWITCH`, `beta` the named candidate switch-ins, `top`, `switch_p`, and a
  `text` rendered by `engine.opp_intent_text` so no surface re-derives the sentence; `None` on a run
  without the heads, which is every trace before v67 — plus **`actual`**, what the opponent then
  DID, with the matching `alpha` option flagged `was_actual` and `actual_unlisted` when `α` never
  named it at all. A prediction is only readable beside its outcome, and the match needs
  normalizing (`α` holds display names, the recorder an id) plus a Hidden-Power rule, so it is done
  once here rather than in each surface), and the critic's read (`value` · `delta_v` ·
  `td_residual` · `reward_total` + components · `events` · `flags`) — now with the two readings the
  scalar V cannot supply: the win-prob head's calibrated **`win_prob`/`delta_win_prob`** (V is a
  shaped, discounted return whose zero is NOT "even", so only this one reads as odds) and the
  distributional head's **`p_loss`/`p_win`/`p_tail`/`knew`** (the awareness fold, joined by decision
  index; `knew` is true from the sustained onset onward). `p_win` is `1 - p_loss` carried on the
  payload rather than left to each surface to flip — a view computing `1 - x` is a view deriving a
  number — and it is what `/battle` renders, so one card reads in ONE direction. Every threshold
  stays defined on `p_loss`. All `None`/`False` on a run without those heads.
  Plus the per-decision DETAIL a
  deeper read wants — the full recorded **`actions`** distribution (`label`/`prob`/`valid`/`chosen`,
  passed through in the recorder's action-index order, NEVER re-sorted — see the move-label gotcha
  below) and the raw Showdown **`protocol`** lines for that turn (parsed ONCE per battle, not once
  per decision). Plus the same `notable` block `battle_overview` returns and a
  `decision_turns[inv] → turn` lookup, so a surface can link "the worst drop" to a turn without
  arithmetic of its own. Also an **`awareness`** block (see `awareness_scan` below — the same
  verdict for THIS battle; `None` on a run without a dist head). No checkpoint: **17–20 ms** for
  the longest real battle measured (249 turns, 821 KB). CLI: `query turns <battle_id>`; the
  browser view is `/battle`.
- `awareness_scan(outcome="loss", opponent=, step=, lead_bar=5, cap_turn=240, stall_bar=0.25)` —
  **model-free 'did it KNOW?'**: the distributional head's battle-level loss-awareness verdicts
  (`main/prober/awareness.py`, pure + unit-tested) over every matching battle, aggregated. Per
  battle: `knew_by_turn` (first game turn from which P(loss) > 0.5 holds SUSTAINED to the end),
  `lead_time`, `blind_loss` (a loss it never saw coming), and `mean_tail_divergence` — the STALL
  SIGNATURE (bottom-atom mass piling up while the distribution MEAN still reads positive; the
  exact shape a scalar critic cannot surface, and the gen-9 pathology: positive V the turn
  before a cap loss). Aggregate: `blind_loss_fraction`, `aware_ge_bar_fraction`,
  **`cap_aware_ge_bar_fraction`** (the runbook's deadline-clock regression readout: fraction of
  cap losses tail-aware ≥ `lead_bar` turns early), `median_lead_time`,
  `stall_signature_fraction`. The atom support is read model-free from the run root's
  `model_config.json` (`value_dist_vmin/vmax/bins`), and the PopArt denorm is FIT per battle
  from the trace's own `(dist mean, recorded scalar V)` pairs (`fit_denorm` — exact under
  `value_from_dist`, an adequate approximation under `shaping`; identity without PopArt), so it
  runs on any dist-head run regardless of architecture drift. Battles ranked blind-first then by
  divergence. The aggregate also carries **`quantile_coverage`** (runbook §3): the pooled
  mid-PIT of the realized MC return under each predicted distribution (`coverage_stats` —
  continuity-corrected, so a perfectly-centered prediction reads 0.5, not 0.5+half an atom) —
  calibrated ⟺ `pit_mean` ≈ 0.5 and `coverage80` ≈ 0.80. ⚠️ Selection caveat: the default
  `outcome="loss"` filter biases PIT low BY CONSTRUCTION; judge calibration on `outcome=None`
  and use the filtered read for direction only. G is the MC discounted-reward return (the
  calibration probe's convention) — an approximation of the bootstrapped training target.
  Measured on gen-10 (1396 losses): 7.2% blind, median lead 7 turns, 12 cap losses of which
  only 50% were aware ≥5 turns early — the top-ranked row is a turn-249 cap loss with
  P(loss)=0.15 at its FINAL decision. Coverage (ALL outcomes, 109k decisions): pit_mean 0.396,
  **coverage80 0.44 vs nominal 0.80** — the head is optimistic AND over-confident (narrow);
  losses-only pit_mean 0.085. CLI: `query awareness <run_dir>` — **`--outcome all` is the
  unfiltered read** the coverage baseline is comparable to (win/loss were the only choices, which
  made the probe's own "judge calibration unfiltered" caveat name a reading the CLI could not
  produce). Those baselines are **not prose**: they live in `awareness.AWARENESS_BASELINES` and
  ride the result as `aggregate.baseline`, so the CLI and the web quote ONE reference point and
  neither keeps a copy. The result also carries `caveats` (baseline provenance · the cap-loss
  small-n · the selection bias · counted-not-judged).
  **The verdict's SENTENCE is `engine.awareness_text`** — same rule as `timeline`'s `text`: a
  surface prints it, never re-derives it. Each folded row also records the **DECISION index** it
  came from (`decisions`, parallel to `turns`/`p_loss`), because a faint puts two decisions on ONE
  game turn and a turn-keyed join silently collapses them.

  **The verdict is carried by the views that need it, not only by its own command:**
  `battle_turns` puts `p_loss`/`p_tail`/`knew` on every decision row (beside the battle-level
  `awareness` block); `scan` rows carry `knew_by_turn`/`lead_time`/`blind_loss`/`awareness_text`
  beside the crater, because one value cliff means opposite things with and without warning;
  `triage` gives each category an `awareness` split (`n_judged`/`n_blind`/`blind_fraction`/
  `median_lead_time`) **reported BESIDE the taxonomy, never folded into it** — the category names
  which lever to pull, the split says whether the model had warning to act on. Measured on
  gen-11: `positional_grind` median lead **22** turns vs `attrition_death` **4** — a slow known
  decline against a death that arrives fast. All `None` on a run with no dist head (never `0.0`,
  never `False`: absent must not read as measured).
- `battle_overview(battle_id)` — **model-free digest**: per-decision rows
  (chosen, top prob, `our_active`/`opp_active` board summary, recorded V(s), **ΔV**,
  **TD residual** = critic surprise, reward total, events, flags) + a `notable`
  block (faints, switches, `biggest_value_drops`) + how a deep analyze resolves.
  Full board state (`board`: both sides' active + bench + our moves) and `field`
  (weather/spikes/screens, when captured) are in `analyze`'s output.
- `find(battle_id, criterion, limit=)` — ranked/listed invocations:
  `switch`/`uncertain`/`faint` (flags, model-free), `value_drop`/`low_value`/
  `high_value` (ranked by recorded V, model-free), or `disagree` (loads the
  model; chosen ≠ the model's argmax).
- `analyze(battle_id, inv)` — full `InvocationAnalysis` as a dict (**loads the model — so it raises
  `ArchDriftError` on any run not at the current architecture, which today is every archived run;
  see the drift section above**). `model_resolution` carries `dropped_kwargs`: non-empty ⇒ flags the
  current code no longer accepts were dropped to make the load possible, so faithfulness is
  approximate and a surface must say so. The value block gains a γ-discounted `td_residual` and, on a `--use-popart` model,
  the PopArt `popart_mu`/`popart_sigma` + `normalized_recorded`/`normalized_rerun`
  (`(V − μ)/σ`, the critic's normalized learning scale; all `None` without PopArt). Also carries a `win_prob`
  block (`WinProbView`: recorded `P(win|s)` + `delta` ΔP to the next decision) — model-free, read
  from the trace's `win_probs` npz array (NaN/absent → `None` on a non-`--win-prob-mode` run; recorded
  at trace-capture by `RLPlayer._win_prob` → `BattleRecorder.states_arrays`). Also a **`value_dist`**
  block (`ValueDistView`, v29 — `None` unless the run trained `--value-dist-mode`): the distributional
  value head's predicted **return DISTRIBUTION** — `probs`/`support` (the histogram) + `mean` (E[Z]) /
  `std` / `p10`/`p50`/`p90` / `entropy` / `bimodality` (+ `mean_real` = de-normalized E[Z] under PopArt).
  Model-free from the trace's `value_dist` npz array (key absent / NaN → `None`); the atom support comes
  from the loaded model (`ProbeModel.value_dist_support` → `value_dist_vmin`/`vmax`/`bins`). The Summary
  panel renders it as a one-line **eighth-block histogram** + the shape stats (`_append_dist_hist`;
  sharp = confident, wide = uncertain, `⑂ bimodal` = the critic sees a coinflip) below the CRITIC /
  WIN-PROB lines — the interpretability read the scalar V collapses. Engine: `engine.build_value_dist`.
  Also an **`opp_intent`** block (`OppIntentView`, v67 — `None` unless the run trained
  `--opp-intent-coef>0`): what the model expected the OPPONENT to do — `alpha` (ranked NAMED believed
  moves + `SWITCH`, each carrying `is_switch` so no surface compares a magic string itself), `beta`
  (candidate switch-ins — see **β name provenance** below), `top`, `switch_p`. It is
  **model-free and stays that way** — unlike `belief`, which prefers a re-computed read, `α`/`β` are
  supervised against what the opponent then DID, so the honest question is what *this* decision's
  model expected, not what a later checkpoint would. Rendered as the Summary's `EXPECT` line and the
  web replay's per-turn *expect* line. Engine: `engine.build_opp_intent` / `opp_intent_text`.

  🚨 **β name PROVENANCE — `revealed` / `caveat`, and it is not cosmetic.** `β` points at a SLOT, and
  what names that slot decides what the row MEANS. A candidate carries `revealed=True` when the
  RECORDER read the mon off the board; otherwise the name is the model's species POSTERIOR, which is
  **un-supervised on a revealed slot** (`β`'s candidate mask is alive-and-not-active, so it includes
  mons already seen, while the species aux scores only the *believed* slots). Measured over a
  843-battle sentinel sweep (2026-08-19), the posterior-decoded name was a mon not on the opponent's
  team **at all in 73.3% of 6,876 pivots** (88.3% on revealed slots) — and one such label was read as
  *"β predicts porygon2"* on a turn where `β`'s slot held the revealed Salamence and `β` was
  **CORRECT**. That is a wrong research conclusion caused entirely by a label.

  **Every trace written before `gen3_beta_revealed_naming_v1` carries no `revealed` key**, so it
  reads as `False` — correct, because those names all ARE posterior decodes. Read time attaches
  `engine.BELIEF_NAME_CAVEAT` (`"believed (posterior decode)"`) to any candidate that is not
  `revealed` and has a name to qualify, and **never re-derives a name**: the board those traces
  should have shown is not in them, so a substituted name would be the same defect facing the other
  way. The caveat rides `opp_intent_text` as well as the candidate, because a surface that prints
  only the sentence would otherwise drop it silently. A `species: None` row (no species head at all)
  gets no caveat — a bare `slot 4` already claims nothing.
  Carries two incoming-threat decodes — **distinguish them**:
  - `threats` (model-free, from `their_matchups`): raw type-*effectiveness* —
    `present`, `revealed_frac` (how much opp coverage is revealed), `max_incoming`
    (worst eff ×4 on the board), `per_our_slot_max`. Effectiveness only — no
    power/Atk/Def/HP, so an OHKO still has to be inferred from it.
  - `incoming` (model-free, from the `incoming_damage` block, `incoming_damage_v1`):
    the calibrated P(KO) / expected-chip **belief** that already prices base-power ×
    Atk·Def × HP × roll — `present`, `max_pko` (worst P(KO) across our team),
    `active_pko` / `active_exp` / `active_outspeed` (our on-field mon, slot found via
    its per-mon active flag), `per_slot_pko`, and the opp recovery scalars
    (`recovery_rate` / `cures_status` / `recovery_known`). This is the direct lens on
    "did the critic-tail-blindness obs gap get filled."
  - `damage_op` (model, re-computed via `ProbeModel.damage_op_view` → `decode_damage_block`): the unified
    **DamageOperator**'s LEARNED-belief view (v23), `None` unless the checkpoint trained `--damage-op`. Per
    our mon the incoming threat `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance` in
    **TEAM-SLOT order** (`incoming[i]` = our team slot i; the op reads `ctx.species_ids[:, :TEAM_SIZE]`, so
    the active is whichever slot holds the active flag — NOT necessarily slot 0 — and the bench slots are
    the safe-switch reads), the `choice_band` tail, and (on `--unified-damage
    both`) our 4 moves' **outgoing** damage `[low,high,crit,pko]` (request-slot/action order — the
    equal-effectiveness move tie-break) + `outgoing.secondary` (per OUR move — "what status can it cause",
    keyed by the 7 live `_OUT_SEC_COLS`; `gen3_op_block_trim_v1` dropped slp/psn/tox, which no gen3 move an
    OUR-side team runs can inflict). **`gen3_op_block_trim_v1` REMOVED three keys** the decode used to
    carry — the opp-active `effect` and `incoming_secondary` collapses (ledger P1: 1.2% / 0.1% of the op's
    measured dependence, no defender axis) and the LEAN `incoming_topk` (0 calls/forward — the matrix
    superseded it). A stale reader now KeyErrors instead of silently mis-reading the Choice-Band bytes. The
    discrete per-move read is `incoming_matrix` (see the Gotcha below).
  - `move_belief` (model, via `ProbeModel.move_belief` → `engine.move_belief_view`, `MoveBeliefView`):
    the model's MOVE belief for each **REVEALED opponent mon** — `None` unless the checkpoint trained
    `--move-belief-mode != off`. Per revealed opp mon (gated on the `species_known` obs bit, so un-revealed
    bench slots are excluded — the run predicts hidden mons' SPECIES not their moves): each `revealed` move
    WITH its belief (pinned ≈100% under `--move-prior-fusion`, so it CONFIRMS the belief tracks the known
    moveset) **plus** the `believed` still-UNSEEN moves `(move, P(in set))` from the multi-label posterior
    (already-revealed filtered, kept if `P ≥ 0.10`; type-collapsed Hidden-Power num → bare `hiddenpower`).
    The unseen list is **CAPPED at the open move slots** `min(top_k, 4 − n_revealed)` — a mon with k known
    moves has ≤`4−k` more, and the multi-label head doesn't enforce that 4-move constraint, so its raw
    top-K over-shows (2 known ⇒ at most 2 unseen, not 4). Also carries `our_labels`
    `(team_slot, species, is_active)` so the op's team-slot incoming rows can be labeled. Pure decode is
    unit-tested (`engine_test::test_move_belief_view_*`). The **move belief (✓ revealed · ≈ unseen)** block —
    `metagross ✓ meteormash 100%  ≈ explosion 32% · …` (green ✓ revealed, magenta ≈ unseen) — now renders in
    the **Beliefs** section (`#beliefs-moves`); the op's outgoing line, the `opp 2ndary:` incoming-status
    line, and the per-OUR-mon op **incoming** damage (worst-channel %HP →KO%, species-labeled, active ▶,
    red-graded by P(KO)) render `🔷`-primary in the **Threats** `#threats-gpu` panel. All fields ride the
    `analyze` CLI JSON.
    **NB:** the op view (`damage_op`) stashes on the **DamageOperator submodule** (`op.last_raw_block`),
    not the extractor — `damage_op_view` reads it there (a prior read of `extractor.last_raw_block`
    silently returned None, hiding the entire incoming/outgoing op view; regression-guarded by
    `model_test.py`).
  Plus `value_saliency` — the **critic** lens: `|d V(s)/d obs|` aggregated into the
  SAME named blocks as the policy `saliency`, so you can see whether the VALUE head
  (where OHKO tail-blindness lives) actually reads `incoming_damage(33)` vs the rest.
  (On run_20260606 the critic's per-dim value-saliency on `incoming_damage` ran ~5×
  the overall mean, vs ~0.3× for the old `their_matchups` — the critic strongly uses
  the new belief.)
- `lookahead(battle_id, inv=, invs=, worst=, n_seeds=0, followup=)` — **one-ply VALUE-DELTA**
  (`lookahead.py`): for an anchored `move_selection` decision, RE-ROLL the turn under each LEGAL action
  (the opponent plays its RECORDED move), materialize the resulting one-sided successor obs through the
  real encoder, and read the loaded model's **V(s′)** — per-action ΔV, "what would the critic have
  valued each alternative at" (the model-scored variant the model-free `falsify` deliberately defers,
  + the distributional / win-prob heads on the successor via `ProbeModel.value_dist_at`/`win_prob_at`
  when the run trained them). Two faithful modes share one call: the **CRN** headline (the `"original"`
  seed — hold the realized dice, vary only OUR action, so ΔV isolates the action's effect; the CHOSEN
  action's CRN successor reproduces the REAL next state so its `value_crn` ≈ the trace's
  `recorded_next_value`, a built-in consistency anchor) and a **dice-averaged** `value_mean`±`value_std`
  over `n_seeds`>0 fresh seeds. A candidate whose turn ENDS the battle is reported `terminal` (win/loss),
  not a numeric V. Loads the exact→nearest→recent model; **requires the trace's `*_reconstruction.json`
  sibling**. (`ProbeModel.value` is the V(s′) primitive; the successor obs is materialized from
  `reroll.prefix_pN_chunks + reroll.pN_chunks` per the obs-materializer recipe.) **The whole
  `(candidate × seed)` sweep resolves in ONE Node process** via `reconstruction.reroll_many` (each arm
  = a fresh session = byte-identical to a single `reroll_turn`, modulo `|t:|`), so the lookahead pays the
  ~677 ms Node-spawn cost ONCE instead of once per candidate (~9× on a full 9-action sweep; pinned by
  `utils/bridge/reroll_many_parity_fuzz_test.py`).
- `better_line(battle_id, inv, depth=2, beam=3, top_k=4, interior_opponent=, opponent_ckpt=,
  confirm_rollouts=0)` — **SEARCH for a better line** (`better_line.py`): the depth-≥2 generalization of
  `lookahead` (which IS its depth-1 instance). A shallow, CRN-anchored **beam over the critic** that
  branches a search TREE by CLONING mid-battle states in the warm `SearchSession`
  (`utils/bridge/search_session.py` → `search_driver.js`'s `State.serializeBattle` on node, or the rust
  binary's `BridgeSession::snapshot` under `--impl rust` — the only primitive that makes depth>1
  feasible), expands OUR top-k actions by policy prior, scores each
  successor's V(s′) on the materialized ONE-SIDED obs (`model.values_batch`, one critic forward per
  ply), keeps the top-`beam`, and recurses; backup is **max-over-our-continuations** and the returned
  LINE is the principal variation. Returns ONE human-legible **contrastive trajectory**: the divergence
  (`best_alternative` + ΔV / `win_prob`), the per-ply `principal_variation`, and the chosen-vs-best
  `candidates`. **Faithful-conditional opponent:** the RECORDED move at the divergence ply (the chosen
  action is `recorded_exact` → the `value_crn` anchor, identical to lookahead), and at INTERIOR plies the
  reloaded opponent reacts greedily on ITS OWN one-sided obs (materialized via the opponent's
  action-history from `obs_materializer.infer_action_indices`) — `interior_opponent`: `"self"` (the
  trainee as a flagged proxy, default), `"ckpt"` (`opponent_ckpt`), or `"none"` (sim default). Depth-1 is
  faithful regardless; only depth≥2 leans on the interior model. `confirm_rollouts>0` CONFIRMS the
  recommended first action with an actual Monte-Carlo replay-to-end vs the RELOADED REAL opponent
  (`replay_counterfactual`) → win-% ± Wilson CI — the ground-truth check on the critic's claim (the
  three-tier eval: search by V, report by ΔP(win), confirm by rollout). Loads the
  exact→nearest→recent model; **requires the trace's `*_reconstruction.json` sibling**. Default depth 2
  is the legibility/cost sweet spot (≈4 s search; depth 1 ≈2 s, depth 3 ≈5 s — try `--depth 3` for setups
  that pay off a turn later). Faithfulness pinned by `utils/bridge/search_clone_parity_fuzz_test.py`.
  **Perf:** the bottleneck is obs materialization (the serializeBattle clones are ~2%), so three levers
  cut it ~1.5× (5.7 s → 3.7 s on a depth-2 decision): (1) ONE shared `replay_battle` feeds both the anchor
  choice-map AND the opponent's `infer_action_indices` history (was two full replays); (2) the per-node
  policy forwards are BATCHED (`action_probs_batch`, was one forward per node); (3) `materialize_decisions`
  `encode_only_at={target}` encodes the obs ONLY at the decision the search reads — every other decision is
  TRACK-ONLY (the tracker still advances faithfully, the ~80%-cost encode is skipped), so a node replays
  its prefix as cheap tracking and encodes one obs. Lever 3 rides `Gen3Player.track_decision` (the tracking
  half of `embed_battle`, extracted byte-identically — the live obs path is unchanged, pinned by
  `obs_roundtrip_fuzz_test`); its bit-for-bit obs equivalence (track-only prefix vs full encode) is pinned
  by the clone-parity fuzz. **L4 (training-scale):** `better_line_decision(session=…)` accepts an injected
  WARM `SearchSession` reused across battles (`open_root(record=…)` + the driver `NODES.clear()`s per root),
  so a background worker pays the ~0.68 s Node spawn ONCE not per search (~1.66× cumulative, ~3.4 s warm).
  This is the expert tier of the **search-as-teacher** plateau-breaker (`designs/ai_v6/design_search_teacher.md`):
  selective offline ExIt — search + rollout-confirm the worst `falsify`-flagged loss craters (exact reloaded
  opponent), CI-gate strictly-better corrections, distil into a policy/value **aux loss** (no on-policy /
  PPO-core change). NOT built; the search + 3-tier-confirm here are its ready foundation.
- `replay_counterfactual(battle_id, inv, action, n_rollouts=1, opponent_ckpt=, opponent_source=)` —
  **COUNTERFACTUAL replay-to-end** (`replay.py` → `utils/bridge/counterfactual.py`): "could the model
  have won if it hadn't choked this turn?". Pick up the recorded battle at `inv`'s turn, substitute
  `action` (a legal action index) for OUR side, then play the rest LIVE — the trainee's GREEDY policy vs
  the **RELOADED real opponent** — to a win/loss. The driver reuses `run_local_battles` with both
  players' `choose_move` SCRIPTED to replay the recorded commands until the divergence (faithful: the
  bridge `START` uses the recorded resolved seed + both recorded packed teams, and each scripted
  Gen3Player decision runs `embed_battle` + `tracker.advance(recorded_idx)` — the recorded index
  recovered by inverting the recorded choice string — so the post-divergence turn-history stays faithful,
  proven bit-for-bit by `counterfactual_fuzz_test.py`). The opponent is RELOADED: a reproducible bot is
  rebuilt exactly; `opponent_ckpt` loads any checkpoint (e.g. a self-play sentinel) as the opponent;
  else the trainee's own model stands in (a flagged `self_model_approx`). `n_rollouts`>1 resamples the
  post-divergence dice (`local_sim_bridge.js`'s `resumeReseed` PRNG swap at the divergence turn) for a
  Monte-Carlo win-rate ± **Wilson CI**; `n_rollouts`==1 is the single realized-dice line (NOT a
  probability — a `caveat` says so). Loads the model; **requires the `*_reconstruction.json` sibling**.
  Each rollout is a full in-process game (seconds); the `caveats` flag the self-play approximation +
  that it best illuminates THROWN-LATE losses, not matchup-lost-from-turn-1 ones. `narrate=True` (the
  CLI `--narrate`, and on by default from the web) additionally captures the **move-by-move play-by-play**
  of the first recovered WIN + first LOSS (`winning_trajectory` / `losing_trajectory`: per-turn
  `{turn, events}` from OUR one-sided view — moves / switches / damage / faints / crits / status / win)
  via `run_local_battles`'s `chunk_sink` → `counterfactual.summarize_trajectory` — so you can read HOW a
  different move wins and what the bot did (e.g. spamming Earthquake into a Levitate Gengar).
- `falsify(battle_id, invs=, worst=, n_seeds=, n_alts=, followup=)` — **dice
  attribution** (`falsifier.py`): was a loss decision LUCK or a reducible MISTAKE?
  RE-ROLLS the real turn via the battle-reconstruction layer
  (`utils/bridge/reconstruction.py`): fix-both-actions under N fresh PRNG seeds →
  where does the REALIZED outcome (the special `"original"` seed — no PRNG swap,
  exact recorded follow-ups) sit in the dice distribution (`luck_percentile`);
  plus a **paired** alternative-action sweep (top-k legal alts by saved logits,
  SAME seeds = common random numbers, mapped to sim choices by the real action
  mapper via `obs_materializer.map_actions_at`) → `paired_advantage` ± SE per alt.
  Both axes score an omniscient **material margin** (`alive diff + hp-frac diff` —
  referee-side analysis; the one-sided wall constrains the encoder, not analysis).
  Verdicts `LUCK`/`MISTAKE`/`MIXED`/`NEUTRAL` (thresholds in `falsifier.py`,
  echoed in the output). Default anchors = the `worst` most-negative-δ
  `move_selection` decisions on distinct turns (a forced-switch crater attributes
  to its turn's move decision — re-rolls anchor at start-of-turn rounds only).
  Model-free (no checkpoint). **Requires the trace's `*_reconstruction.json`
  sibling** (bridge-eval traces written by the reconstruction layer; older /
  websocket traces raise with that explanation). Alternatives the sim refuses
  (maybe-trapped) are detected from the `[Unavailable choice]` error in the
  one-sided suffix and excluded from the verdict when mostly refused.
  ~1–2 s per arm at 40 seeds (fresh Node replay per seed); a decision with 3
  alts ≈ 5–10 s. Each falsified decision is annotated with `anchor_delta` — the
  TD-residual δ that selected it — the one source of truth `falsify_scan` weights
  craters by (`falsifier.anchor_deltas`, which `select_anchors` now ranks over).
- `falsify_scan(outcome="loss", opponent=, step=, limit=20, worst=2, n_seeds=32,
  n_alts=2, followup=, concurrency=1)` — the **RUN-LEVEL** generalization of
  `falsify`, an **input to** the distributional-critic decision (it *brackets* the
  headroom, it does not measure it — **read `caveats`**). Falsifies the worst δ-craters
  of every matching battle (default losses) that carries a `*_reconstruction.json`
  sibling, then aggregates the per-decision verdicts **weighted by crater magnitude
  (|anchor δ|)** into four levers — a **measurement-time attribution at one frozen
  checkpoint, NOT independent root causes**: **`LUCK` → `aleatoric`** (the chosen
  line's realized outcome sat in the dice bad-tail — reducible *only* by a
  risk-SENSITIVE policy avoiding a lower-variance line, which today's risk-neutral
  PPO/this scan don't have/test), **`NEUTRAL` → `unattributed`** (a real crater the
  sweep pinned on NEITHER luck NOR a better action — **NOT proven critic error**;
  "not bad-tail luck" = a TYPICAL outcome, equally a genuinely-lost position;
  splitting it needs the model-based V(s)-vs-return calibration probe, *not run here*),
  **`MISTAKE` → `policy_reducible`** (a top-k alt provably beat the chosen action —
  the **only proven** leg; but actor-critic coupled, so a better critic still reduces
  these over training), **`MIXED`** (both). `gate.critic_headroom_upper_bound` =
  LUCK + NEUTRAL share is an **UPPER BOUND** — it can only inflate as the shallow
  mean-only alt-sweep (top-`n_alts` by logit · single re-rolled turn · `followup`
  mid-turn) fails to *prove* a mistake (mass falls into LUCK/NEUTRAL), and it folds
  in the unproven `unattributed` leg. Both **`weighted_shares` (|δ|) and `count_shares`**
  are reported — a large gap means a few big ambiguous craters dominate (anchors are
  pre-selected by worst δ). `dominant_lever` is `None` on an empty scan or a near-tie
  (within 0.05). A **`caveats`** list (mirroring `triage`/`probe`) carries all of the
  above in the data. `coverage` reports `n_matched` / `n_with_record` / `n_falsified`
  / `n_capped_by_limit` / `n_skipped_no_record` / `n_battle_errors` / `n_decision_errors`
  (nothing **silently dropped**). Weighting falls back to count-based
  (`weighting="uniform_fallback"`, announced) when every δ≈0 (e.g. placeholder values);
  `"none"` on an empty scan. Model-free. `concurrency` > 1 falsifies battles in
  parallel (each re-roll spawns Node → raise it only on an **idle** box; it contends
  with a live training run). The coarser defaults (worst=2, 32 seeds) keep a 20-loss
  scan to a few minutes — the run-level statistic gets its power from MANY decisions,
  not deep per-decision seeds. `include_decisions=True` adds each battle's full
  per-decision list to its row (the calibration probe reads it).
- `calibration(outcome="loss", step=, opponent=, limit=20, worst=2, n_seeds=32,
  n_alts=2, concurrency=8, n_bins=10, overvalue_tau=5.0)` — resolve `falsify_scan`'s
  **unattributed** (NEUTRAL) bucket into **`critic_overvalued`** (epistemic — a
  better/distributional critic helps) vs **`lost_position`** (the critic was right),
  by comparing the RECORDED value V(s) to the REALIZED discounted return G(s) =
  `Σ γ^k r_{t+k}` (the MC value target). **Model-free** (uses recorded V — no
  checkpoint); the falsify pass that finds the unattributed craters runs at
  `concurrency` (default 8). **Selection-aware (the crux):** a loss-conditioned V−G
  is biased positive *by construction* (losses are the below-V tail of any critic),
  so the baseline is a **reliability curve over BOTH wins and losses, binned by V**
  (`_reliability_curve`/`_calibration_stats`/`_reliability_gap_at`, pure + unit-tested);
  a crater is `critic_overvalued` only if the critic SYSTEMATICALLY over-values at its
  V-level (reliability `gap` > `overvalue_tau`). **The output self-diagnoses the
  remaining confound**: `overall_calibration.bias_on_wins` (<0) / `bias_on_losses`
  (>0) is the CALIBRATED-critic signature, and `captured_win_fraction` ≠ the true win
  rate (eval QUOTA over-captures losses), so the unconditional bias and the reliability
  gaps are SELECTION-SKEWED — `critic_mean_reducible_upper_bound` is a LOOSE upper bound
  until reweighted to the true win rate, or replaced by the selection-free **gold-standard
  re-roll → policy-rollout → return PIT** (the true distributional-critic validator,
  deferred — needs a mid-game rollout primitive). Reads the `caveats`; this is the cheap
  aggregate proxy, knowingly confounded on a quota-captured sample.
- `decision_table(steps=, opponents=, outcomes=, categories=, max_battles=)` — a complementary
  MODEL-FREE per-decision FORENSIC TABLE (`forensics.py`): one row per captured decision with `cat`
  (`move_category`: selfko/recovery/**cure**/setup/stall/status/switch/attack_or_other — `cure` is
  its own bucket because clearing status, healing HP and inflicting status are three different acts,
  and it is derived from the move data, not a hardcoded set), our/opp species+HP,
  policy `conf` (`softmax(logits)[chosen]` — learned vs exploration-tail), `reward`, critic `dV`
  (`V[i+1]−V[i]`, the self-KO over-valuation signal), incoming-KO `pko` belief, faint flags, outcome.
  The single source for the softmax/dV/`decode_incoming_belief` plumbing every behavioural-hypothesis
  check reuses (the shipped self-KO finding used the `selfko` `dV_med`). Distinct from `falsify_scan`
  (the luck/mistake bracket) — this is the raw per-decision table. `move_category` /
  `decision_table_digest` are pure (unit-tested). Each row also carries `our_status` / `cure_avail` /
  `cure_prob` / `chose_cure`, and the digest a **`cure_uptake`** block — over the decisions where a
  status cure was genuinely available (statused AND legal), how often the policy took it, the median
  probability it put there, and what it did `instead`. That is the run-level form of the `cure-skipped`
  flag; pair it with the `cure` category count, since a cure chosen with **no** `cure_avail` is a
  WASTED self-cure (the same NO_OP `progress_clock._is_wasted_self_cure` charges). Measured on
  `ai_v9_09 @16M`: uptake **32/474 = 6.8%** (median P(cure) 0.036, `instead` led by `recover` ×127)
  while **104 of 136** cure uses (76%) had nothing to cure — the policy is picking these moves close
  to independently of whether it is statused. See `designs/research_state/` for the hypothesis ledger.

CLI mirror — prints JSON to stdout (and `{"error": …}` + exit 1 on failure, so an
agent always gets parseable output). `--help` carries a worked example sequence:
```bash
python -m main.prober.query triage   <run_dir> [--step N] [--opponent X]
python -m main.prober.query probe    <run_dir> <is_faster|damage_taken|faint_soon|faint_healthy|big_hit_incoming|opp_switches|opp_status_move> [--which vf|pi] [--step N] [--max-decisions K]
python -m main.prober.query switch-vs-info <run_dir> [--step N] [--opponent X] [--outcome win|loss] [--max-battles K]   # MODEL-FREE: do we switch more when we know less?
python -m main.prober.query summary  <run_dir>
python -m main.prober.query list     <run_dir> --outcome loss --step 8000000
python -m main.prober.query scan     <run_dir> --outcome loss --opponent X [--metric td_residual] [--limit K]
python -m main.prober.query awareness <run_dir> [--outcome loss] [--opponent X] [--step N] [--lead-bar 5] [--cap-turn 240] [--stall-bar 0.25]
python -m main.prober.query overview <battle_id>
python -m main.prober.query turns    <battle_id>          # MODEL-FREE turn-by-turn replay of the game
python -m main.prober.query find     <battle_id> value_drop --limit 5
python -m main.prober.query analyze  <battle_id> <inv> [--ckpt PATH] [--tier auto|nearest|recent]
python -m main.prober.query lookahead <battle_id> [--inv N] [--worst K] [--seeds N] [--followup random|default]
python -m main.prober.query better-line <battle_id> <inv> [--depth 2] [--beam 3] [--top-k 4] [--interior-opponent self|ckpt|none] [--opponent-ckpt PATH] [--confirm-rollouts N]
python -m main.prober.query replay-counterfactual <battle_id> <inv> <action> [--rollouts N] [--opponent-ckpt PATH] [--opponent-source auto|bot|self|ckpt] [--narrate]
python -m main.prober.query falsify  <battle_id> [--inv N]... [--worst K] [--seeds N] [--alts K] [--followup random|default]
python -m main.prober.query falsify-scan <run_dir> [--outcome loss|win] [--opponent X] [--step N] [--limit K] [--worst K] [--seeds N] [--alts K] [--concurrency N]
python -m main.prober.query calibration  <run_dir> [--step N] [--opponent X] [--limit K] [--worst K] [--seeds N] [--concurrency N] [--bins N] [--overvalue-tau F]
python -m main.prober.query decision-table <run_dir> [--step N]... [--opponent X]... [--outcome loss] [--cat selfko]... [--out t.jsonl] [--limit K]

# GLOBAL flags (before the subcommand): --compile (torch.compile the rollout models) and
# --impl {node,rust} (which sim engine the search/replay children run — see below)
python -m main.prober.query --impl rust better-line <battle_id> <inv>
```
**Investigation recipe:** `triage` (which LEVER recovers the most rating — start here
for "what next") → `summary` → `scan --outcome loss [--opponent X]` (the worst turn in
*every* matching battle, ranked — model-free, fast) → `overview` the top battles (read
`notable.biggest_value_drops` / `faints`) → `find disagree` / `find value_drop` →
`analyze` the worst turn → `falsify` it (was that crater dice or a reducible
mistake — separates irreducible aleatoric variance from real policy errors) →
`falsify-scan` the whole run (aggregate that split across every loss into the
**crater-fraction bracket** — `aleatoric` [LUCK] · `unattributed` [NEUTRAL, the
residual the shallow sweep couldn't pin] · proven `policy_reducible` [MISTAKE];
`critic_headroom_upper_bound` = LUCK+NEUTRAL is an **upper bound**, not a
measurement — read its `caveats`) → `calibration` to split the `unattributed`
bucket (`critic_overvalued` vs `lost_position`) via recorded V(s) vs realized
return G(s) — **selection-aware** (reliability over wins+losses) and
self-diagnosing (`bias_on_wins`/`bias_on_losses`/`captured_win_fraction` expose the
eval-quota selection skew; on a quota-captured sample it is knowingly confounded,
so its number is a loose upper bound pending true-WR reweighting or the rollout-PIT).
γ is read from the run's `metadata.json`. (`triage` aggregates
`scan`'s per-battle worst turns into ranked failure CATEGORIES; `scan` is the
cross-battle generalization of a single battle's `notable.biggest_value_drops`.)
`ProbeSession(..., model_loader=fn)` injects a fake model in tests (no torch).

## Obs-offset dependence (regression-guarded)

**`gen3_cpu_damage_deleted_v1` (v48):** two of these regions no longer EXIST in the obs — the
active-move type multipliers and the `incoming_damage` block were deleted (the DamageOperator
computes both GPU-side from the learned belief). `ObsOffsets.mm_off` / `incoming_off` / `incoming_dim`
now resolve to **0 = absent**, and every consumer no-ops on 0 (the saliency block list drops the
"active move_multipliers(4)" row, `_active_move_mults` returns zeros, the intervention sweep skips
its write). The fields are KEPT so archived pre-v48 traces still decode.
**`gen3_entity_rehome_v1` (v60) extends the same convention to the matchup matrices**: the
`our_matchups`/`their_matchups` blocks are DELETED from the obs (pair effectiveness is GPU-side —
the D/V edge families), so `om_off`/`tm_off` also resolve to **0 = absent** — ThreatView returns
`None` and the two saliency rows drop. The engine's one remaining live obs region beyond the
per-mon/global blocks is the turn-history span — all resolved at runtime from
`Gen3ObservationEncoder.get_layout()`. **If the obs layout changes, these move
automatically** (e.g. `gen3_move_effects_v1` inserted a block before `our_matchups`,
shifting it; `gen3_cpu_damage_deleted_v1` REMOVED three of them, moving the matchups
1568 → 1465), and `engine_test.py` pins the resolved values
(`test_offsets_resolve_matches_layout`) so a silent shift fails loudly. (Mirror note
in `src/agents/observation/CLAUDE.md`.)

## Blocking work (the concern outlives the TUI)

The checkpoint load and every torch forward/backward are BLOCKING, and the surface that shows them
must not stall on them. The Textual answer (exclusive worker threads + a staleness token) died with
the TUI; the web answer is in `web/CLAUDE.md` and is the same shape for the same reason —
`def` handlers run on a worker thread, `/analyze` arrives via an HTMX fragment so a checkpoint load
never blocks first paint, and the minutes-long probes go through the job registry instead of a
request. `probe_replay.py` and `query.py` are one-shot processes and simply block, which is correct
for a CLI.

## The counterfactual tier (`lookahead` · `better_line` · `replay_counterfactual`)

The three re-roll/clone-powered probes, bridge-eval traces only (each needs the
`*_reconstruction.json` sibling) and each spawning Node. **The surface is `/analyze`** — they are
per-DECISION probes, so they launch from the bottom of that page as password-gated background jobs
(`web/CLAUDE.md`); the CLI equivalents are `query lookahead|better-line|replay-counterfactual`.

- **one-ply lookahead** — per legal action, the re-rolled successor's **V(s')** under common random
  numbers (hold the realized dice, vary only our action), the **ΔV** vs the line actually played,
  and `terminal` win/loss where an action ends the battle. The chosen action's CRN successor
  reproduces the real next state, so its value is a built-in consistency anchor.
- **better-line search** — a CRN-anchored beam returning ONE contrastive trajectory: *"turn T: you
  played X → better line Y"*, the headline ΔV / ΔP(win), the principal variation ply by ply, and the
  depth/beam/opponent provenance. At depth ≥ 2 the interior opponent is the trainee standing in for
  the real one, which the surface must FLAG — a contrastive line that hides its proxy reads as fact.
- **replay-to-end** — substitute an action and play the rest live vs the reloaded opponent to a
  win/loss; `n_rollouts > 1` resamples the post-divergence dice for a win-% ± Wilson CI. At
  `n_rollouts == 1` it is a single realized-dice line and **not** a probability, which the payload's
  own `caveats` say and every surface must repeat.

`model.py` carries `value_dist_at` / `win_prob_at` for these — the counterfactual analog of the
trace's recorded distributional/win-prob arrays, since a re-rolled successor has no saved row, so
they re-read the head stash after a forward on s' (mirroring `belief` / `damage_op_view`).

## `--compile` (search-shaped commands)

`python -m main.prober.query --compile <cmd> …` `torch.compile`s the no-grad replay/rollout models
that `session._load` builds (`ProbeSession(..., compile_extractor=True)`), for a measured **~6.5×** per
B=1 CPU forward at a ~10-20 s one-time cost.

**Off by default, and use it selectively.** A one-off `summary` / `list` / `analyze` does a handful of
forwards and would never amortize the compile. It pays for the SEARCH-shaped commands, which do
thousands: `better-line` (a CRN-anchored beam), `falsify` / `falsify-scan` (paired alternative-action
sweeps × seeds), `replay-counterfactual` (Monte-Carlo re-rolls to a win/loss), `lookahead`.

**Gradient saliency is unaffected.** `history-saliency` and the gradient paths backprop through this
same extractor, and the compiled artifact is inference-only (AOTAutograd's CPU backward codegen fails
on the model's scatter/`index_add`). `maybe_compile_extractor`'s wrapper routes any **grad-enabled**
call to the eager forward, so `--compile` cannot change or break a saliency result — it simply does
not apply there. Detail: `src/agents/training/CLAUDE.md` → Compiled CPU opponents.

## `--impl {node,rust}` (which sim engine the search/replay children run)

`python -m main.prober.query --impl rust <cmd> …` — the **offline analogue of the trainer's
`--use-bridge={node,rust}`**, and like `--compile` it is a global flag placed BEFORE the subcommand.
Default `node` = today's behavior byte-for-byte.

It picks the child process the re-roll-backed probes exec — `better-line` / `lookahead` / `falsify`
/ `falsify-scan` / `calibration` / `replay-counterfactual`. The model-free, no-replay commands
(`summary`, `list`, `scan`, `triage`, `overview`, `find`, `analyze`, `probe`, `decision-table`)
spawn no sim child, so the flag is inert for them. Under `node` the work is split across
`search_driver.js` (the clone-and-branch server) and `replay_driver.js` (replay / reroll); under
`rust` a single `src/rust_sim` `search_driver` binary serves both — resolved (and built, once) by
`utils/bridge/sim_bridge_bin.resolve_search_driver_bin`, overridable with
`$POKESIM_SEARCH_DRIVER_BIN`. `replay-counterfactual`'s live post-divergence rollouts additionally
ride the LIVE bridge seam (`$POKESIM_SIM_BRIDGE_BIN` / the `sim_bridge` binary), since that leg
plays a real game. **It NEVER falls back to node** — an unbuildable binary is a clear error, because
a "rust" probe that silently ran on node would answer a different question than the one asked.

**The default lives on the SESSION, not the call**: `ProbeSession(root, …, impl="node")` stores it
and every probe reads it — the same shape as `compile_extractor`, and deliberate, since two probes
of one run answering under different engines would not be comparable. `better_line` REFUSES an
injected warm `SearchSession` whose `impl` differs from the session's (the search-teacher's reuse
path), so a correction can't be half-searched on one engine and half-confirmed on the other.

**The rust binary is BUILT and gated** (`gen3_rust_search_driver_v1` + `gen3_rust_replay_driver_v1`,
over the `gen3_bridge_clone_branch_v1` snapshot primitive), so `--impl rust` works today. Equivalence
is pinned node-vs-rust at 18873 + 30689 leaf fields, and — the claim that matters for a probe — the
cross-impl `better_line_integration_test` asserts node and rust yield IDENTICAL candidate V; since
that fake model is `V = obs.sum()`, an exact match is an obs-level bit-identity claim at every ply of
the beam. Two known divergences are printed by the parity harnesses rather than hidden: the
choice-reject framing (no `|error|` frame, boundary re-opens to both sides) and reconstructed
`pre_state` volatile names. **Perf: per-op the rust driver is 7–20× (clone-and-branch 13–20×), but
end-to-end `better_line` is only ~1.9×** — child-wait falls from 51% to 4% of a call, so Python-side
obs materialization is now the bottleneck and it is impl-invariant. See
`src/utils/bridge/README.md` → Offline driver transport.

## Gotchas

- **Move-action labels are ALREADY in action-index order — do NOT re-sort them.** The recorded
  `summary.actions` dict is built by `BattleRecorder._all_action_labels`, which iterates action index 0..10
  and keys move slot *m* (action 6+*m*) on **`legal.move_ids[m]`** — the SAME request-slot order the action
  mask, the `DamageOperator`'s per-move blocks, and the policy logits (action 6+k) all use. So
  `list(acts.keys())[i]` ↔ action index *i* ↔ `model.action_dist(...)[i]` directly, and `analyze_invocation`
  zips them with NO realign. A former `_reorder_move_labels` step (+ `ProbeModel.our_active_move_slots`)
  *re-sorted* the move labels to the per-mon obs block's **moveset** order — which differs from request order
  after a server reorder — and thereby SCRAMBLED the already-correct labels (transposing e.g.
  hiddenpower↔thunderbolt), producing a spurious `disagree` flag, a wrong re-run argmax, and backwards
  Matchups ×mults / op-outgoing labels on `exact`-tier replays. **Both were removed** (the recorded order is
  authoritative). Invariant pinned by `engine_test::test_recorded_actions_are_action_index_aligned` (an
  exact-reproducing model with a scrambling `our_active_move_slots` must still AGREE with the recorded
  choice). The outgoing-damage panel renders a non-damaging move EXPLICITLY as `— (non-damaging)`.
- **Per-move incoming threat = the `incoming_matrix` (`--damage-matrices incoming`).** `ProbeModel.damage_op_view`
  threads `matrices_incoming_k` / `matrices_outgoing` into `decode_damage_block` (since
  `gen3_op_block_trim_v1` there is no lean-top-K arm to disambiguate — `decode_damage_block` lost its
  `topk_k` parameter entirely). The Threats panel renders the rich
  `incoming_matrix`: per opp candidate move (decoded name + belief + acc + phys/spec + notable effect/secondary)
  → per OUR mon the FULL cell `low–high · crit · →KO · ×type-mult · status` (immune ⇒ `safe`). This is the
  "which opp move threatens which of my mons, by how much" read. (A prior bug omitted the matrices-decode flags,
  so on a `--damage-matrices incoming` run the prober mis-read the absent lean top-K block — the garbage
  `acc-580` render — and never decoded the matrix; deleting the lean block made that class unrepresentable.)
- **The rest of the decoded op fields now render too** (display-only): the OUTGOING **status-landing** (`our
  status (land)` — per OUR move P(a dedicated status move lands on the opp) + ✓certain/?prior), the per-defender
  **`incoming extras`** (the op's belief-aware `p_outspeed` + `provenance`), the **`opp Choice Band`** belief
  (`p_cb` + per-our-mon CB-conditional physical →KO), and the OUTGOING **`our damage vs switch-ins`** matrix (our
  moves × each REVEALED opp mon). Opp-mon columns are labeled by the obs-slot→species map (`mb.opp[*].slot`),
  NOT the board active+bench order (the op reads `ctx.species_ids[:, TEAM_SIZE:]` raw, active at any slot).
- **Op OUR-move blocks are ACTION-ordered — the old "op move order ≠ action order" caveat is GONE, and
  this entry exists so nobody re-derives a plan from it.** `gen3_op_move_align_v1` fixed it at the
  MODEL: the op's OUTGOING blocks (`our damage (out)` / `our status (land)` / `our damage vs
  switch-ins`) now read the request-ordered obs slice (`ctx.our_active_req_move_*`), so slot *k* ↔
  action 6+*k*, the same axis as `a.matchups.move_labels`, the faithfulness table and the policy
  logits. The prober therefore labels them with the recorded action labels; `ProbeModel._our_active_moves`
  and the `dop["our_moves"]` relabel are DELETED, and `app_test.py` asserts the caveat string never
  comes back. (Before the fix the blocks were indexed by `ctx.all_move_ids[our_active]` — the per-mon
  moveset order — which differed from action order in ~90% of decisions, so the v23 outgoing tie-break /
  v27 status-landing / v34 outgoing-matrix were positionally misaligned with the actions they informed.
  Kept here as history because this doc told two readers otherwise after it was already fixed.)
- **Faithfulness is exact only on the `exact` tier.** On `nearest`/`recent` the
  model differs from the one that generated the trace, so recorded ≠ re-run (the
  re-run cell is colored by the drift) — expected, and the badge says which tier.
  For bit-exact replay, train with `--keep-eval-snapshots` (then the `exact` tier
  loads the retained snapshot), or pass `--ckpt`.
- A trace whose `_states.npz` is missing, or an invocation with `has_state=0`,
  yields an analysis with `warnings` and no panels (the engine never touches the
  model) — handled, not a crash.
- **`ArchDriftError` is the EXPECTED outcome of loading an archived checkpoint**, not an
  exceptional one (measured: 79/79 runs). Any surface that loads a model should render its message
  — it is written to be read by a human, multi-line, and ends with the `git checkout` to run — and
  should NOT collapse it to "analysis failed". See the drift section above.
- `models/` is gitignored and lives only in the **main checkout**, not in a
  worktree — point the prober at an absolute `models/...` path when running from
  a worktree.
- **Obs-version mismatch** (`a.obs_mismatch`): when the trace's obs length ≠ the CURRENT encoder's
  `total_dim` (an obs change — e.g. `gen3_protect_odds_v1`'s +2 scalars — landed AFTER the probed
  model was trained), every obs-OFFSET decode past the divergence (incoming P(KO)/outspeed, THREAT
  incoming-eff, RESULT crit/boost/move-order, Matchups, Saliency) is misaligned. The Summary shows a
  red **⚠ OBS MISMATCH** banner; the board / items / movesets (front-of-obs + summary-sourced) stay
  correct. The model itself can't be re-run on the new obs (its policy expects the old dim), so the
  fix is to probe a model trained on the current obs. `engine_test.py::test_obs_version_mismatch_is_flagged`
  guards the detection.
- The obs decode (`describe_team`) only OVERLAYS info onto the summary teams block via `_merge_team` —
  an empty obs item never erases a known item (the bug where an own bench mon showed no item);
  `test_obs_item_overlay_does_not_erase_a_known_item` guards it.
- **`y`** yanks a precise pointer to the CURRENT decision — **`<replay.html path> inv<N>`** (N =
  the highlighted invocation) — onto a dedicated full-width **`#replay-path-bar`**, the ref on its OWN
  line so it's cleanly selectable under **`v`** copy mode (the portable path; the old toast wrapped it
  with its label). It also best-effort `copy_to_clipboard`s the ref (OSC-52 terminals only —
  kitty/iTerm2/WezTerm; dead on Terminal.app, hence the bar) **and RECORDS the ref in this decision's
  review notes** (deduped — a re-yank doesn't pile up), so the exact issue is one paste away to hand a
  model (`query analyze <battle> <N>`). The bar is hidden until used and cleared when a new battle is
  selected; warns if the file is missing.
- **Scroll stability:** the three scroll regions (`#trace-tree`, `#invocation-list`, `#analysis-scroll`)
  set **`scrollbar-gutter: stable`** so the 1-cell scrollbar column is always reserved — content no
  longer shifts a cell ("off by a pixel") the moment a scrollbar appears on scroll.

## Retention / grooming (`groom.py`)

Training writes a trace pair per sampled eval battle (+ a ~27MB snapshot per cycle
when `--keep-eval-snapshots`, default 10, is on), so `eval_traces/` grows. The
groomer prunes it — **scoped strictly to `eval_traces/`**:

```bash
python -m main.prober.groom <run_dir> [--keep-trace-steps 10] [--keep-snapshots 10] [--apply]
```

Keeps full traces for the K most-recent eval steps (deletes older step dirs) and
`snapshot.zip` for the N most-recent. **Dry-run by default** — it prints a JSON
report (`removed_steps`, `dropped_snapshots`, `mb_reclaimed`); pass `--apply` to
delete.

This CLI is a **manual fallback**. The producer grooms its own data: the **trainer**
(rl_agent eval callback) prunes after every cycle — `_prune_eval_traces`
(`--keep-eval-trace-steps`, default 20) and `_prune_eval_snapshots`
(`--keep-eval-snapshots`, default 10) — so a live run stays bounded on its own. The
prober is read-only and **never** grooms. Use this CLI for finished runs, a
different retention, or a one-off deep clean.

## Tests

`engine_test.py` (pure, FakeProbeModel + offset regression, + the loss-attribution
taxonomy and the `fit_probe` stats as pure cases — decodable-vs-noise,
regression, too-few-graceful), `session_test.py` (tmp_path traces for the agent
API incl. `triage` orchestration / recoverable-ranking / no-eval-results fallback,
`probe` end-to-end with a fake feature-model — label recovery, noise→baseline,
unknown-target, CLI — the `falsify_scan` aggregation with a monkeypatched
falsifier: coverage accounting, |δ|-weighted shares, the uniform-fallback, and
concurrency-order-independence — plus the `calibration` pure helpers
(`_discounted_returns`/`_reliability_curve`/`_reliability_gap_at`/`_calibration_stats`)
and the unattributed-split integration: over-valued vs lost via the reliability gap,
+ the selection-confound diagnostics), `discovery_test.py` (tmp_path trees, checkpoint
precedence, **sharded `<outcome>_s<shard>_<idx>` parsing → distinct index**),
`forensics_test.py` (pure `move_category` + `build_decision_table`/`decision_table_digest` over a
hand-written tmp trace via a fake session — no torch, no bridge),
`falsifier_test.py` (pure: margin/percentile/paired-stats/verdict
matrix, seed determinism, δ-anchor selection incl. the forced-switch remap, and
the `anchor_deltas` δ-map) + `falsifier_integration_test.py` (`@integration`,
real bridge battle → full falsify pipeline, determinism re-run, and the run-level
`falsify_scan` over a recorded discoverable tree), `review_test.py` (pure
`ReviewStore` — flag/note roundtrip, persistence, prune, export),
`better_line_test.py` (pure: the SEARCH backup logic — terminal sentinels, max-over-continuations,
beam pruning, principal variation) + `better_line_integration_test.py` (`@integration`, real bridge,
fake `V=obs.sum()` model: the depth-1 chosen value == sum(recorded next obs) value_crn anchor, the
depth-2 beam principal variation, determinism), `model_test.py` (the torch boundary — where each
forward stash LIVES, plus the `ArchDriftError` diagnosis and the dropped-kwarg recovery):

and the **web** suite under `web/` (`charts_test.py` pure Vega-Lite specs · `app_test.py`
`TestClient` over a synthetic run, each endpoint compared against a direct `ProbeSession` call ·
`openapi_snapshot_test.py` the committed-contract drift gate · `render_integration_test.py`
`@integration`, headless chrome with the network blocked — see `web/CLAUDE.md`):

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/main/prober -q
```

`textual`, `fastapi`, `uvicorn`, `jinja2` and `httpx` are pinned in `environment.yml`
(`httpx` is what starlette's `TestClient` runs on, so the web unit tests need it). The shared
Textual base lives in `src/main/tui/` — still used by the LAUNCHER's UI, which is why it
survived the prober's TUI. See its CLAUDE.md.

## Web front end (`web/`)

A third sibling over the engine — FastAPI + server-rendered Jinja2/HTMX, charts as Vega-Lite specs
emitted from Python, all JS **vendored** (no CDN, no build step, no `node_modules`). Read-only,
adapts `ProbeSession` and nothing else — it is now the ONLY human-facing surface.

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m main.prober.web models/   # :6008, pick any run
python3 -m main.prober.web --check-openapi                                # contract drift gate
```

Pointed at `models/` it enumerates the runs and offers a picker; a run is selected by NAME and the
name must be in the server's own listing, so no client string ever reaches a path join. Reading is
anonymous; `falsify_scan` / `calibration` need the shared password. `--impl {node,rust}` picks the
offline replay/search driver those two spawn — a **startup** flag, matching `ProbeSession`'s
session-wide treatment of `impl` rather than a per-request knob.

**Deployed at prober.g5d.io** (reads anonymous, probes password-gated/fail-closed; verified serving 2026-08-19). Local remains the debugging default — from elsewhere:
`ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io`.

Full detail — the one rule (every number comes back from a session method verbatim), the job
registry for the minutes-long probes, what the headless render test actually verifies, and the
two gotchas (`starlette.HTTPException` dispatch; `build_trace_tree` tolerating a nonexistent path)
— is in **`src/main/prober/web/CLAUDE.md`**.
