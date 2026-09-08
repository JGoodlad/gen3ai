# CLAUDE.md — `src/main/prober/` (forensic-replay inspector)

The **prober**: browse the `eval_traces` a training run writes and inspect *why the policy chose
what it did* at any saved decision point.

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober <models_dir | run_dir>        # the browser front end -> :6008
python3 -m main.prober.query <cmd> ...               # the JSON CLI, for agents and scripts
```


**The DETAIL lives in `designs/prober/`** — an always-current tree that OWNS what it holds and is
updated in the same pass as the code, exactly like this leaf. This file is a CONSTITUTION, a
COMMAND CARD and a MAP: the rules, the invocations, the module maps and every hazard that has
already cost a wrong reading. Follow the pointer when you touch the subject.

| I am about to… | Read |
|---|---|
| read a decision's panels field by field | `designs/prober/analyze_panels.md` |
| change what the RESULT timeline says happened | `designs/prober/result_timeline.md` |
| read or change the belief / threat views | `designs/prober/beliefs_and_threats.md` |
| call `loops` / `triage` / `probe` | `designs/prober/session_scans.md` |
| call any model-free reading method | `designs/prober/session_reading.md` |
| read `analyze`'s JSON block by block | `designs/prober/analyze_output.md` |
| run a re-roll / search / rollout probe | `designs/prober/counterfactual_probes.md` |
| touch `ProbeModel` or trace discovery | `designs/prober/engine_and_model.md` |
| diagnose a checkpoint that will not load | `designs/prober/arch_drift.md` |
| choose `--impl node` vs `rust` | `designs/prober/sim_impl.md` |
| add or find a test | `designs/prober/tests.md` |

History — the retired Textual TUI, its panels, its keys and manual review mode — is in
`designs/research_state/claude_md_archive/prober_leaf_history.md`. Nothing there is current.

**TWO surfaces over one engine, and that is the whole design.** `engine/` + `session/` are the
analysis; `web/` renders it for a human and `query.py` prints it for an agent. Neither is a layer
on the other.

**Both are PACKAGES whose `__init__.py` is a re-export hub** (2026-08-23 — they were single
3,058- and 2,573-line modules). `from main.prober.engine import <anything>` and
`from main.prober.session import <anything>` resolve exactly as they always did; the module maps
are in *Engine / app split* and *Agent API & JSON CLI* below, and `hub_contract_test.py` pins the
full pre-split export set, the no-cycle rule, and `ProbeSession`'s mixin base list.


**⚠ THE TEXTUAL TUI IS GONE** — `python -m main.prober` starts the WEB app, and its two TUI-only
flags (`--ckpt`, `--inv`) print what replaced them instead of failing. `web/` is the only
human-facing surface. What was dropped with it, and why:
`designs/research_state/claude_md_archive/prober_leaf_history.md`.

**Reading a game turn by turn** is `battle_turns()` (below) — model-free, so it opens instantly.
`query turns` prints the whole game as JSON; `/battle` renders it as a phone-readable replay. The
plain-text battle log is `engine.timeline_entry_text`, and the vocabulary it draws on
(`engine.CANT_PHRASE` / `NO_EFFECT_TEXT` / `surprise_phrase`) lives in the ENGINE precisely so that
a reason one surface learns cannot go missing on another.

## Engine / app split (the important seam)

The analysis is a **pure, framework-agnostic engine** (`engine/` + `model.py`); the web front end
(`web/`), the JSON CLI (`query.py`) and the one-shot `probe_replay.py` are all thin callers. This is
the single source of truth — change the analysis once, every surface follows. It is also why
retiring the TUI cost no analysis: the deleted 4,400 lines were rendering, not reasoning.

- **`engine/`** — `analyze_invocation(model, summary, npz, inv_index) →
  InvocationAnalysis` (a tree of frozen dataclasses: `ActionRow`, `MatchupView`,
  `InterventionSweep`, `Saliency`, …). No printing, no Textual, no file IO beyond
  the passed-in arrays. **Every torch call goes through the
  injected `model`**, so the whole engine is unit-tested with a `FakeProbeModel`
  (no torch) — see `engine_test.py`. One module per concern, a strict DAG (leaves first),
  with `__init__.py` the re-export hub:

  | module | holds |
  |---|---|
  | `views.py` | every frozen dataclass an analysis returns — the DATA MODEL. No numpy, no IO |
  | `util.py` | the shared leaves: percent parsing, npz access, species keys, move predicates |
  | `opponents.py` | opponent-NAME ordering (sentinels first, strongest first) |
  | `flags.py` | `summary_flags` + the cure-option ("heal ≠ cure") helpers |
  | `protocol.py` | reading the RAW Showdown protocol out of a trace's `*_replay.html` |
  | `board.py` | the BOARD read-model (`build_board`) |
  | `timeline.py` | the RESULT timeline — HP loss re-attributed, one line per action |
  | `beliefs.py` | species / move / exclusive-species beliefs + the refinement trajectory |
  | `intent.py` | the α/β opponent-intent read + `awareness_text` |
  | `spread.py` | believed vs TRUE derived spreads (the DamageOperator's stat input) |
  | `switch_in.py` | forced-switch OUTGOING damage per bench candidate |
  | `decode.py` | `_faithfulness` / `_matchups` / `_intervention_sweep` / `_saliency` / `_threats` / `decode_incoming_belief` |
  | `analyze.py` | `analyze_invocation` — the top-level entry — plus `build_meta` / `build_value_dist` |
  | `taxonomy.py` | loss attribution: the turning-point category table |
  | `probes.py` | representation probing (`fit_probe`) |
- **`model.py`** — `ProbeModel`: the torch boundary, and the ONLY place a forward or backward runs.
  `load(ckpt)` does a raw `MaskablePPO.load` (no env, no `ModelVersion` check), resolves `ObsOffsets`
  once from `enc.get_layout()`, **silences the policy's `ObservationDebugger`** (a `--log-level
  periodic` checkpoint would otherwise print a DEEP TRACE banner into the output), and raises
  `ArchDriftError` on a stale checkpoint. `action_dist` / `logit_grad` are the forward/backward pair;
  `belief`, `damage_op_view`, `move_belief`, `value_dist_at` / `win_prob_at` and `architecture()` each
  read a head's stash after one clean forward. The three non-torch decode helpers (`describe_global`,
  `describe_team`, `describe_turn_outcome`) live here because they need the encoder. 🚨 **A turn's
  events land in the NEXT decision's obs**, so `describe_turn_outcome` is read from decision *T+1*.
  Detail: `designs/prober/engine_and_model.md`.
- **`discovery.py`** — pure filesystem, and it never opens a JSON or an npz: `build_trace_tree`
  groups step → opponent → battle by **parsing path strings only**, so a 1000-battle run opens
  instantly. It reads each cycle's `eval_manifest.json` for model identity, and
  `resolve_model_for_step` picks the model **per battle**. 🚨 **The `<outcome>` alternation is BUILT
  from `agents.training.trace_result.OUTCOMES`**, never retyped here. A trace's siblings:
  `*_replay.html` (the raw protocol every honest claim is checked against) and, on bridge-eval
  traces, `*_reconstruction.json` (what every counterfactual probe needs).
  Detail: `designs/prober/engine_and_model.md`.
- **`web/`** — the browser front end (FastAPI + Jinja2/HTMX over `ProbeSession`). It is
  **first-class for the GPU obs**: the learned belief/op signals tagged `🔷 GPU` render PRIMARY
  and the decoded CPU obs regions they subsume tagged `📋 CPU-obs` render dimmed, because the
  operator's physics supersede the older type-effectiveness decode and a reader must be able to see
  which is which. **Beliefs** is the model's world-model vs ground truth; **Threats** leads with the
  DamageOperator. See `web/CLAUDE.md`, and `designs/prober/beliefs_and_threats.md` for
  what those panels MEAN.

## THE RESULT VOCABULARY — `win` · `loss` · `draw`, and what an old tree's zero means

One declaration, `agents/training/trace_result.py` (`gen3_trace_result_v2`, pure stdlib — the
prober imports it without pulling in the training stack, exactly like `trace_selection`). The
recorder writes it, the filename prefix carries it and every filter here is built from it.

| `meta.result` | `meta.draw_kind` | how the battle layer reports it |
|---|---|---|
| `WIN` | — | `won` |
| `LOSS` | — | `lost`, before the turn cap |
| `DRAW` | `timeout` | `lost` **and** `turn >= MAX_TURNS` (250) — the trainee FORFEITED at the deadline |
| `DRAW` | `tie` | `won`/`lost` both falsy, finished — the sim's `\|tie\|` |

**A TIMEOUT ARRIVES WEARING A LOSS'S FLAGS.** The trainee forfeits at the cap
(`inference/player._handle_stall`), so poke-env reports `lost=True`. The training reward never
agreed — `reward_manager`'s terminal fold pays `draw_penalty` for exactly that state, keyed on the
TURN COUNT — and `classify_result` now tests the cap **before** the loss, on the same constant
(`reward_weights._TIMEOUT_TURN_CAP` == `MAX_TURNS`; parity pinned by
`trace_result_test.test_the_classification_matches_the_training_rewards_own_timeout_rule`).

🚨 **A PRE-DRAW-BUCKET TREE'S `draw: 0` IS NOT A MEASUREMENT.** Every trace written before
2026-09-07 carries no `meta.result_vocabulary`, and that ABSENCE is what dates it. In that era a
timeout was written as an ordinary `loss_*` (separable only by `meta.turns >= MAX_TURNS`, which is
what the G7 kill clause has always done) and a tie was **dropped before the summary was written** —
no file, no count. So such a tree can estimate a STALL rate but a TIE rate is **NOT MEASURABLE**
there. `run_summary()` reports `result_vocabulary` (the eras present, sampled) and
`result_vocabulary_note` (`trace_result.era_note`, `None` when the tree is all-current); `/` prints
the note above the outcome chart, and the chart omits the draw series entirely rather than drawing
a flat zero line. A MIXED tree — a run that restarted onto new code mid-flight — gets the note too.

🚨 **AN UNKNOWN RESULT IS REFUSED, NEVER RENDERED.** `battle_overview` / `battle_turns` call
`trace_result.result_of`, which raises `UnknownTraceResult` on a token outside the vocabulary
(`"TIE"` included — it is the string the *previous* writer would have used). A forensic tool that
displays an outcome it cannot classify is how a mislabelled bucket survives being looked at. An
ABSENT result is not an unknown one and passes through.

**Where draws sit in the CAPTURE QUOTA: their own bucket** (`_FORENSIC_DRAW_QUOTA` = 5, beside
win 5 / loss 10). Folding them into the loss quota — which is what the old code did — lets a stall
storm evict the decisive losses the prober exists to study. `calibration` EXCLUDES draws (no binary
realized label) and reports `n_draw_excluded` rather than dropping them silently; `awareness_scan`'s
`cap_loss` accepts `loss` **or** `draw` at the cap so the row means the same thing across the whole
archive.

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

**The three walls behind the diagnosis** — a deleted flag still baked into the zip's
`features_extractor_kwargs` (recovered by dropping unknown kwargs, and *which* ones is reported),
a value the code now validates (deliberately NOT recovered), and weight shapes that no longer fit
(not recoverable in principle) — plus what the error message names and the ~5 ms `peek_checkpoint`
that makes diagnosing cheap: `designs/prober/arch_drift.md`. Tests: `model_test.py`.

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
3. **most recent** — **the run's LAST SNAPSHOT**, resolved through the ONE choke point
   `agents.training.fixed_opponent_pool.resolve_model_ref`, so a bare run dir means here exactly
   what it means to a `--distill-teacher` or `--stable-opponents` spec.

🚨 **This tier was `best_model` → latest until 2026-09-06, which is the ordering
`gen3_last_snapshot_resolution_v1` INVERTED everywhere else** — `best_model/best_model.zip` is the
BOT-WIN-RATE export and is now the LAST rung, a fallback for a run with nothing else. The prober
kept a second opinion, and it was not academic: measured over five archived runs on this box, **all
five disagreed**, the prober loading the bot-selected export while its own tier label read "most
recent" (one pick differed by ~23.3M steps). `resolve_checkpoint_with_rung` returns the RUNG beside
the file and `ModelChoice.detail` prints it, so a `best_model_fallback` read says outright that
those weights were chosen by bot win rate rather than by being latest — a different claim, which
must not render as the same sentence. It falls back to the historical ladder (reported as its own
rung, never silently) when the choke point cannot resolve, since it needs a `model_config.json` and

`--ckpt` forces an override. The badge shows the active tier + the trace's `git_hash` /
`arch_signature` from the manifest, so any faithfulness drift is explained, and loaded models are
cached by path (`_model_cache`) so revisiting a step is instant. Even for runs that predate the
manifest (no snapshots), the ladder picks the **nearest checkpoint** — strictly better than always
using `best_model`.

## What one decision's analysis CONTAINS (the `/analyze` panels)

Everything below renders purely from one `InvocationAnalysis` — this section is about what the
FIELDS mean, which is why it survived the TUI that used to draw them. Where it still describes a
terminal (glyphs, colour ramps, fixed-width columns), read that as the SEMANTIC it encodes: `/analyze`
renders the same distinction in HTML, and `web/CLAUDE.md` says how. Panel-by-panel field map, kept
current with the renderer, lives there.
Panel by panel — the Summary header's three groups, MOVES / SWITCHES / OPP TEAM, the two belief
forms, Board, Faithfulness, Intervention, Saliency, Flow and Outcome — is
`designs/prober/analyze_panels.md`. What stays here is the part a surface can get WRONG.

### What the timeline may CLAIM

🚨 **"— no effect" IS A CLAIM, AND IT IS ONLY OURS TO MAKE WHEN THE EVIDENCE SUPPORTS IT**
(`engine._no_effect_supported`, 2026-09-07). Exactly three things support it: the recorder DECODED
the move's fate/effectiveness, the SIM said so (`|-fail|` / `|-immune|` / `|-miss|` in the move's own
protocol window), or that window was LOCATED and is EMPTY of effect tags. Anything else renders
**`— outcome unrecorded`** — a gap in the evidence, said out loud, and *not* a synonym. And a window
that CONTRADICTS the claim (it carries effect tags) beats every recorded outcome, because the log is
the sim's own transcript while the recorded `events` list is known to have had a hole. The assertion
form of the same rule is **`verify_timeline_against_protocol`** → raises `TimelineContradiction`;
it is deliberately NOT called from `build_result_timeline` (a forensic view must still render a trace
it cannot fully explain) — it is for tests and for a surface that would rather stop than mislead.

⚠️ **"Nothing happened" had THREE causes and one sentence, so the line described the wrong thing.**
The recorded outcome says what a side CHOSE; nothing in a model-free trace says whether the choice
ever ran, so a move that never executed was explained as one that executed and achieved nothing — a
claim about the MOVE on a turn where the move never happened. Reported on gen-16 `loss_s0_004`:
Forretress CHOSE Explosion on turn 6, was outsped by a +2 Tyranitar and killed, and the timeline
read `we explosion — no effect`. Two pure readers over the turn's protocol slice fix it, siblings of
`move_order_from_protocol` and matching sides the same way:

| the log says | now reads | before |
|---|---|---|
| no `\|move\|` for that side, and it fainted | `— never moved (fainted first)` | `— no effect` |
| `\|cant\|<mon>\|frz` (or par/slp/flinch/…) | `— couldn't move (frozen)` | `— no effect` |
| `\|-immune\|<target>` | `— no effect (immune)` | `— no effect` |


The three pure protocol readers that supply that evidence (`protocol_action_fate` /
`protocol_move_result` / `protocol_move_effects`), the precedence between a RECORDED outcome and
the log, and the measured 9.8% of move lines they repaired, are in
`designs/prober/result_timeline.md`.

🚨 **A DETECTOR READS THE RAW PROTOCOL; ONLY A HUMAN SURFACE READS THE RENDERED TIMELINE.**
`loops.py` keys on each battle's `*_replay.html` protocol lines, never on
`engine.timeline_entry_text` — the rendering is a SENTENCE, and a detector must key on the fact
underneath it. The timeline's job is to be readable, the detector's is to be exact, and the two
must not be wired together.

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

It hangs off the RE-COMPUTED branch only, never the summary fallback (whose top-3 rows do not sum
to 1), which is also why the model-free `battle_turns` / `/battle` do not carry it. The two
distinct defects it separates (`max_expected_count`/`illegal_mass` vs `duplicate_top1`), the clean
`revealed_leak_max` reading and its dependence on `--species-prior-fusion`, and the contents of
both GPU-first sections, are in `designs/prober/beliefs_and_threats.md`.

**Per-invocation flags** (`engine.summary_flags`, model-free): `switch` · `uncertain` (top recorded
prob < `UNCERTAIN_THRESHOLD` = 0.34, a genuine tossup) · `faint` · `opp-switch` (the OPPONENT
voluntarily pivoted — `engine.opp_voluntary_switch`) · `cure-skipped` · plus `disagree`, added per
analysis when the loaded model's argmax ≠ chosen. `query find <battle> <flag>` lists them. Two of
them exist because a reader got the position wrong without them:

### The "heal ≠ cure" trap (`cure-skipped`)

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

### The "computed-vs ≠ resolved-vs" guard (`opp-switch`)

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

## Agent API & JSON CLI (`session/`, `query.py`)

`ProbeSession` is a framework-agnostic facade so **agents/scripts** can probe a
model without a UI — all methods return JSON-serializable dicts and model
loading uses the same exact→nearest→recent ladder (cached per process). A
`battle_id` is the trace's `*_summary.json` path **or** a short
`step_<N>/<Opponent>/<outcome>_<idx>` id.

**`session/` is a package; `ProbeSession` is assembled from one MIXIN per command family.** The
class alone was 2,068 lines, so splitting the module without splitting the class would not have
got under the size bound — hence a base list rather than one `class` block, and
`hub_contract_test.py` pins that no family silently drops out of it.

| module | holds |
|---|---|
| `core.py` | `ProbeSession` itself — construction, the shared internals, the resolution ladder |
| `reading.py` | MODEL-FREE orientation: `run_summary` · `battles` · `decision_table` · `battle_overview` · `battle_turns` |
| `scans.py` | RUN-LEVEL model-free folds: `scan` · `awareness_scan` · `loops` · `triage` |
| `trace_io.py` | the trace's sibling files (protocol log, privileged teams, our HP types) — file IO kept OUT of the pure engine |
| `analysis.py` | the per-decision deep read: `analyze` (loads the model) · `find` |
| `counterfactual.py` | `falsify` · `lookahead` · `better_line` · `replay_counterfactual` |
| `aggregate.py` | `falsify_scan` · `calibration` — the two run-level counterfactual folds |
| `probes.py` | `probe` · `switch_vs_info` · `history_saliency` |
| `serialize.py` | the JSON-shaping leaves |
| `stats.py` | the pure statistics (loop aggregation, discounted returns, reliability bins) |
| `probe_targets.py` | the representation-probe target table |


**The per-method reference moved out of this leaf** — one doc per family, each stating what the
method returns and what it costs:

| family | methods | reference |
|---|---|---|
| model-free reading | `battles` · `scan` · `switch_vs_info` · `battle_turns` · `awareness_scan` · `battle_overview` · `find` · `probe_model` · `decision_table` | `designs/prober/session_reading.md` |
| run-level scans | `loops` · `triage` · `probe` | `designs/prober/session_scans.md` |
| the deep read | `analyze` and every block on its JSON | `designs/prober/analyze_output.md` |
| counterfactual | `lookahead` · `better_line` · `replay_counterfactual` · `falsify` · `falsify_scan` · `calibration` | `designs/prober/counterfactual_probes.md` |

Four rules from that reference are binding wherever a number of theirs is read, so they stay here.

### `critic_currency` — what every V on every view MEANS

- `run_summary()` — **orient** (model-free): steps, per-step model identity
  (git/arch/snapshot-available), opponents with win/loss tallies, persisted
  checkpoints, γ, and **`critic_currency`**. The natural first call.

  **`critic_currency` says WHICH READOUT IS THE CRITIC, and therefore what every V on every other
  view MEANS** (`ProbeSession.critic_mode()` / `.critic_currency()`, model-free off the run's
  `model_config.json` `critic` key, cached exactly like `_dist_support`). `{mode, units, low, high,
  even, span, is_probability, default_overvalue_tau, note}` — `shaped` (V is a shaped, discounted
  return of roughly ±30, and its zero is NOT "even": a self-mirror 50/50 reads V≈−6.5) or
  `winprob` (V = sigmoid(win-prob logit) ∈ [0,1], `values` EQUALS `win_probs`, PopArt absent,
  G(s) the terminal win indicator at γ=1, and **0.5 really is even**).
  **An ABSENT `critic` key means `shaped`** — a fact about the archive rather than a chosen
  default: the flag landed at config version 109, so every run recorded before it has no key and
  every one of them is shaped (**214 of the 215** on this box, measured 2026-09-06). An unreadable
  config is shaped too, so a failed read can never silently re-scale an old run's numbers.

### β name provenance — a label decided a research conclusion

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

### The trace QUOTA prefers losses — read `selection` FIRST

🚨 **THE QUOTA IS NOW STATED, NOT ASSUMED — read `selection` FIRST**
(`gen3_trace_selection_manifest_v1`). `captured_win_fraction` says what the sample's mix IS;
`selection` says what the recorder's RULE WAS, which is what distinguishes a loss-enriched quota
from a genuinely losing population — and until this shipped nothing in the trace tree recorded
it, so every curve here silently inherited the skew (measured on `ai_v9_59_R2ACTION_0827`:
captured outcome rate **0.46** against the same cycles' recorded **0.901 vs bots / 0.702 vs
pool**). Both `calibration` and `falsify_scan` return the block, scoped to the `step` filter,
from `ProbeSession.trace_selection(step)`: per step the rule in words plus per-opponent
`battles_played` / `battles_won` / `traces_written` / `traces_won` and the derived
`capture_rate_win` / `capture_rate_loss` (traces per battle PLAYED, by outcome — the pair whose
DIFFERENCE is the skew; a zero denominator reads `None`, never `0.0`). ⚠️ **A tree that records
no selection is `known: false` and carries the standing UNKNOWN label — never read as uniform**;
`calibration` additionally puts that label FIRST in its `caveats`, because it says whether the
selection confound below it can be sized on this tree at all. Every archived run is in that
state; only cycles collected after this shipped carry a record. The `/calibration` web view
renders the per-opponent capture-rate table and marks a selection-unknown curve as such.
Declaration: `agents/training/trace_selection.py` — the one module the recorder, this session,
and `main.scaffolding_gauge` all read, so the three cannot drift on what the quota was.


### `overvalue_tau` is in the CRITIC'S OWN UNITS

`calibration`'s over-value threshold defaults **per critic currency** (`None` resolves it from the
run): 5.0 shaped return units on a critic spanning roughly ±30, ≈**0.083** P(win) under
`--critic winprob` — the same 1/12-of-span fraction either way. Carried across unchanged, the
shaped 5.0 **exceeds the entire representable range of a probability gap**, so no crater can clear
it and `critic_overvalued` reads a confident **0** — a units error in the shape of a finding
(measured: the headline moved 0.0 → 0.4997 once the tau was in the right currency). An EXPLICIT
value is honoured verbatim, the result carries `params.overvalue_tau_source` + `critic_currency`,
and a tau no gap in the run can reach raises a loud `threshold_warning`. Any future currency change
re-opens this for every value-unit threshold in the tree. Detail:
`designs/prober/counterfactual_probes.md`.

### The JSON CLI

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
python -m main.prober.query replay-counterfactual <battle_id> <inv> <action> [--rollouts N] [--opponent-ckpt PATH] [--opponent-source auto|bot|self|ckpt] [--opponent-regime recorded|stochastic|greedy] [--narrate]
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


Equivalence is pinned node-vs-rust at 18873 + 30689 leaf fields, and the cross-impl
`better_line_integration_test` asserts node and rust yield IDENTICAL candidate V — an obs-level
bit-identity claim at every ply of the beam. Per-op the rust driver is 7–20×; end-to-end
`better_line` is only ~1.9×, because Python-side obs materialization is the bottleneck and it is
impl-invariant. The two known divergences and the build/override path: `designs/prober/sim_impl.md`.

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
- **Per-move incoming threat = the `incoming_matrix` (`--damage-matrices incoming`)**, which
  `ProbeModel.damage_op_view` threads into `decode_damage_block` — since `gen3_op_block_trim_v1`
  there is no lean top-K arm left to disambiguate, and a run whose matrices-decode flags are not
  threaded mis-reads an ABSENT block rather than reporting one. What the panel renders per cell:
  `designs/prober/analyze_panels.md`.
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

Everything under `src/main/prober/` is unmarked (pure, no torch, no bridge) except the three
`*_integration_test.py` files that carry `@sim` (real bridge battles), `web/`'s headless-chrome
render test (`@integration @browser`), and `belief_obs_fuzz_test.py`, which is run as a script. What each file pins — the engine's `FakeProbeModel` +
offset regression, the session API and its `falsify_scan` / `calibration` folds, the pure
falsifier / better-line / awareness / loops / forensics cases, the torch boundary's stash-location
and `ArchDriftError` tests, and the `web/` suite — is `designs/prober/tests.md`.

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/main/prober -q
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
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober.web models/   # :6008, pick any run
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
