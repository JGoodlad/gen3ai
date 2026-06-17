# CLAUDE.md — `src/main/prober/` (forensic-replay inspector)

An interactive Textual TUI for the **prober**: browse the `eval_traces` a
training run writes and inspect *why the policy chose what it did* at any saved
decision point. It is the navigable successor to the one-shot `probe_replay.py`
CLI — same analysis, but you click through battles and invocations instead of
re-running a script per turn.

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m main.prober <run_dir | eval_traces_dir | summary.json> [--ckpt PATH] [--inv N]
```

## Engine / app split (the important seam)

The analysis is a **pure, framework-agnostic engine** (`engine.py` + `model.py`);
the TUI (`app.py`) and the `probe_replay.py` CLI are both thin callers. This is
the single source of truth — change the analysis once, both surfaces follow.

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
  Textual screen). Three **non-torch decode helpers** also live here (they need the encoder,
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
  replay/re-roll record (`utils/bridge/reconstruction.py`) — which the prober
  also ignores today (a future counterfactual probe consumes it).
- **`app.py`** — `ProberApp(Gen3App)`: trace `Tree` | invocation `ListView` |
  a `VerticalScroll` of `Collapsible` analysis sections (Summary · Team · Review · Board ·
  Faithfulness · Matchups · Intervention · Saliency · Flow · Outcome).
- **`review.py`** — `ReviewStore`: persistent manual-review annotations (a *funky* flag +
  a **timestamped note append-log** per decision) at `<run_dir>/review_notes.json`; pure
  (no Textual), unit-tested, exports to `<run_dir>/review_notes.md`. Each saved comment is a
  `{ts, text}` entry appended (not overwritten), so the history + *when* each was added is kept;
  the clock is injectable for tests; legacy single-string `note` entries read transparently.

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

## Panels & navigation

Analysis sections (collapsible — **multiple open at once**, in a scroll; toggle
by clicking a title or pressing its number key) render purely from one
`InvocationAnalysis`. Keys are **in display order** — `1`–`9` then `0` for the 10th
(`Outcome`) — and **shown in each title** (`1  Summary`, `2  Team`, … `9  Flow`, `0  Outcome`); `_SECTIONS` is the
single source — `_SEC_TITLE` builds the titles and the `BINDINGS` are generated from it, so
key/label/binding never drift. The top one is **Summary** (`1`, open by default) — the
decision dashboard for walking "funky turns". The context header is chunked into **three blank-line
groups** for scannability — SITUATION (matchup + FIELD + THREAT), DECISION (CHOSE), OUTCOME
(RESULT + REWARD + CRITIC): line 1 the matchup, each active as **species + colour-graded HP bar**
(`_hp_bar`) + bundled **status/volatiles** in `[...]` (e.g. `[TOX(5)|SUB]`) + **boosts** in
`{...}` magenta (e.g. `{atk:-1 spa:+6}`) + held **item** as `@item` (incl. the **opponent's once
revealed** — Choice items highlighted) + outcome; then **FIELD** (weather/hazards/screens/turn +
a `💧wish: our/opp` tag when a Wish is floating — `gen3_wish_wired_v1`, the ~50% end-of-turn heal —
the same `_field_text` the Board shows) · **THREAT** (STACKED, so the Summary is self-sufficient —
line 1 incoming P(KO)·outspeed·worst-on-team·opp-recovery, line 2 the incoming type-**effectiveness**
`worst N× · revealed X%` folded in from Matchups; P(KO) reds with danger in BOTH places —
`gradient_color(1 − pko)`) · **CHOSE** chosen+confidence [+ a `⚠ now prefers X` on disagree] ·
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
but `move_order` wasn't recorded (a no-state / model-free decision), the engine flags the entries
`order_certain=False` and the renderer drops the implied sequence (neutral `·` bullets + a *(move order
not recorded)* note) instead of guessing. One line per move / switch / forced replacement /
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
and **OPP TEAM** (the opponent's REVEALED mons — active ▶ then bench — name · hp · status · item,
the mirror of our switches; Gen3 has no team preview so only revealed mons appear) — and, when the
**hidden-opponent belief** was enabled for the run (`--opp-belief-aux-coef>0`), the model's guess for
the still-hidden mons below it. Two forms, best-available wins:
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

The belief itself is **re-computed from the loaded model** each analysis (`ProbeModel.belief` → the belief
head's per-slot species logits + believed mask → `engine.belief_view_from_logits`; one clean forward, since
the intervention-sweep/saliency passes clobber the extractor's stash), so it works for **any belief-on
checkpoint** — including runs whose recorder predates the summary's per-decision `belief` block. `engine.build_belief`
reads that summary block as a **model-free fallback** (available even without a captured `.npz`). Both
`belief` and `belief_truth` ride the `analyze` JSON output (`asdict`). `None`/absent on a belief-off run (then
only the revealed mons show). **Mon names are
blue** (`_MON_COLOR`); **disabled slots** (a fainted mon / an illegal switch / a no-PP move) render
**grey** (`_DISABLED_GREY`), NOT the red of a low value — so "dead/unavailable" reads differently
from "alive but low HP = real danger". Hidden Power shows its **type** (`hiddenpower(fire)`) — for a
revealed HP from the obs type channel, and **for OUR OWN un-revealed HP from the reconstruction
record** (`engine.build_our_hp_types` → `_retype_hp`, threaded as `analyze_invocation(our_hp_types=…)`,
loaded by `app._load_our_hp_types` / `session._our_hp_types`): Showdown's request carries only the bare
`hiddenpower` id (the type is IV-derived), so without this our own mons showed an untyped HP until they
used it. **OUR side only** — an opponent's un-revealed HP MUST stay bare (no leak), and the retype is a
no-op on websocket/older traces with no `reconstruction.json`. Helpers
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
(recorded vs re-run probs), **Matchups** (our active move type-multipliers — a non-damaging move shows
`—  n/a (non-damaging)` per `MatchupView.applicable`, see MOVES above — + two incoming lines: an
**incoming eff** line decoded from `their_matchups` — `worst N×` / `revealed XX%`, or
`BLANK` when the opponent's coverage is unrevealed; and an **incoming P(KO)** line
decoded from the `incoming_damage` belief block — `active NN%` (our on-field mon's KO
belief) · `outspd NN%` · `worst-on-team NN%` · opp-recovery — the calibrated
DAMAGE belief, not raw effectiveness),
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
`+ outgoing` under `damage_outgoing`, `MoveBelief` `+ prior-fusion` + `· PRE-transformer` under
`move_belief_prefuse` (v32 — the reinjection ran BEFORE the transformer; the band still draws it in
BELIEF, the label carries the timing), `BeliefHead` `+ latent`). The
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
`faint` (a faint in this turn's events); plus `disagree` (added per-analysis when
the loaded model's argmax ≠ chosen). The list shows `?`/`✗` glyphs; `n`/`N` jump
to the **discrete** flags (faint/switch — `uncertain` is the norm for a
low-confidence policy, so it's a glyph, not a jump target). `f` cycles a
battle-outcome filter (all → loss → win), rebuilding the tree.

**Manual review mode (model's own games).** The top **Review** section is the
human-walkthrough surface: a one-glance card of *what the model EXPECTED → what it
DID → what HAPPENED → how surprised* (chosen + prob · incoming-P(KO) belief · V(s) ·
ΔV · TD δ surprise · a `⚠ now prefers X` when the re-run argmax disagrees · the actual
outcome+events), so you can judge each choice. Notes are a **timestamped append log** — the
card lists every comment with the date/time it was added (newest last); the Summary mirrors the
same EXPECTED→DID→HAPPENED story. While stepping turn-by-turn (`j`/`k`) you annotate:
**`space`** toggles a *funky* flag, **`e`** focuses the note input (Enter **appends** a new
timestamped entry — it does NOT overwrite the prior one), **`[`/`]`** jump to the prev/next
annotated decision, **`E`** exports all notes (with timestamps) to `<run>/review_notes.md`.
Annotations persist in `<run>/review_notes.json` (`review.ReviewStore`, keyed by run-relative
trace path + invocation index) and show a `⚑`/`✎` glyph in the invocation list — distinct from
the auto `summary_flags`.

Layout: the trace `Tree` (`NavTree`) uses file-explorer `←`/`→` to collapse/
expand; the three panes are separated by draggable `PaneSplitter` bars — drag with
the mouse to resize (the `#analysis` pane is `1fr` and absorbs the slack). The
analysis sections are `Collapsible`s in a `VerticalScroll`, so the render methods
(unchanged, keyed by widget id) populate them whether open or collapsed.

## Agent API & JSON CLI (`session.py`, `query.py`)

`ProbeSession` is a framework-agnostic facade so **agents/scripts** can probe a
model without the TUI — all methods return JSON-serializable dicts and model
loading uses the same exact→nearest→recent ladder (cached per process). A
`battle_id` is the trace's `*_summary.json` path **or** a short
`step_<N>/<Opponent>/<outcome>_<idx>` id.

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
  no-death value craters split on the critic's **pre-cliff value sign** (scale-invariant):
  V(s)>0 = the critic thought it was WINNING then craters = `critic_blindspot` (CRITIC
  CAPACITY / a missing obs feature — the "more value capacity / transformer layers"
  lever); V(s)≤0 = it already knew = `positional_grind` (upstream/material). Reads the
  true per-opponent win-rates from `eval_results.jsonl` (falls back to ranking by raw
  loss volume, announced in the metric + a caveat, when absent). Carries explicit
  `caveats` (loss-weighted sampling; one-cause-per-loss; bot-only rating weight).
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
  `{id, short_id, opponent, step, outcome, turns, worst:{inv, turn, chosen,
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
- `analyze(battle_id, inv)` — full `InvocationAnalysis` as a dict (loads the
  model); the value block gains a γ-discounted `td_residual` and, on a `--use-popart` model,
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
    the safe-switch reads), the opp-active effect scalars, and (on `--unified-damage
    both`) our 4 moves' **outgoing** damage `[low,high,crit,pko]` (request-slot/action order — the
    equal-effectiveness move tie-break). v24 (gen3_unified_move_system_v1) ADDS per-status SECONDARY
    probabilities: `incoming_secondary` (the opp active's damaging-move para/flinch/freeze threat,
    accuracy-folded + ×Serene Grace) and `outgoing.secondary` (per OUR move — "what status can it cause").
    v30 (gen3_unified_topk_incoming_v1) ADDS `incoming_topk` (`None` unless the run trained `--damage-topk`):
    the opp active's K most-believed CANDIDATE moves INDIVIDUALLY — `moves` (each with its exact decoded
    `move` NAME — resolved from the op's stashed candidate indices, typed HP → `hiddenpower(type)` — plus
    `belief`/`accuracy`/`is_phys`/`latent`) and `per_defender` (6 × K of `[high, pko, status_lands]`, the
    discrete-move + per-pivot safe-switch read incl. damage-immunity AND status-immunity = 0). The app's
    Matchups panel renders an **"opp likely (op):"** block — per move: name + belief, the read on OUR active
    (▶), and the SAFEST bench pivot (min combined damage/status threat) — the literal "anticipate the move
    → which mon switches in safe (immune)" read.
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
    unit-tested (`engine_test::test_move_belief_view_*`). The app's **Matchups** panel renders a
    **move belief (✓ revealed · ≈ unseen)** block — `metagross ✓ meteormash 100%  ≈ explosion 32% · …`
    (green ✓ revealed, magenta ≈ unseen) — plus the op's outgoing line, the `opp 2ndary:` incoming-status
    line, and the per-OUR-mon op **incoming** damage (worst-channel %HP →KO%, species-labeled, active ▶,
    red-graded by P(KO)). All fields ride the `analyze` CLI JSON.
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
  (`move_category`: selfko/recovery/setup/stall/status/switch/attack_or_other), our/opp species+HP,
  policy `conf` (`softmax(logits)[chosen]` — learned vs exploration-tail), `reward`, critic `dV`
  (`V[i+1]−V[i]`, the self-KO over-valuation signal), incoming-KO `pko` belief, faint flags, outcome.
  The single source for the softmax/dV/`decode_incoming_belief` plumbing every behavioural-hypothesis
  check reuses (the shipped self-KO finding used the `selfko` `dV_med`). Distinct from `falsify_scan`
  (the luck/mistake bracket) — this is the raw per-decision table. `move_category` /
  `decision_table_digest` are pure (unit-tested). See `designs/research_state/` for the hypothesis ledger.

CLI mirror — prints JSON to stdout (and `{"error": …}` + exit 1 on failure, so an
agent always gets parseable output). `--help` carries a worked example sequence:
```bash
python -m main.prober.query triage   <run_dir> [--step N] [--opponent X]
python -m main.prober.query probe    <run_dir> <is_faster|damage_taken|faint_soon|faint_healthy|big_hit_incoming|opp_switches|opp_status_move> [--which vf|pi] [--step N] [--max-decisions K]
python -m main.prober.query switch-vs-info <run_dir> [--step N] [--opponent X] [--outcome win|loss] [--max-battles K]   # MODEL-FREE: do we switch more when we know less?
python -m main.prober.query summary  <run_dir>
python -m main.prober.query list     <run_dir> --outcome loss --step 8000000
python -m main.prober.query scan     <run_dir> --outcome loss --opponent X [--metric td_residual] [--limit K]
python -m main.prober.query overview <battle_id>
python -m main.prober.query find     <battle_id> value_drop --limit 5
python -m main.prober.query analyze  <battle_id> <inv> [--ckpt PATH] [--tier auto|nearest|recent]
python -m main.prober.query falsify  <battle_id> [--inv N]... [--worst K] [--seeds N] [--alts K] [--followup random|default]
python -m main.prober.query falsify-scan <run_dir> [--outcome loss|win] [--opponent X] [--step N] [--limit K] [--worst K] [--seeds N] [--alts K] [--concurrency N]
python -m main.prober.query calibration  <run_dir> [--step N] [--opponent X] [--limit K] [--worst K] [--seeds N] [--concurrency N] [--bins N] [--overvalue-tau F]
python -m main.prober.query decision-table <run_dir> [--step N]... [--opponent X]... [--outcome loss] [--cat selfko]... [--out t.jsonl] [--limit K]
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

The engine reads five semantic obs regions by offset: the active-move type
multipliers (`OFFSET_REACTIVE + move_multiplier`, currently dim 1458), the
`our_matchups` block (`+ our_matchups`, currently 1568), the **`their_matchups`**
block (`+ their_matchups`, currently 1712 — the raw-effectiveness decode + saliency),
the **`incoming_damage`** block (`+ incoming_damage`, currently 1517, dim 51 — the
P(KO)/chip belief decode + critic/policy saliency; per-mon active flag read via
`pokemon_full_dim`), and the turn-history span — all resolved at runtime from
`Gen3ObservationEncoder.get_layout()`. **If the obs layout changes, these move
automatically** (e.g. `gen3_move_effects_v1` inserted a block before `our_matchups`,
shifting it; `gen3_incoming_damage_v1` inserted the 33-dim belief block at reactive
offset 50), and `engine_test.py` pins the resolved values
(`test_offsets_resolve_matches_layout`) so a silent shift fails loudly. (Mirror note
in `src/agents/observation/CLAUDE.md`.)

## Worker-thread model (event-loop safety)

Textual runs on asyncio; torch forward/backward and the checkpoint load are
blocking. So:
- The checkpoint loads in `@work(thread=True, exclusive=True, group="load")`; a
  selection made before it's ready is queued (`_pending_inv`) and run on ready.
- Each analysis runs in `@work(thread=True, exclusive=True, group="analyze")`;
  fast re-selection cancels the prior worker. A monotonic `_analyze_token` guards
  against a stale (already-superseded) result painting the panels.
- Workers **compute and return**; widgets are only touched on the event loop via
  `call_from_thread`. The npz is opened, the single obs row copied out, and the
  archive closed immediately (no handle accumulation while browsing).

## Gotchas

- **Move-action labels are realigned to request-slot order (`engine._reorder_move_labels`).** The
  recorded `summary.actions` dict orders the 4 move actions by poke-env **`available_moves`**, which can
  DIFFER from the obs **request-slot** order that the action mask, the `DamageOperator`, and the policy
  logits (action 6+k) all use (`gen3_move_slot_align_v1`) — e.g. after a Disable / server reorder. Left
  unaligned, EVERY action-6+k-indexed consumer (the mask, the matchup ×mult, the op outgoing damage, the
  re-run argmax) pairs with the WRONG move (the bug that made Hypnosis — a 0-BP status move — display
  Thunderbolt's 27% / `par 10%`). `analyze_invocation` calls `ProbeModel.our_active_move_slots(obs)`
  (decodes our active mon's move block = request order) and reorders `labels[MOVE_START:MOVE_END]` to it
  BEFORE building the mask, matching by normalized name (HP type stripped), with a safe fallback (unmatched
  → unchanged, never a wrong reorder). The op itself was always correct (`usable = legal·(bp>0)` zeroes
  status moves). Guarded by `engine_test::test_reorder_move_labels_realigns_to_request_slot_order`. The
  outgoing-damage panel now also renders a non-damaging move EXPLICITLY as `— (non-damaging)`.
- **Faithfulness is exact only on the `exact` tier.** On `nearest`/`recent` the
  model differs from the one that generated the trace, so recorded ≠ re-run (the
  re-run cell is colored by the drift) — expected, and the badge says which tier.
  For bit-exact replay, train with `--keep-eval-snapshots` (then the `exact` tier
  loads the retained snapshot), or pass `--ckpt`.
- A trace whose `_states.npz` is missing, or an invocation with `has_state=0`,
  yields an analysis with `warnings` and no panels (the engine never touches the
  model) — handled, not a crash.
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
- **`y`** reveals the current battle's `*_replay.html` path (browser-watchable Showdown replay) on a
  dedicated full-width **`#replay-path-bar`** — the path on its OWN line so it's cleanly selectable
  under **`v`** copy mode (the portable path; the old toast wrapped the path with its label). It also
  best-effort `copy_to_clipboard`s it (OSC-52 terminals only — kitty/iTerm2/WezTerm; dead on
  Terminal.app, hence the bar). The bar is hidden until used and cleared when a new battle is selected;
  warns if the file is missing.
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
`ReviewStore` — flag/note roundtrip, persistence, prune, export), `app_test.py`
(Textual `run_test` Pilot with an injected fake model — never loads a real checkpoint,
so it stays fast; incl. the review flag/note/glyph flow):

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/main/prober -q
```

`textual` is pinned in `environment.yml`. The shared Textual base lives in
`src/main/tui/` — see its CLAUDE.md.
