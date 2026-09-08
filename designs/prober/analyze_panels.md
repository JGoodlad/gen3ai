# The `/analyze` panels — what every field on a decision view means

Owned by this tree. `src/main/prober/CLAUDE.md` keeps the rule this file serves: everything here
renders purely from one `InvocationAnalysis`, so this is about what the FIELDS mean; where the text
describes a terminal (glyphs, colour ramps, fixed-width columns), read that as the SEMANTIC it
encodes — `/analyze` renders the same distinction in HTML, and `web/CLAUDE.md` says how.

## The Summary panel — SITUATION · DECISION · OUTCOME


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
heal / boost move (whose effect is legitimately invisible in the outcome) is left bare.

## AFTER · REWARD · CRITIC · WIN-PROB, and the three side-by-side panels

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


## Team · Board · Faithfulness · Intervention · Saliency · Flow · Outcome

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

## The decoded op fields that render display-only

- **The rest of the decoded op fields now render too** (display-only): the OUTGOING **status-landing** (`our
  status (land)` — per OUR move P(a dedicated status move lands on the opp) + ✓certain/?prior), the per-defender
  **`incoming extras`** (the op's belief-aware `p_outspeed` + `provenance`), the **`opp Choice Band`** belief
  (`p_cb` + per-our-mon CB-conditional physical →KO), and the OUTGOING **`our damage vs switch-ins`** matrix (our
  moves × each REVEALED opp mon). Opp-mon columns are labeled by the obs-slot→species map (`mb.opp[*].slot`),
  NOT the board active+bench order (the op reads `ctx.species_ids[:, TEAM_SIZE:]` raw, active at any slot).

## The per-move incoming matrix

- **Per-move incoming threat = the `incoming_matrix` (`--damage-matrices incoming`).** `ProbeModel.damage_op_view`
  threads `matrices_incoming_k` / `matrices_outgoing` into `decode_damage_block` (since
  `gen3_op_block_trim_v1` there is no lean-top-K arm to disambiguate — `decode_damage_block` lost its
  `topk_k` parameter entirely). The Threats panel renders the rich
  `incoming_matrix`: per opp candidate move (decoded name + belief + acc + phys/spec + notable effect/secondary)
  → per OUR mon the FULL cell `low–high · crit · →KO · ×type-mult · status` (immune ⇒ `safe`). This is the
  "which opp move threatens which of my mons, by how much" read. (A prior bug omitted the matrices-decode flags,
  so on a `--damage-matrices incoming` run the prober mis-read the absent lean top-K block — the garbage
  `acc-580` render — and never decoded the matrix; deleting the lean block made that class unrepresentable.)
