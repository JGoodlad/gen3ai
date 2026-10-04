# ARCHITECTURE.md — what is true NOW

**Scope: current state only.** History lives in [`CHANGELOG.md`](CHANGELOG.md); versioning
*mechanics* live in `src/agents/model/CLAUDE.md`. If a claim here disagrees with the code, the code
wins and this file is a bug — fix it in the same change.

**"Current" means the production configuration as HEAD resolves it.** Two objects, deliberately
distinct — a generation now turns over every ~2 days, so conflating them is how this file went
stale twice:

| | |
|---|---|
| Production run | **`ai_v12_02_winprob_critic`** (the WIN-PROB CRITIC era, 2026-09-06) — `config_version` **122** in the mirror this file is gated against (the `gen3_event_record_v2` signature-bump window, row 3: the mirror follows the code, so the obs-architecture batch's surface — 2761-dim obs, the reshaped event rows — and the shaped-reward deletion's field removal are what this file describes); the run's own `model_config.json` records 110, the frozen-phi bump it launched on, and signature `gen3_critic_route_wave_v1`, which HEAD no longer loads. `arch_signature` **`gen3_event_record_v2`** in the mirror. It is gen-17's architecture surface with the CRITIC swapped and nothing else: the substrate cells stay ON in the base (`pair_outcome_cell` / `pair_outcome_switch` / `switch_branch_cell` / `conditional_threat_cell`), all 17 edge families, the entity seats, the event window and the belief stack are unchanged. The 13 rows that moved are the critic family alone — see §3.4 and §6. Its predecessor `models/ai_v9_21_gen17_pfspoff_0820/` (gen-17, v97) is what every §4/§5 measurement below was taken on |
| Code on HEAD | `MODEL_CONFIG_VERSION` / `ARCH_SIGNATURE` — **read them from `agents/model/model_version/constants.py`**, never from prose (at this writing: 132 / `gen3_event_record_v2`) |
| `designs/production_config.json` | the live run's config **carried forward to HEAD's schema** — a verbatim mirror of the production run's `model_config.json`, refreshed with `python -m agents.model.delivery_graph --sync-config <run>/model_config.json`, never hand-edited, and carrying its provenance in the sibling [`production_config.README.md`](production_config.README.md) (JSON has no comment syntax, so the record cannot live in the file). The `gen3_event_record_v2` **signature-bump window is OPEN** (2026-09-26, the observation-architecture batch): the production run records `gen3_critic_route_wave_v1`, HEAD builds `gen3_event_record_v2`, so the mirror follows the CODE until the first run at the new signature exists — then it closes and the mirror tracks that run. (Inside such a window the two requirements pull in opposite directions — the compile gate needs the mirror to match live code, the drift gate needs it to mirror the newest run, and neither can be relaxed — so `arch_tables_test` DETECTS the window from the run's recorded signature and lets the mirror follow the code until a run at the new signature exists.) It exists so this file, the compile gate, the delivery graph and the viewer all derive from ONE real feature set |

Everything below describes what HEAD builds under `designs/production_config.json`. The
machine-derived tables are **generated** (`python -m agents.model.arch_tables`, pinned by
`arch_tables_test.py`) — regenerate them in the same commit as any architecture change; the prose
is hand-written and the generator never touches it.

**The companion artifact is [`architecture_graph.dot`](architecture_graph.dot)** — a *generated*
delivery digraph (seats and sinks as nodes; edges typed by what they physically carry: `bias` = a
softmax-normalised RATIO, `content` = an absolute as token content, `concat` = an absolute at the
head input, `cell` = an absolute per-action, `aux` = training-only and never in the forward). It is
produced by `src/agents/model/delivery_graph.py` and pinned by `delivery_graph_test.py` against a
committed JSON snapshot, so an architecture change that is not reflected in the graph fails a test
rather than quietly rotting. The snapshot compares the graph against *itself*, which cannot see a
module nobody drew — so a second gate enumerates the extractor's parametered top-level modules and
fails unless each one is reachable in the graph or allowlisted with a reason
(`delivery_graph.MODULE_GRAPH_TOKENS` / `NON_DELIVERY_MODULES`, checked by
`test_every_parametered_module_is_reachable_in_the_graph` and by `--check`). Regenerate both
artifacts in the same commit:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python -m agents.model.delivery_graph \
    --dot designs/architecture_graph.dot \
    --json src/agents/model/delivery_graph_snapshot.json
```

**Conventions used in this file**
- No version numbers in prose. Blocks and families are named **structurally** — write
  `outgoing_matrix (our moves × opp mons)`, never "the v34 matrix". Where a version tag is
  genuinely needed it goes in a trailing parenthetical, never in the noun phrase.
- Every measured number carries checkpoint / step / state-count / date inline.
- A claim that could not be verified against code or the run config is marked **UNVERIFIED**.

---

## 1. Observation

One flat `float32` vector of **2761** dims, plus an 11-dim `action_mask`, delivered as a Dict obs.
Every number below comes from `agents/observation/constants.py` and
`Gen3ObservationEncoder.get_layout()`. **Never hardcode an offset — read the layout.**

### 1.1 Top-level blocks

| Block | Start | End | Dims | Constant |
|---|---|---|---|---|
| Our team — 6 × per-mon slot | 0 | 732 | 732 | `OFFSET_OUR_TEAM`, `6 × POKEMON_FULL_DIM` |
| Opp team — 6 × per-mon slot | 732 | 1464 | 732 | `OFFSET_OPP_TEAM` |
| Active context ×2 (ours, theirs) | 1464 | 1584 | 120 | `OFFSET_CONTEXT`, `2 × ACTIVE_CONTEXT_DIM` (60) |
| Global env | 1584 | 1604 | 20 | `OFFSET_GLOBAL`, `GLOBAL_ENV_DIM` |
| Board (reactive) | 1604 | 1621 | 17 | `OFFSET_REACTIVE`, `REACTIVE_DIM` |
| Pair history — 6×6×5 h[i,j] | 1621 | 1801 | 180 | `OFFSET_PAIR_HISTORY`, `PAIR_HISTORY_DIM` (`gen3_pair_history_v1`) |
| Event window — 32 × 30 event records | 1801 | 2761 | 960 | `OFFSET_EVENT_WINDOW`, `EVENT_WINDOW_DIM` (`gen3_event_record_v2`) |
| **Total** *(= `base_dim`)* | | **2761** | | `Gen3ObservationEncoder.dimension` |

The event window is the LAST block: `total_dim == base_dim`, and the encoder's output IS the
observation. There is no appended tail — the row the Rust core builds IS `encode(...)`, unchanged.

**The event window** (Tier H-B, `gen3_event_window_v1`, reshaped by `gen3_event_record_v2`): the
last 32 decision-relevant EVENTS as typed 30-column records — type id · actor/target species +
side · move id · attributed `hp_delta` · outcome/crit/effectiveness · `we_first` · status id ·
log-saturated recency · forced-window phase tag · valid · cant-reason id · faint-cause id ·
item-transition id · REL species + side · entry reason · denial reason · caller move · boost stat ·
Spikes layers · Pursuit-on-switch — folded by `EventWindowTracker`
(`agents/training/event_window_tracker.py`; the Rust core's `trackers::history::EventWindow`,
slice T / O byte-equal) from PUBLIC protocol events (seq-idempotent), most-recent LAST with
zero-padding at the front. The per-row-type schema is §1.6. Ids are
embedding ids; **no Linear reads the block raw** — its only consumer is the opt-in
`history_events` event-seat encoder (§ flag table). The columns are documented at
`agents/observation/constants.py` (`EVENT_TOKEN_DIM`).

**What the columns say** (`gen3_event_window_semantics_fixes_v1`, the Rust core M3 loss
catalogue's findings W1–W5): a BOOST row's MAGNITUDE is the SIGNED stage change / 6 (a drop is
negative); a HAZARD row's MAGNITUDE is +1 when a side condition STARTS (a Spikes layer, a screen)
and −1 when one ENDS (Rapid Spin's clear, a screen expiring); a MOVE row stopped by the target's
Protect / Detect reads OUT_FAIL; a MOVE row's attributed `hp_delta` counts a bare `-damage` only
while that move's user is the side MOVING (the other side's own Substitute / Belly Drum cost is not
the hit); a FAINT that no damage line caused (Destiny Bond, Perish Song, Memento) reads `other`,
never `attack`; an item line `[from]` Trick / Thief / Covet reads SWAPPED on both mons; a MOVE
row's TARGET is the move's dex target class (`gen3_move_target_class_v1`) — the USER for a self /
side / field move (Protect, Recover, Refresh, Calm Mind, Rain Dance, a non-Ghost Curse), none for
`adjacentAlly`, else the other side's active — and the reading's event target follows the same class
when the protocol blanked the field (`[still]`). The same
rules hold in the Rust core's trackers (slice T, 0 divergences). Measured on the golden battle
set: 698 / 991 decisions' obs change, in those five fields and the clock scalar only
(`designs/research_state/measurements/training_input_gigo_fixes_2026-09-24/`).

`gen3_entity_rehome_v1` (Stage 3): the two 144-dim matchup matrices and 6 of the 11 reactive
scalars are **deleted** — the D/V edge families compute a strict superset of the matchup signal
GPU-side, `active_status` was byte-redundant with the per-mon condition one-hot, and
`forced_struggle` is derivable (all-zero `active_req_moves` legal bits / the action mask).
`protect_odds`, `trapped` and `maybe_trapped` moved **onto the per-mon slots** (the facts ride
the entities they describe).

> The `OFFSET_*` trailing comments inside `constants.py` used to carry stale evaluated numbers
> (642 / 1284 / …). They are deleted (2026-08-14) — only the expressions remain, with a comment
> forbidding evaluated values there; read `get_layout()` for live offsets.

### 1.2 Per-Pokémon slot — 122 dims (`POKEMON_FULL_DIM`)

`POKEMON_VECTOR_DIM` is 119; `state_encoder` appends the two OUR-side trapping bits and then the
active flag → 122. The active flag stays the **last** dim of the slot on purpose — the model's
`hp_and_active[:, :, -1]` convention is load-bearing (ObsUnpack / DamageOperator / entity seats).

| Field | Offset | Dims | Constant |
|---|---|---|---|
| species id + 6 base stats | 0 | 7 | `POKEMON_SPECIES_OFFSET` |
| item `[id, known, consumed]` | 7 | 3 | `POKEMON_ITEMS_OFFSET` |
| type ids ×2 | 10 | 2 | `POKEMON_TYPES_OFFSET` |
| ability `[id1, id2, dominance, known]` | 12 | 4 | `POKEMON_ABILITIES_OFFSET` |
| status one-hot | 16 | 7 | `POKEMON_CONDITION_OFFSET`, `CONDITION_DIM` |
| 4 × move slot (11 each) | 23 | 44 | `POKEMON_MOVES_OFFSET`, `MOVE_SLOT_DIM` |
| HP fraction | 67 | 1 | `POKEMON_HP_OFFSET` |
| species_known | 68 | 1 | `POKEMON_SPECIES_KNOWN_OFFSET` |
| sleep / toxic counters | 69 | 2 | `POKEMON_COUNTER_OFFSET`, `POKEMON_COUNTER_DIM` |
| spread (IV×6, EV×6, known, nature×5) | 71 | 18 | `POKEMON_SPREAD_OFFSET` |
| Hidden-Power block (`hp_revealed`, 16 type probs) | 89 | 17 | `POKEMON_HP_BLOCK_OFFSET` |
| sleep-wake belief | 106 | 3 | `POKEMON_SLEEP_BELIEF_OFFSET` |
| recency `[since_seen, since_acted, since_hit]` | 109 | 3 | `POKEMON_RECENCY_OFFSET` |
| protect-success odds | 112 | 1 | `POKEMON_PROTECT_OFFSET` |
| last action `[move_id, was_switch, hit, miss, fail, crit]` (active only) | 113 | 6 | `POKEMON_LAST_ACTION_OFFSET` (`gen3_pair_history_v1` — the id is embedding-routed, its raw column zeroed at the slice) |
| trapped (our active only) | 119 | 1 | `POKEMON_TRAPPED_OFFSET`, appended by `state_encoder` |
| maybe_trapped (our active only) | 120 | 1 | `POKEMON_MAYBE_TRAPPED_OFFSET`, appended |
| active flag | 121 | 1 | `POKEMON_ACTIVE_OFFSET`, appended (LAST — load-bearing) |

Move slot (11, `moves.py`): `[id, power/200, has_secondary, has_recoil, type_id, category, known,
current_pp, max_pp, accuracy, never_miss]`.

The two counters are read off the vendored poke-env's `status_counter`: sleep `min(n, 4)/4` (the
`|cant|`/`|move|` lines since the sleep began), toxic `min(stage, 8)/8` where `stage` is the SIM's
`tox` stage — the residual `[from] psn` chips since the switch-in, 0 on entry and while benched
(`gen3_pe_reading_fixes_v1`). A FAINTED active's `active_context` boosts are zero (the sim's faint
clears them), and the `flashfire` volatile slot stays set until its holder leaves the field.

An OPPONENT's Hidden-Power block is `HiddenPowerTracker`'s per-species vector (our own mons read
`hp_revealed` 1, probs all zero): all-zero until an HP hit with
an effectiveness EMISSION is observed (a neutral hit prints none and is never observed); then the
species' Smogon usage row (flat 1/16 without one), each type zeroed that could not produce an
observed bucket against the resolved target. A usage zero is not an impossibility
(`gen3_hp_prior_support_v1`): a species whose own observations refute its row — every type the row
gives mass eliminated while some type explains them all, e.g. an IV-less set's HP Dark — restarts
from the flat prior and replays its log; only a log NO type explains raises. `hp_revealed` is 1
once the species is narrowed or its four revealed moves hold no Hidden Power (probs then all zero).
Exposure (2026-09-25, `hp_belief` fix): 0 of the training pool's 1,912 HP users carry a type their
row gives 0, 289 of the ladder corpus's 52,007 (all HP Dark).

**The opponent block under `--oracle-reveal species` (a DIAGNOSTIC mode, NOT production; `off` is the production row).**
The Rust env core can write the opponent's TRUE species into this block from turn 1 (team-preview semantics; the X5 A/B's
oracle-species reference arm, `endstate/design_x5_belief_tokens.md` §7.6). The seen opponent mons keep their reveal-order
slots and bytes; after them come one slot per UNSEEN mon, in dex-num order, each the encoder's own row for a never-seen mon
(`species_known` 1, full HP, "never seen" recency, the dex row's stats / types and the ability prior; item, status, moves,
spread, Hidden-Power block, sleep belief, last action, trapped and active flag all 0). A revealed mon leaves the tail and
joins the seen prefix, matched by dex num. Nothing else in the row moves, and obs dims are unchanged. The model reads it
through the shared trunk like any state of the block (with every species stated, no slot is "believed"); the Python encoder
has no reveal. Mechanism, cells and gates: [`rust_sim/encoder.md`](rust_sim/encoder.md) §11.

### 1.3 Board (reactive) block — 17 dims

**5 raw board scalars, then the request-ordered active moves.** Everything derived is gone
(`gen3_entity_rehome_v1`): the matchup matrices live GPU-side as D/V edges, and the per-entity
scalars moved to the mon slots (§1.2).

| Field | Offset in reactive | Dims |
|---|---|---|
| `fainted` (ours, theirs) | 0 | 2 |
| `turns_since_progress` | 2 | 1 |
| `wish_floating_our` | 3 | 1 |
| `wish_floating_opp` | 4 | 1 |
| `active_req_moves` — `[move_num ×4, type_id ×4, legal_now ×4]` | 5 | 12 |

`active_req_moves` is in **request-slot order** (slot *k* ↔ action logit 6+*k*) and is sliced
straight into `ExtractorContext`; it never enters the raw-scalar projection path. The per-mon move
block (§1.2) stays **sorted by id** because the role token concatenates the 4 encodings and is
therefore order-sensitive. Both orders are live simultaneously — that is the reason the pointer
head permutes by move-num identity (§3).

`non_matchup_rest` — the raw-scalar tail the global token and both projection heads read — is
`GLOBAL_ENV_DIM (20) + the 5 board scalars = 25` dims. It stops at the `active_req_moves` offset,
so the embedding-ID block is excluded from it by construction.

### 1.4 Global env — 20 dims

`weather` 7 · `hazards` (spikes ×2) 2 · `clock` **3** · `screens` 8. (`global_env.py`)

**The clock group is 3 scalars** (`CLOCK_DIM`), all sharing the `log(1 + MAX_TURNS)` denominator:

| idx | scalar | formula | where its resolution sits |
|---|---|---|---|
| 0 | log-ELAPSED | `log(1+turn) / log(1+MAX_TURNS)` | the OPENING — turns 1–50 are 58.6% of its range |
| 1 | remaining LINEAR | `(MAX_TURNS − turn) / MAX_TURNS` | uniform — the proportional budget left |
| 2 | log-REMAINING | `log(1 + MAX_TURNS − turn) / log(1+MAX_TURNS)` | the DEADLINE — the last 20 turns are 55.1% of its range |

`MAX_TURNS` (250) is **also the forfeit deadline**: `StallConfig.threshold` imports it, so the
turn the trainee actually loses on and the clock's normaliser cannot drift apart
(`global_env_test.py::test_max_turns_is_the_forfeit_deadline`). Remaining is clamped at 0, so an
over-cap turn saturates rather than going negative or NaN.

Why three and not one: value near a forfeit cap is a function of turns REMAINING, and the
log-elapsed scalar alone gave the last 20 turns **1.5%** of its range (per-turn sensitivity at
turn 249 is 125× lower than at turn 1). A critic cannot price a cliff it has no resolution on,
and that cliff is the link TD must fit FIRST before it can bootstrap value back down a 200-turn
episode. Both remaining forms are provided as raw facts; which one matters is the model's to
learn.

### 1.5 Active context — 60 dims per side

`BOOSTS_DIM` 14 + `VOLATILES_DIM` 46 (`gen3_effects.VOLATILE_DIM`, source-derived).

The LAST two volatile columns are the gen-3 **field sports** (`gen3_field_sport_slots_v1`):
`mudsport`, `watersport`. Verified on the vendored sim (the gen4 mod the gen3 dex inherits): each
is a `volatileStatus` on its USER, with no duration, announced by `-start|<mon>|Mud Sport` /
`-start|<mon>|move: Water Sport`; it ends silently when the holder leaves the field (switch or
faint); `noCopy: false`, so **Baton Pass carries it** (poke-env's `BATON_PASS_COPIED_EFFECTS` and
the core's reading rule V4 both reconstruct the silent copy); while the holder is active,
`onAnyBasePower` halves the base power of EVERY Electric (Mud) / Fire (Water) move, the holder's
own included. The `DamageOperator` honours both (§4). The Rust SIM does not implement either move
(its census lists them fail-loud), so they reach the core only through a parsed stream (the
ladder).

### 1.6 What happened last turn

Carried by the **event window** (§1.1), not by lag frames. The 7 × 159 TurnDelta frames and the
11-dim prev-turn action mask are DELETED; `TurnDelta` itself survives as the reward manager's
per-decision input and as the α/β intent label source, but it no longer has an obs encoding.

Every fact the frames delivered has an event-window column — move id, outcome, crit,
effectiveness, status applied/cured, boosts, switch-ins, forced-switch phase, move order,
and the attempted switch target:

- **`cant_id` (column 19) was ADDED for the deletion.** "This mon could not move, and why" (full
  paralysis / sleep / flinch / recharge) had NO event-window column: `EventKind.CANT` was in the
  battle event log with its reason and the TurnDelta fold read it, but the window emitted no row.
  It now emits `EVENT_T_CANT` with the reason as a 1-based id into `gen3_effects.CANT_REASONS`
  (0 = not a cant row). It has its own column rather than riding `status_id` — the two are
  mutually exclusive by `type_id`, so overloading would encode compactly and read wrongly.
- **`faint_cause_id` (column 20) and `item_transition` (column 21) were ADDED**
  (`gen3_event_semantics_v1`), closing the other two gaps. The FAINT row now carries WHY a mon
  died — 1..8 into `turn_view.FAINT_CAUSE_VOCAB`, via the same `_classify_faint_cause` the
  TurnDelta fold uses, so the two cannot drift. The sequence-makes-it-inferable argument only
  ever covered {attack, recoil, selfko}: weather, status, hazard and Leech Seed deaths emit NO
  preceding event, because residual damage is not an event. `item_transition` is an ENUM, not a
  consumed flag — gen3 has three item-GONE routes (consumed berries/herbs · REMOVED by Knock
  Off, permanent in ADV · SWAPPED by Trick/Thief/Covet) and one flag would leave the conflation
  half-alive.
- **The refused-switch target is carried (E4, `gen3_event_record_v2`).** When a switch is refused
  while trapped, the `SWITCH_REJECTED` row's TARGET is the bench mon the switch aimed at. The
  server's `|error|[Unavailable choice]` does not name it, so each path supplies it from what it
  SENT: the Python fold from the previous decision's action index (`EpisodeTracker.
  _attempted_switch`, TurnDelta's decode), the core from the noted choice token
  (`trackers::attempted_switch_species`: `switch N` against the request's `side.pokemon`, or
  `switch <name>`). A search successor notes no choice, so its leaf row leaves the target 0.

**The event record (`gen3_event_record_v2`, E12 — the event-block RESHAPE).** The Rust core's native
per-action record (`record::Window`: ordered actions with attributed effects, denials, the
information boundary as a type) is carried into the observation instead of flattened into the old
22-column window. The shape stays ENTITY-aligned: a row names up to three mons (ACTOR on its own
side, TARGET on the other, and a REL mon on the side its `REL_SIDE` states), each linked to its mon
token by the `r` edge family's three reference cells (§5); a fact that is a property of a mon
(the boosts and volatiles a Baton Pass carried, the item a mon now holds) stays on that mon's
entity block rather than being copied into a row. Columns 22–29 (`constants.EventCol`):

| column | meaning, per row type |
|---|---|
| `REL_SPECIES` / `REL_SIDE` (22, 23) | SWITCH_IN: the mon it replaced (the fainted mon for a replacement, the PASSER for a Baton Pass, the dragged-out mon for a drag). DENIED: the mon that denied it (fainted first: the KOer; turn cut: the turn's first faint). ITEM: the other party of a transfer / removal. FAINT with cause `attack`: the KOer |
| `ENTRY` (24) | SWITCH_IN: 1 chosen · 2 REPLACEMENT (the free switch after a faint) · 3 DRAG (Roar / Whirlwind; MOVE = the phazing move, TARGET = the phazer) · 4 BATON PASS (MOVE = `batonpass`) |
| `DENIAL` (25) | DENIED (row type 11): 1 fainted first · 2 TURN CUT |
| `CALLER` (26) | MOVE: the calling move of a called move (Sleep Talk, Metronome, Mirror Move, Assist, Nature Power, Magic Coat, Snatch) |
| `STAT` (27) | BOOST: atk def spa spd spe accuracy evasion = 1..7 |
| `LAYERS` (28) | Spikes layers / 3 — SWITCH_IN: met by the entrant; HAZARD: on that side after a Spikes change; FAINT (hazard): at the KO |
| `PURSUIT_SWITCH` (29) | MOVE: Pursuit struck a switching target (the line's `[from] Pursuit`, the MOVE event's `pursuit_switch`); the row precedes the switch |

Changed meanings of existing columns: a SWITCH_IN row's MAGNITUDE is the Spikes chip on entry
(signed HP fraction); MOVE on a SWITCH_IN / DENIED row is the move that caused the entry / the
denying action's move; `FAINT_CAUSE` reads the LIVE vocabulary
(`turn_view.FAINT_CAUSE_VOCAB_LIVE` — the archive's eight plus `destinybond` / `perishsong`); an
item line `[from]` Trick / Thief / Covet is `ITEM_TR_RECEIVED` (5) on the `|-item|` side and
`ITEM_TR_SWAPPED` (4, taken from this mon) on the `|-enditem|` side.

**ACTION DENIAL** (`EVENT_T_DENIED`): every turn ACTOR (a side's active at the turn's first event)
that was chosen for the turn and never acted gets one row. In gen-3 singles ANY faint in the action
phase cancels every remaining queued action (`sim/battle.ts` `faintMessages`: "in gen 3, fainting
skips all moves and switches", verified at the vendored source), so: an actor that FAINTS before
acting is `fainted first` (REL/MOVE = the action that KOed it); every other actor still waiting is
`turn cut` (REL = the first faint, `FAINT_CAUSE` = its cause, MOVE = its action's move) — a faster
Explosion or a recoil self-KO denies the survivor. A mon acted if it moved, chose a switch (or
passed), was refused (`CANT`), or hit itself in confusion; a faint in the residual block (a `[from]`
psn / tox / brn / sandstorm / hail / Leech Seed / Nightmare / Curse / Leftovers / Wish / Ingrain /
Future Sight / Doom Desire line opens it) cuts nothing. 🚨 **The information boundary holds by
construction:** the fold receives no choices, so a DENIED row cannot carry what the denied side
chose (the opponent's intent label is masked on these windows, `gen3_intent_label_semantics_fixes_v1`).
REFUSED actions keep their `CANT` row. A Roar that drags out a mon that had not moved yet is NOT a
denial row (the native record does not model it either).

Constructed-battle fixtures, one per mechanic (denial fainted-first / turn cut by self-KO / by
recoil, trade KOs by Destiny Bond and Perish Song, Roar into Spikes, Baton Pass into Spikes,
Thief / Trick / Knock Off, a Spikes sack and its free replacement, Pursuit on a switch, called
moves, the E4 trap), each run through the core and the Python path with slices T and O on:
`src/agents/battle/event_record_v2_fixture_test.py`.

**The α/β intent label and the progress clock read `TurnDelta`, and both mask or discount what
was not a choice or not our doing** (`gen3_intent_label_semantics_fixes_v1`,
`gen3_progress_clock_attribution_fix_v1`): the label is MASKED when the opponent's switch-in was
DRAGGED by our Roar / Whirlwind, was the free replacement for a faint in the previous window, or
when our Encore overrode its move before it acted, and a CALLED move (Sleep Talk → Rest, Mirror
Move, Magic Coat, …) is labelled as its CALLER. The clock's clause (i) ("our move dealt damage")
requires our move's OWN hit to clear the 3 % floor as well as the target's net fall, and a
blocked attack (outcome `fail` against Protect / Detect) is the exogenous FREEZE, not a charged
no-op.

Per-slot layout of the event record, and the embedded-ID manifest that routes raw ids to
embedding tables, live in `src/agents/observation/CLAUDE.md`.

---

## 2. Feature extractor — the production chain

`Gen3FeaturesExtractor` (`src/agents/model/features_extractor.py`), paired **mandatorily** with
`Gen3DualHeadMaskablePolicy` (`policy.py`) — the extractor returns a `(pi_features, vf_features)`
tuple, which stock SB3 policies cannot consume.

Modules actually built under the production config (`named_children()`) — GENERATED:

<!-- BEGIN GENERATED: modules -->
```
embeddings · unpack · pokemon_encoder · entity_seats · edge_bias · team_transformer · cls_pool ·
hidden_opp_belief · intent_move_cell · intent_threshold_move · intent_conditional ·
pair_outcome_move · pair_outcome_switch · switch_branch · conditional_threat · t0_species_prior ·
belief_slots · belief_head · move_belief · spread_belief · hp_type_belief_head ·
item_belief_head · damage_op · prefuse_proj · assembler · win_head · value_entity_pool ·
history_events · pre_proj_norm · projection · value_pre_norm · value_projection · activation ·
alpha_head · beta_head
```
<!-- END GENERATED: modules -->

### 2.1 Order of operations — the TIER ORDER, and the only order

The belief + physics stack runs **once, before attention**, on every config. There is no flag: the
forward resolves the game in the order the game resolves in, and the four tiers are an **asserted
invariant** (`tier_contract.py`, `tier_contract_test.py`) rather than a property of how the code
happens to be written.

| tier | question | modules |
|---|---|---|
| **T0 RESOLVE** | what is on the board? | `pokemon_encoder`, `belief_slots`, `move_belief`, `hp_type_belief_head`, `spread_belief`, `item_belief_head` |
| **T1 REASON** | what follows from it? | `damage_op`, `entity_seats`, `edge_bias`, `team_transformer` |
| **T2 DECIDE** | what will they do, what are my moves worth? | `belief_head`, `cls_pool`, `alpha_head`, `beta_head`, `intent_threshold_move` / `intent_conditional` / `pair_outcome_move` / `pair_outcome_switch` / `switch_branch` / `conditional_threat`; `cls_pool` additionally owns the token-content critic injection (`value_threat_proj`) |
| **T3 DELIVER** | one contract, two pools | `hidden_opp_belief`, `assembler`, `win_head` |

The contract asserts two things per forward: tier-declared entry points are entered in
**non-decreasing** tier order, and no entry point receives a tensor whose storage was produced by a
strictly **later** tier (checked across two forwards, so a stale stash counts). It is a check on
data flow, not on meaning — it cannot see a T0 leg *recomputing* something intent-like from raw
tokens. Every `nn.Module` child of the extractor must declare a tier or be listed as untiered, so a
new phase cannot escape it.

`BeliefSlots` (T0, injects unknown-mon tokens pre-trunk) and `BeliefHead` (T2, reads refined tokens)
are the one deliberate split: `BeliefHead` is a **training-only side readout** whose output is
stashed for the aux loss and never fed forward, which is exactly what its T2 declaration records.

**`--belief-tokens` (X5) is `blob` in production**: the representation described here, and nothing
of X5 is built. The `fixed_mass` arm ([`endstate/design_x5_belief_tokens.md`](endstate/design_x5_belief_tokens.md))
adds `HypothesisBuilder` (`agents/model/hypothesis_set.py`, T0) — concrete species hypotheses with
logistic fixed-size presence (Σπ = 6 − revealed), a learned delta on the T0 prior, OTHER and the
opponent active's move group — and READS it (`gen3_x5_belief_tokens_v1`,
`agents/model/hypothesis_tokens.py`): each hidden opponent slot holds its hypothesis's dex row encoded
by THE `PokemonEncoder` + a learned `hypothesis_marker` (replacing `BeliefSlots`' constant token, which
that arm does not call); the T0 belief heads read those seats with their species; OTHER_species joins
the trunk as one extra seat after the entity seats; and every expectation-type reduction over opponent
tokens — the trunk's attention, `their_cls`, `value_cls`, `HiddenOppBeliefPool`, `value_entity_pool` —
adds log π of each opponent key (detached; FLOAT key masks in the three `nn` pools). The opponent
active's MOVE axis reads the move group (`FixedMassMoves`): ONE order (revealed first, then the top
unrevealed by π_m) gives the E4 seats, the op's top-K seat axis (pair cells, α's seats), the D3 / S3
cells and the intent operands (no `torch.topk`); the op's incoming maxes weight each candidate by its
DETACHED fixed-mass presence (the presence-scaled max); a revealed Hidden Power's seat is priced as its
typed mixture; the active's E5 seat is OTHER_move — and the active's move REINJECTION soft-embeds its
row by that detached presence (the other slots keep sigmoid weights). The op's opponent-MON axis reads an
`OpRoster` (`hypothesis_tokens.py`): the op runs on the HYPOTHESIS context, so a hidden slot is priced as
its concrete species at first appearance (full HP, no status); "alive" is `opp_addressable`, never an HP
cell; a hypothesis DEFENDER is the expected-latent read on a per-slot one-hot (its own species, P(KO)
defined — not nulled); every live opponent mon is an ATTACKER (C1b / C2 / C3 / D4) on its own fixed-mass
move presence (k = 4 − revealed, revealed moves pinned at 1), selected by one stable per-mon order;
`p_pur_vs_us` is a presence-scaled max over the mons; Beat Up's opponent party sum reads π / k over every
candidate; the bench E5 seats are presence-aware (tail beyond rank K, summed presence, presence-scaled
worst case). OTHER_species gets the blob's AVERAGED construction on the renormalised tail
(`other_roster`): defender `P_tail @ tables` with P(KO) nulled, attacker E_tail[base stats], E_tail[STAB],
E_tail[speed], moves = the parameter-free E10 mixture over the tail through the k = 4 construction; its
edge column is written for D1, C1, C3, D4 and V (`EdgeBias.OTHER_FAMILIES`; C2 / S1 / T / X / G carry
no OTHER edge), and at the Pursuit max it enters with presence 1 − Π_tail(1 − π). `BeliefSlots` is not
built in that arm. The opponent pointer there is ONE flat list (`gen3_x5_flat_pointer_v1`,
`agents/model/flat_intent.py`, T2): the opponent active's K move seats, OTHER_move (token: the active's E5
seat), a switch to each of their six slots (a revealed mon, or the hypothesis a hidden slot holds) and
OTHER_species (its refined trunk token), scored by one shared scorer plus each candidate's DETACHED log π,
one softmax; masks are structural. It REPLACES α / β in that arm (they are constructed and see SB3's
orthogonal re-init, then the policy's `_build` retires them before the optimizer is made, so no non-X5
initial byte moves). Its labels come from the existing intent label: a move beyond the seats is an
OTHER_move label, a hidden switch-in not among the hypotheses an OTHER_species label (a belief miss is
supervised, never masked); a typed Hidden Power label names a revealed HP's seat. The seven α consumers
read its re-expression: α over the K seats + OTHER_move (a PRICED (K+1)-th seat — every seat-axis operand
gets OTHER_move's column, the op's per-candidate cells on the full move axis contracted with the
renormalised tail weights `FixedMassMoves.other_u`) + the total switch mass, β over the six slots +
OTHER_species (`out_cells` gets OTHER's column from the OTHER-mode D1 pass, `opp_p_ghost` the tail's
P(Ghost)); `threshold_probs`, `IntentMoveCell` and `IntentConditionalMoveCell` now apply the seats'
meaningful-K gate as `pair_alpha` does. The ride-along B head (when on) is re-based onto the flat list.
Config v136, no `ARCH_SIGNATURE` change while both arms build.

The concrete steps:

1. **`ObsUnpack`** — slices the 2761-dim vector into `ExtractorContext` (~30 named tensors:
   per-mon blocks, categorical ids, active-slot indices, fainted key-masks,
   `our_active_req_move_{ids,type_ids,legal}`).
2. **`PokemonEncoder`** — per-move network (`MOVE_NET_HIDDEN` `[96,32]`, with the `MoveLatentEncoder`
   latent concatenated in) → within-mon move self-attention → role encoder
   (`ROLE_ENCODER_HIDDEN` `[256,128]`) → **12 × 128 role tokens**. The role input carries the
   E2 active-context injection: each side's 60-dim boosts+volatiles block scattered onto its
   ACTIVE mon's row, bench rows zero (the §6-audited entity home; the global-token/projection
   routes remain — additive delivery, pinned by `e2_ctx_injection_test.py`). Stashes
   `last_move_tokens` `[B,12,4,32]` (sorted-by-id) for the seats and the pointer head.
3. **`MoveBelief`** (T0, `move_belief_mode` = `"both"` — every opp slot, revealed and hidden)
   — reads the opp **role** tokens, predicts each opp slot's moveset,
   fuses the Smogon log-odds prior, pins revealed moves, and reinjects the soft-embedded moveset
   into the opp role tokens. Stash: `last_move_belief_logits` `[B,6,400]`.
   The prior buffer `[n_species, n_moves]` is **learnset-gated unconditionally**: a move the species
   cannot learn is `logit(1e-6)` (impossible), a legal move keeps its **true** Smogon usage (no rarity
   cap — a surprise tech is never zeroed), a legal move absent from the usage data gets the
   `move_candidate_floor` base, and a row about which nothing is known (national-dex num 0 — the
   unknown-species sentinel an unrevealed slot carries — or a dex gap) is the **flat floor**, never
   "no moves". Non-persistent, recomputed from `data/` at build.
   **The Smogon usage is `P(m in set | s)` = the chaos `Moves[m]` over the species' RATING-WEIGHTED
   set total W** (`gen3_smogon_prior_denominator_v1`: W = `Σ Abilities`, Smogon's own `p.raw.weight`;
   per species `Σ_m P + empty-slot mass = 4` exactly — median 4.000, min 3.79). The facade THROWS at
   load on a table that breaks that identity. Every run before 2026-10-04 trained on a DEFLATED prior
   (over the unweighted `Raw count`: a species-dependent ×0.10–0.94, Skarmory Spikes 0.547 vs 0.997,
   sums 0.41–3.75; F-X5-41); a pre-fix checkpoint reads the corrected prior at HEAD.
   **The Smogon SPECIES-USAGE marginal is each species' share of the same RATING-WEIGHTED W**
   (`gen3_smogon_species_usage_weighted_v1`: `gen3_data.priors.species_usage()` → `build_species_usage_prior`,
   read by the op's `SPECIES_USAGE_PRIOR`, the `T0SpeciesPrior` / `BeliefHead` log-marginal and the
   co-occurrence lift's independence baseline, whose numerator — the chaos `Teammates` conditional — is
   weighted too). The facade THROWS on a marginal that is not W (checked against `Σ Abilities` and
   `Σ Moves / 4`). Every run before 2026-10-04 trained on the UNWEIGHTED `Raw count` share (F-X5-47: new /
   old share ×0.19–×1.75 over the 216 covered species, ×0.92–×1.19 over the top 25, Tyranitar 0.078 → 0.086,
   Shuckle ×0.19); a pre-fix checkpoint reads the corrected marginal at HEAD.
   **A HIDDEN slot does not read that flat row** (E10, `gen3_hidden_slot_move_mixture_v1`): its
   prior is the parameter-free Smogon mixture `P(m | hidden) = Σ_s P_T0(s | revealed) · P(m | s)` —
   the `T0SpeciesPrior` team-composition posterior (step 0 of this chain, Species Clause applied)
   times the same per-species prior as probabilities, one `[B,S] @ [S,M]` matmul, clamped to
   `[HIDDEN_SLOT_MIX_EPS, 1 − ε]` before the logit. It has NO learned parameters (it cannot
   memorise the pool), so the hidden slot's posterior now moves with the revealed team; the head's
   learned delta on the (state-independent) unknown-slot token still adds on top. The flat row
   remains only when `t0_species_prior` is off. Before this, the hidden-slot posterior was a
   state-independent constant (max deviation 0.0 over 57k decisions, belief-calibration read
   2026-09-24). Pinned by `hidden_slot_move_mixture_test.py`.
4. **`HPTypeBelief.compose_typed_hp`** (`hp_belief_mode` = `"composed"`) — inside the same step:
   rewrites the posterior so Hidden
   Power exists only at the 16 typed move-nums **355–370** (each `logit(presence · P(type))`) and
   the bare typeless 237 is driven to a finite `-30`. `Σ_t P(HP_t) == presence`, and presence is
   reveal-pinned, so a seen Hidden Power can never be believed away. Every downstream consumer
   (op, edges, seats, BCE, prober) reads this one typed posterior.
   *(Also live: **`ItemBelief`** — `gen3_item_belief_v1`, `--item-belief` — resolves
   each opp slot's hidden item as a posterior over item nums, Smogon usage prior ⊕ zero-init trunk
   delta; cold start == prior. Published leak-mode-aware (`last_item_logits`); the op's p_cb
   unrevealed branch consumes P(Choice Band) from the publication instead of the static
   `SPECIES_CB_PRIOR` scalar — the revealed 0/1 exactness gate is unchanged. Supervised as the
   BeliefBank's seventh row.)*
5. **`DamageOperator`** — the full 660-dim block (§4), computed on the pre-attention tokens.
6. **`prefuse_proj`** — the op's per-our-mon incoming rows `[B,6,12]` projected to `d_model` and
   **added** to our 6 role tokens. Zero-init ⇒ exactly 0 at init.
7. **`EntityMoveSeats`** — builds 16 extra seats (§2.3).
8. **Edge cells** — 15 per-pair physics tensors computed here, pre-transformer (§5).
9. **`TeamTransformer`** — 36 tokens, 2 `BiasedEncoderLayer`s, `d_model` 128, 4 heads, FFN 256,
   post-LN. One `[B,4,36,36]` float bias carries both the key-padding addend (`-1e9`) and every
   edge family; it is built once and shared by both layers.
10. **`CLSPool`** — three learned queries: `our_cls` over our 6 refined tokens, `their_cls` over
    theirs, `value_cls` over **all 12**. Also extracts `our_active_refined`.
11. **`ProjectionAssembler`** → `pre_proj_norm`/`projection`/ReLU (policy) and
    `value_pre_norm`/`value_projection`/ReLU (value), both emitting `PROJECTION_DIM` = 512.

### 2.2 Dims that flow between phases

| Constant | Value | File |
|---|---|---|
| `ROLE_TOKEN_SIZE` = `D_MODEL` | 128 | `arch_constants.py` |
| `PROJECTION_DIM` | 512 | " |
| `MOVE_NET_HIDDEN` | `[96, 32]` | " |
| `MOVE_LATENT_DIM` / `MOVE_LATENT_HIDDEN` | 32 / 64 | " |
| `ROLE_ENCODER_HIDDEN` | `[256, 128]` | " |
| `ACTIVE_CTX_HIDDEN` | `[64, 32]` | " |
| `POINTER_HIDDEN` | 64 | " |
| `TRANSFORMER_N_LAYERS` / `N_HEADS` / `FFN_DIM` | 2 / 4 / 256 | " |
| `NET_ARCH` (SB3 mlp_extractor) | `[512, 512]` | " |

Embedding tables (`Embeddings`, registered exactly once, passed as a forward argument):
species 400×32, move 400×16, item 600×16, ability 100×16, type 20×16.

### 2.3 The 29-token sequence

| Seats | Index range | Token type | Content |
|---|---|---|---|
| our mons | 0–5 | `TOKEN_TYPE_OUR_TEAM` | role token (+ `prefuse_proj` incoming residual) |
| opp mons | 6–11 | `TOKEN_TYPE_THEIR_TEAM` | role token (+ move-belief reinjection) |
| global | 12 | `TOKEN_TYPE_GLOBAL` | `[our_ctx, opp_ctx, non_matchup_rest]` → `global_proj` |
| **E3** our active's moves | 13–16 | `TOKEN_TYPE_OUR_MOVE` | move token in **request order**, `move_seat_proj` 32→128 |
| **E4** opp threat moves | 17–22 | `TOKEN_TYPE_THEIR_THREAT` | `threat_seat_proj([latent(32), w, acc, is_phys])`, K = `entity_topk_seats` = 6 |
| **E5** tail threats | 23–28 | `TOKEN_TYPE_THEIR_THREAT` + `tail_marker` | per-opp-mon beyond-top-K residual `tail_proj([p_tail, worst_phys, worst_spec, revealed])` |

There are **no `TOKEN_TYPE_HISTORY` seats in the base sequence** — the seven of them went with the
lag frames (`gen3_frame_deletion_v1`), which is what took the sequence from 36 tokens to 29 and
shifted every extra seat down by seven. `TOKEN_TYPE_HISTORY` itself survives in the token-type
table and is what an opt-in event seat takes.

`entity_seats.n_seats` = 16 (4 + 6 + 6). Base seat count = `2·TEAM_SIZE + 1` = 13 (the
`N_HISTORY_TURNS` history seats went with the lag frames), so **every extra seat index is
`13 + offset`** — that is what makes the base slices position-stable.
E5 deliberately reuses `TOKEN_TYPE_THEIR_THREAT` rather than adding a 7th token-type row (growing
the table changes every model's state_dict).

---

## 3. Heads — exactly what each consumes

This section is the canonical answer to "what does head X read". Widths verified by forward pass,
2026-08-08.

### 3.1 / 3.2 The head inputs — GENERATED

**The op head-concat is DEAD (`gen3_no_concat_v1`, v61)** — the flat block enters neither head;
the op reaches the policy via the pointer cells (lossless per-action), the prefuse token
injection, and the edge cells. **The active-ctx concat is DEAD too (`gen3_ctx_dedup_v1`, v76)** —
the per-side encoded ctx pair was duplicated delivery (the E2 injection carries each side's full
raw ctx block on its active token; the global token is a second route). `non_matchup_rest` stays:
its only token route is the global token, which no pool reads directly.

The exact concat composition and widths of both projections, under the production config on
HEAD, are generated below — never hand-edit inside the markers.

<!-- BEGIN GENERATED: head-inputs -->
**`pi_projection` — `Linear(1177, 512)`** (LayerNorm → Linear → ReLU). Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `our_team_pooled` | 128 | `CLSPool.our_cls` over our 6 refined tokens |
| `their_team_pooled` | 128 | `CLSPool.their_cls` over their 6 |
| `our_active_refined` | 128 | our active slot's refined token |
| `non_matchup_rest` | 25 | global env + board scalars (`_non_matchup_rest_dim`) |
| `hidden_opp_belief` | 768 | `HiddenOppBeliefPool` — k=6 × `D_MODEL` (POLICY only — the vf half read dV 0.0000 and was deleted) |
| **total** | **1177** | == `projection.in_features`, asserted at generation |

**`vf_projection` — `Linear(128, 512)`** (LayerNorm → Linear → ReLU). Input concat, in order:

| Part | Dims | Source |
|---|---|---|
| `value_pooled` | 128 | `CLSPool.value_cls` over **all 12** team tokens |
| **total** | **128** | == `value_projection.in_features`, asserted at generation |
<!-- END GENERATED: head-inputs -->

**Every value route INJECTS into `value_pooled`** (v89 `gen3_value_pooled_routes_v1`): the
routes below add a zero-init `D_MODEL` contribution to the tensor the win-prob critic actually
reads (and `vf_parts[0]`, so the scalar critic sees the same wiring). The old post-assembler
vf-concat delivery was structurally bypassed by the (since deleted) distributional-critic route —
gen-12 proof: `value_entity_pool.out_proj` and the then-live α-reduce projection bit-exact ZERO
after 25M steps, while `value_threat_proj` (the one `value_pooled` route) trained to 0.117. The
gradient-connectivity guard (`value_route_gradient_test.py`) backprops the critic through every
registered route each suite run.

**The seam has ONE member**, `value_entity_pool` (production, below); the PRIVILEGED true-team route that was its second is deleted, so every route on the seam reads the shared observation and none adds information. The seam takes no `obs` argument.

**Four of the seam's five original routes were deleted** (v96 `gen3_critic_route_wave_v1`), on
measured dependence against a 0.39 dV bar: `intent_value_reduce` 0.3176 and `value_clock` 0.2169
(both re-audited at 2× sample first), `intent_threshold`'s p_KO vf half 0.155/0.136, and
`value_intent` 0.156. `value_intent`'s **re-entry condition survives its deletion**: any future
α/β-to-critic proposal passes the C4-style offline gate FIRST (ledger C6 — the delivery line is
EXHAUSTED). The seam is kept generic because its value is covering the NEXT route on the day it is
written — which is the point of keeping it.

**ON in production: `value_entity_pool`** (v80, `UnifiedValueReadout` — Stage-3 T3-DELIVER of
`design_unified_belief.md` §3). ONE attention pool over the critic's entity-row set (the 12
post-transformer team tokens + the op's 6 per-our-mon incoming rows, each projected to
`UVR_DIM`=64 with a per-source type embedding, `UVR_K`=4 queries, explicit NaN-safe softmax)
adds its zero-init `D_MODEL` output into `value_pooled` — the policy is untouched at
any weight. It was the designed SUCCESSOR contract of the bolt-on vf routes, it ran ALONGSIDE
them for two generations so the `critic_route_audit` could price them against each other on one
trained run, and **it WON**: gen-14 read it at dV **5.490 = 97% of `all_off` 5.635**, against
`threat` 1.0686 and every other route below 0.32. The succession is complete — the seed readout,
the `nmr` vf concat and the hidden-opp vf half are all deleted at v96, and this pool plus
`--value-threat-inject` are what the critic reads. **`value_entity_pool_full`** (v82, **ON**)
completes the row set — +the refined GLOBAL token and +the hidden-opp belief queries — which is
where the content of the two deleted concats now lives (its own flag/shape: a v80-shape pool
keeps loading under `full=False`).

**ON in production: `intent_threshold`** (v84, `gen3_intent_threshold_v1` — the §3.0
threshold operator of `design_conditional_execution.md`). `threshold_probs` contracts the op's
per-candidate pair cells with the published α into `p_KO` / `p_sub_broken` / `p_fp_broken`, and
ONE zero-init projection delivers five mechanic channels (Focus Punch / Substitute / Endure /
Destiny Bond / Endeavor) plus `p_KO` as per-slot context through the pointer MOVE cell
(+`INTENT_THRESH_MOVE_DIM`). ⚠️ **v84 also built a SECOND consumer — the `[p_KO, …]` vf route,
the ledger-H1 payoff — and v96 DELETED it** (dV 0.155 on gen-13, 0.136 on gen-14, against a 0.39
bar). What H1 asked for stands; this delivery of it did not, and any successor owes the C4-style
offline gate first. The POLICY half is a per-action pointer channel measured in KL/flips rather
than |dV|, was never part of that verdict, and is pinned by `intent_threshold_test.py`.
**`intent_conditional`** (v85) is its sibling: the Counter/Mirror-Coat category sums,
flinch's `(1−α_SWITCH)` term, Explosion's execute/into-switch facts and Pursuit's ×2 never-miss
doubling trigger (the port-verified departing-target rule — no β), one more zero-init block on
the move cell (+`INTENT_COND_MOVE_DIM`). Enabling either is a gen-13+ decision gated on
gen-12's `intent_move_cell` audit (the G3 verdict); the pre-build G2 usage baseline is
`measurements/gen12_mechanic_usage_baseline.json` (Endure 0.0% / Sub 0.9% / Counter 5.6%).

**LIVE (ON in the gen-17 base): `pair_outcome_cell`** (v93, `gen3_pair_outcome_v1` — component 1 + 3 of
`design_opponent_intent.md`, §2.1/§9a of `design_pair_reduction.md`). It closes a **currency**
failure, not a reduction failure: incoming status reaches the policy only through the `s3` edge
family, i.e. as a softmax-normalised RATIO, so "35% of my HP" and "80% chance of burn" never meet
in one vector and no reducer can trade them. When on, the op builds ONE **`pair_in[their believed
seat k, our mon j, :]`** of width `_PAIR_OUTCOME_RAW` = **14** — the six existing damage channels
(`[low, high, crit, ko_ramp, acc, is_phys]`) concatenated with eight new ones:

| # | coordinate | source |
|---|---|---|
| 6-11 | `p_par p_brn p_frz p_slp p_psn p_tox` | `_incoming_status_lands` (the per-pivot immunity physics, unchanged) SPLIT by the seat's status IDENTITY — `MOVE_STATUS_IDENT` for a dedicated status move (read from the raw `status_inflicted`, so **tox and psn stay apart** where `MOVE_STATUS_CAT` folds them), `MOVE_SECONDARY`'s L1-normalised major prefix for a damaging move's secondary |
| 12 | `neutralization` | fraction of this mon's per-turn contribution destroyed WITHOUT a KO: burn → `0.5·base_atk/(base_atk+base_spa)`, paralysis → `0.25 + 0.75·Δp_outspeed` (the op's OWN outspeed logistic re-evaluated at ×0.25 speed), freeze/sleep → 1.0, psn/tox → the 1/8 and 1/16 residual ticks. Every scalar is a gen3 RULE; no tuned prior |
| 13 | `tempo_cost` | `P(any major status) × undo_turns(j)`, where `undo_turns` is the **CHEAPEST available undo path**: 1 turn for a cure MOVE (`MOVE_CURES_SELF_STATUS`: Refresh / Heal Bell / Aromatherapy), **1 for the Natural Cure ABILITY** (the status is shed on switch-out and a switch consumes exactly one of our actions), the op's own `rest_sleep_noeb` (**2**) for Rest, **2 for the bench-CLERIC path** (switch to an ALIVE teammate carrying a party-wide Heal Bell / Aromatherapy, then click it), else **0**. `0` means *no path exists* — never *the path is free* — which is why Natural Cure is priced at its literal switch. Every input is OUR mon's (moveset, ability, HP), so all of it is exact and no marginalisation arises on this axis. `neutralization` deliberately does NOT read the ability: it is a per-TURN rate and Natural Cure changes DURATION, which this reduction refuses to model without a rule to source a number from |

ONE α over the move axis then reduces it — **Contract W**: α has no defender axis and no channel
axis, so the flat block's nine-independent-maxima incoherence (D2) and a per-defender α (D3) are
**shape errors** here rather than properties under test. α is the softmax of the PUBLISHED α
logits, **move slice only, unrenormalized and stop-grad** (a high `α_SWITCH` correctly shrinks
every coordinate toward zero; the detach is unconditional, so no PPO→`alpha_head` route depends on
a training flag). **With `--opp-intent` OFF it falls back to the shipped R1 `belief_mean` rung
(α := w/Σw)**, which is what makes the flag independently enableable — the DELIVERY claim is
testable apart from the DISTRIBUTION claim. A seat closed by the meaningful-K gate is MASKED, its
mass not reassigned. The reduced row for our ACTIVE defender rides every move cell as decorrelated
context through a zero-init `Linear(14, 14)`.

Known limits of the coordinate table, named rather than approximated: status DURATION (which is
also why the Natural Cure ability rides `tempo_cost` and not `neutralization`), physics mutation
(Marvel Scale), and a held berry's auto-cure.

**LIVE (ON in the gen-17 base): `pair_outcome_switch`** (v94, `gen3_pair_outcome_switch_v1`). The same
reduction, at **every** defender (`Σ_k α_k · pair_in[k, j, :]`), delivered to mon *j*'s own pointer
**SWITCH** cell through a second zero-init projection. This is the sink `design_pair_reduction.md`
§2.1 traced the defect to: the switch cell carries ten damage numbers, one speed number, two
belief-mass numbers and **no status coordinate in any currency**, so *"they will click Will-O-Wisp,
so bring the Natural Cure mon"* is unrepresentable there. It is the **first module to widen the
switch cell**. One α still serves all six rows — D3 (a per-defender α) stays a shape error — and
the module is equivariant in our team axis by construction. One extra per-defender coordinate rides
with the row, `spin_denied` = `is_ghost(our mon j) · Σ_k α_k·is_rapidspin(k) · their_side_hazards`
(the defensive half of the Pursuit mirror: a gen-3 Rapid Spin fails outright against a Ghost, so a
Ghost switch-in is hazard insurance; the stake is what makes it a value rather than a fact).
Requires `damage_op`, **not** `pair_outcome_cell` — the two deliver one tensor to two sinks and
coupling them would make a measured result unattributable. `PAIR_OUTCOME_SWITCH_DIM` = **15**.

**LIVE (ON in the gen-17 base): `conditional_threat_cell`** (v95, `gen3_conditional_threat_v1` —
`design_conditional_opponent_cells.md` §1's **OA1**, the defensive pivot). The **second** module to
widen the pointer SWITCH cell, and it carries exactly the quantities the α-reduced outcome row
structurally cannot. Four coordinates, all `Σ_k α_k · f(k, j)` against the same one α:

| # | coordinate | meaning |
|---|---|---|
| 0 | `e_pko_acc` | `Σ_k α_k · ko_ramp(k,j) · acc(k)` — §0.2(2)'s rule (*precompute every nonlinearity of two numbers IN THE OP*). `ko_ramp` and `acc` ride the reduced row DECORRELATED and a thin `tanh` scorer does not multiply two of its own inputs; two of our mons can be identical in `Σα·ko_ramp` AND in `Σα·acc` while their true P(dies) differ |
| 1 | `e_type_mult` | `Σ_k α_k · type_mult(k,j)` — the one cell channel NOT divided by the defender's own bulk, so a structural immunity (`0.0`) reads apart from an incidental zero and the read survives the mon's own HP moving |
| 2-3 | `margin_high` `margin_crit` | `Σ_k α_k · high(k,j) − hp_frac(j)` and the same on the crit roll (§0.2(3): *probabilities SATURATE; ship the MARGIN too*; `> 0` ⇒ dead). They separate two mons a saturated `pko` cannot — at the bottom (*both survive; by how much?*) and at the top (*both die; does a low roll save one?*) — and the crit margin is the *safe pivot vs coinflip pivot* distinction |

Three of §1.2's five clauses are **superseded and were substituted rather than built** (the table is
in `conditional_threat.py`): its `λ`-weighted `w = softmax(λ·threat + log belief)` is NOT built —
`pair_alpha` is the shipped distribution and a second one would be a second α; `high` / `pko` /
`status_lands` are already delivered by `pair_outcome_switch`, and `status_lands = Σ_s p_s` is
additionally barred by §9a's derivability rule; and §1.3's *"also turn on
`--damage-matrices-outgoing-all`"* is **VOID**, that flag having been deleted at v88. The op
stashes the per-(defender, seat) `type_mult` at α's own seat alignment behind a new seam
(`stash_pair_type_mult`) rather than letting a consumer re-derive it — the `op move-order` bug class
with extra steps. Requires `damage_op` + `damage_matrices_incoming`, **not** `opp_intent` (the R1
`belief_mean` fallback is MEANINGFUL here: every coordinate is a *what lands on me if they attack*
contraction, so the missing SWITCH mass correctly shrinks it) and **not** `pair_outcome_switch` (two
quantities, one sink, attributable separately). `CONDITIONAL_THREAT_SWITCH_DIM` = **4**.

**LIVE (ON in the gen-17 base): `switch_branch_cell`** (v94, `gen3_switch_branch_v1` —
`design_conditional_opponent_cells.md` §2's OA2, plus two owner-specified mechanics of the same
shape). Everything in it is `Σ over their options of (usage probability) × (a property of the
option)`, contracted over the branch in which they **switch**. Gen-3 is simultaneous-move, so
`P(they switch)` is ONE scalar for the turn (§2.1); the CONSEQUENCE is per-move, because switches
resolve first and our move lands on the arrival, which **β** names. Nine coordinates on the move
cell:

| # | coordinate | meaning |
|---|---|---|
| 0-2 | `e_high_switch` `e_pko_switch` `e_mult_switch` | `Σ_j β_j · omx[k, j, ·]` — the SWITCH branch of our own move, from the outgoing matrix. §2.3's rule is followed literally: the branches ship DECORRELATED (the stay branch already rides the op's move cell), never the collapsed `(1−p)·stay + p·switch` |
| 3 | `wasted_ko` | `pko_stay(k) · α_SWITCH` — §2.3's named interaction, *"don't click the KO into the obvious switch"* |
| 4 | `a_switch` | the ONE per-turn switch scalar, broadcast over all four slots |
| 5-6 | `p_spin_blocked` `spin_value_lost` | `is_ghost(their active)·a_stay + α_SWITCH·Σ_j β_j·P(slot j is Ghost)`, gated to the Rapid Spin request slot, and that probability × our-side hazards. **The Pursuit mirror**: v85's Pursuit is `α_SWITCH` against a property of the DEPARTING mon with positive valence and no β (the sim strikes before the switch resolves); this is `α_SWITCH` through β against a property of the ARRIVING mon with negative valence (Rapid Spin resolves after). `P(slot is Ghost)` is leak-free — revealed types where revealed, the hidden-team species posterior through `SPECIES_IS_GHOST` where not |
| 7-8 | `protect_attack_mass` `protect_blocked_mass` | `Σ_k α_k · is_damaging(k)` gated to Protect/Detect, and that × the obs `p_success` decay. The `c4` successor: Protect's cell carried the mechanical decay and never asked *will they attack*. Decorrelated from v85's `e_dmg_avoided`, which is a MAGNITUDE where this is a MASS — they come apart in both directions (a believed Spore has mass and no magnitude; a 4×-resisted Hidden Power the reverse). `is_damaging` is typed from the data facade, so an immune damaging move cannot masquerade as a status move |

α and β are read from the PUBLICATIONS and **stop-grad unconditionally**. This flag **requires
`opp_intent` with no fallback**, and the asymmetry with the pair-outcome pair is substantive: the
R1 `belief_mean` rung is a PRESENCE belief over their MOVES and carries no switch class, so
`α_SWITCH` would be identically 0 and every coordinate would assert *"they never switch"* — a
claim, not an absence. §4.1's hard prerequisite for OA2 is **CLOSED**
(`gen3_unrevealed_outgoing_prior_v1` prices an unrevealed arrival against the expected-latent
defender); the one residue, stated rather than hidden, is that **`pko` stays NULLED at unrevealed
slots** by the op's owner rule, so `e_pko_switch` is deflated in proportion to β's hidden mass
while `e_high_switch` carries the magnitude there. `SWITCH_BRANCH_MOVE_DIM` = **9**. Not modelled:
Rapid Spin also clears Leech Seed and partial-trap from its user, and a Ghost KO'd on the switch-in
denies nothing.

Route availability is **width-neutral by construction** (additive injection changes no
projection width), so the old ede5a88 discovery-sizing bug class — a fall-through branch hiding
a vf part from the forward that sized `value_pre_norm` — is unrepresentable. Both projection
input widths are **static arithmetic** (`compute_projection_widths`, `gen3_static_widths_v1`;
the construction-time discovery forward is deleted): pi = 3·D_MODEL + the `non_matchup_rest`
scalar tail + k·D_MODEL (hidden-opp belief pool, `opp_belief_cls_k`, POLICY side only);
**vf = D_MODEL, a constant no flag can move**. `projection_width_test.py` verifies the arithmetic
against a real forward per flag combo. Any NEW value route goes through `_value_pooled_routes`
(the registry the gradient guard iterates) — never a new vf-concat part; there is no vf concat.

**`vf_combined` IS `value_pooled`** (v96 `gen3_critic_route_wave_v1`). The value head reads one
tensor and nothing else: not `our_active_refined` (the `value_active_readout` toggle was deleted
at v88), not either team pool, not the `non_matchup_rest` scalar tail, not the hidden-opp belief.
The whole post-assembler vf tail was retired on measurement — its three members read dV **0.0000**
on gen-14 at n=12,391 (`nmr` and `hidden_opp_vf`) and **0.0000 bit-exact on two consecutive
audits** (the seed window). Since the win-prob critic reads `value_pooled` directly, this
also makes the v89/M2 orphaned-vf-branch class **unrepresentable** rather than merely fixed: there
is no second path for a critic to bypass. Every critic enrichment is an additive injection into `value_pooled` (the v89 seam) or a
token-content injection on the value pool's local copy inside `CLSPool`.

**The v61 multi-seed window is DELETED with the rest of that tail.** It was k=4 learned queries
cross-attending over the op's per-our-mon incoming rows — MULTIPLICITY rather than width (ledger
P3 refuted width only) — and it carried the `value_seeds/*` collapse contract, which went with it.
**The trigger it existed to catch FIRED on gen-5** (`out_effective_rank` 1.0 sustained
196k→15.7M steps — the k=4 outputs identical), and the two pressures then applied to it are the
reason the line closed:

**Both pressures are deleted (v78). The measurement is why.**
`--value-seed-vicreg-coef` (v62) was the repulsive one — a scale-relative
variance+**covariance** floor on the seed outputs. Gen-6 ran it and every term moved (std_rel
0.002 → 0.53, correlation → 0.19) while effective rank stayed **1.05**, because the deviations
occupy **less than one direction** (centered PR 0.846): seeds 0/1/2 kept near-identical attention
while seed 3 alone broke away. Repulsion buys spread, not multiplicity. `--seed-quantile-coef`
(v63) was the positive counterpart — seed k predicts **quantile τ_k** of the return through **one
shared** `Linear(dim,1)`, so k different predictions require k different seed reads. Gen-7 ran it
and it worked on its own terms (`quantile_crossing_rate` 0.456 → **0.000**, `quantile_spread`
0.007 → **1.016** at 10.6M steps — the four seeds do predict four ordered quantiles) yet
`out_effective_rank` reached only **1.157** against a ceiling of k=4.

**The structural reason is shared, which is what closed the line**: a **shared** readout constrains
only each seed output's component along its own weight vector, leaving every orthogonal direction
free. Seed MULTIPLICITY is therefore not the axis the critic was missing, and no coefficient on
either term reaches it — so both flags and both modules were deleted rather than retuned, and the
READOUT itself followed at v96 once two end-of-run audits read its dependence at 0.0000.

**That null is what `--value-threat-inject` (v64) responds to.**

`--value-threat-inject` takes the third route instead — **magnitude as token content, per entity**.
For each of OUR mons `j`, the op's α-weighted incoming row (`Σ_k α_k · pair_in[k, j, :]`, α = the R1
`belief_mean` rung, which the flag forces on because R0 `hard_max` builds no reducer) is projected
by ONE shared zero-init `Linear(13, 128)` and added to that mon's token on **the value pool's copy
only**. `value_cls` then pools augmented tokens; `our_cls`, `our_active_refined` and the pointer
head all read the untouched tensor, so `pi` is bit-identical at **any** weight — asserted against a
large random projection, not merely at init. The route is invariant under permuting their moves (α
is shared across defenders by Contract W), equivariant under permuting ours (the row rides mon `j`'s
token), and invariant at the pool — unlike the deleted flat concat, whose meaning was slot-ordered.
`W_inj` is covered by `restore_identity_init()` (ledger M1) and that is gated on a real
`MaskablePPO` build, not a bare extractor. Structural + version-checked, fresh runs only; OFF
(production) builds no module and leaves the op on `hard_max`. **v1 substitutes α := normalize(w),
a PRESENCE belief where the design wants a supervised USAGE belief** — deliberately, so a null
indicts the delivery route rather than the belief.

### 3.3 The action head is the pointer head — there is no flat `action_net`

`Gen3DualHeadMaskablePolicy._build` replaces SB3's flat `Linear(latent, 11)` with a **raising stub**
and rebuilds the optimizer; `PointerNativeActionHead` produces the logits, and
`_get_action_dist_from_latent` is the single funnel all three logit sites pass through.

Shared context for all three families: **`latent_pi`** — the policy tower's output, i.e. everything
in §3.1 after the mlp_extractor. So the op block, the beliefs, and any head-level modulation
condition every pointer score.

Output layout is `[switch ×6, move ×4, struggle]` (`agents/action/constants.py`).

| Logit | Entity token | Physics cells | Cell width |
|---|---|---|---|
| **move k** (logit 6+k) | the **refined E3 seat k** (`last_pointer_inputs[0]`, `[B,4,128]`) — post-attention, board-aware, already permuted sorted-by-id → **request** order by move-num identity | `[low, high, crit, pko, p_land, known, sec×7]` | **13** (`_PTR_MOVE_CELL`) |
| **switch j** | our-team token *j* (`our_team_out[:, j]`, `[B,6,128]`) — the same post-transformer token the CLS pools read | the incoming per-defender row (12) + `[phys_high_cb_j, pko_cb_j, p_cb]` | **15** (`_PTR_SWITCH_CELL_IN`), +15 under `pair_outcome_switch`, +4 under `conditional_threat_cell` |
| **struggle** | none — context only | none | 0 |

The move cell WIDENS under the opt-in α cells, each appending its own zero-init block:
`intent_move_cell` (+`INTENT_MOVE_CELL_DIM`), `intent_threshold` (+`INTENT_THRESH_MOVE_DIM`),
`intent_conditional` (+`INTENT_COND_MOVE_DIM`), `pair_outcome_cell`
(+`PAIR_OUTCOME_MOVE_DIM` = 14) and `switch_branch_cell` (+`SWITCH_BRANCH_MOVE_DIM` = 9).
`pointer_move_cell_dim` is the single sum the policy sizes the move scorer's `in_features` from —
a missing block narrows the `Linear` rather than silently feeding it zeros at a learned weight.
The **switch** cell likewise widens under `pair_outcome_switch`
(+`PAIR_OUTCOME_SWITCH_DIM` = 15) and `conditional_threat_cell`
(+`CONDITIONAL_THREAT_SWITCH_DIM` = 4), summed by `pointer_switch_cell_dim` and appended in that
order; until v94 nothing widened it at all.

Scoring: `tanh(proj(token ⊕ cells) + ctx_proj(latent_pi))` → a zero-init `Linear(64, 1)`.
Move logits are multiplied by `move_valid`, so an unresolved request slot contributes **exactly 0**
rather than a score computed from a zero token.

**What the switch logit does NOT see** (with every v94 flag off): a per-candidate **offense** read,
and — the defect `design_pair_reduction.md` §2.1 names — any **status** coordinate in any currency
(`pair_outcome_switch` closes the second one; `conditional_threat_cell` adds the conditional-threat
coordinates that row cannot carry). The OAX attacker row
(`damage_matrices_outgoing_all`) was deleted with its flag (v88 `gen3_dead_flag_purge_v1` — never
enabled in a gen-8+ run), so the flags-off switch cell is 15 dims and its physics is purely defensive
(what this mon takes on the switch-in) plus whatever the trunk carried into `our_team_out`. **In the
gen-17 production config both flags are ON, so the switch cell is 15 + 15 + 4 = 34** and both gaps
above are closed. The
`d2` edge family (§5) — whose engine is the same `_outgoing_attacker_matrix` kernel — is the route
by which a bench mon's offense reaches its own token.

Secondary channel widths: `sec×7`, not `sec×10` — the outgoing block prices only the 7 secondary
columns an our-side gen3 move can inflict (`_OUT_SEC_COLS`; slp/psn/tox were dropped as structural
zeros). The `PointerNativeActionHead` docstring still says `sec×10`; the code is right (§8).

Position-equivariance is structural: one shared scorer per entity family, so permuting the team
permutes the logits, and a sorted-vs-request misalignment is unrepresentable at the logits.
Cold start: all three scorers are zero-init and built **after** SB3's ortho-init pass, so every
logit is exactly 0 at step 0 ⇒ uniform-over-legal.

**The pointer route is POLICY-ONLY.** `pointer_head` is reached solely through
`_get_action_dist_from_latent(latent_pi)`; every value path is
`forward_critic(vf_features) → _critic_value`, which never touches it. So the per-action `cell`
channel exists for the actor and **not** for the critic — the critic's op-physics routes are the
entity pool's injection into `value_pooled` (dV 5.490 — 97% of the whole critic route joint),
`--value-threat-inject`'s token content on the value pool's copy (1.0686) — both vf-only, both reading the op through `OpTensors` views rather
than flat offsets.

### 3.4 Side readouts

A side readout hangs off `value_pooled` AFTER the pools and stashes its logits; none of them ever
enters `pi` or `vf`, so none changes a projection width. **One is built in this config, and it is
the critic:**

| head | flag | grad flow |
|---|---|---|
| `win_head` | `win_prob_mode` **`shaping`** | live `value_pooled` — the win objective also shapes the trunk (`read_only` would stop-grad it) |

#### WHICH readout is the critic — the recorded `critic` field {shaped,winprob}

`policy._critic_value` has a MODE, and the production config is on the second of two. There is no `--critic` flag (deleted, deletion pass P11b batch (b)): the win-prob critic is a constant of every trainer namespace and `critic` is the recorded field below. **This is
the ONLY axis on which this generation differs from gen-17** — the trunk, the seats, the edge
families, the pointer cells and the belief stack are that run's, unchanged.

| recorded `critic` | `V(s)` is | trained by | reward stream | `gamma` |
|---|---|---|---|---|
| `shaped` (the historical critic; LOADABLE only — not trainable since the Python env core was deleted, so an old shaped checkpoint serves as an opponent / in the meters, and a resume or fork of one is refused `FATAL_CONFIG`, D4) | `value_net` | the MSE at `vf_coef`, in raw return units | the SIGNED terminal alone, ±30 with a −35 timeout (the shaping it once carried is deleted, §6.3) | 0.9999 |
| **`winprob`** (**this config**) | `sigmoid(win_head logit)` ∈ **[0, 1]** | the win-prob head's **BCE against the terminal WIN INDICATOR**, at `vf_coef` **0.5** | the TERMINAL **WIN INDICATOR** alone — `+victory_value` (**1.0**) on a win, `0.0` on a loss, a tie and a 250-turn timeout alike | **1.0** |

**PopArt, the distributional value head (`value_dist_head`) and the `value_from_dist` critic route
were DELETED** (deletion pass L1, config v131 — `designs/ops/deletion_pass_manifest.md` §2): the
three of them, the CVaR value-tail weight, the win-prob aux-BCE coefficient and both win-prob PBRS
rungs (self-φ, frozen-φ) were levers of the shaped critic or the Python env core, and every
v121+ checkpoint recorded them OFF. A checkpoint that recorded one ON is refused
(`model_version.retired_levers`). (On a shaped checkpoint — loadable, not trainable — the win-prob BCE
was an auxiliary readout at a fixed weight of 1.0.)

The critic and the return are the same quantity by construction: with the win-indicator terminal (victory 1.0, a constant of the namespace) the
undiscounted return from any state is exactly `1{win}`, so **`V(s) = P(win | s)` with no
approximation term**. `value_net` is in no loss graph (its scalar term is dropped), the BCE joins
the **`value`** noise-scale group rather than `aux`, and there is one critic and one coefficient
(`--vf-coef`).

**Both critic-side enrichment routes SURVIVE** (`value_entity_pool` / `value_entity_pool_full` /
`value_threat_inject`, all still true): they inject additively into `value_pooled`, which is
exactly what the win head reads, so they enrich a probability critic the same way they enriched a
scalar one.

The critic mode is STRUCTURAL — it selects a different set of heads to carry the value — so
`critic` is recorded in `model_config.json` and string-compared by `check_compatible`. It carries
**no `ARCH_SIGNATURE` bump**: a flipped mode produces no shape error anywhere (both routes return
`[B,1]`), which is exactly why the recorded-and-compared field is the whole safety.

⚠️ **A critic bounded in [0,1] cannot represent "a timeout is worse than a loss."** The `−35 < −30`
ordering a draw penalty would set is not merely unused here, it is unrepresentable — so
the draw value is fixed at 0.0 by construction (the `--draw-penalty` flag is DELETED), and the anti-stall pressure comes from the obs
deadline clock (§1.4) — the reward has no anti-stall term at all since the shaped reward path was
deleted (§6.3). **Stall rate and mean episode length
are PRIMARY endpoints, not monitored ones.** The 250-turn cap, forfeits and ties are TERMINAL under
this mode rather than SB3 truncations — as truncations at γ = 1 the bootstrapped `γ·V(s_last)` made
every timeout's TD error identically zero, so the critic could not see them at all.

ONE flag is IMPLIED by the win-prob critic (the only critic; `resolve_critic_mode` applies it when untyped): `--win-prob-mode shaping`, because its argparse default is the `None` sentinel, so "unset" is
representable and an implication can never overwrite a typed value. The discount (`WINPROB_GAMMA` = 1.0) and the win-indicator terminal (indicator, victory 1.0, draw 0.0) are not flags: `--gamma`, `--victory-value`, `--draw-penalty` and `--terminal-indicator` are DELETED (P11b batch (c)) and are CONSTANTS of every trainer namespace (`src/main/train/parser/objective.py`, `parser.set_defaults`); a typed one is refused with its reason (`designs/deleted_flags.md`), and a resume whose recorded reward is not the production one is REFUSED by `check_reward_config`. `resolve_critic_mode` runs BEFORE the resume-inheritance sweep, so the implied value lands on the `None` sentinel before `_resolve` can inherit a recorded one (a fork of a `shaped` parent never gets that far: it is REFUSED, D4). Design of record:
[`designs/ai_v12/design_winprob_only_critic.md`](ai_v12/design_winprob_only_critic.md).

**`--fork-fraction` FORKS CONTESTED STATES INTO THE BUFFER** (`gen3_fork_v1`, config v120 — THE
FORK ARM). Default **`0.0` = OFF and BIT-identical**: no fork pass is built, no obs key is declared
and no row is injected. It runs on the win-prob critic (the only critic). It runs on the Rust core (the only env core). Above 0.0 it is the FRACTION of the buffer's
decisions that are FORKED: at a CONTESTED decision (a move round, turn 2-40, ≥3 legal actions, top-2
masked-logit gap under the `--fork-contested-gap` quantile of this rollout's own candidate pool) the
episode is replayed to that turn (on the Rust core, from its own finished input log) and `--fork-branches` continuations —
the policy's top-2 candidates plus ONE uniformly random legal action — are played to a terminal by
the CURRENT policy on both sides at temperature 1.0. Their transitions enter the SAME PPO buffer,
with the branch's own GAE and its own outcome as `win_target`. **Plain BCE, no ranking
term.**

🚨 **The measurement that registers it:** the promoted win-prob critic's held-out PAIRWISE ACCURACY
on successors ONE MOVE APART is **0.5169 [0.4800, 0.5524]** — a coin — while a FROZEN trunk with
only `WinProbHead`'s four tensors refit on counterfactual successors reaches **0.6032
[0.5690, 0.6374]** (+0.0863 [+0.0384, +0.1347], DETECTED), and a pairwise RANKING term on the same
rows buys **nothing** (−0.0107 [−0.0249, +0.0028], NOT DETECTED, negative on points). Pairwise
accuracy is a RANK statistic and is invariant to monotone recalibration, so the ordering was in
`value_pooled` all along — **the DATA is the lever, not the loss form**
([`paired_refit_discrimination_2026-09-14`](research_state/measurements/paired_refit_discrimination_2026-09-14/README.md)).
The RANDOM branch is where the new information is: on 5,076 measured forks the policy's top-1 and
top-2 were outcome-INTERCHANGEABLE (0.7082 vs 0.7078) while a uniformly random legal alternative
beat BOTH candidates in **4.5 % [3.99, 5.16]** of forks.

🚨 **THE FORK STEP IS MASKED OUT OF THE POLICY TERM FOR EVERY BRANCH**, the top-2 included, and the
term is RENORMALISED over the kept rows — an exclusion that depended on WHICH branch a row came from
would re-weight the policy gradient by the branch mix, and a masked `.mean()` would lower the
effective policy learning rate by the fork rate. The carrier is the `fork_pg_m` obs key, declared
only when the flag is on. **The shared PREFIX is counted ONCE** — a branch's rows begin AT the fork
step; the fork STATE appears once per branch with a DIFFERENT action, which is the exploring start.
🚨 **`--fork-crn dice_and_draws` (the default) pairs the sim dice AND both sides' policy sampling
streams**, so branches differ in exactly the action at the fork; `dice` alone is the `cf_q_labels`
regime and is kept as the control. ⚠️ A branch is played against the parent's REAL policy opponent
where that slot still serves the parent's model, else against a SELF-LIKE one (a bot parent, an
external route, a reloaded slot) — labelled on the row (`opp_class = POOL`) and priced by
`fork/opp_substituted` / `fork/branch_share` / `fork/bot_share`. 🚨 **Cost is
`forks × branches × remaining decisions`** — ~1.5-2× a plain run's simulation at fraction 0.02 with
3 branches — published as `fork/sim_steps_share`, and bounded by a fork cap, a row budget
(branch rows may at most match the rollout's own rows and COMPETE with them for the update's `D`) and,
because a fork dropped at that budget has already been PLAYED, by the previous pass's measured
`fork/rows_per_fork`. The six are training-only — no
forward pass, no weight shape, no `check_compatible` compare, **not** `flag_registry.py` rows — so
they do not appear in §6's flag table until a production config adopts them. Mechanics:
`designs/training/forks.md`.

The `--win-prob-pbrs-*` family (self-φ and frozen-φ potential-based shaping from the win-prob head) was
**DELETED** with the shaped critic's levers (deletion pass L1, config v131): with `V ≡ φ` the self-φ
form is the TD residual GAE already turns into the advantage, and the frozen-φ actor-only rung was
registered but never read. `designs/deleted_flags.md` carries the citations; UNDERSTANDING keeps the
open question "do we need PBRS?".

**`QWinProbHead` is the per-action scorer the ride-along A head is built from — and nothing else.** It
scores each of the eleven actions from the token of the entity that action selects — the SAME per-action
tokens the pointer head scores (`stash.pointer_inputs`) — with `value_pooled` as its context, one shared
zero-init scorer over the eleven slots. The extractor has NO per-action win-probability head of its own
(no `q_winprob_mode`), and no evidential Beta head, twin win-prob heads or shadow critic (no
`cf_evidential` / `cf_twin_heads` / `cf_shadow_critic`): each was a state_dict delta whose only trainer
was a counterfactual-label loss, and no run writes those labels. A checkpoint that recorded one of the
four is refused on load; `designs/deleted_flags.md` carries the flags and the pin that still has them.

**The DETACHED RIDE-ALONG heads EXIST in the code and are OFF here** (`gen3_ridealong_heads_v1`,
config v126; owner 2026-09-30, EXPERIMENT_BACKLOG X25 / X4a; `agents/model/ridealong_heads.py`).
Four STRUCTURAL flags, each 0 / false by default: `--ridealong-ensemble K` (K win-prob members on
LayerNorm(`value_pooled`), each with a per-STATE bootstrap mask and a randomized prior; their spread is
V's epistemic uncertainty), `--ridealong-rnd` (RND over the running-normalised RAW observation, not
the trunk features, so novelty is not confounded by representation drift), `--ridealong-adv K`
(per-action A over the pointer head's own tokens, `QWinProbHead`'s shared-scorer shape, centred
under π, MSE on the GAE advantage of the action taken), `--ridealong-opp K` (B over α's support
— their believed move seats by MOVE ID + SWITCH — centred under α, MSE on the same advantage where
their actual action is named; requires `opp_intent`). Q = V + A + B is a derived `ridealong/*` meter;
there is no I term. **They cannot change what the run learns, and that is tested, not asserted**:
the extractor only RECORDS the four kwargs; `Gen3DualHeadMaskablePolicy` builds `policy.ridealong`
after SB3's `_build` (so the heads are in no `policy.optimizer` group and outside the ortho-init
apply), from a private seed inside `fork_rng`; the frozen networks are BUFFERS; the forward never calls
them; and the learner steps them with their own Adam on `.detach()`ed stashes right after the
minibatch forward, returning their grads to None before PPO's loss is assembled. They train on
PPO's first epoch only, so each rollout row is seen once. `ridealong_heads_test`
shows no ride-along loss reaches any trunk, policy or V parameter; `ridealong_update_test` shows one
real update with the heads ON vs OFF leaves the policy, trunk, V, PPO optimizer state and RNG
BIT-IDENTICAL. The compile gate's coverage count excludes `ridealong.*` (never in a compiled graph).
Their Adam state is the LEARNER's and is not checkpointed (a restart resumes the weights with a fresh
Adam). It is ACQUIRED AT STARTUP (the declared lifecycle): `_setup_model` ends in `_ridealong_acquire`,
which builds every ride-along optimizer and pre-allocates its Adam state, bit-identically to torch's
lazy init. There is NO lazy build path: a step that finds an optimizer missing raises
`RideAlongLifecycleViolation`. `ridealong_update_test` pins that raise, and unchanged optimizer,
state and buffer identities across a real update. **Only the trainee acquires** (`gen3_opponent_inference_load_v1`):
an OPPONENT load (the self-play pool, eval sentinels, stable / exploiter / teacher loads) is an
`InferenceMaskablePPO`, with policy weights only and no optimizer of any kind. It may differ from the
trainee in the ride-along keys (`RIDEALONG_FLAGS`) alone, in either direction, and the T2 slot identity
and every served replica leave the heads out. The trainee's resume stays strict on every ride-along
key, and every checkpoint load (learner, opponent, reader) is strict on state-dict keys — a missing or
unexpected key raises (`gen3_strict_checkpoint_load_v1`: `StrictCheckpointLoad`, which `OwnedLoop` and
`StrictMaskablePPO` carry, and `snapshot.load_checkpoint_strict` for every reader;
`designs/training/learner_lifecycle.md`). The heads' step is the K8 inventory's candidate compile region
R-ride; it stays EAGER. `family=CRITIC`, so they are off the ARCH surface; §6's table carries them
OFF. The pre-registered baseline that turns them on is EXPERIMENT_BACKLOG's X26.

**The RND VARIANT ENSEMBLE rides beside the RND head and is OFF here** (`gen3_ridealong_rnd_variants_v1`,
config v127; owner 2026-09-30; `--ridealong-rnd-variants`, a canonical comma list or `all`; it requires
`--ridealong-rnd`). Each variant is its own detached predictor with its own Adam, gradient clip,
error z-score, fail-closed switch and `ridealong/rndv_<name>_*` series. The `--ridealong-rnd` head
(`base`) is unchanged and is the reference. The declarations are `RND_VARIANT_DECLS`:

| variant | input | how it forgets | predictor vs target |
|---|---|---|---|
| `fast` | base's normalised observation | 10× base's predictor rate (3e-3) | base's predictor, started from base's exact weights; the same ReLU-MLP family as the target, one layer deeper |
| `decay` | base's normalised observation | pulled toward its own init once per PPO update, half-life 10 updates (≈ half an eval cycle) | as `fast` |
| `small` | base's normalised observation | cannot memorise: obs→32→64, 90,496 parameters (11.5 % of base's) | deliberately LESS expressive than the 256-wide target, so it cannot identify the target exactly |
| `feat` | the detached `value_pooled` (D_MODEL), with its own feature normalisation | — (it measures representation drift live) | base's shapes over the features, with its own frozen target |

The three observation variants share base's frozen target and observation normalisation. The same
stream gives identical statistics, so every comparison with base is PAIRED. Adding the variants
leaves every other ride-along tensor, base included, BIT-IDENTICAL after an update, and
`ridealong_update_test` pins that with a third arm. Every variant is inside the same detachment proofs
(`ridealong_heads_test`: no variant loss reaches the trunk, policy or V, and each trains only its own
predictor; `feat` additionally detaches its own input). Each logs its raw-error median, IQR and IQR ÷
median (the saturation series). Base and the observation variants also log `*_ident_ratio`: the error
on the minibatch's deterministic block chimeras (`block_chimera`, no RNG) over the error on its real
rows. That is the identification monitor, and it stays high while a predictor learns only the states
it visits. MEASURED overhead, 2026-09-30, `ridealong_step_benchmark.py` on the GPU at load ~10 (box
idle): the four heads add 0.65 s to arm C's 67.05 s update (0.97 %). The four heads plus all four
variants add 0.97 s (1.45 %). Inside that configuration each variant's marginal cost is 0.11–0.16 s
(`measurements/ridealong_baseline/overhead_variants_2026-09-30.json`).

Belief heads run under `belief_grad_mode` **`shaping`** (production mirror `belief_grad_mode:
"shaping"`; §6's table carries it ACTIVE): all four routes are live — the label loss trains the
head AND shapes the shared trunk through the head's read, and the PPO loss reaches the head's
parameters through the reinject write. The opponent-INTENT head is the exception and runs
**`detached`** (`opp_intent_grad_mode: "detached"`): its trunk read is stop-grad, so the intent
labels cannot reshape the trunk. `label_only` (publish stop-grad; labels alone train the head) is
a built, non-default mode — see `src/agents/model/CLAUDE.md` for the four-route table.

---

## 4. The `DamageOperator` output block

**`out_dim` = 138** under the production config — that is the width of the FLAT block, which is
what the `ProjectionAssembler` concat and `pointer_cells` slice. Layout is contiguous, in this
order, and every sub-block is appended after the previous one (so enabling a later one never moves
an earlier offset).

| # | Sub-block | Width | In the flat block? | Gate |
|---|---|---|---|---|
| 1 | incoming per-mon — 6 × `[phys(low,high,crit,pko,acc), spec(…), p_outspeed, provenance]` | 6 × 12 = **72** | ✅ | `damage_op` |
| 2 | Choice-Band tail — `phys_high_cb ×6`, `pko_cb ×6`, `p_cb` | **13** | ✅ | `damage_op` |
| | *(1 + 2 = `incoming_dim` = **85**)* | | | |
| 3 | outgoing single-active — 4 moves × `[low,high,crit,pko]`, `p_outspeed`, 4 × 7 secondary | **45** | ✅ | `damage_outgoing` |
| 4 | status-landing — `P(lands) ×4`, `known ×4` | **8** | ✅ | `damage_outgoing` |
| 5 | `outgoing_matrix` — our 4 moves × opp 6 mons | 126 | ❌ **not rendered** | `damage_matrices_outgoing` = true, and `op_drop_renders` = **true** |
| 6 | `incoming_matrix` — K=6 headers (51 each) + 6 mons × 6 moves × 6-wide cells | 6×51 + 6×6×6 = 522 | ❌ **not rendered** | `damage_matrices_incoming` = true, K = `damage_topk_k` = 6, and `op_drop_renders` = **true** |
| | **Total** | **85 + 45 + 8 = 138** | | |

🚨 **"Not rendered" is not "off".** Sub-blocks 5 and 6 are COMPUTED on every forward and both
flags are `true`; `op_drop_renders` (v86, `gen3_op_lean_forward_v1`) drops only their
serialization into the flat block, which had no consumer. The `outgoing_matrix` call is what
`stash.out_cells` / `stash.out_pko` are a view of (the OA2 switch-branch magnitudes and the
Explosion `pko`), and the `incoming_matrix` call is where `last_topk_idx` / `last_topk_cand_idx`
— the seat axis α aligns to — are selected. Turning either matrix flag off deletes those; turning
`op_drop_renders` off re-widens the flat block to 660 and changes nothing else.

**Non-formula damage (`gen3_nonformula_damage_v1`, Beat Up `gen3_beatup_exact_v1`).** Every move whose damage is not the gen-3
base-power formula has ONE declared model in `damage_tables.DAMAGE_MODELS` (each row cites its
`deps/pokemon-showdown` source), and every kernel — incoming, the three outgoing blocks, the d3
refine, c1/c2/c3, the recovery cell and d4 — applies it through `damage_kinds.py`:

| kind | moves | damage the op prices |
|---|---|---|
| `fixed` | Seismic Toss, Night Shade (the level, 100 — every pool team is L100), Dragon Rage 40, Sonic Boom 20, Psywave (its expectation, 100 — an approximation) | that HP, no roll / crit / screen |
| `target_hp_frac` | Super Fang ½; Guillotine / Horn Drill / Fissure / Sheer Cold 1 (gen-3 OHKO = the target's current HP; 30% accuracy rides `MOVE_ACCURACY`; Sturdy NOT modelled) | that fraction of the target's CURRENT HP |
| `endeavor` | Endeavor | `max(0, target HP − attacker HP)` |
| `bp_flail` / `bp_hp_scaled` | Flail, Reversal (the gen-3 48-step table) / Eruption, Water Spout (150 × HP fraction) | the BP from the ATTACKER's HP fraction |
| `bp` | Return, Frustration 102 (the set that maximises them — the obs has no happiness), Magnitude 71 and Present 52 (expectations; Present's heal branch priced 0) | the formula at that BP |
| `beatup_party` | Beat Up — EXACT (typeless `'???'`: no STAB, no effectiveness, no ability read; SPECIAL: Light Screen halves it, Reflect and burn do not) | one hit per healthy party member, `Σᵢ [(42/50)·10·Aᵢ/D + 2]` with `Aᵢ` = ally i's species BASE Atk and `D` = the target's species BASE Def (no stat stage, item or ability reaches either), then the op's usual mean roll / screen / crit / KO ramp |
| `unmodelled` | Counter, Mirror Coat, Bide (need the turn's incoming damage), Low Kick (no weight in `data/`), Spit Up (no Stockpile count) | 0, BY DECLARATION |

The `fixed` / `target_hp_frac` / `endeavor` kinds respect type / ability immunity (Fighting Seismic
Toss into a Ghost is 0) and a kind's KO is `acc · [the hit KOs]` (an OHKO always, a fixed amount iff ≥
the remaining HP).
`MOVE_BP` holds the BP at the attacker's full HP for every formula-priced move, and every
dex-damaging move rides its TYPE's gen-3 channel in `MOVE_PHYS` (Return is physical). 🚨 **The table
build RAISES on a damaging (dex category ≠ Status) move with base power 0 and no row** — the
`category` field of `gen3_moves.json` exists for exactly this guard. The pointer head's E5 tail
score (`w · BP/150 · acc`, defender-free) reads `MOVE_BP` only, so a non-formula move reads 0 THERE and
Beat Up reads its per-hit 10.
Pinned by `src/agents/model/nonformula_damage_test.py`.

**Beat Up (`gen3_beatup_exact_v1`).** The one kind that stays the BP formula with its inputs swapped
(`damage_kinds.beatup_swap`): a Beat Up cell reads `A → S` (the attacking side's Σ base Atk over its healthy
mons), `D → ` the target's base Def and `+2 → +2N` (one `+2` per hit, `N` = the healthy count; no eligible ally
= no hit = 0), and rides every kernel's roll / screen / crit / KO line unchanged. The `MOVE_TYPE_IDX` row is
`'???'` (the outgoing kernels, which read the obs' resolved type — Dark, the dex type — route it through
`typeless_move_type`), so STAB, the chart, the ability multipliers and the weather / sport modifiers read
neutral. **Who is an ally.** Ours is fully known: alive and no major status, the user counted iff healthy. The
opponent's: a revealed mon counts iff alive and unstatused; **every unrevealed slot counts with certainty** (a
mon that never entered cannot have fainted or been statused) at its EXPECTED base Atk under the op's one
hidden-mon belief — the T0 species belief the extractor hands every pricing site (`t0_species_probs`), else the
Species-Clause usage prior (`unrevealed_species_probs`); the sum's expectation is exact. An unrevealed
DEFENDER (`_outgoing_matrix`) uses `D = 1 / E[1/D_base]` under the same belief, which returns E[damage]
exactly. Declared limits: a forme reads its base species' stats (the op's num-keyed convention); Beat Up into a
Substitute and the per-hit crit roll are not modelled; the KO ramp is the op's one shared roll over the summed
damage, which over-states the spread by ≈ √N (the certain-KO / certain-no-KO ends are exact). Measured against
the real sim (`src/agents/model/beatup_sim_parity_test.py`, 11 constructed scenarios, fixed seeds): the sim's
mean total into a Blissey for the pinned six-mon party is 507.2 HP, the integer-exact mean 505.9, the op's
smooth mean 511.5 — the op reads 0.9–1.0 HP per hit above the integer process (its documented smooth-vs-floor
bias); the parent's one-hit price was 8.7 HP. Pinned by `src/agents/model/beatup_damage_test.py`; evidence
`designs/research_state/measurements/beatup_golden_2026-10-03/`.

**Field base-power modifiers.** Every damage kernel multiplies the candidate's base power by the
gen-3 weather modifier (rain ×1.5 Water / ×0.5 Fire, sun the reverse) and by the **field sports**
(`gen3_field_sport_slots_v1`): ×0.5 on an Electric move while EITHER active holds `mudsport`, ×0.5
on a Fire move while either holds `watersport` (read off both actives' context blocks,
`DamageOperator._sport_mult`). World approximation, the weather convention: a hypothetical world
that swaps a holder out still reads the current actives' sports. Pinned by
`damage_op_test.py::test_op_field_sports_halve_electric_and_fire_from_either_active`.

The block passes through a learned per-channel `out_gain` (a Parameter, multiplicative only, so the
"no threat ⇒ exactly 0" gates stay clean) before it reaches the heads and before `pointer_cells`
slices it — so the pointer path and the flat concat can never disagree on a value.

**The pair-reduction rungs exist but are INERT in production** (`agents/model/pair_reduce.py`,
`design_pair_reduction.md` §8.1 steps 3–4): `DamageOperator(reduce_how=…)` — constructor-only, no
CLI flag, no config field — can build Contract-W/L reducers (R1 `belief_mean` / R2W `learned` /
R2L `deepsets_{sum,max}` / R3 `multi`) BESIDE the legacy per-channel hard max. The production
default `"hard_max"` builds **nothing**: no params, no state_dict keys, no forward work. A
non-default rung only stashes `last_reduced_extra` [B,6,extra_dim]; nothing consumes it — delivery
+ versioning is gen-6 work, gated on the §8.1 step-0 audit.

Also **not present anywhere** (deleted with the op block trim, not merely off): the opp-active
collapsed effect scalars, the opp-active collapsed incoming-secondary scalars, the outgoing
slp/psn/tox columns, and the lean top-K block. `damage_topk_k` now sizes the incoming matrix and
nothing else; `K > 0` without `damage_matrices_incoming` **raises** in both the extractor and the
op.

### 4.1 Per-block dependence — the current-config measurement

Source: [`research_state/measurements/gen3_op_block_dependence_6k.json`](research_state/measurements/gen3_op_block_dependence_6k.json)
— **`models/run_20260807_135637_gen3/checkpoints/checkpoint_9600000_steps.zip`, 6000 real eval
states, 2026-08-07.** ⚠️ That is gen-3, an EARLIER generation — not the production run. The
architecture surface it measured is the same family, but the numbers are a fact about that model,
and it was taken with the op's render tail still in the flat block (`op_drop_renders` off) — which
is why its `FULL_CONCAT` ceiling is 660 wide where production's is 138.
Method: zero
each sub-block as a contiguous slice of the op's output **at the `ProjectionAssembler` concat only**
(edges, the `prefuse_proj` injection and the pointer cells stay live) → masked KL against the
policy's own distribution. It answers *what does the HEAD still lean on*, not *what does the model
use*.

| Sub-block | Width | KL | Argmax flips | Shuffle-control KL |
|---|---|---|---|---|
| `FULL_CONCAT` (ceiling) | 660 | 0.2444 | 23.6% | 0.1818 |
| `incoming_matrix` | 522 | **0.2534** | **24.2%** | 0.1357 |
| outgoing single-active | 45 | 0.0176 | 5.4% | **0.0254** |
| incoming per-mon + CB | 85 | 0.0174 | 6.2% | 0.0158 |
| incoming per-mon | 72 | 0.0160 | 6.0% | 0.0130 |
| Choice-Band tail | 13 | 0.0013 | 1.4% | 0.0014 |
| status landing | 8 | 0.0006 | 0.8% | 0.0010 |

**Read the shuffle column first.** Shuffling a block across the batch preserves its marginal
statistics and destroys its state-specific content. For the outgoing single-active block, the
Choice-Band tail and status landing, the shuffle arm **meets or exceeds** the zero arm — at this n
those blocks show no dependence the probe can separate from noise. The one clean signal is the
`incoming_matrix`: zeroing it costs essentially the entire concat ceiling.

Two caveats that bound this: it is **mid-training** (9.6M of 40M, and edge dependence grew ~3× with
training in earlier generations), and it has not been re-run at end of run.

### 4.2 ⚠️ The older P1 table is NOT current — do not quote it

The frequently-cited per-block ablation table (`tmp/op_block_ablation_probe.py`, **2026-07-25**,
4000 real eval states, per-block zero → masked KL, ceiling 0.9385 = zeroing the whole op) was
measured on a **different model and a different config**, and §4.1 above supersedes it. Three things
make it non-transferable:

1. **It predates this generation entirely.** It was taken before the pointer-native action head
   existed, when the op block reached only a flat positional `action_net`. The current model routes
   the same numbers through per-action pointer cells *as well as* the concat, so "how much does the
   policy depend on block X" is a different question with a different mechanism.
2. **Two of the blocks it ranks do not exist here.**

| Block in that table | % of that run's ceiling | Exists in production config? |
|---|---|---|
| OUTGOING (per-action, un-collapsed) | 65.7% | ✅ yes (sub-block 3) |
| `outgoing_attacker_matrix` | 21.4% | ❌ **no** — the OAX flat block is deleted (v88); the kernel survives as `d2`'s engine |
| `incoming_matrix` (mon × move) | 15.4% | computed, but **not in the flat block** — `op_drop_renders` = **true** |
| incoming per-mon | 12.7% | ✅ yes (sub-block 1) |
| status-landing | 8.8% | ✅ yes (sub-block 4) |
| `outgoing_matrix` | 6.3% | computed, but **not in the flat block** — `damage_matrices_outgoing` = true, `op_drop_renders` = **true** |
| Choice-Band | 2.9% | ✅ yes (sub-block 2) |
| incoming effect (collapsed) | 1.2% | ❌ deleted from the code |
| incoming secondary (collapsed) | 0.1% | ❌ deleted from the code |

3. **Its headline is reversed by §4.1.** That table says the OUTGOING families dominate; the
   current-config measurement puts the outgoing single-active block at its own shuffle-control level
   and the `incoming_matrix` at the whole ceiling. Both cannot be true of the same model. The
   sub-block ordering is a fact about a model, not about the architecture.

Its raw output is not in version control anywhere (only the derived table in
`designs/learning/shortcut_learning_and_feature_delivery.md` survives), so it could not be archived
under `research_state/measurements/`.

---

## 5. Edge families — physics as attention bias

`edge_bias_families = "d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r"` — **all 17 families are on**
in the production config. Each maps its per-pair cell through a **zero-init**
`Linear(cell_width, 2 · n_heads)`: one head-set for `row→col`, one for `col→row`. Zero-init ⇒ the
whole edge system is bitwise-identical to `off` at initialisation.

Seat indices as in §2.3: our mons `[0:6]`, opp mons `[6:12]`, global `19`, E3 `[20:24]`,
E4 `[24:30]`, E5 `[30:36]`.

### 5.1 The from × to grid

| Family | Placed at (row, col) + transpose | Cell width | Cell contents |
|---|---|---|---|
| **d1** | E3 move seat *k* × opp mon *d* | 6 | `[low, high, crit, pko, type_mult, revealed]` — our active's move vs each opp mon |
| **s1** | E3 seat *k* × opp mon *d* | 2 | `[land, land·immob]` — will this status move land on that mon |
| **c1** | E3 seat *k* × opp mon *d* | 7 | `[is_boost, d_best_high, d_best_pko, d_outspeed, hp_cost, d_in_high, d_in_pko]` — post-setup deltas, offensive **and** defensive halves |
| **c2** | E3 seat *k* × opp mon *d* | 7 | `[is_status, land, d_their_outspeed, d_in_phys_high, d_sched, d_in_all_slp, e_slp_free_turns]` — what *landing* would do |
| **c3** | E3 seat *k* × opp mon *d* | 3 | `[is_recovery, d_in_pko, rest_sleep_turns]` — does healing beat their KO |
| **c5** | E3 seat *k* × **our** mon *j* | 4 | `[is_bp, d_best_high, d_best_pko, d_outspeed]` — Baton-Pass receiver axis |
| **c4** | E3 seat *k* × **global** (19) | 4 | `[is_protect, p_success, net_ours, net_theirs]` — the turn a successful Protect banks |
| **d3** | E4 threat seat *c* × our mon *i* | 5 | `[high, pko, eff, is_phys, w]` — their believed move vs each of our mons |
| **s3** | E4 seat *c* × our mon *i* | 3 | `[land, land·immob, w]` |
| **d2** | our mon *i* × opp **ACTIVE** (one-hot column) | 4 | `[best_high, best_pko, p_outspeed, alive]` — our bench's offense vs their active |
| **d4** | our mon *i* × opp mon *j* (active column pre-zeroed) | 4 | `[phys_high, spec_high, phys_pko, spec_pko]` — the opp **bench**'s believed threat |
| **v** | our mon *i* × opp mon *j* | 3 | `[p_outspeed, both_alive, revealed_j]` |
| **h** | our mon *i* × opp mon *j* | 5 | `[switch_ins, attacks, status_clicks, shared_field_turns, pairing_recency]` — obs-fed pair-history TENDENCIES (`gen3_pair_history_v1`; EpisodeTracker-folded, log-saturated; **IN the production families string** since gen-12 — the one family whose cell the GPU cannot recompute, since it IS compiled battle history) |
| **r** | event seat *e* (the LAST-N tokens) × mon *m* (all 12) | 3 | `[is_actor, is_target, is_rel]` — STRUCTURAL reference edges (`gen3_event_ref_edges_v1`, Tier H-C; `is_rel` added by `gen3_event_record_v2`): event *e*'s recorded actor/target/REL mon IS mon *m* (species-num equality, side-gated against mirror false-links — actor on the row's side, target on the other, REL on its own `REL_SIDE`; `_event_reference_cells`, pure). **IN the production string** — requires `--history-events`, which is ON (the seats are the rows) |
| **t** | our mon *i* × opp mon *j* | 2 | `[P(i traps j), P(j traps i)]` |
| **x** | each mon × **global** (both sides) | 4 | `[entry_chip, pursuit_p, pursuit_eff, grounded]` |
| **g** | each mon × **global** (both sides) | 4 | `[leftovers, weather_chip, status_tick, leech]` — signed maxhp fractions, Toxic at its ramped next tick |

**No family targets the E5 tail seats** — they are token content only.

### 5.2 Requirements (enforced at extractor build, `ValueError`)

| Families | Require |
|---|---|
| d1, s1, c1, c2 | `damage_op` **and** `damage_outgoing` |
| c3, c4, c5, d2, d4, g, t, v | `damage_op` |
| x | `damage_op` |
| d3, s3 | `entity_topk_seats > 0` (the bias rows *are* the E4 seats) |
| r | `history_events` (the bias rows *are* the H-B event seats) |

All are satisfied in the production config, whose string carries every family including `h` and `r`
(`r`'s requirement holds because `history_events` is ON).

### 5.3 What an edge can and cannot carry

Attention weights are softmax-**normalised**, so an edge bias moves *who attends to whom*: what it
writes is a **ratio within its row**, not an absolute magnitude ("53% of max HP"). The two channels
that can carry an absolute are **token content** (`prefuse_proj`) and **per-action pointer cells**
(§3.3). This is a capacity/conditioning argument, not an impossibility proof; the reasoning is in
`designs/learning/shortcut_learning_and_feature_delivery.md`.

### 5.4 Edge-family audit — an EARLIER generation's measurement

Source: [`research_state/measurements/gen3_edge_family_audit_9p6M.json`](research_state/measurements/gen3_edge_family_audit_9p6M.json)
— **`models/run_20260807_135637_gen3/checkpoints/checkpoint_9600000_steps.zip`, 6000 eval-trace
states, 2026-08-07** (gen-3, NOT the production run), produced by
`src/agents/model/edge_ablation_audit.py`. Each row zeroes one family's bias map and measures masked
KL / argmax flips / |dV| against the unablated policy.

| Family | KL | Argmax flips | \|dV\| |
|---|---|---|---|
| `d2` (bench offense → their active) | 0.0426 | 7.6% | 1.308 |
| `d1` (our moves → their mons) | 0.0345 | 6.0% | 0.274 |
| `v` (speed) | 0.0035 | 2.9% | 0.651 |
| `d3` (their threats → our mons) | 0.0013 | 1.9% | 0.141 |
| `d4` (their bench threats) | 0.0015 | 1.1% | 0.492 |
| `s3` | 0.0003 | 0.9% | 0.063 |
| `s1` | 0.0017 | 0.8% | 0.022 |
| `t` (trapping) | 0.0003 | 0.7% | 0.121 |
| `c1` (setup consequence) | 0.0002 | 0.4% | 0.015 |
| `c2` (status consequence) | 0.0007 | 0.4% | 0.017 |
| `x` (entry/exit) | 0.0000 | 0.3% | 0.036 |
| `c3` (recovery) | 0.0002 | 0.2% | 0.018 |
| `c5` (Baton Pass) | 0.0000 | 0.2% | 0.018 |
| `g` (end-of-turn ledger) | 0.0000 | 0.1% | 0.016 |
| `c4` (Protect) | 0.0000 | 0.1% | 0.001 |
| **all families off** | 0.1011 | **13.9%** | 1.857 |
| **op head-concat off** | 0.2444 | **23.6%** | 5.669 |
| **concat + pointer cells off** | 0.6369 | **37.8%** | 5.669 |

Two things to carry from this, both **mid-training (9.6M of 40M)** and provisional:

- **The OUTGOING damage families dominate** (`d2`, `d1`), the same ordering gen-1 and gen-2 showed;
  every consequence family (`c1`–`c5`) and the board-level `g`/`x` are at or below 0.4% flips so
  far. Edge dependence grew ~3× with training in earlier generations, so a low number here is not
  yet a verdict on a family.
- **The concat is not starved by the edges, and vice versa.** Zeroing the head concat flips more
  actions (23.6%) than zeroing the *entire* 15-family edge system (13.9%) — replicated in gen-1,
  gen-2 and gen-2.5 (`research_state/measurements/README.md` has the cross-run table). That is the
  expected result if the two carry different things: a bias is a softmax-normalised ratio, the
  concat is an absolute.

Earlier generations' audits are archived alongside for the cross-run comparison. **The gen-1 and
gen-2 `v` rows were measured on the speed-stat GIGO bug** (§8) and describe the buggy feature.

---

## 6. Flags — production value and status

The flag/status and loss-coefficient tables are GENERATED from `designs/production_config.json`
resolved against HEAD (`python -m agents.model.arch_tables`; drift pinned by
`arch_tables_test.py`) — a hand-derived version of this table went stale twice within one week of
generation turnover, and its stale rows are precisely what mis-briefed downstream readers.

**Status legend** — `ACTIVE`: on and doing work. `OFF`: not enabled. `INERT`: nominally set but
does nothing given another setting.

<!-- BEGIN GENERATED: flag-table -->
| Flag | Production value | Status |
|---|---|---|
| `attend_unrevealed_opponents` | `true` | ACTIVE |
| `belief_grad_mode` | `"shaping"` | ACTIVE |
| `belief_tokens` | `"blob"` | OFF |
| `conditional_threat_cell` | `true` | ACTIVE |
| `consequence_topk` | `6` | ACTIVE |
| `damage_candidate_k` | `0` | OFF |
| `damage_matrices_incoming` | `true` | ACTIVE |
| `damage_matrices_outgoing` | `true` | ACTIVE |
| `damage_op` | `true` | ACTIVE |
| `damage_outgoing` | `true` | ACTIVE |
| `damage_topk_k` | `6` | ACTIVE |
| `edge_bias_families` | `"d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r"` | ACTIVE |
| `entity_tail_seats` | `true` | ACTIVE |
| `entity_topk_seats` | `6` | ACTIVE |
| `history_events` | `true` | ACTIVE |
| `hp_belief_mode` | `"composed"` | ACTIVE |
| `intent_conditional` | `true` | ACTIVE |
| `intent_move_cell` | `true` | ACTIVE |
| `intent_threshold` | `true` | ACTIVE |
| `item_belief` | `true` | ACTIVE |
| `move_belief_mode` | `"both"` | ACTIVE |
| `move_candidate_floor` | `0.02` | ACTIVE |
| `move_latent` | `true` | ACTIVE |
| `move_prior_fusion` | `true` | ACTIVE |
| `op_believed_lean` | `true` | ACTIVE |
| `op_drop_renders` | `true` | ACTIVE |
| `opp_belief_cls_k` | `6` | ACTIVE |
| `opp_belief_slots` | `true` | ACTIVE |
| `opp_intent` | `true` | ACTIVE |
| `opp_intent_grad_mode` | `"detached"` | ACTIVE |
| `pair_outcome_cell` | `true` | ACTIVE |
| `pair_outcome_switch` | `true` | ACTIVE |
| `ridealong_adv` | `0` | OFF |
| `ridealong_ensemble` | `0` | OFF |
| `ridealong_opp` | `0` | OFF |
| `ridealong_rnd` | `false` | OFF |
| `ridealong_rnd_variants` | `"off"` | OFF |
| `species_prior_fusion` | `true` | ACTIVE |
| `spread_belief` | `true` | ACTIVE |
| `spread_belief_nature` | `true` | ACTIVE |
| `switch_branch_cell` | `true` | ACTIVE |
| `t0_species_prior` | `true` | ACTIVE |
| `value_entity_pool` | `true` | ACTIVE |
| `value_entity_pool_full` | `true` | ACTIVE |
| `value_threat_inject` | `true` | ACTIVE |
| `win_prob_mode` | `"shaping"` | ACTIVE |
| `hp_type_belief_coef` | `0.05` | ACTIVE |
| `intent_label_bot_weight` | `0.25` | ACTIVE |
| `item_belief_coef` | `0.05` | ACTIVE |
| `move_belief_coef` | `0.05` | ACTIVE |
| `move_belief_latent_coef` | `0.05` | ACTIVE |
| `opp_belief_aux_coef` | `0.05` | ACTIVE |
| `policy_grad_coef` | `1.0` | ACTIVE |
| `spread_belief_coef` | `0.05` | ACTIVE |
| `vf_coef` | `0.5` | ACTIVE |
<!-- END GENERATED: flag-table -->

**`--oracle-reveal {off,species}` is not a row of the generated table above:** it builds no module and moves no weight, so it is
a `resume_immutable` OBSERVATION-MODE flag (config v137), `off` in production and never on the ARCH surface. It is recorded in
`model_config.json` and `metadata.json`, inherited by a flagless resume, refused if a resume flips it (`check_oracle_reveal`),
and shown by `main.checkargs` and the launch banner. `main.h2h`, `main.anchors`, `main.play`, `main.belief_roles` and
`main.policy_spectrum` refuse a checkpoint that records `species` (they build observations without the reveal). It is refused
with the fork arm. See §1.2 for what the row carries.

### 6.3 Reward config (resume-immutable, `check_reward_config`)

**The production reward is ONE TERMINAL TERM.** `terminal_indicator` **true** · `victory_value`
**1.0** · `draw_penalty` **0.0** · γ **1.0**. The composition is **1 TERMINAL (`win_loss`) + 0 PBRS +
0 BIAS**: `+1.0` on a win, `0.0` on a loss, a draw and a 250-turn timeout alike. `draw_penalty` is
INERT under the indicator (the resolved reading `inert_reward_flags` names, written beside the
values in `model_config.json` and in `metadata.json`'s `reward_composition` block).

**It is the ONLY reward the code has.** The hand-shaped reward path — eight PBRS potentials, ~25
BIAS terms, the bias-additivity refund, the no-progress tax — and its 14 recorded fields were
DELETED (`gen3_shaped_reward_deletion_v1`, config v122, 2026-09-26; the fields are listed in
`designs/deleted_flags.md`). Without the indicator terminal (an old checkpoint that recorded none) the terminal is SIGNED: `±victory_value`,
a pre-cap tie scoring as a loss, and `draw_penalty` (default −35) at the 250-turn cap.
`reward_golden_test` was recorded at the last pre-deletion commit (`029cee83`) and passes unchanged
after it — the measured proof that production's reward, and the `win_margin` training-only obs key
(`material_margin.py`, the material balance that used to be Φ_mat's by-product), did not move.

**Resume:** every field here is enforced on the training-resume path only (a flagless resume of a
win-indicator run under the signed defaults FATALs and the error names the flags to re-pass).
🚨 **A resume or FORK of a checkpoint trained WITH the shaped reward REFUSES LOUDLY**
(`agents.model.model_version.shaped_reward`, enforced in `resolve_config` and `main.checkargs`,
typed `ShapedRewardCheckpointError`): it would otherwise continue on the terminal alone under the
same run name. Run it pinned to ≤ `029cee83`. Frozen eval / pool opponents are
unaffected — `check_compatible` excludes reward fields and `_migrate_config` pops the deleted ones,
because a frozen forward never reads the reward.

### 6.4 Runtime knobs (never versioned, must be re-passed on every resume)

The Rust bridge and the Rust env core (the only transport and the only core since deletion pass U3 deleted the Python env core; their flags `--use-bridge` / `--env-core` were deleted, P11b, so there is nothing to re-pass; `designs/production_config.json` `recipe.sizing` holds the sizes, the M5 switch `gen3_env_core_switch_v1`; on a `--model` launch a python-era checkpoint (produced on `python`, or before the core was stamped) moves onto `rust`, announced as a CORE SWITCH, and one that trained the SHAPED critic is REFUSED (`FATAL_CONFIG` — run it pinned; deletion pass D4)) · `--compile-trainer` (ON by default for cuda) · `--grad-accum-steps` at whatever `--batch-size` the run
uses · `--grad-checkpointing`. **Matmul precision is not a knob: fp32 `highest` (full
FP32, no TF32 — PyTorch's default) is the only precision** (TF32 was retired and `--matmul-precision` deleted,
deletion pass K2); `metadata.json` records the realized value as provenance, every parity gate refuses any
other, and a run that recorded `high` resumes pinned or not at all. **The CUDA learner
compile is ONE DECLARED REGION** — R1, the micro-step (`evaluate_actions` + the fold), `fullgraph=True`
at its one declared signature, `batch_size` rows (`agents/model/compile_regions.py`,
`designs/training/compile_flags.md` "K8 — DECLARED COMPILE REGIONS"; the rollout forward is NOT a
learner-compiled region — the Rust collector serves it through the T2 inference service, and the
learner's own `policy.forward` is eager); the startup region gate holds it to eager on the K9 learner
golden's real labelled rows (loss and every policy gradient), then the compile sentinel
(`gen3_compile_sentinel_v1`) prewarms the declared signature and LOCKS before the first iteration: a
later recompile or a dynamo cache-limit hit (a silent eager fallback) exits `FATAL_CONFIG`. **The code runs on torch >= 2.8 only**
(`src/utils/torch_floor.py`, 2026-10-02): torch 2.8 (`gen3ai_torch28`) is the interpreter for every
new run, and a run that recorded torch 2.5.1 (or none) resumes PINNED to its own commit on
`gen3ai_stable` (`src/main/launcher/torch_runtime.py`). Eager, the CPU compile and the weights are
unchanged either way. On torch 2.5.1 the whole extractor as one CUDA Inductor graph miscompiled on
real observations (argmax agreement 70.9%, gradient cosine 0.778 vs eager, measured 2026-09-28 on
`ai_v14_01_base`; every default cuda run from 2026-08-17, `28eaef29`, to 2026-09-28), which a trunk
split worked around there; on 2.8 the unsplit graph passed (eval and train) and the
split was deleted with HEAD's 2.5.1 support. **UNVERIFIED:** the root-cause op. These do not appear in `model_config.json` and
are **not** inherited on resume — with the compile flags defaulting ON it is the OPT-OUT that must
be re-passed each launch, not the flag.

⚠️ **`gamma` is not in `model_config.json` either.** It is a constant of the namespace (γ 1.0; no flag) and the
resume path restores the checkpoint's own γ, so the value in force is visible in `metadata.json`'s
`cli_args` and in the startup lines — not in the mirror, and therefore not in §6's table.

**The policy's GAE λ is 0.80** (`--policy-gae-lambda`, `gen3_policy_gae_lambda_v1`): the λ of the
advantages the clipped surrogate trains on — the critic's BCE target is the terminal outcome (§3.4, *WHICH readout is the critic*). It was a hardcoded literal until config v123;
from v123 it is a recorded, `_resolve`-inherited training field (`policy_gae_lambda` in
`model_config.json`). The mirror is at v122 and so does not carry the key; every run this code can
load (config ≥ v121) trained at the hardcoded 0.80 (older eras ran 0.95 and 0.85 — `eff7ddee` set 0.80).

---

## 7. Training-only obs keys — the leak-safety list

These are Dict-obs keys the Rust env core emits for supervision (declared by `agents/training/trainee_spaces.py`). **The forward reads only
`obs["observation"]`** (`ObsUnpack.forward`), so none of them can reach `pi`/`vf` or any pointer
logit. Declared conditionally, so a key absent from the space is simply not emitted.

| Key | Shape | Consumer | Emitted when | In production? |
|---|---|---|---|---|
| `belief_species` | int64 `[6]` | `BeliefHead` species CE | `opp_belief_aux_coef > 0` **or** `move_belief_mode != off` | ✅ emitted and consumed (`opp_belief_aux_coef` 0.05) |
| `belief_moves` | int64 `[6,4]` | `BeliefHead` moves BCE (Hungarian) | " | ✅ emitted and consumed |
| `known_moves` | int64 `[6,4]` | `MoveBelief` BCE | `move_belief_mode` ∈ {revealed, both} | ✅ emitted and consumed (`move_belief_coef` 0.05) |
| `belief_spread` / `belief_spread_mask` | f32 `[6,5]` / `[6]` | `SpreadBelief` regression | `spread_belief` **and** `spread_belief_coef > 0` | ✅ emitted and consumed |
| `belief_nature` / `belief_nature_mask` | int64 `[6]` / f32 `[6]` | nature CE — agent2's TRUE declared nature, guarded against its stats; mask == `belief_spread_mask` (`gen3_true_spread_labels_v1`) | " | ✅ (`spread_belief_nature` true) |
| `belief_ev` / `belief_ev_mask` | f32 `[6,5]` / `[6]` | EV smooth-L1 — the declared EVs at `4·⌊ev/4⌋` | " | ✅ |
| `hp_type_label` / `hp_type_mask` | int64 `[6]` / f32 `[6]` | HP-type CE | `move_belief_mode != off` **and** `hp_belief_mode == composed` **and** `hp_type_belief_coef > 0` | ✅ **emitted and consumed** |
| `item_label` / `item_mask` | int64 `[6]` / f32 `[6]` | item CE (`gen3_item_belief_v1`) | `item_belief` **and** `item_belief_coef > 0` | ✅ emitted and consumed (`item_belief_coef` 0.05) |
| `win_target` / `win_mask` / `win_margin` | f32 `[1]` each | the win-prob head's BCE — under the win-prob critic (the only critic) **the value loss itself** (MC outcome, a **future** label back-filled by the Rust collector — `rust_rollout/store.py` `fill_complete`) | `win_prob_mode != none` | ✅ **emitted and consumed — this is the critic's target** |
| `opp_action_kind` / `opp_action_num` / `opp_switch_slot` / `opp_switch_species` | int64 `[1]` each | opponent-intent CE (`gen3_opp_intent_v1`) — what they did at the PREVIOUS decision, shifted one row back in `train()` | `opp_intent_coef > 0` | ✅ emitted and consumed (`opp_intent` true ⇒ `--arch production` sets `opp_intent_coef` 0.05; recorded in `model_config.json` from config v125, so a flagless resume or a launcher restart inherits it) |
| `opp_class` | int64 `[1]` | **two consumers**: the intent metrics, which it SPLITS (bot / pool / stable / exploiter — one pooled intent accuracy over random bots, heuristics and frozen selves cannot be read); and the training-side value sidecar's per-class calibration slice | `opp_intent_coef > 0` **or** `win_prob_mode != none` | ✅ emitted (both gates hold), read by the intent metrics and the sidecar, not by any loss |
| `fork_pg_m` | f32 `[1]` | the policy term's per-row mask (`gen3_fork_v1`; placeholder 1.0) | `--fork-fraction > 0` | ❌ |

🚨 **In this config a privileged key is no longer merely auxiliary — `win_target` IS the critic's
training target.** The leak-safety property is unchanged and is exactly what makes that safe: the
forward reads `obs["observation"]` alone, so a future outcome can label the value head without ever
being visible to it at decision time. But the consequence for reasoning is real — "training-only"
now means "not in the forward", never "not load-bearing".

Every belief label above is both emitted AND consumed here (all six supervised coefficients are at
0.05), and so are the four intent labels (`opp_intent_coef` 0.05), so the whole emitted set is read:
**21 keys** (the list, each key's producer, consumer and where the Rust env gets it:
`src/utils/rust_env/label_inventory.py`, pinned against the declared trainee space by
`agents/training/rust_env_label_inventory_test.py`, which also reads this table's ✅/❌). **Do not infer supervision from emission**, though: the
emit gates and the loss coefficients are separate conditions, and a config that drops a coefficient
to 0 keeps paying the buffer cost while training nothing — which reads identically in every metric.
`--fork-fraction` is off, so its key (`fork_pg_m`) is not emitted at all.

🚨 **`opp_class` IS EMITTED UNDER TWO GATES, AND THAT IS DELIBERATE** (`gen3_value_sidecar_v1`,
2026-09-08). It used to ride the intent labels alone. A win-prob arm normally runs with no intent
loss, so the one instrument that reads the class per state — the training-side value sidecar — had
an empty by-opponent-class table on precisely the runs it exists for. Widening the gate cannot
change a forward pass (the key is never read by `ObsUnpack`), and `train()`'s one-ahead intent
SHIFT is still gated on `opp_intent_coef > 0`, so a win-prob-only run carries the env's own
per-episode value with no shift applied. This is the one row in the table whose production consumer
is a DIAGNOSTIC rather than a loss.

Only the **trainee** carries any of these. Eval and self-play opponents play through
`RLPlayer` (and, in training, the inference service's slots), which never construct them.

**Where the trainee's `observation` row comes from:** the Rust env core builds it from the trainee's
own per-side stream (parse -> reading -> view -> trackers -> encode) — the only source since the Python
env core, whose `Gen3Env` could also encode it in the env worker, was deleted (deletion pass U3). Every
key in the table above is computed in the core from both sides' readings and the engine board, or filled
by the host (`label_inventory.py` says which, per key), and the leak-safety property is unchanged — the
forward reads `obs["observation"]` alone.

**Which env runs the rollout:** the Rust env core, the only one (no flag selects it;
`designs/production_config.json` `recipe.sizing` holds its sizes; the M5 switch `gen3_env_core_switch_v1` made it the
production core, ledger *THE M5 SWITCH*, and deletion pass U3 deleted the Python core). Its label keys come
from the core, the host or a refusal (the inventory of record: `src/utils/rust_env/label_inventory.py`,
`designs/rust_sim/env_labels.md`).

Two side-channel stashes are also never fed forward: `last_belief_target_latent` (computed only
under `torch.is_grad_enabled()`) and `last_move_latent_table`. The pinned no-leak tests are
`belief_slots_test.test_latent_target_is_no_leak`,
`damage_op_test.test_op_is_leak_free_of_privileged_keys`, the bridge fuzz
`poke_env_gaps/belief_labels_fuzz_test.py`, and — as a **graph invariant** —
`delivery_graph_test.test_no_aux_edge_reaches_the_forward`, which asserts that no `aux` edge
terminates at `pi_projection`, `vf_projection`, or any pointer logit.

---

## 8. Known contradictions between the old prose and the code

> **Update 2026-08-14:** several entries below are FIXED by the ctx-dedup / OpTensors /
> generated-tables pass: this file's header and flag/head tables no longer hand-state config
> values (generated from `production_config.json`), the delivery graph no longer draws the dead
> op→head concat edges (it drew them for five days after the v61 deletion, pinned by its own
> test — the exact rot class it exists to prevent), and `designs/CLAUDE.md`'s state table was
> brought to gen-9/v76, and the `constants.py` stale offset comments are deleted. (The
> observation leaf's opening had already been fixed separately — item 2 below is resolved.)
> The list below is kept as found (2026-08-08) for the record.

Found while deriving this file (2026-08-08). Each is a place where a doc asserted something the
code does not do. None are fixed in `src/` by this pass — they are recorded so the next reader does
not re-derive them.

1. **Root `CLAUDE.md` reactive-block prose was two revisions stale.** It described "the 414-dim
   reactive block (**19 scalars**)" with `turns_since_progress` at `vec[14]`, protect-odds at
   `vec[15]`/`vec[16]`, wish at `vec[17]`/`vec[18]`. Real: **311 dims, 11 scalars**, progress at
   offset **6**, protect at **7**/**8**, wish at **9**/**10**. The summary table 60 lines above it
   was correct — the table and the prose contradicted each other in the same section.
2. **`src/agents/observation/CLAUDE.md` opens with "2889-dim"**; the live obs at audit time
   was **2925**; since `gen3_entity_rehome_v1` it was **2667**, and since
   `gen3_deadline_clock_v1` it was **2669** (the
   per-mon recency block added 12 × 3). ⚠️ **Every obs dim in this section is AS-FOUND in 2026-08;
   live is 2761** — see §1. Its per-block reference section then describes the
   pre-deletion 414-dim reactive layout and the 51-dim incoming-damage / 44-dim move-effect blocks
   as if present. Its own inline banner says to treat the deletion note as authoritative — i.e. the
   file tells you not to trust the rest of the file.
3. **`src/agents/model/CLAUDE.md` states `ARCH_SIGNATURE` in three places with two different
   values** (`gen3_op_block_trim_v1` in the phase-structure rules, `gen3_edge_bias_trunk_v1`
   later) and states `MODEL_CONFIG_VERSION` as 31/32/37/38/40/41/43/44/45/46/47/53/55/57 in
   different paragraphs. Live at audit time: **59** and `gen3_edge_bias_trunk_v1` (now **60** /
   `gen3_entity_rehome_v1`). It also contains a duplicated,
   partly-conflicting pair of paragraphs (two "`MODEL_CONFIG_VERSION` was **38** at v38" endings).
4. **Root `CLAUDE.md` claimed `MODEL_CONFIG_VERSION` = 57.** Live at audit time: **59** (58 = the
   speed-stat GIGO stamp, 59 = `consequence_topk`; now 60 = the re-home stamp). Neither v58 nor
   v59 was described anywhere in the root file.
5. **`src/agents/model/CLAUDE.md` describes `ObsUnpack` as peeling "the flat 3390-dim
   observation"** — obs-layout generations out of date (2669 at audit time; **2761 live**).
6. **`PointerNativeActionHead`'s docstring says the move cell is `[low,high,crit,pko,p_land,known,
   sec×10]`.** It is `sec×7` (`_PTR_MOVE_CELL` = 13, not 16) since the outgoing slp/psn/tox columns
   were dropped. `pointer_cells`' own docstring, 900 lines away, says 7 correctly.
7. **`constants.py`'s `OFFSET_*` trailing comments are stale** (`# 642`, `# 1284`, `# 1400`,
   `# 1418`, and "base dim = 1790, full obs = 3391"). The expressions are correct; only the
   comments are wrong — so anyone grepping for an offset by eye gets the wrong number.
8. **`designs/CLAUDE.md` named the active run as gen-2 `run_20260805_060807`.** The live run is
   gen-3 `run_20260807_135637_gen3`. Corrected in this pass.
9. **The speed-stat GIGO is fixed in code but its consequence is under-flagged in docs.**
   `pairwise_speed` (edge `v`) and `pairwise_boost`'s outspeed channel read stat index 4 — Special
   Defense — as "speed" through two trained generations. Fixed 2026-08-06 with named `_BS_*` /
   `_NAT_*` indices and a discriminating regression test. Consequence: **every `v`-edge and
   C1-outspeed number measured before 2026-08-06 describes the buggy feature**, including the
   gen-1 audit in §5.4.

---

## 9. Where to look next

| Question | File |
|---|---|
| **This document as a clickable digraph** — the **120 nodes / 1103 edges** above (counted 2026-08-23 from `delivery_graph_snapshot.json`, which the viewer is built from — read it there rather than trusting this cell), hue-coded by what each channel physically carries, with a per-checkpoint measured-dependence overlay and a path filter (pick `vf_projection` to see exactly what the critic reads) | **https://model.g5d.io** (served live from the workstation checkout, so it is never a stale copy), or `designs/architecture_viewer.html` via `file://`. **Generated — never hand-edit it**: rebuild with `python -m agents.model.build_arch_viewer`, and `--check` fails if the committed artifact has drifted from the graph. |
| Obs-build performance gate (mandatory benchmark) + per-slot detail | `src/agents/observation/CLAUDE.md` |
| Phase contract, `ExtractorContext`, versioning playbook | `src/agents/model/CLAUDE.md` |
| How it got here — every version entry, verbatim | `designs/CHANGELOG.md` |
| Which `ai_vN` folder is relevant | `designs/CLAUDE.md` |
| Hypothesis status / what has been killed | `designs/research_state/ledger.md` |
| **The raw audit outputs behind every measured number here** | `designs/research_state/measurements/` (+ its README for how to read one) |
| Delivery-channel theory (edge vs content vs cell) | `designs/learning/shortcut_learning_and_feature_delivery.md` |
| Event-sourced battle layer + read-models | `src/agents/battle/CLAUDE.md` |
| Training loop, eval sharding, ELO | `src/agents/training/CLAUDE.md` |
