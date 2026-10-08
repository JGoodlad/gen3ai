# Entity coverage audit: is every gen-3 battle fact reachable in the entity trunk? (2026-10-06)

**Status: AUDIT, read-only. Nothing in it is built.** Asked by the owner on 2026-10-06: *"audit our global
attention to make sure everything is appropriately represented in an entity based world"*. It serves the
**static-token rebuild** the owner approved the same day ([`design_arch_audit.md`](design_arch_audit.md) Decision
record, 2026-10-06: per-mon tokens become static identities, per-mon dynamic state becomes a small separate per-mon
input, and board context reaches tokens only through attention and the edges). The question this doc answers for
that rebuild: **does every board fact have a home that attention or an edge can read, in an entity-appropriate
form, once it leaves the per-mon encoder?**

Audited at `079dee3e` against [`../ARCHITECTURE.md`](../ARCHITECTURE.md), the sorting rule in
[`../learning/entity_tokens_biases_pointers.md`](../learning/entity_tokens_biases_pointers.md) (Part 3, §6.1, §6.9),
[`design_arch_audit.md`](design_arch_audit.md) (F4, F12, F13, F19, §4),
[`design_x5_belief_tokens.md`](design_x5_belief_tokens.md), [`obs_enrichment_backlog.md`](obs_enrichment_backlog.md),
`src/agents/observation/CLAUDE.md` and the code (`agents/observation/{global_env,active_context,gen3_effects,wish_belief}.py`,
`agents/model/{encoders,team_transformer,pointer_head,hypothesis_encode,damage_op_blocks,damage_op_pairwise}.py`,
`agents/model/extractor_ctx.py`). Gen-3 mechanics were checked in the pinned `deps/pokemon-showdown` (`e0551883`),
resolving the mod chain gen3 → gen4 → … → base the way `sim/dex.ts` merges it. Tags: **VERIFIED** (read at the
cited source), **MEASURED** (a number with its source), **UNVERIFIED**. No code changed; no GPU; no new measurement.

---

## 1. Summary for the owner (one page)

**The short answer.** Every board fact the observation carries today is reachable by the network, but mostly
because it is **copied into every Pokémon's encoder** (and every move encoder) as well as the one global token. Take
that copy away, as the static rebuild does, and three kinds of fact lose their good route:

1. **Counts.** How many Pokémon each side has lost cannot be recovered by attention, because fainted Pokémon are
   *masked out* of attention (nobody can look at them). Today the count rides the global token and the broadcast.
   It needs an explicit home.
2. **"Mine vs theirs".** The global token stores board facts in a fixed (ours, theirs) order. A Pokémon reading it
   must work out which half is *its own side's* Spikes or screens. Today the per-mon encoder does this with no side
   input at all (it infers the side from the "spread known" flag, which is 1 only for our own mons). A token per
   SIDE makes it structural: each Pokémon reads its own side.
3. **Amounts.** An edge bias can only change *who listens to whom* (a ratio); it cannot write "this switch-in takes
   25% from Spikes" into a token. Today a token can compute that amount from the broadcast Spikes count and its own
   types. After the rebuild the only amounts that reach a token as content are the damage operator's incoming rows,
   and only on OUR six tokens. The switch decision has no Spikes number at all in its per-action cell.

**What the model never sees** (ranked in §5): what the opponent has seen of our team; how many turns are left on
Reflect / Light Screen / Safeguard / Mist (the data is already held and thrown away); how long Encore, Taunt,
Disable, confusion and partial traps have run; whether the opponent is Choice-locked; a pending Future Sight or Doom
Desire once its user switches out (it is stored on the wrong Pokémon); a two-turn charge (Fly / Dig). **And one
physics error:** the damage operator's "will this status land" ignores Safeguard in both directions, and incoming
Sleep Clause, our Substitute, Freeze Clause and Yawn, so it can claim a status lands when the game forbids it.

**The recommended layout for the static rebuild (§6.1).** Replace the single global token with **three board
tokens**: OUR SIDE, THEIR SIDE (the same projection for both, side-relative content: Spikes, screens with turns
left, Safeguard, Mist, a pending Wish, a pending Future Sight, how many mons are alive / fainted / revealed, Sleep and
Freeze Clause state) and FIELD (weather with turns, the clock, the no-progress counter). Retarget the three edge
families that point at the global seat today (`x` to the mon's own side, `g` and `c4` to the field), and give every
Pokémon an **op-derived content input** (its incoming damage, Spikes chip on entry, end-of-turn HP change) for both
sides, so amounts survive. 61 → 63 tokens (64 under X5).

**Counts.** 71 inventory rows (§3). Recommendations: **KEEP 11 · EXACT REFACTOR 2 · BEHAVIOUR CHANGE 9** (§6.3).

---

## 2. Route legend, and the board routes today

| code | route | what it carries |
|---|---|---|
| **T** | a per-mon slot field → `PokemonEncoder` → that mon's role token | content, per mon |
| **Tb** | a ROW-LEVEL board input broadcast into all 12 role encoders (`encoders.py`, `global_context`) and, for clock / weather / fainted / Spikes, into all 48 move-network rows (`move_context`) | content, the same vector in every mon |
| **E2** | each side's 60-dim active context (boosts + volatiles) scattered onto its ACTIVE mon's role-encoder input | content, active only |
| **G** | the global token, seat 12: `global_proj([our_ctx, opp_ctx, non_matchup_rest])` (`team_transformer.py`) | content, ONE key |
| **S3 / S4 / S5** | the entity seats: our active's moves (request order) / their top-K believed moves / the beyond-top-K tails | content |
| **EV** | the 32 event seats (29–60) and their `r` reference edges | content + structural edge |
| **edge:f** | edge family *f* (`d1 … r`, ARCHITECTURE §5) | a softmax RATIO only |
| **OP** | an input the damage operator reads as a kernel MODIFIER | physics, reaches tokens only through the op's outputs |
| **PF** | `prefuse_proj`: the op's incoming per-mon rows added as content to OUR six tokens only | content, our side |
| **CELL** | a per-action pointer cell (move cell / switch cell) | content at the logit |
| **H** | head-only: `non_matchup_rest` (25 scalars) straight into the policy projection; the critic's entity pool reads the refined global token | content, outside the trunk |
| **—** | MISSING | |

**F4's routes, counted from the code (VERIFIED, `encoders.py:229-354`, `team_transformer.py:362-365`,
`ARCHITECTURE.md` §3.1).** The broadcast context of the role encoder is `[clock 3, weather 7, fainted 2, Spikes 2,
screens 8]` = 22 dims; the move network's is the same minus screens plus the mon's own HP. The global token's input is
`2 × 60 + 25` = 145 dims through one `Linear(145, 128)`.

| board fact | routes today | count |
|---|---|---|
| weather (type, permanent, turns left) | Tb role · Tb move-net · G · H · OP (BP, `g` weather chip, `c3` heals) | 5 |
| Spikes layers per side | Tb role · Tb move-net · G · H · OP (`x` entry chip) · EV (`LAYERS` column) · CELL (`spin_denied`, `spin_value_lost`) | 7 |
| clock (3 scalars) | Tb role · Tb move-net · G · H | 4 |
| fainted (ours, theirs) | Tb role · Tb move-net · G · H | 4 |
| screens / Safeguard / Mist presence | Tb role · G · H · OP (Reflect, Light Screen only) | 4 |
| boosts + volatiles (active) | E2 · G · OP (boosted stats, Substitute, sports, Leech Seed) | 3 |
| Wish pending, no-progress counter | G · H | 2 |

**One fact about the per-mon encoder that matters for the rebuild (VERIFIED, `encoders.py:184-354`).** The shared
`PokemonEncoder` takes no side input. The token-type embedding (ours / theirs) is added later, in the trunk. Inside
the encoder, a mon's side is recoverable only from slot content: `spread_known` (1 for our mons, 0 for theirs) and our
active's trapping bits. So today each mon reads the broadcast `(fainted ours, fainted theirs)` and `(Spikes ours,
Spikes theirs)` through a side inferred from a flag that also changes meaning under `--oracle-reveal full` (a
diagnostic mode that sets the opponent's `spread_known` to 1). This is the "mine vs theirs" problem in §4, A2.

---

## 3. Inventory: where every fact lives today, and where it should live

Each row: the fact · the gen-3 rule where the mechanics matter (source) · TODAY (routes, §2 codes) · the entity
HOME it should have under the static design · why, and the status. Status words: **OK** (right home), **DUP**
(right home plus duplicates, F4), **PARTIAL** (only part of the fact), **WRONG** (on the wrong entity or wrongly
priced), **MISSING**.

The homes used: **STATIC** (the per-(species, set) table token the owner specified) · **DYN** (the small per-mon
dynamic input) · **OPC** (op-derived per-mon content, recommended in §6, B2) · **SIDE** (an attribute of the OUR SIDE
/ THEIR SIDE token) · **FIELD** (an attribute of the field token) · **EV** · **edge** · **OP** (stays a kernel
modifier) · **CELL** · **S3 / S4**.

### 3.1 Per-mon static (the set)

| # | fact | today | home | why / status |
|---|---|---|---|---|
| 1 | species + 6 base stats | T · OP | STATIC | arity 1, entity-invariant. OK |
| 2 | types | T (poke-env `_temporary_types`, so Conversion / Color Change rewrite it) · OP | STATIC + a DYN type override | species-static except a typechange (VERIFIED: `gen3_effects.NOT_A_VOLATILE["typechange"]`). The override is dynamic; keep it out of the table. PARTIAL-OK |
| 3 | item identity | T (`[id, known, consumed]`) · ItemBelief · OP (`p_cb`, Leftovers in `g`) | STATIC (from the set; reveals override) | the set; `consumed` / transition is DYN (#22) |
| 4 | ability (id1, id2, dominance, known) | T · OP (status block, Levitate, Serene Grace) | STATIC | OK |
| 5 | the four moves (id, BP, type, category, accuracy, secondary, recoil, never-miss) | T (sorted-by-id concat, F14) · S3 (our active) · S4 / S5 (their beliefs) · OP | STATIC, pooled as a set (F14, folded into the rebuild by the owner) | OK once pooled |
| 6 | spread (IVs, EVs, nature) | T (ours exact; theirs zeros) · spread belief → OP | STATIC as actual stats | OK |
| 7 | Hidden Power type | T (HP block: ours `hp_revealed`, theirs the belief) · OP | STATIC (the set) | OK |
| 8 | level | not encoded; the op assumes L100 (every pool team is L100) | STATIC if ever variable | OK for OU |

### 3.2 Per-mon dynamic

| # | fact | today | home | why / status |
|---|---|---|---|---|
| 9 | HP fraction | T · OP · edges (`alive`, `both_alive`) · attention key mask at 0 | DYN | OK |
| 10 | major status (7-way) | T · OP (burn, paralysis speed, already-statused gate) | DYN | OK |
| 11 | sleep counter, toxic stage | T · OP (`g` status tick) | DYN | gen-3 toxic stage resets on switch-in (VERIFIED `data/conditions.ts:151-153`). OK |
| 12 | sleep-wake belief (3) | T | DYN | OK |
| 13 | stat boosts (7, active only) | E2 · G · OP | DYN (owner's spec) | DUP (G is a second copy) |
| 14 | active flag | T (last dim, load-bearing) · OP · `d2` active column | DYN | OK |
| 15 | trapped / maybe-trapped (our active) | T · `t` edge | DYN | OK |
| 16 | Protect success odds | T · `c4` cell | DYN | OK |
| 17 | recency (seen, acted, hit) | T | DYN | OK |
| 18 | last action (move id, switch, hit / miss / fail / crit; active only) | T | DYN | OK |
| 19 | current PP per move | T (ours exact; theirs from sightings) | DYN per move (or S3 content for our active) | opponent PP with Pressure: UNVERIFIED whether the sighting count is Pressure-aware |
| 20 | move legal now (our active) | move-network validity column · S3 key mask · action mask | S3 content + mask | server-authoritative; carries OUR Choice-lock, Disable, Taunt, no-PP. OK |
| 21 | revealed flags (species, item, ability, each move) | T | STATIC selector (which set / hypothesis) | information state, §3.6 |
| 22 | item consumed / removed / swapped | T (`consumed` bit) · EV (`item_transition`) | DYN | OK |

### 3.3 Volatiles (active only, binary unless stated; `gen3_effects.VOLATILE_SLOTS`, 46)

All ride **E2 + G** today (route count 2, plus OP where named). Home for all: **DYN** on the afflicted mon. In singles
the counter-party of every pairwise volatile is the opposing ACTIVE (or its slot), so an attribute is lossless; the
`t` edge already carries trapping physics. Durations below are VERIFIED at the cited source.

| # | volatile | gen-3 rule | today | status |
|---|---|---|---|---|
| 23 | Substitute | costs maxhp/4, sub HP floor(maxhp/4); a hit prints `-activate …\|Substitute\|[damage]` with NO remaining HP (`data/moves.ts:18325-18340`, `gen4/moves.ts:1307`) | presence bit · OP (blocks outgoing status) | PARTIAL: sub HP absent (not public for theirs; ours UNVERIFIED in the request); OP ignores OUR Sub on INCOMING status (§5) |
| 24 | Leech Seed | heals whoever is in the seeder's slot; survives the seeder's switch; ends on the seeded mon's switch or Rapid Spin (`data/moves.ts:10216-10223`) | bit · OP (`g` leech, Grass immunity) | OK |
| 25 | confusion | counter 2–5 (`data/conditions.ts:174`), 50% self-hit (`gen4/conditions.ts:74`) | bit | PARTIAL: elapsed turns absent |
| 26 | Encore | 3–6 turns (`gen3/moves.ts:257`) | bit; the encored move = that mon's last action; ours via legality | PARTIAL: elapsed turns absent |
| 27 | Taunt | 2 turns (`gen3/moves.ts:596-597`) | bit; ours via legality | PARTIAL: elapsed absent |
| 28 | Disable | 2–5 turns, +1 if the target had not moved (`gen3/moves.ts:208`, `gen4/moves.ts:309-311`) | bit; ours via legality | PARTIAL: THEIR disabled move id absent; elapsed absent |
| 29 | Perish count | 3 → 0 (`data/moves.ts:13255-13272`) | normalised level | OK |
| 30 | Mean Look `trapped` | ends when the trapper leaves (`data/moves.ts:11499`, `sim/pokemon.ts:1528`) | bit · `t` edge | OK |
| 31 | partial trap (Wrap, Fire Spin, Whirlpool, Sand Tomb, Clamp, Bind) | 3–6 turns, 1/16 per turn, ends when the trapper leaves (`gen4/conditions.ts:114`, `gen5/conditions.ts:12-21`) | one collapsed bit | PARTIAL: elapsed absent; the 1/16 chip not in `g` (UNVERIFIED) |
| 32 | Ingrain | 1/16 heal, no switching, no phazing; not grounding in gen 3 (`data/moves.ts:9623-9629`, `sim/pokemon.ts:2150`) | bit | OK |
| 33 | Yawn | sleep lands at the end of the NEXT turn (`data/moves.ts:21156-21163`) | bit | PARTIAL: OP does not price the delayed sleep (declared residual, `damage_op_blocks.py`) |
| 34 | Destiny Bond, Grudge | DB removed before the user's next move (`data/moves.ts:3512-3516`) | bits | OK (the owner's 2026-10-06 resolution family prices DB as P(they KO us)) |
| 35 | Choice-lock | `choicelock` volatile stores the move; lasts to switch-out (`gen4/items.ts:37-39`, `gen4/conditions.ts:127`) | ours: legality; THEIRS: **not a field** (poke-env does not surface it) | PARTIAL: their lock only inferable (item belief × last action) |
| 36 | two-turn charge (Solar Beam, Fly, Dig, Dive, Razor Wind, Skull Bash, Sky Attack) | `twoturnmove` + a move-named volatile; Fly / Dig / Dive semi-invulnerable (`data/conditions.ts:287-294`, `data/moves.ts:5919-5923`) | **MISSING**: not a volatile in poke-env (`mon.preparing`), not in the slot, and `-prepare` is NOT folded into the event window (VERIFIED: `EventKind.PREPARE` exists, `event_window_tracker.py` — deleted in T27 P6 — never read it) | MISSING |
| 37 | lockedmove (Outrage, Thrash, Petal Dance), mustrecharge, Bide, Rage, Uproar | Uproar 2–5 turns and blocks sleep for every active (`gen3/moves.ts:617`, `data/moves.ts:20261-20266`) | bits | OK; Uproar's field-wide sleep block not priced (minor) |
| 38 | Attract, Torment, Imprison, Foresight, Lock-On / Mind Reader, Minimize, Defense Curl, Charge, Stockpile (level), Flash Fire, Nightmare, Ghost Curse, Focus Energy | Attract ends when its source leaves (`data/moves.ts:737`) | bits / level | OK (Lock-On's poke-env timing is a known reading finding, obs CLAUDE.md) |
| 39 | single-turn: Protect, Endure, Magic Coat, Snatch, Focus Punch charge, flinch, Pursuit, Follow Me, Helping Hand | — | bits | OK |
| 40 | Mud Sport / Water Sport | a volatile on the USER, Baton-Passable, halves every Electric / Fire move while it is active (ARCHITECTURE §1.5) | bits · OP (`_sport_mult`) | OK: a mon attribute whose field-wide consequence is priced by the op |
| 41 | anti-status ability fired | collapsed `ability_activated` | bit | OK (identity rides the ability block) |
| 42 | Future Sight / Doom Desire | a `futuremove` SLOT condition on the TARGET's slot, hits at the end of turn T+2, damage computed AT USE, typeless `???` (no STAB, neutral), and it SURVIVES the user switching out (`gen4/moves.ts:533-561`, `gen3/moves.ts:221-247`, `data/conditions.ts:384-415`) | a bit on the USER (`futuresight` / `doomdesire`), from the `-start` line on the user | **WRONG**: wrong entity (user, not the target side), lost when the user switches (poke-env clears effects on switch-out), no turns-left, not priced |
| 43 | Truant, Rollout / Ice Ball, Fury Cutter, the stall counter as a volatile | — | excluded (no poke-env `Effect` member; the stall state rides #16) | MISSING, negligible in OU |

### 3.4 Per-side conditions

| # | fact | gen-3 rule | today | home | status |
|---|---|---|---|---|---|
| 44 | Spikes layers | up to 3; 1/8, 1/6, 1/4 on entry; grounded only (Flying / Levitate immune); cleared by Rapid Spin (`data/moves.ts:17518`, `gen4/moves.ts:1251-1252, 1092`, `sim/pokemon.ts:2155`) | Tb ×2 · G · H · OP `x` · EV · CELL | SIDE (layers) + `x` edge (mon → own SIDE) + OPC (the chip) | DUP ×7 |
| 45 | Reflect / Light Screen | 5 turns; ×0.5 unless crit (`data/moves.ts:14853-14855, 10321-10323`, `gen4/moves.ts:1111, 724`) | presence bits: Tb role · G · H · OP | SIDE (presence + turns left) | PARTIAL: turns left absent although poke-env stores the START turn (`abstract_battle.py:1569-1570`) and the encoder writes only a bit (`global_env.py:94-99`) |
| 46 | Safeguard | 5 turns; blocks foe major status, confusion and applying Yawn, not the sleep of an already-applied Yawn (`data/moves.ts:15586-15611`) | presence bits: Tb role · G · H | SIDE (+ turns) and an OP modifier | **WRONG** for physics: OP status landing ignores it both ways (§5) |
| 47 | Mist | 5 turns; deletes foe-caused stat drops (`data/moves.ts:12082-12093`) | presence bits: Tb role · G · H | SIDE (+ turns) | PARTIAL: not priced (negligible) |
| 48 | Wish pending | slot condition, duration 2, heals the RECIPIENT's maxhp/2 (`gen4/moves.ts:1498-1505`) | reactive scalar per side → G · H | SIDE | OK entity (side-keyed = slot-keyed in singles); DUP ×2 |
| 49 | Future Sight / Doom Desire pending on this side | #42 | MISSING as a side fact | SIDE (turns left, the stored damage priced by OP at cast) | MISSING |
| 50 | Sleep Clause state | a foe-inflicted sleep fails while any living mon on the target side is asleep from a foe; Rest neither triggers nor counts (`data/rulesets.ts:1386-1400`) | computed inside OP for OUR sleep moves only (`damage_op_blocks.py:657-662`); incoming not priced | SIDE (a "sleep slot used" bit per side) + OP both ways | PARTIAL |
| 51 | Freeze Clause state | gen-3 OU carries Freeze Clause Mod: one foe frozen at a time (`config/formats.ts:4420`, `data/rulesets.ts:1458-1468`) | MISSING | SIDE bit + OP | MISSING (rare: freezes are ~10% secondaries) |
| 52 | fainted count | — | reactive → Tb ×2 · G · H | SIDE (alive, fainted) | DUP ×4; **the only route after the rebuild must be explicit** (§4, A1) |
| 53 | revealed count / mons unseen | derivable from per-mon `species_known` | derivable only | SIDE (and X5's Σπ = 6 − revealed) | §4, A1 |

### 3.5 Field

| # | fact | gen-3 rule | today | home | status |
|---|---|---|---|---|---|
| 54 | weather type + permanent + turns left | move weather 5 turns; Drizzle / Drought / Sand Stream permanent in gen 3 (`data/conditions.ts:479-678`, `if (this.gen <= 5) duration = 0`) | Tb ×2 · G · H · OP | FIELD + OP modifier + `g` (mon → FIELD) | DUP ×5 |
| 55 | turn clock (log-elapsed, remaining, log-remaining) | `MAX_TURNS` 250 = the forfeit | Tb ×2 · G · H | FIELD | DUP ×4 |
| 56 | turns since progress | — | G · H | FIELD | DUP ×2 |
| 57 | the active pair | — | active flag (T) · `d2` column · OP roles · S3 / S4 | DYN active flag + edges | OK |
| 58 | decision phase (normal turn vs forced replacement) | — | no field; derivable (every S3 seat masked, the action mask, the event rows' phase tag) | FIELD bit (cheap) or keep derivable | PARTIAL |
| 59 | weather effects on non-damage move behaviour (Thunder accuracy in rain / sun, Solar Beam without charge in sun, Weather Ball) | standard gen-3 rules (UNVERIFIED line by line) | move network reads weather (Tb); OP has no such branch (grep: none) | OP / CELL | PARTIAL (rare in OU) |

### 3.6 Information state

| # | fact | today | home | status |
|---|---|---|---|---|
| 60 | their species revealed / hidden | `species_known` (T) · BeliefSlots (blob) or hypothesis tokens + OTHER (X5) · T0 species prior | STATIC selector + X5 presence | OK (gen 3 has no Team Preview: the gen-4 mod's `standardag` drops it, `data/mods/gen4/rulesets.ts:2-7`, VERIFIED) |
| 61 | their moves revealed / believed | per-move `known` (T) · MoveBelief · S4 / S5 · OP | STATIC (reveals pin the set) | OK |
| 62 | their item / ability revealed / believed | `known` (T) · ItemBelief · ability prior in OP | STATIC | OK; poke-env does not disclose Focus Band on `-activate` (obs CLAUDE.md finding) |
| 63 | their spread / HP type believed | spread belief → OP · HP block | STATIC | OK |
| 64 | **what the opponent has seen of OUR team** (E1) | **MISSING** | DYN on our mons (revealed-to-opponent flags, last HP the opponent saw) | MISSING |
| 65 | speed-order evidence (E8) | partly (`we_first` on event rows) | EV + belief | PARTIAL |
| 66 | damage-roll evidence against candidate sets (E9) | learned implicitly by the belief heads | belief | PARTIAL |

### 3.7 History

| # | fact | today | home | status |
|---|---|---|---|---|
| 67 | the last 32 events (typed records, actor / target / REL mons) | EV + `r` edges | EV | OK (the size 32 is F19's question) |
| 68 | pair tendencies h[i, j] (6×6×5) | `h` edge only | edge | OK as a tendency; an edge-only fact is a ratio (§4, A3) |
| 69 | recency per mon | T | DYN | OK |
| 70 | last action per side | T (on the active) | DYN | OK |
| 71 | faint causes, item transitions, denials, drags, Baton Pass, Pursuit-on-switch | EV columns | EV | OK (E12, `gen3_event_record_v2`) |

---

## 4. What an attention-only static design could NOT reach (the key output for the rebuild)

"Board context leaves the per-mon encoder" removes routes **Tb** and **E2**, and the static token removes per-mon
board-dependent content. If the global token stayed as it is, these facts would be badly placed or unreachable:

**A1. Counts are not recoverable by attention.** A fainted mon is a MASKED KEY (`extractor_ctx.py:336-348`: the
key-padding mask is `hp == 0`, except the active), so no query can read it, and a softmax over the alive tokens is a
weighted AVERAGE, which cannot count. Fainted count, alive count, and (blob) how many opponent mons are still unseen
must therefore be explicit attributes. Today they ride G and the broadcast; under the rebuild they belong on the
SIDE tokens. (X5 already makes "unseen" structural: Σπ = 6 − revealed.)

**A2. One global token in a fixed (ours, theirs) frame does not give "my side".** Every board fact that has a side
(Spikes, screens, Safeguard, Mist, Wish, Future Sight, the clause states, the counts) sits in G as an (ours, theirs)
pair. A THEIR-side mon that attends to G gets the same value vector as an OUR-side mon and must route "their Spikes"
vs "our Spikes" through its FFN conditioned on its token type, an interaction learned once per fact. Two SIDE tokens
with the SAME projection over side-relative content make it structural (a mon reads its own side's token; the game's
side symmetry becomes weight sharing), and the per-mon encoder's side inference from `spread_known` (§2) stops
mattering.

**A3. Amounts reach a token only as content, and the rebuild removes most content routes.** An edge bias writes a
ratio within a softmax row (ARCHITECTURE §5.3). Three pairwise board consequences are amounts the game charges per
mon:
- *Spikes chip on entry* (layers × grounded): today `x` (a bias to G) plus the broadcast Spikes count the token can
  combine with its own types. After the rebuild: the bias only. The SWITCH pointer cell (incoming row 12 + Choice Band
  3 + `pair_outcome_switch` 15 + `conditional_threat` 4) has **no entry-chip coordinate** (ARCHITECTURE §3.3), so the
  switch logit would see Spikes only through attention ratios. Spikes is the central gen-3 hazard.
- *End-of-turn ledger* (Leftovers, weather chip, status tick, Leech Seed): `g`, a bias to G. Same problem.
- *Damage to the opponent's mons*: `prefuse_proj` adds the op's incoming rows to OUR six tokens only
  (`extractor_forward.py:479-492`); what our active does to each of THEIR mons reaches their tokens only through the
  `d1` / `d2` biases and the per-action cells.
Recommendation B2 (§6): an **op-derived content input per mon, both sides**, carrying these cells, so the
"absolute via content" route the sorting rule requires survives the rebuild.

**A4. The single global key is a bottleneck by construction.** 145 raw dims (two 60-dim active contexts and 25
board scalars) pass through one `Linear(145, 128)` into one key / value per head, competing with 60 other keys for
softmax mass. Each reader gets the whole board vector scaled by one attention weight. Splitting it into SIDE ×2 +
FIELD gives the heads three separately addressable keys, and the owner's spec removes the two active contexts from
it (they become DYN on the active mons).

**A5. Facts with no source anywhere** cannot be recovered by any routing: the turn counters (#25-28, #31, #45-47), the
opponent's Choice-lock and Disable target (#28, #35), the two-turn charge (#36), what the opponent knows of us (#64),
Future Sight after its user leaves (#42, #49). The event window can supply an elapsed count only while the starting
row is still among the last 32 events, which is a window limit rather than a home.

**A6. Move-token semantics that read the board.** Today the move network reads clock, weather, fainted and Spikes, so
our active's move seats (S3) are board-aware as content. Under static move tokens, the weather's effect on DAMAGE
still arrives through the op's cells, and recovery-move weather (Moonlight / Synthesis / Morning Sun) through `c3`; the
non-damage weather rules (#59) would reach S3 only by attention to FIELD. Rare in OU.

**What stays reachable without change** (verified): the active pair (active flag + `d2`), every pairwise damage /
status / speed / trap fact (the edges + cells), history (EV + `r`), hidden information (X5 or BeliefSlots), and every
physics MODIFIER, because the op reads raw board facts as kernel inputs (sorting rule, fifth row) and keeps doing so.

---

## 5. Gaps: facts the model never sees, ranked

Ranked by plausible strength impact, then by cost. Impact is a judgement (**UNVERIFIED** for every row: no usage
census of these mechanics on the ladder corpus was run here). "Rebuild hazard" rows are facts seen TODAY that the
static rebuild would lose unless §6's layout carries them.

| rank | gap | kind | impact | cost | cross-reference |
|---|---|---|---|---|---|
| 1 | Spikes chip / end-of-turn ledger / outgoing damage as per-mon AMOUNTS after the broadcast goes (A3) | rebuild hazard | HIGH (switch choice under Spikes) | S: the cells exist, only a content projection is new | arch audit F4 / §4 |
| 2 | fainted / alive counts and "my side" after G is reshaped (A1, A2) | rebuild hazard | HIGH if lost | XS: the scalars exist | arch audit F4 |
| 3 | status landing ignores Safeguard (both ways), incoming Sleep Clause, our Substitute, Freeze Clause, Yawn | WRONG physics (GIGO) | MEDIUM (wrong P(lands) in `s1`, `s3`, `c2`, `pair_outcome`); Safeguard usage UNVERIFIED | XS-S: the Sleep-Clause code is the template | `damage_op_blocks.py` "v2 residual"; owner's GIGO rule |
| 4 | what the opponent has seen of our team (#64) | never seen | MEDIUM-HIGH (information play) | S | backlog E1, X18 |
| 5 | the opponent's Choice-lock (#35) | never seen as a fact | MEDIUM (Choice Band is common in ADV OU, UNVERIFIED share) | S: P(Band) × their last move | — |
| 6 | turns left on Reflect / Light Screen / Safeguard / Mist (#45-47) | never seen | MEDIUM-LOW (screen and Safeguard timing) | XS: poke-env holds the start turn | — |
| 7 | elapsed turns of Encore, Taunt, Disable, confusion, partial trap, Uproar; Yawn stage (#25-28, #31, #33, #37) | never seen | LOW-MEDIUM | S | backlog E3 (stall resources) |
| 8 | Future Sight / Doom Desire pending on the target side (#42, #49) | WRONG entity / lost on switch | LOW-MEDIUM (Doom Desire Jirachi) | S | backlog E12 lists "pending effect and its target" |
| 9 | the opponent's disabled move (#28) | never seen | LOW | XS (the `-start …\|Disable\|<move>` line names it, UNVERIFIED in poke-env's reading) | — |
| 10 | two-turn charge and semi-invulnerability (#36) | never seen | LOW (rare in OU) | S: fold `-prepare` | E12 "charge and recharge turns" |
| 11 | Substitute HP (#23) | never seen | LOW-MEDIUM (Sub + Leech Seed / SubPunch lines) | M: theirs is a belief (not public); ours UNVERIFIED | — |
| 12 | speed-order and damage-roll evidence (#65, #66) | partial | MEDIUM long-run (belief sharpness) | M | backlog E8, E9 |
| 13 | weather rules for non-damage move behaviour (#59) | partial | LOW | XS-S | — |
| 14 | Mist pricing, Uproar's sleep block, partial-trap chip in `g` (#47, #37, #31) | not priced | VERY LOW | XS | — |

---

## 6. Recommendations

### 6.1 The token layout for the static rebuild

| seats | today (61) | proposed (63; 64 under X5) |
|---|---|---|
| 0–11 | 12 mon role tokens (board broadcast inside) | 12 STATIC identity tokens + DYN input + OPC input (both sides) |
| 12 | GLOBAL `[our_ctx 60, opp_ctx 60, non_matchup_rest 25]` | **OUR SIDE** |
| 13 | E3 starts here | **THEIR SIDE** (the same `side_proj` as OUR SIDE; token type distinguishes) |
| 14 | — | **FIELD** |
| then | E3 (4), E4 (6), E5 (6), events (32) | unchanged, shifted by 2 (`base = 2·TEAM_SIZE + 3`) |

**SIDE token content (side-relative, one projection for both):** Spikes layers; Reflect, Light Screen, Safeguard,
Mist (presence + turns left, B5); Wish pending; Future Sight / Doom Desire pending (turns left + the op's priced
damage, B6); alive count, fainted count, revealed count; Sleep Clause used, Freeze Clause used. **FIELD content:**
weather one-hot, permanent, turns left; the three clock scalars; turns since progress; (optional) a forced-
replacement phase bit.

**Edges.** Retarget the three families that point at the global seat: `x` → (mon, its OWN SIDE token); `g` →
(mon, FIELD) (the end-of-turn residual is a field phase); `c4` → (Protect seat, FIELD). Optionally add a structural
membership family `m` (mon, side token: `[same_side]`, zero-init, like `r`); the token types already let a query
find its side, so `m` is a cheap guard, not a requirement. No other family changes.

**Readers.** The critic's `value_entity_pool_full` reads the three refined board tokens where it reads the refined
global token today. The head-only `non_matchup_rest` route into the policy projection is deleted (F4's fourth route;
the policy reads the board through tokens and cells only).

**Active context.** Per the owner's spec, boosts and volatiles become DYN on each mon (nonzero only on the actives);
they leave G. Option A of the arch audit (an active-context TOKEN per side) is superseded by the owner's 2026-10-06
spec and is not recommended here.

### 6.2 Why three board tokens and not one, or one per condition

- **Not one** (keep G): A1, A2, A4 above. A single token is the only design in which a side-relative fact needs a
  learned side-conditional read in every consumer.
- **Not one per condition** (a Spikes token, a Reflect token, …): a condition is an attribute of a side, not an
  entity (learning note Part 3, *things get seats; conditions of things get columns*). Per-condition tokens would add
  ~10 seats of mostly-zero content to a trunk whose cost is the binding constraint (X5's +16.7 % `train_ms`).
- **Pending effects** (Wish, Future Sight) are slot-keyed, and in singles one slot is one side, so they are SIDE
  attributes, not tokens. A pending-effect token would only pay in doubles.

### 6.3 Buckets

**KEEP (11).**

| id | what | why |
|---|---|---|
| K1 | The op reads raw board facts (weather, screens, sports, Spikes, boosts, status, Substitute) as kernel modifiers | the sorting rule's fifth row: a modifier lives inside the kernel; the owner's "let the damage op do the heavy lifting" |
| K2 | History as event seats + `r` edges | typed, entity-referenced, exact for forensic replay; size is F19 |
| K3 | `h` pair-history edges | a tendency between two mons is arity 2 |
| K4 | Volatiles as attributes of the afflicted mon, including the pairwise ones | singles: the counter-party is always the opposing active (or slot) |
| K5 | Mud / Water Sport as user volatiles priced field-wide by the op | the verified gen-3 semantics (ARCHITECTURE §1.5) |
| K6 | Fainted mons key-masked | a fainted mon is not a participant; counts move to SIDE (6.1) |
| K7 | The active pair via the active flag + `d2`'s active column + the op's roles | structural and cheap |
| K8 | The clock as three raw scalars | the deadline-resolution argument (ARCHITECTURE §1.4); moved to FIELD |
| K9 | Wish as a side-keyed fact | verified slot-keyed (gen4 `Wish` condition); moved to SIDE |
| K10 | X5 hypothesis tokens + OTHER for hidden information | unchanged by this audit |
| K11 | Our Choice-lock / Disable / Taunt / no-PP through server legality | server-authoritative; the correct home for OUR side |

**EXACT REFACTOR (2).**

| id | what | note |
|---|---|---|
| X1 | A typed board view in `ExtractorContext` (named ours / theirs / field fields, side-relative accessors) replacing the slices of the 25-dim `non_matchup_rest` tail and the five per-feature board slices | bitwise identical; makes the rebuild a move of named fields instead of offset arithmetic. Bundle with the rebuild's first commit |
| X2 | Documentation: state this audit's declared limits where the current truth lives (ARCHITECTURE §1.4 / §1.5 and `src/agents/observation/CLAUDE.md`: screens are presence-only; Future Sight / Doom Desire live on the user; `-prepare` is not folded; Safeguard, Freeze Clause, incoming Sleep Clause and our Substitute are not priced) | doc-only; not done in this unit (scope) |

**BEHAVIOUR CHANGE (9), one lever each unless stated.**

| id | what | lever | priority |
|---|---|---|---|
| B1 | **The board layout of §6.1** (SIDE ×2 + FIELD replacing G; board context out of the per-mon and move encoders; `x` / `g` / `c4` retargeted; the critic reads the three board tokens; the head-only `non_matchup_rest` route removed) | NOT a separate lever: it IS the layout of the owner's GO'd static rebuild (arch audit §4 L1) | HIGH |
| B2 | **The op-derived content input (OPC), both sides**: each mon's incoming rows (today PF, ours only), its Spikes chip on entry and its end-of-turn ledger (the `x` / `g` cells), and for their mons what our active does to them (`d1`-style row) | a REQUIREMENT of B1: without it a null on L1 indicts the missing amount route, not attention. Inside L1 | HIGH |
| B3 | The entry chip (and its `grounded` gate) in the SWITCH pointer cell | own lever (a switch-cell widening, the `pair_outcome_switch` precedent) | MEDIUM-HIGH |
| B4 | Status-landing physics: Safeguard both ways (major status, confusion, Yawn application), incoming Sleep Clause, our Substitute on incoming, Freeze Clause, Yawn's delayed sleep | a GIGO fix (the owner's rule: fix ASAP, a throwing guard, verify end to end): one unit with a sim-parity test per clause; rides the rebuild's version break or lands before it | MEDIUM |
| B5 | Turns left for Reflect / Light Screen / Safeguard / Mist (SIDE attributes; the start turn is already in LiveView) | obs enrichment | MEDIUM-LOW |
| B6 | Future Sight / Doom Desire as a pending attack on the TARGET side (turns left; damage priced at cast, typeless) instead of a user volatile | obs + op | LOW-MEDIUM |
| B7 | What the opponent has seen of our team (backlog E1) as DYN flags on our mons | obs enrichment (X18 already ranks it) | MEDIUM-HIGH |
| B8 | Elapsed-turn counters for Encore, Taunt, Disable, confusion, partial trap, Uproar, and the Yawn stage (elapsed, which is public; never the hidden remaining roll) | obs enrichment | LOW-MEDIUM |
| B9 | Opponent lock facts as DYN: P(Choice-locked into its last move), the disabled move id, the two-turn charge / semi-invulnerable state (fold `-prepare`) | obs enrichment | MEDIUM |

**Batching.** B5–B9 are new observation facts. The owner's 2026-09-25 precedent batched four into one retrain
boundary; if the rebuild's version break is that boundary, they should enter as ONE separately-flagged lever
(screened after L1, not inside it), so L1's verdict stays about attention. B4 is a correctness fix and is not held for
a screen.

---

## 7. What this audit could not verify

- Usage rates on the ladder corpus of Safeguard, Choice Band, Future Sight / Doom Desire, Substitute, Fly / Dig
  (every impact in §5 is a judgement).
- Whether our own Substitute's HP is in the Showdown `|request|` (#23).
- Whether poke-env's opponent PP estimate applies Pressure (#19), and whether it reads the opponent's disabled move
  from the `-start` line (#28 / B9).
- That the partial-trap 1/16 chip is absent from `g` (#31): the `g` cell's four terms were read from the
  ARCHITECTURE table, not from the kernel line by line.
- The Rust core's twin of the side-condition start turn (B5 needs the core to carry it; the core is byte-equal on the
  obs, not on LiveView internals).
- Dig and Dive's invulnerability conditions were not read line by line (Fly's was).
- Gen-3 non-damage weather rules (#59) were taken from the standard mechanics, not re-read at source.
- Nothing here is measured on a trained model: whether the trunk ALREADY recovers, e.g., the opponent's Choice-lock
  from item belief and last action is a probe question (a CPU linear probe on the X5 blob seeds would answer it, the
  P9 rule: dependence is not coverage).

---

## 8. As built: the OBS-FACTS block — APPENDED to the observation at the X5 version break (`gen3_obs_facts_v1`, 2026-10-06 / 2026-10-07)

The owner put four of §5's gaps IN SCOPE on 2026-10-06 as ONE observation-enrichment lever, screened
on its own and NOT bundled into the static-token rebuild's arm: rank 4 (B7, backlog E1), rank 5 (B9's
Choice-lock half), rank 6 (B5) and rank 7 (B8's Encore / Taunt, plus Disable, Uproar and the partial
trap). Layout and sources: [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §1.7 and
`src/agents/observation/CLAUDE.md` (the OBS-FACTS block).

**Status: IN THE OBSERVATION since the X5 version break (config v144, part 3, 2026-10-07).** The block is
the observation's LAST block (`OFFSET_OBS_FACTS` = 2761, obs 2761 → 2845; the 2761-dim prefix byte-identical:
all 991 obs-golden vectors' prefixes hash to their pre-append values), written every decision by both
encoders and gated in both languages (the reading flags, the fold, slice O's whole-row byte comparison, the
engine truth test). The model reads it only under `--obs-facts v1`; production is `off`
(`designs/production_config.json`). History: the block was first COMPUTED but held OUT of the observation
(orchestrator decision under the owner's delegation, 2026-10-06, option A), because appending it changes
`total_dim`, a weight field, and HEAD would have refused every existing checkpoint mid-X5-A/B; the append
and its consumer were kept ready on branch `obs-facts-append` (`60ebf659`, config v138 there) and landed at
the ONE planned checkpoint break, re-based onto that tree (config v144, no bump of its own). Rejected:
prefix-compatible loading of 2761-wide checkpoints (a compatibility shim, which root `CLAUDE.md` rules out).

**What the block carries (84 dims; laid out to be appended after the event window).**

| fact | dims | home it is laid out for |
|---|---|---|
| what the opponent has seen of each of OUR mons: on the field once, each of its four moves, its item, its ability | 6 × 7 (row = our team slot) | DYN on our mon (B7) |
| the opponent active's Choice-lock EVIDENCE: NOT-locked by item, NOT-locked by two distinct moves this stint, the stint's first move (an embedding id), the trailing run | 4 | DYN on their active (B9) |
| each active's Encore / Taunt / Disable / Uproar / partial trap: elapsed, min left, max left | 2 × 5 × 3 | DYN on each active (B8) |
| each side's turns left on Reflect / Light Screen / Safeguard / Mist | 2 × 4 | SIDE (B5) |

**Mechanics, verified in the pinned `deps/pokemon-showdown` (the gen3 → gen4 → … → base chain).**
- Durations count RESIDUALS: `sim/battle.ts` `fieldEvent('Residual')` decrements `handler.state.duration` and
  ends the condition at 0; `|upkeep|` follows the residual (`sim/battle.ts:2838`).
- Reflect / Light Screen / Safeguard / Mist: `duration: 5` (`data/moves.ts`); the `durationCallback`
  extensions are Light Clay and Persistent, neither in gen 3; the gen-4 mod changes only residual order.
- Encore: `data/mods/gen3/moves.ts` `durationCallback() { return this.random(3, 7) }` = 3–6 (the PRNG's
  upper bound is exclusive; the owner's "3–7?" is 3–6), and base `encore.onStart` adds one when
  `!this.queue.willMove(target)` (the target already acted this turn) — RANDOM, so the block gives
  bounds, never a claimed value.
- Taunt: `data/mods/gen3/moves.ts` `duration: 2`, `durationCallback: undefined`; the gen-4 `onStart` override
  carries no `willMove` adjustment, so exactly 2.
- Disable: gen 3 `this.random(2, 6)` = 2–5, plus one from the gen-4 `onStart` when `!willMove`.
  ⚠️ §3.3 row 28 reads this as "+1 if the target had NOT moved"; the source adds it when the target will
  not move again this turn, i.e. it HAD moved (corrected here, a FINDING).
- Uproar: gen 3 `this.random(2, 6)` = 2–5. Partial trap: `data/mods/gen4/conditions.ts`
  `this.random(3, 7)` = 3–6, ending early when the trapper leaves (gen-5 `onResidual`).
- Choice Band: `data/mods/gen4/items.ts` adds `choicelock` `onAfterMove`; its gen-4 `onStart` stores
  `pokemon.lastMove`; base `choicelock.onBeforeMove` drops the lock when the item is no longer a
  Choice item and exempts Struggle. So: a revealed non-Choice item, or a known empty hand, proves NOT
  locked; two distinct freely selected moves in one stint prove NOT locked; the first move of a stint is
  the lock's move if there is a lock.

**Engine truth.** `src/rust_sim/tests/obs_facts_truth_test.rs` plays 60 seeded random-legal battles
built around these mechanics and checks every encoded fact against the referee at every decision:
2,879 decisions, 1,711 live screen readings exactly equal to the engine's remaining duration, the
engine's Encore / Taunt / Disable / Uproar / trap durations inside the encoded bounds at 591 / 284 /
266 / 270 / 580 readings, and 652 decisions where the engine's opponent active was choice-locked:
no NOT-locked proof fired and the stint's first move was the locked move. 0 failures. Teeth: ignoring
the residual phase fails it 26 times, ignoring the Encore / Disable adjustment 154 times.

**The lever, as built.** `--obs-facts {off,v1}` (config v144, a STRUCTURAL `ModelFlag`): `off` (production)
builds nothing and reads none of the block (the forward is bitwise invariant to its contents); `v1` builds
`agents/model/obs_facts_inject.py`, four zero-init `IsolatedLinear`s adding each fact to its entity's token,
built LAST. Identity at init and no RNG draw: a `v1` arm's every other initial weight equals the `off` arm's at
the same seed (pinned on a real SB3-built policy, `obs_facts_inject_test.py`). Each sub-block is CLASSIFIED for
`--token-encoding static` (`FACTS_TOKEN_CLASS`): `seen` / `choice` / `vol` are the mon's own dynamic state (D:
added to the mon's token after the encoder, never into the static identity S), `screens` is a SIDE fact —
under `legacy` it rides every token of its side, under `static` it goes to that side's BOARD token (OUR SIDE /
THEIR SIDE, `TeamTransformer.board_tokens`' `side_extra`) and never to a per-mon token.

**Version consequence (of the append).** It rode the X5 version break's ONE bump (v144, `ARCH_SIGNATURE`
`gen3_x5_version_break_v1`); `total_dim` 2761 → 2845 refuses a 2761-wide checkpoint on its own. Measured at the
break: the 2761-dim prefix of all 991 obs-golden vectors and of every row the Rust env core's oracle-reveal
corpus pins (27k decisions at each of `off` / `species` / `full`) is byte-identical. Measured on the pre-break
branch (the model unchanged there): with `off` one real PPO update was byte-identical. At the break the
learner golden was re-recorded once for the whole break, so the append's own effect on it is not separately
measured there.

**Not built (still open):** confusion's elapsed count (its counter is decremented per move attempt, not per
residual, and poke-env keeps no count); the Yawn stage; the opponent's disabled move id and the two-turn
charge (the rest of B9); the last HP the opponent saw of our benched mons (backlog E1's fifth item); a
move revealed by a `|cant|` line naming it (poke-env does not read one for either side).

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-06 | Scope | Inventory every gen-3 OU fact by entity, map each to its route today and its home under the owner's static-token spec, rank the gaps; read-only, mechanics verified at the pinned Showdown source | Re-auditing the trunk's size or the edge families' dependence (arch audit F5, F12, F22 own those) | the dispatch brief; owner 2026-10-06 |
| 2026-10-06 | Board layout recommended for the rebuild | Three board tokens: OUR SIDE and THEIR SIDE (one shared side-relative projection) + FIELD; `x` to own side, `g` and `c4` to field; the critic reads all three; the head-only board route removed | Keeping one global token (A1, A2, A4); a token per side condition (conditions are attributes, and trunk cost binds); a token per pending effect (slot-keyed = side-keyed in singles); arch audit option A's active-context token (superseded by the owner's DYN spec) | §4, §6.1, §6.2 |
| 2026-10-06 | Amounts after the broadcast goes | An op-derived content input per mon, both sides (incoming rows, entry chip, end-of-turn ledger, outgoing damage to their mons), as a requirement of L1 | Relying on edge biases alone (ratios cannot carry amounts, ARCHITECTURE §5.3) | §4 A3 |
| 2026-10-06 | New facts | B5–B9 as one separately-flagged enrichment lever screened after L1; B4 (status-landing physics) as a GIGO fix, not held for a screen | Folding them into L1 (two levers in one arm) | §6.3 |
| 2026-10-06 | **As built (owner scope):** B7 (what the opponent has seen of us), B9's Choice-lock EVIDENCE, B5 (screen turns) and B8's Encore / Taunt / Disable / Uproar / partial trap | One 84-dim OBS-FACTS block, COMPUTED and gated in both languages; Choice-lock as evidence only (two NOT-locked proofs, the first move, the run), never a claimed lock state (owner) | A lock-state bit or P(locked) (the owner: "we can never 100 % know"); bundling into the rebuild's arm | §8; `gen3_obs_facts_v1` |
| 2026-10-06 | **The append waits for the version break (orchestrator, owner's delegation, option A):** block computed, append deferred to the X5 adoption version break, with the exact-refactor bundle and the baselines re-pointed / era-marked in the same unit; the append + `--obs-facts {off,v1}` (v138) consumer kept ready on branch `obs-facts-append` | (B) prefix-compatible loading of 2761-wide checkpoints (a shim; root `CLAUDE.md` rules it out); (C) the break now (it breaks every checkpoint mid-X5 read, the look-3 cross and the untaught-meter opponent included); a flag that forks the obs width | §8 |
| 2026-10-07 | **The append LANDED at the X5 version break (part 3, config v144, no bump of its own):** obs 2761 → 2845; `--obs-facts {off,v1}` born at v144 (no migration branch: every pre-144 config is refused at the floor); production `off`; under `--token-encoding static` the `screens` sub-block (a SIDE fact) goes to the side BOARD tokens, the rest to the per-mon tokens as D content (`FACTS_TOKEN_CLASS`); slice O's separate `[FACTS]` comparison and `core_events`' `facts` field DELETED (redundant with the whole-row compare) | Refusing `v1` × `static` (the static rebuild then could not screen the lever); screens on every per-mon token under static (breaks the static encoder's "no board fact in a token" rule); keeping the `[FACTS]` compare (a second check of bytes the row compare already holds) | §8; CHANGELOG v144 part 3 |
| 2026-10-06 | Disable's adjustment | +1 when the target had ALREADY acted this turn (`!willMove`), for Encore and Disable alike | §3.3 row 28's "+1 if the target had not moved" (inverted) | `data/mods/gen4/moves.ts` `disable.condition.onStart`; §8 |
