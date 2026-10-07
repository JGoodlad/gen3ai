# The gen3ou FORMAT SPEC — one declaration, one story per rule

**ALWAYS-CURRENT** (`designs/endstate/README.md`). Status 2026-10-07: BUILT and LANDED — the spec, team validation,
the drift gate and the board-state readers (`5e8d2956`), then the prior filter (§5.1, `gen3_format_spec_priors_v1`),
landed once the orchestrator confirmed the X5 look-3 cross had finished (10:02, `c344ccd1`) with no pinned run live.

Owner direction (2026-10-07): *"I want the clauses to have a clear story, prefer setting the prior to 0 for
banned items; for more complex things like sleep clause, the observation can help it out"* ("or whatever you
think would be better").

## 1. What the spec is, and why one

Every game the project creates is `format_id: "gen3ou"` (the Rust env core REFUSES any other,
`src/rust_env/src/core/spec.rs:203`); the model never sees another format. Before this doc the format's rules
lived in four places that did not know about each other: the pinned Showdown engine, comments citing
`rulesets.ts` lines in the damage op and the move-resolution family, the acquisition tool's ability floor (which
gave a BANNED ability 1 % of the prior), and nowhere at all for the ladder's own changes since the pin.

**`src/agents/gen3_data/format_spec.py` is now the ONE declaration** (poke-env-free, a facade module). For gen3ou
it declares:

* every RULE the format's ruleset reaches (the entry → gen-3 `Standard` → gen-4 `Standard AG`), each with
  exactly one handling STORY (§3), the reason, and where it is defined;
* every BAN (species incl. the Uber tier, items, moves, abilities, a move banned for one species, per-set
  combos), each with its SOURCE: the pinned file and line, master's, or a derivation;
* the team-building clauses' own lists (One Boost Passer / Speed Pass boosting effects, Accuracy Trap's
  trapping and accuracy-drop moves), verbatim;
* `PINNED_DIFFERENCES`: where the pinned engine differs from the ladder.

Readers ask it, never a constant: `format_spec.active()`, `sleep_clause_mod()`, `freeze_clause_mod()`,
`GEN3OU.banned_{species,items,moves,abilities}`, `species_move_bans()`, `is_banned(kind, id, species=)`.

## 2. Which Showdown: the LADDER's rules

The spec is the LIVE LADDER's format — Showdown master, the server a rated game is played on. Our engine is
pinned (`deps/pokemon-showdown` @ `e0551883`, 2026-05-09); master was snapshotted at `c046106c`
(2026-10-07). The gen3ou entry, gen-3 `Standard`, gen-4 `Standard AG`, the Uber tier and every clause body the
spec restates are identical on both, except (`main.format_drift check --pinned` finds exactly these):

| difference | master | pinned | consequence |
|---|---|---|---|
| **Quick Claw** | banned (`config/formats.ts:4699`) | legal (`:4421`) | a team holding it PLAYS in training and is ILLEGAL on the ladder |
| **One Boost Passer Clause** `boostingEffects` | includes `recycle` (`data/rulesets.ts:1167`) | does not | a Baton Passer with one boost + Recycle is ladder-illegal |

The second was missed by the first drift survey (which diffed only the three format files); it lives in a
clause BODY in `data/rulesets.ts`, which the drift gate now reads (§7).

## 3. The clause table — one story per rule

Stories: **PRIOR = 0** (a banned entity; §5.1) · **BOARD STATE** (engine + observation + the op reads the spec;
§5.2) · **TEAM BUILDING** (engine validator + our team validation; the model needs nothing; §5.3) · **ENGINE**
(a battle mechanic the engine applies, whose consequence the observation already carries) · **COSMETIC** ·
**RULESET** (a container).

| rule (parent) | story | why |
|---|---|---|
| Banlist entities (format): Uber, Sand Veil, Soundproof, Quick Claw *(master)*, Assist, Swagger, Smeargle + Ingrain | PRIOR = 0 | a banned thing is never on a legal team; Smeargle + Ingrain is a move banned FOR one species |
| Banlist combos (format): Baton Pass + Block / Mean Look / Spider Web | TEAM BUILDING | each part is legal alone; only the per-set pair is banned (§8 limitation) |
| One Boost Passer Clause (format) | TEAM BUILDING | a team property |
| Accuracy Trap Clause (format) | TEAM BUILDING | a per-set combination (a trapper + a guaranteed accuracy drop) |
| Freeze Clause Mod (format) | BOARD STATE | in-battle: no foe frozen while one of its side is frozen |
| Speed Pass Clause (format) | TEAM BUILDING | no Baton Passer with a Speed boost |
| Standard (format) | RULESET | its parts below |
| Sleep Clause Mod (Standard) | BOARD STATE | in-battle: no foe slept while one of its side sleeps from a non-Rest source |
| Switch Priority Clause Mod (Standard) | ENGINE | the faster mon switches first on a double switch |
| Species Clause (Standard) | TEAM BUILDING | one of each species; its consequence (a revealed species is not also hidden) is already a rule of the hidden-team belief (`belief_tables.SPECIES_CLAUSE_LOGIT`, the op's Species-Clause-filtered prior) |
| Nickname Clause (Standard) | COSMETIC | nicknames never reach a decision |
| OHKO Clause (Standard) | PRIOR = 0 | = the banned moves Fissure, Guillotine, Horn Drill, Sheer Cold |
| Evasion Items Clause (Standard) | PRIOR = 0 | = Bright Powder, Lax Incense |
| Evasion Moves Clause (Standard) | PRIOR = 0 | = Double Team, Minimize (Acupressure: gen 4, listed verbatim) |
| Standard AG (Standard) | RULESET | gen 3 inherits gen 4's: no Team Preview |
| Obtainable (Standard AG) | PRIOR = 0 | learnset legality is already the move prior's ILLEGAL value (`gen3_data.learnset`); the format consequence is the ABILITY-LOCKED species (below) |
| HP Percentage Mod (Standard AG) | ENGINE | the opponent's HP is a percentage; the observation encodes what the protocol shows |
| Cancel Mod (Standard AG) | COSMETIC | a UI affordance |
| Beat Up Nicknames Mod (Standard AG) | ENGINE | Beat Up never announces the ally; `gen3_effect_sources.RULE_GATED_LINES` relies on it |
| Endless Battle Clause (Standard AG) | ENGINE | our stall forfeit (`agents.training.stall`) ends a game far earlier |

**The ability-locked species** (derived; `format_spec.ABILITY_LOCKED_SPECIES`): Sandshrew, Sandslash, Cacnea,
Cacturne (Sand Veil only), Whismur, Loudred, Exploud, Mr. Mime (Soundproof only — Mr. Mime's Filter is a gen-4
ability). `Obtainable Abilities` requires an ability and the only one is banned, so the validator rejects every
set of theirs (`sim/team-validator.ts:738-750`, `:2018`); Showdown never names them, the spec does so their
species prior can be 0. Confirmed on the pool: three dump teams with Exploud / Mr. Mime were rejected by the
acquisition-time Showdown validator for exactly this.

`format_spec_test.py` fails if a rule is declared twice, lacks a story / reason / source, or the declared rule
set differs from the pinned rule closure; it re-derives the Uber tier, the OHKO moves, Accuracy Trap's accuracy
moves and the ability-locked species from `deps/pokemon-showdown`.

## 4. Where each story lives (code map)

| story | code |
|---|---|
| the declaration | `src/agents/gen3_data/format_spec.py` |
| PRIOR = 0 — acquisition filter | `tools/smogon_stats_downloader/compute_priors.py` (the four `compute_*` filters + `check_format_legal`) |
| PRIOR = 0 — the facade's throwing guard | `src/agents/gen3_data/priors.py` (`_checked_format_legal`) |
| PRIOR = 0 — the model tables' ILLEGAL value | `belief_tables.build_move_prior_logits` / `build_item_prior`, `dex_ids.build_species_usage_prior`, `belief_tables.build_species_cooccur_prior` |
| BOARD STATE readers | `agents/model/status_rules.py`, `damage_op_blocks.py` (`_outgoing_status_land`), `move_resolution.py` (Yawn, the switch branch) |
| TEAM BUILDING validation | `src/agents/gen3_data/team_legality.py` + `python -m main.team_legality` |
| the drift gate | `src/main/format_drift.py` (+ `format_drift_snapshot/`), run by `src/main/ladder_drift_scan.py` |
| format-gated mechanics | `agents/model/move_order.quick_claw_live` (F7b's Quick Claw term reads the banlist) |

## 5. The stories in detail

### 5.1 PRIOR = 0 (banned entities) — `gen3_format_spec_priors_v1`

**Where the filter sits: at ACQUISITION, not at load.** The Rust encoder reads `gen3_ability_priors.json`
directly (`src/rust_sim/src/encoder/data.rs`; the unrevealed-ability observation and the Early Bird sleep
belief), as does the procedural team generator (`ou_random_teams.js`, items / abilities / moves). A Python-only
load-time filter would make the Rust and Python observations disagree. So `compute_priors.py` removes the
illegal mass and renormalises, writing the files every reader shares, and the facade re-checks:

* **ability** prior: banned abilities leave the dex-anchored ability set BEFORE the coverage tiers, so the old
  1 % floor no longer lands on them; a species left with no legal ability (exactly the ability-locked list —
  the tool THROWS if the two disagree) has no row;
* **item** prior: banned items are dropped and the row renormalised (sum 1);
* **move** prior (`P(move in set)`, Σ + empty-slot mass = 4): a banned move (or a move banned for that species)
  is dropped and the legal moves scaled by `(4 − e) / (4 − e − b)` so the slot identity holds; a scaled value
  above 1 THROWS (no joint to renormalise against) — `b = 0` on every species of the current window;
* **teammate** prior: banned species are dropped from every row (renormalised), and no row is keyed by one;
* **species usage** (`priors.species_usage()`, computed at call time from the raw chaos stats): banned species
  are excluded;
* **the facade guard** (`priors._checked_format_legal`) THROWS `PriorInvariantError` on load if any prior file
  gives a banned entity mass > 0 — a regenerated-without-the-spec file can never reach a model.

**The model tables give a banned entity the ILLEGAL value, not a liftable floor.** The Smogon files carry no
mass, but four table builders add a FLOOR for "legal but unobserved", and before this change that floor landed
on banned things too:

| table | before | after |
|---|---|---|
| move-belief prior (`build_move_prior_logits`) | 0.02 on a banned move the species can learn (789 cells: Swagger and Double Team are TMs) | `_ILLEGAL_PROB` (1e-6), as an unlearnable move |
| item-belief prior (`build_item_prior`) | 1e-5 floor on Quick Claw / Bright Powder / Lax Incense in every row | 0 |
| species usage prior (`build_species_usage_prior`) | 1e-6 floor on the 20 banned base species | 0 |
| species co-occurrence log-marginal | `log 1e-4` on them | `log 1e-6` (`_SPECIES_CLAUSE_PROB`, the species-side ILLEGAL value) |

**Mass moved** (on the committed 12-month window, 2025-05..2026-04, 2.53M battles):

* **Quick Claw**: 182 species carry it; **0.32 % of all sets** (usage-weighted); per species up to 51 % (Hypno),
  35 % Trapinch, 33 % Swalot, 29 % Masquerain; among top-15-usage species Magneton 2.9 %, Snorlax 1.1 %,
  the rest ≤ 0.2 %. Bright Powder and Lax Incense carry none (the ladder's validator already rejects them).
* **Abilities**: Dugtrio (a top-15-usage species) and Diglett Sand Veil 0.01 → 0 (Arena Trap 1.0); Gligar
  Sand Veil 0.01 → 0; Electrode Soundproof 0.01 → 0; Voltorb Soundproof 0.5 → 0 (no usage data: was uniform);
  8 ability-locked rows dropped (every one had zero usage).
* **Moves, teammates, species usage**: no banned entry carries mass (the ladder's own validator kept them out).
* **Model-table floors**: 789 move cells, 3 item columns, 20 species.

It is a **TRAINING-INPUT BOUNDARY with no ARCH bump** (the precedent of `2d29c4c0` and `f0d673fd`): weights are
unchanged and loadable; every HEAD-code read of any checkpoint sees the new inputs. The observation changes
only for an UNREVEALED opponent Dugtrio / Diglett / Gligar / Electrode / Voltorb / Mr. Mime (the ability prior
columns): on the golden observation battles **148 of 991 decisions moved, in exactly 2 columns** (the unrevealed
slot's second-ability id, Sand Veil 8 → 0, and P(top ability) 0.99 → 1.0). Re-recorded deliberately, each with the
reason: the golden observation fixture, the X5 dex-row table (`hypothesis_dex_rows.json`, 6 species rows), the Rust
env core's oracle-reveal `off` / `species` digests, the K9 learner golden (both arms, on REBUILT buffers — their
behaviour log-probs were the old tables') and `main.h2h`'s off/off play digests; the Lane S bank's byte gate skips by
its own rule (the encoder identity moved).

**Both belief modes.** The filter sits in the tables every prior consumer shares, so it covers the blob arm and X5's
`fixed_mass` (adopted `c471c2a8`, production at the coming version break) alike: `fixed_mass` builds each hypothesis
mon's candidate moves from the move prior's own legality (`hypothesis_set.build_move_legality`, so a banned move is
never a candidate — `format_spec_priors_test.test_fixed_mass_hypothesis_moves_exclude_banned_moves`), its species
score from the same co-occurrence prior, and its dex rows from the same observation encoder.

**Why it waited:** a `data/` change reaches pinned runs (a pin isolates code, not data), and the orchestrator gated
every model-input change on the X5 look-3 cross finishing. It finished at 10:02 (`c344ccd1`) with nothing pinned live.

### 5.2 BOARD STATE (Sleep Clause Mod, Freeze Clause Mod)

* **The engine enforces them.** The Rust core gates BOTH on one `sleep_clause` flag set for any format not
  ending `customgame` (`src/rust_sim/src/state.rs:1960`, `turn/status.rs:599`, `:621`).
* **The observation carries the state each reads.** Each mon's status one-hot (`slp`, `frz`) and the Rest flag
  (`POKEMON_SLEEP_BELIEF_OFFSET`) are per-mon observation columns; the static encoder's SIDE tokens expose
  `sleep_clause_used` per side (`board_tokens.SIDE_FACTS`). There is no `freeze_clause_used` side column — the
  per-mon `frz` bit carries it; a side column is a separate observation lever (§8).
* **The op and move-resolution rules read the spec.** `status_rules.incoming_status_mask` (incoming sleep and
  freeze), `DamageOperatorBlocks._outgoing_status_land` (outgoing sleep) and `move_resolution` (Yawn's sleep, the
  switch branch's sleep) compute the clause's mechanics as before and ask `format_spec.sleep_clause_mod()` /
  `freeze_clause_mod()` whether it is in force. Production is byte-identical (both are in force);
  `format_spec_readers_test.py` plants a spec without each clause and fails on revert of every reader.
* **The mechanics they restate are drift-gated:** a master body of either clause that differs from the pinned
  one verified in `status_rules.py` FAILS the drift check ("re-verify").

### 5.3 TEAM BUILDING

The engine's validator (the pinned one at acquisition: `tools/others_team_downloader` marks a rejected team
`valid: false` and `TeamLoader` skips it) and our spec validation (`team_legality.validate_team`, the LADDER's
rules) keep illegal teams out of every pool, so **the model only ever sees legal teams and needs nothing more**.
Validated rules: the bans, the per-set combos, Accuracy Trap, Species Clause, One Boost Passer, Speed Pass. Not
re-implemented: the rest of `Obtainable` (learnsets, EVs, IVs, event combos) — Showdown's validator owns it. A
problem caused only by a master-only rule is tagged `ladder_only`.

**Results (2026-10-07, `python -m main.team_legality`):** 747 teams.

| pool | teams | illegal | ladder-only |
|---|---|---|---|
| TRAINING POOL (TeamLoader: 32 Smogon sample + 40 promoted + the `others/` dumps not marked invalid) | 719 | **0** | 0 |
| Smogon sample (the ladder campaign's 32) | 32 | 0 | 0 |
| promoted fleet teams | 40 | 0 | 0 |
| specialist bot teams | 3 | 0 | 0 |
| `others/yak_attack` entries marked `valid: false` (not played) | — | 8 | 0 |

The 8 are exactly teams the acquisition-time Showdown validator already rejected (Speed Pass ×7, One Boost
Passer ×5, Exploud / Mr. Mime's Soundproof ×4, Baton Pass + Mean Look ×1, Swagger ×1 — counts are problems,
a team can carry several). **No pool team holds Quick Claw**, so the master-only ban costs the pool nothing.
`team_legality_test.py` fails if a team the training pool serves becomes illegal.

The PROCEDURAL generator (`ou_random_teams.js`, fuzz / parity gates, not training) draws items from the Smogon
item prior and validates with the PINNED validator, so **19 of 1,000 procedural teams (1.9 %) hold a Quick
Claw** — ladder-illegal. The prior filter removes Quick Claw from the item prior the generator samples.

### 5.4 Format-gated mechanics

`--speed-physics` (F7b, `d02aded6`, OFF in production) implements Quick Claw's shared 1-in-5 roll and gates it
on ONE read, `move_order.quick_claw_live()`, which is now `not format_spec.active().is_banned("item",
"quickclaw")` — the spec is the one source (it landed with a hard-coded `False` waiting for this spec). Quick Claw
is banned, so the term stays OFF and nothing changes; a planted spec without the ban turns it on
(`move_order_test.test_quick_claw_live_reads_the_format_spec`). The pinned engine still rolls Quick Claw for an
illegal team; no pool team holds one (§5.3).

## 6. Team validation — how to run

```bash
python -m main.team_legality                 # per-pool table + every illegal team, with the rule
python -m main.team_legality --json out.json # the per-team record
```

It REPORTS; it never edits a team file.

## 7. The drift gate

`python -m main.format_drift check` compares the spec with a Showdown tree:

* the `[Gen 3] OU` entry's `ruleset` and `banlist` (sets);
* gen-3 `Standard` and gen-4 `Standard AG` (sets);
* the gen-3 Uber tier;
* the clause LISTS the spec restates: Evasion Items / Moves banlists, One Boost Passer / Speed Pass boosting
  effects, Accuracy Trap's trapping list;
* the clause BODIES the op restates — Sleep Clause Mod, Freeze Clause Mod, OHKO Clause — as text against the
  pinned engine's.

Any new or removed entry FAILS, named. Offline by default against the committed snapshot
`src/main/format_drift_snapshot/` (master `c046106c`: the three small files whole, the format entry and the
cited `data/rulesets.ts` blocks as excerpts, `MANIFEST.json` with each full file's sha256); `--pinned` checks
`deps/pokemon-showdown` and must find exactly `PINNED_DIFFERENCES` (a pin bump that closes one fails until the
entry is deleted). `python src/main/ladder_drift_scan.py` runs it against its fresh master clone before any live
session (`--no-format-spec` skips it). Refresh: `python -m main.format_drift snapshot --showdown <master
checkout> --commit <sha>`, then update the spec and `MASTER_SNAPSHOT_COMMIT` in the same commit.

`format_drift_test.py` plants ten master differences (a new ban, a lifted ban, a new format clause, a clause
dropped from Standard, a rule added to Standard AG, a new Uber, a new Evasion item, a removed One Boost Passer
effect, a new trapper, a changed Sleep Clause body) and requires each to fail with exactly its diff.

## 8. Limitations and their triggers

* **Per-set combos are not in the prior.** The belief's per-move marginals can still put a little mass on
  Baton Pass AND Mean Look on one Umbreon. Trigger to revisit: a joint (set-level) belief.
* **Freeze Clause has no side token** (Sleep Clause does). Trigger: an observation-enrichment unit
  (`obs_enrichment_backlog.md`), not this spec.
* **The engine's one flag** gates both board clauses together (`state.rs`); harmless while gen3ou is the only
  format.
* **`Obtainable` is not re-implemented**; acquisition relies on the pinned validator, which is master-equal for
  every rule the spec checks except the two `PINNED_DIFFERENCES`.
* **The drift gate reads only the rules the spec cites.** A brand-new rule that a format adds by NAME fails
  (the name is new); a change inside a rule body the spec does not cite (e.g. `Obtainable Misc`) does not.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-07 | Which Showdown the spec states | the LADDER (master), with `PINNED_DIFFERENCES` declared and drift-gated | the pinned engine's rules (would keep Quick Claw legal in the prior while the ladder bans it) | owner: the model never sees another format; the ladder is the deployment target |
| 2026-10-07 | One story per rule | six stories; every resolved rule has exactly one (a test) | ad-hoc handling per site | owner direction 2026-10-07 |
| 2026-10-07 | Banned entities | PRIOR = 0 everywhere a model reads a prior (Smogon files, the facade guard, the four model-table floors) | leaving the floors (they put 0.02 on Swagger for 789 species) | owner: "prefer setting the prior to 0 for banned items" |
| 2026-10-07 | Where the prior filter sits | at ACQUISITION (`compute_priors.py`), files shared by Rust and Python, + a throwing load guard | a load-time filter in the facade (the Rust encoder reads the same files directly; it would split the observation) | `src/rust_sim/src/encoder/data.rs` |
| 2026-10-07 | The move prior's renormalisation | scale the legal moves by `(4 − e) / (4 − e − b)`, THROW above 1 | moving the banned mass to the empty slot (says "a set runs fewer moves"); no renormalisation | the slot identity `Σ P + e = 4`; `b = 0` today |
| 2026-10-07 | Species-conditioned rows of a banned species | kept when they carry legal mass (an Uber's ability row is inert in-format); dropped when every entry is banned (ability-locked species) | dropping every banned species' rows (an off-format fuzz battle would lose its encoding for no in-format gain) | §5.1 |
| 2026-10-07 | Board-state clauses | engine + observation + the op / resolution rules ask the spec whether the clause is in force | a hard-coded clause per site (the state before); a new side column for Freeze Clause (an observation lever, separate) | owner: "the observation can help it out" |
| 2026-10-07 | Team-building clauses | engine + team validation; the model needs nothing | modelling combos in the belief | §5.3: the training pool is 719 / 719 legal |
| 2026-10-07 | The drift gate's reach | the three format files + the clause bodies the spec restates (found master's `recycle`) | the three format files only (the first survey missed `recycle`) | §2 |
| 2026-10-07 | F7b's Quick Claw gate | `move_order.quick_claw_live()` reads `format_spec` (one source) | its landed hard-coded `False` (a second banlist) | `d02aded6` left the read for this spec |
| 2026-10-07 | Sequencing | ship the spec, validation, drift gate and readers (production byte-identical, `5e8d2956`); HOLD the prior filter until the X5 look-3 cross finishes, then land it (orchestrator OK 10:02) | landing all at once | orchestrator brief 2026-10-07 |
| 2026-10-07 | Versioning of the prior filter | a TRAINING-INPUT BOUNDARY, no `ARCH_SIGNATURE` / config bump (`gen3_format_spec_priors_v1` in the docs and goldens) | an ARCH bump (would strand every archived checkpoint; the weights are unchanged and loadable) | precedent `2d29c4c0`, `f0d673fd` |
