# The four-move closure, and the garbled nicknames (2026-10-10)

**Status: MEASURED (CPU, constructed and real boards). A fix is BUILT behind a default-OFF flag
(`--move-set-closure on`, `gen3_move_set_closure_v1`, config v154). Production is byte-identical. The closing test
runs pinned at `95d014fa` and is untouched.**

**Addendum (2026-10-10, owner "Do A"):** the closure is now ON in the closing test's END-STATE arm (`--arch endstate`),
a pre-data amendment; production stays OFF at P_prod `95d014fa`, and P_end moved to `b132b099`
(`../endstate_closing_test_2026-10-09/p_end_2026-10-10/`, `design_endstate_closing_test.md` Decision record).

Two possible garbage-in problems turned up on 2026-10-09 while the battle viewer was being built
(`designs/prober/battle_viewer_ux_2026-10-09.md`).

## In plain language

**1. "It has shown all four moves, so it has no other move."** A gen-3 Pokémon knows at most four moves. Once all
four have been seen, the chance that it carries any other move is exactly zero. The viewer showed a Skarmory with
all four moves seen and the model still "believing" Roar at 92 %. We checked where that belief goes.

- **Most of the model already gets the fact right.** For the opponent's Pokémon on the field, every part of the
  model that reads its moves (the damage calculator, the threat seats, the "some other move" slot, the opponent
  pointer) uses a construction that spreads exactly 4 − (moves seen) of probability over the unseen moves. With four
  seen, that is zero. The damage calculator also uses it for every benched mon. We pushed the unseen moves' beliefs
  hard in both directions and checked the output. For a four-revealed ACTIVE mon nothing moved, not by one bit. That
  held on 1,481 real decisions with a trained model and on 35 with an untrained one.
- **One path leaked: a benched mon's move belief fed back into its token.** The model writes what it believes
  about each opponent mon's moves back into that mon's token (the "reinjection"). For every slot except the active
  one, that step used the raw belief per move, with no four-move limit. On a trained screen model (`rb_st_static_s1008`),
  pushing the beliefs about a benched four-revealed mon's unseen moves changed the win estimate by a median 4.0
  points (90th percentile 26) and changed the chosen action on 13.6 % of those decisions. That is GARBAGE IN: a fact
  the model cannot learn. Its move head is one linear read plus a fixed prior, so it cannot express "four seen, so
  zero". Training did not teach it either. A trained model's raw belief still puts a summed 2.5 expected moves on a
  four-revealed mon's unseen moves (untrained: 2.4, which is the Smogon prior). On 57 % of such mons, some unseen move
  sits at 50 % or more.
- **The viewer shows the raw belief.** The 92 % Roar is what the move head says. Before this fix, a benched mon's
  token READ that number. The model's damage physics and threat seats never did.
- **How often it matters.** On the probe-battery bank (25,000 decisions from 704 games of the screen arms), 10.5 % of
  decisions have at least one alive opponent mon with all four moves seen. In 7.6 % it is the active mon, which was
  already handled. In 3.4 % it is a benched mon, the leak.
- **The fix.** With `--move-set-closure on`, every opponent slot's reinjection uses the same fixed-mass belief the
  rest of the model uses: four seen means only those four, and fewer seen spreads 4 − seen over the rest. With it on,
  the same push moves nothing (0 change on 860 + 1,481 trained-model decisions and on every untrained one). It is a
  FACT by the owner's test, a candidate for production after the closing test, and needs its own screen.

**2. The garbled nicknames** (`MÃ©talosse`, French nicknames saved with the wrong text encoding). **No nickname
reaches anything the model reads.** Species always comes from the species field of a team, or from the "details" part
of a battle message, never from the name. We played 60 seeded battles three ways: with the garbled names, with the
names blanked, and with the names replaced by nonsense. The model's input bytes were identical all three times. The
garbled names break nothing either: all 20 affected pool teams load, validate and play. Repairing the files is a
`data/` change, so it waits until after the closing test. It also gives the model nothing.

## 1. The four-move closure: what reads what

Where a mon's unrevealed-move presence comes from (HEAD `3190a05d`):

| reader | what it reads | the four-move fact |
|---|---|---|
| (a) the published posterior `last_move_belief_logits` [B,6,400] (`MoveBelief.move_logits`) | Smogon prior ⊕ learned delta per move, independent sigmoids; revealed moves pinned at `_REVEAL_LOGIT`; Hidden Power's typed composition rules HP out when four non-HP moves are revealed (`compose_typed_hp`) | **NOT enforced** (except for Hidden Power). This is what the prober displays and what the move BCE reads |
| (b) the opponent ACTIVE's move group (`HypothesisBuilder.move_group` → `FixedMassMoves`): E4 seats, E5 = OTHER_move, the op's incoming candidates, the flat pointer, the active's reinjection | fixed-mass π, Σ = 4 − r over its legal unrevealed moves | **ENFORCED** (k = 0 → π ≡ 0, OTHER_move masked) |
| (b′) the op's per-mon roster (`build_op_roster` → `slot_move_presence`): every opponent attacker, the bench E5 tail seats | fixed-mass π per slot, k = 4 − r | **ENFORCED** |
| (c) `MoveBelief.reinject_moves` on every slot but the active | `sigmoid(logits)` | **NOT enforced: the leak** (X5's F-X5-33, recorded as a known semantics, never measured) |
| (d) the prober's scouting notes (`model_capture.py`) | `sigmoid(last_move_belief_logits)` | display only. The number is the raw belief, which the bench reinjection read (c) and the physics did not (b) |

### Evidence

The driver is `sensitivity.py` (this directory). It adds ±6 to the logit of every unrevealed move (num ≥ 1) of every
opponent mon with four revealed moves, then reads the masked policy probabilities and the value. A change means the
model reads the unrevealed belief of a four-revealed mon. Any nonzero change counts. The closed cases are exact zeros.

| model · rows | four-revealed ACTIVE only | an alive BENCH mon has four | file |
|---|---|---|---|
| HEAD fresh production build (perturbed 0.02, the K9 noise) · 60 seeded Rust-core fixture battles | 35 rows, **0** change | 16 rows, all change: Δvalue median 0.50 pp, max 1.9 pp; argmax flips 1 / 16 | `head_fresh_bump_off.json` |
| same, MoveBelief reinjection zeroed (localisation control) | 0 | **0**: the reinjection is the ONLY leak | `head_fresh_bump_off_zero_reinject.json` |
| same, `--move-set-closure on` | 0 | **0** | `head_fresh_bump_on.json` |
| trained `rb_st_static_s1008` final, at its pin `6c6d2e09` (read-only checkout) · probe bank `bank_v1` | 1,481 rows, **0** change | 860 rows, all change: Δvalue median **4.0 pp**, p90 26 pp, max 68 pp; max Δprob median 4.3 pp; argmax flips **13.6 %** | `pin_s1008_bump_off.json` |
| same, the closure EMULATED at the pin (the reinjection's weights replaced by `slot_move_presence`, the flag's values) | 0 | **0** | `pin_s1008_bump_on_emulated.json` |

The pin runs fixed_mass X5 at config v143 with the same reinjection code as HEAD (`extractor_forward.py`, the active
row by `fm.w_all`, every other row by `torch.sigmoid(logits)`). So the trained-model read is the leak as it stands at
HEAD. The pin predates the flag, which is why the closure there is emulated with the same construction.

**The raw belief on a four-revealed alive mon** (Σ and max of sigmoid over the unrevealed moves, typeless 237
excluded): trained s1008 Σ mean 2.52, max-single mean 0.54, 57 % of mons with an unrevealed move ≥ 0.5 (2,124 mons);
the fresh build Σ 2.40 (the Smogon prior alone: `Σ_m P(m | s) = 4` minus the revealed). Training did not learn the
closure. The move BCE does label those moves 0 on a revealed slot, but the head is ONE linear read of the slot token
plus the species prior, averaged over 400 moves at coefficient 0.05, and it cannot express "four seen, so zero".

**What enforcing it does to a model trained without it** (`pin_s1008_effect.json`, closure on vs off, no bump; an
OUT-OF-DISTRIBUTION read: it measures how much the model leans on the leaked mass, not what a model trained with the
fact would do): bench-four rows Δvalue median 0.8 pp, p90 3.8 pp, argmax flips 4.8 %; rows with no four-revealed mon
(the renormalisation alone: a hidden or partly revealed bench mon now carries 4 − r instead of its sigmoid sum) Δvalue
median 0.5 pp, argmax flips 1.3 %.

### How often it matters (`bank_count.py` → `bank_count.json`)

Probe bank `bank_v1` (25,000 decisions, 704 games, the screen arms' own games; archetype- and phase-stratified, so the
rates are of that sample, DESCRIPTIVE):

| decisions where … | share |
|---|---|
| some opponent mon (alive or fainted) has 4 revealed moves | 15.1 % |
| some ALIVE opponent mon has 4 revealed | **10.5 %** |
| … the opponent ACTIVE (already closed) | 7.6 % |
| … an alive BENCHED mon (the leak) | **3.4 %** |
| some alive opponent mon has ≥ 3 revealed (the renormalisation matters most) | 33.2 % |

### What was built

`--move-set-closure {off,on}` (`gen3_move_set_closure_v1`, config v154, STRUCTURAL, OFF in production, not in any
named arm). Under `on`, `_apply_move_belief` gives every non-active slot's reinjection the per-slot fixed-mass presence
`hypothesis_tokens.slot_move_presence(..., graph=True)`: the same construction and values the op roster reads (1 on a
revealed move; σ(a + τ) with Σ = 4 − r over the slot species' legal unrevealed moves; exactly 0 at r = 4; a revealed
Hidden Power as its typed weights). Unlike the roster's copy it CARRIES THE GRAPH through σ(a + τ), with τ a no-grad
shift, so the move head keeps the PPO route these rows had under the sigmoid. Only the values change. The active row
is unchanged (the move group's detached π). It builds nothing and draws no RNG. It needs the belief family and is
refused at build without it.

Tests (`src/agents/model/move_set_closure_test.py`, fail on revert):

- the construction: the same values as the detached roster presence, mass 4 − r, exactly 0 at r = 4, and a gradient
  into the logits;
- `on` on constructed boards (the committed parity rows with a benched mon given four legal moves): bit-identical under
  the ±6 bump;
- `off`: the same bump moves the outputs. This pins the leak and keeps the test from passing vacuously;
- the active case: bit-identical in BOTH modes;
- the build refusal and the versioning.

Reverting the forward branch fails the bench test (checked: Δvalue 0.12).

**A production candidate after the closing test.** It is a fact by `design_hand_computed_features.md` §1's test,
it closes a measured leak, and it costs nothing at build. It changes what every bench and hidden slot's token
carries (the renormalisation), so it is a behaviour change and needs its own screen. It is not a silent fix.

## 2. The garbled nicknames: no nickname reaches the model

**Verdict: NO GIGO.** The trace was read-only, by a sub-agent, at HEAD `3190a05d`.

**Behavioural check.** 60 seeded Rust-core battles (2,332 decision rows) with both sides drawing the 20 mojibake
pool teams, run three ways: the mojibake nicknames as stored, the nicknames blanked, and the nicknames renamed `Zq0..Zq5`.
The observation and mask bytes were IDENTICAL (one sha256, prefix `06d1f0cc19f966cc`, for all three).

**The chain, hop by hop.**

- **Packing.** `utils/team_packing.py:311-330` parses `Nick (Species) (M) @ Item`, taking the species from the
  parentheses. `packed` (`:134-150`) writes the nickname verbatim into field 0 and `to_id_str(species)` into field 1.
  On the Rust side, `team::unpack` / `build_set` (`rust_sim/src/team.rs:236-242`) resolves species from field 1 and
  uses the name only when field 1 is empty, which means the name IS the species. The spread match
  (`present/mon.rs:243-270`, `board_reading.rs:460-478`) keys on the species field. Real files:
  `sample/d1ed25a242.txt` packs as `MÃ©talosse|metagross|…`, `d07725903ef50648.txt` as `PtÃ©ra|aerodactyl|…` and
  `Ã\x89lecthor|zapdos|…`.
- **Rust core and encoder.** The sim prints the nickname in idents (`p2a: Nick`). The readers use it only as an
  IDENTITY key. `board_reading.rs:284-337` re-keys by the DETAILS species (`identifies_as`), and
  `core_events/line.rs:28` says the ident name is "never parsed as a species". `reading.rs` takes species from details
  (`update_from_details`). `MonView` (`present/view.rs:39-70`) has NO name field, and the encoder (`encoder/slot.rs`,
  `encoder/mod.rs`, `encoder/facts.rs`) and the trackers read `mon.species`.
- **The live reader** (`main.live`) feeds the server text, decoded as UTF-8, into the same `SideReader` chain, so
  species comes from details there too. Non-ASCII ident slicing is safe on the ASCII prefix (`line.rs:37-46`).
  Showdown itself bounds a nickname: `dex.getName` strips `| [ ] ,` and whitespace and truncates to 18 characters,
  the validator rejects a name that is another species' name, and Nickname Clause rejects duplicates.
- **The Python side.** The extractor reads only the observation tensor, and `agents/battle` reads no names.
  `team_archetypes.py:126` puts the species first. The prober uses names only for display and side matching.

**The mojibake itself breaks nothing.** On disk: 33 lines in 21 files, 7 distinct nicknames (`MÃ©talosse` 13). In the
pool: 20 of 719 teams, all of which pass `validate_teams_locally('gen3ou')`, and `packed_teams("pool")` returns all 719
intact. `committed_team_bytes_test.py` pins these bytes, and `tools/CLAUDE.md` records the decision to keep them.

**The proposed `data/` repair (NOT made; `data/` is frozen while pinned runs are live).**

- **Cause:** `tools/others_team_downloader` (`a1c18731`, 2026-05-14) read PokePaste replies with no charset as
  ISO-8859-1. Both downloaders are fixed since.
- **The repair**, per file at the byte level: `raw.decode("utf-8").encode("latin-1")`. Checked in memory on all 21
  files, it round-trips to Métalosse / Magnéton / Ptéra / Léviator / Libégon / Séléroc / Électhor.
- **It re-ids 20 pool teams** (`sha256(text)[:16]`). With it go the sample manifest's `promoted.from`, the
  `team_sha` keys in `gen3_team_archetypes.json`, the team-PFSP keys and `committed_team_bytes_test.py`.
- **Recommendation:** since the model gains nothing, do it only at a deliberate pool / era boundary with no pinned run
  live, or not at all.

**Findings from the trace** (none model-facing):

- **N-F1.** `main/prober/engine/protocol.py:61-63` claims "our teams are packed without nicknames", which is false.
  A nicknamed actor's move-order and action-fate reads come out unknown in the prober (display only).
- **N-F2 (latent; no pool name triggers it).** `team_packing.py:325-329` takes the FIRST `(`-token as the species. A
  nickname like `Big (Boss) (Metagross)` would pass Showdown's validator and pack the wrong species, and a
  double-spaced token raises `IndexError`.
- **N-F3.** `Gen3Teambuilder` packs its OWN parse rather than Showdown's cleaned one (`teambuilder.py:85-109`).
- **N-F4 (latent).** The R10 path takes species from the NAME when a mon is referenced with no details
  (`board_reading.rs:333`, `reading.rs:284`). A well-formed gen-3 stream never reaches it, a bogus name refuses
  loudly, and Nickname Clause forbids a real species name as a nickname.
- **N-F5.** `TeamLoader` opens team files with no `encoding=` (`team_loader/loader.py:68`), so it depends on the
  locale.
- **Not verified:** the live path against a real Showdown server (the code was traced, nothing was played live),
  and how the Rust side truncates a nickname longer than 18 characters (the pool has none).

## Files

- `bank_count.py` / `bank_count.json`: how often the fact matters on the probe bank.
- `sensitivity.py`: the bump and effect reads. `head_fresh_bump_{off,on}.json` and
  `head_fresh_bump_off_zero_reinject.json` come from HEAD; `pin_s1008_bump_off.json`,
  `pin_s1008_bump_on_emulated.json` and `pin_s1008_effect.json` come from the trained screen final at its pin. The
  arguments are recorded in each file (`bank`, `checkpoint`, `closure`, `perturb`, …); `sensitivity.py`'s docstring has the command.
