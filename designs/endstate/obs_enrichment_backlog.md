# Observation enrichment backlog — what the Rust-core observation could carry, and on what terms

**Status: BACKLOG, not scheduled.** Authored 2026-09-23 at the owner's request. It lists the facts
the current 2,501-dim observation lacks (or carries poorly), because deriving them through
`LiveView` / `TurnDelta` / the trackers was expensive, and which become cheap once the Rust core
([`program_rust_core.md`](program_rust_core.md)) owns the battle as typed, attributed events plus a
per-side projection. **Explicit-only** like every document in `endstate/`: update it on the owner's
word.

**Companions:** [`design_ladder_campaign.md`](design_ladder_campaign.md) (several entries serve its
stage-2 piloting question) · [`design_three_tier_environment.md`](design_three_tier_environment.md)
§3.3 (the reading-vs-truth rules) · [`../ARCHITECTURE.md`](../ARCHITECTURE.md) (the observation as
it IS).

---

## 0. The three rules every entry is held to

1. **Parity first, enrichment after.** M4 (the Rust encoder) must reproduce today's observation
   byte-for-byte, because that is the cutover gate. No entry here lands before the cutover. After
   it, each entry is an ordinary architecture change: behind a flag, recorded in `ARCHITECTURE.md`
   and `CHANGELOG.md`, benchmarked with `obs_build_benchmark`, and trained into a fresh arm.
2. **The imperfect-information boundary is a TYPE.** The encoder accepts only the per-side
   `OneSidedView` (plus its events), never the omniscient board. Each entry states which side's
   projection it reads. A fact the side cannot see on the ladder must not compile into its
   observation.
3. **Every entry earns its place.** Each carries a HYPOTHESIS (what should improve, and by which
   meter), a COVERAGE check (is the fact already reachable through another block? The frame-deletion
   lesson: a dV ablation says whether the model leans on a block, never whether each fact has a home
   elsewhere), and a COST. It is kept only if a registered arm moves its meter.

## 1. The backlog

| # | fact | visible to the side? how | today | hypothesis → meter | cost after cutover |
|---|---|---|---|---|---|
| **E1** | **What the OPPONENT knows about US**: per own mon, whether its species, each move, its item and its ability have been REVEALED to the opponent, and the HP% the opponent last saw | **yes**: it is the fold of the PUBLIC protocol lines about our own side (switch-ins, `\|move\|`, `-item`/`-enditem`/`[from] item:`, `[from] ability:`/`-ability`, public HP%), the same reading rules we apply to the opponent, pointed at our mons. It is derivable from our own stream online (the Showdown client's own-team tooltip shows the human version) | **not modelled at all** (`learning/imperfect_information_and_equilibria.md` names it as a scope cut) | information management (keeping a set hidden, a surprise move, a switch-in that reveals nothing) → stage-2 piloting per team; the anchors on the sample teams | S: the core already computes both sides' projections; this is the opponent's projection of our side |
| **E2** | **Per-cause HP accounting**: per mon per turn, HP lost/gained by cause (move, recoil, hazard, weather, status, Leech Seed, Leftovers, Wish, Substitute cost) | yes: attributed public lines (exact for our side, percent for theirs) | partly, through the event window's rows; the attribution was historically error-prone (the dead-code guard that credited residuals to the attacker's move, fixed since) | the residual race that decides stall games → stall piloting interaction (the campaign's §5.1 split) | S: attribution is a fact at the source |
| **E3** | **Stall resources**: PP remaining (ours exact; theirs from sightings), Toxic counter progress, turns-to-wake beliefs, Leftovers-vs-residual net per turn, Wish pending | ours exact; theirs by presentation rules (Pressure-aware sighting counts) | partly: sleep and Toxic counters, our own PP, the Wish pair and the sleep-wake belief exist; the per-turn net and the opponent's PP estimate do not | stall is hard to pilot, perhaps because its resources are not legible → stall piloting | S–M |
| **E4** | **The refused-switch target**: which bench mon a trapped switch aimed at | yes: our own action | knowingly lost ("structural": the event window folds from events alone) | small; trap situations → the trap-archetype piloting row | XS: the core knows the intended action |
| **E5** | **Faint cause and item transitions** | yes | CLOSED already (`faint_cause_id`, `item_transition`, `gen3_event_semantics_v1`) | — | — (listed so nobody re-adds them) |
| **E6** | **Mimic / Transform move overlays; Mist** | yes | deferred (one-sided view D8, D9) | correctness, not strength | XS each |
| **E7** | **One view type for every sub-encoder**: `items`, `abilities`, `types`, `moves` read the raw poke-env `Pokemon`, not `LivePokemon` | — | tech debt (one-sided view D2) | none; it removes a second source for the same fact | disappears at M4 by construction |
| **E8** | **Speed-order facts**: who moved first each turn, and whether it contradicts the believed speed tier (Choice Scarf does not exist in gen3; Quick Claw, paralysis and priority do) | yes | partly (`TurnView.we_moved_first`) | the speed-tie and Quick Claw inference the belief heads approximate → belief calibration | S |
| **E9** | **Damage-roll consistency**: for each observed hit, which of the opponent's candidate spreads/items the damage is consistent with | yes: the observed HP% change plus our exact stats | the belief heads learn this implicitly | sharper item/spread beliefs → belief-head calibration | M: needs the damage operator run per candidate |
| **E10** | **A STATE-DEPENDENT posterior for the opponent's HIDDEN move slots** (moved from `designs/ops/TECH_DEBT_BACKLOG.md` 2026-09-24, where it was filed P2: *The HIDDEN-slot MoveBelief posterior is a state-INDEPENDENT constant*) | yes: a Smogon mixture over the T0 species prior, built from inputs the model already computes | the HIDDEN-slot MoveBelief posterior is a state-INDEPENDENT constant (max deviation 0.0 over 57k decisions; belief-calibration read, 2026-09-24) | the Smogon mixture beats the constant in every arm (recall@4 0.28 vs 0.10 on pool) → belief calibration; a structural gap on an input the policy reads | M; a TRAINING-INPUT change, the owner's call. It does not need the Rust core, so whether rule 1's after-the-cutover ordering binds it is also the owner's call |
| **E11** | **`[of]` attribution on damage/heal lines** — who caused it (moved from `designs/ops/TECH_DEBT_BACKLOG.md` 2026-09-24, where it was filed P3: *Reading vs truth for `[of]` (M1 rule R9)*) | yes: `[of]` is on the public line | poke-env's reading DISCARDS it; the core carries the truth beside the reading (M1 rule R9) | the model cannot see who caused residual damage/healing on those lines → overlaps **E2** (per-cause HP accounting); decide at the next retrain whether to flip the reading to the truth, as a registered experiment (it changes the obs) | S, owner-gated, retrain-class |

## 2. Ordering (recommended, after the cutover)

1. **E1**: the cheapest entry with the clearest new capability, and the owner's own tooltip shows
   the information is used by strong players.
2. **E2 + E3**: they belong together and serve the campaign's open stall question. Run them only
   after the strength/piloting split says whether stall is a piloting deficit.
3. **E4, E6, E7**: correctness housekeeping; E7 comes free with M4.
4. **E8, E9**: belief-side, measured by the belief meters rather than by strength.

## 3. What this backlog does NOT license

- Any fact from the omniscient board that the side cannot see. The truth may be used for GATES and
  DIAGNOSTICS, never as model input.
- Pool-derived priors: the owner's Smogon-only rule applies to every prior an entry introduces.
- Shipping an entry because it is cheap. Cheapness is why it is on the list, not why it lands.
