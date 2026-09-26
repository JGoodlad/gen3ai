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
| **E4** — **DONE** 2026-09-26 (`gen3_event_record_v2`, SHA_BATCH) | **The refused-switch target**: which bench mon a trapped switch aimed at | yes: our own action | **carried**: the `SWITCH_REJECTED` row's TARGET (Python from the previous action index; the core from the noted choice token, a step-built version from the transport's `choice_log`) | small; trap situations → the trap-archetype piloting row | XS: the core knows the intended action |
| **E5** | **Faint cause and item transitions** | yes | CLOSED already (`faint_cause_id`, `item_transition`, `gen3_event_semantics_v1`) | — | — (listed so nobody re-adds them) |
| **E6** | **Mimic / Transform move overlays; Mist** | yes | deferred (one-sided view D8, D9) | correctness, not strength | XS each |
| **E7** | **One view type for every sub-encoder**: `items`, `abilities`, `types`, `moves` read the raw poke-env `Pokemon`, not `LivePokemon` | — | tech debt (one-sided view D2) | none; it removes a second source for the same fact | disappears at M4 by construction |
| **E8** | **Speed-order facts**: who moved first each turn, and whether it contradicts the believed speed tier (Choice Scarf does not exist in gen3; Quick Claw, paralysis and priority do) | yes | partly (`TurnView.we_moved_first`) | the speed-tie and Quick Claw inference the belief heads approximate → belief calibration | S |
| **E9** | **Damage-roll consistency**: for each observed hit, which of the opponent's candidate spreads/items the damage is consistent with | yes: the observed HP% change plus our exact stats | the belief heads learn this implicitly | sharper item/spread beliefs → belief-head calibration | M: needs the damage operator run per candidate |
| **E10** — the PARAMETER-FREE half **DONE** 2026-09-26 (`gen3_hidden_slot_move_mixture_v1`, SHA_BATCH: a hidden slot's prior is `Σ_s P_T0(s \| revealed) · P(m \| s)`, no learned parameters; the learned mixture of §1c stays open) | **A STATE-DEPENDENT posterior for the opponent's HIDDEN move slots** (moved from `designs/ops/TECH_DEBT_BACKLOG.md` 2026-09-24, where it was filed P2: *The HIDDEN-slot MoveBelief posterior is a state-INDEPENDENT constant*) | yes: a Smogon mixture over the T0 species prior, built from inputs the model already computes | the HIDDEN-slot MoveBelief posterior is a state-INDEPENDENT constant (max deviation 0.0 over 57k decisions; belief-calibration read, 2026-09-24) | the Smogon mixture beats the constant in every arm (recall@4 0.28 vs 0.10 on pool) → belief calibration; a structural gap on an input the policy reads | M; a TRAINING-INPUT change, the owner's call. It does not need the Rust core, so whether rule 1's after-the-cutover ordering binds it is also the owner's call |
| **E11** | **`[of]` attribution on damage/heal lines** — who caused it (moved from `designs/ops/TECH_DEBT_BACKLOG.md` 2026-09-24, where it was filed P3: *Reading vs truth for `[of]` (M1 rule R9)*) | yes: `[of]` is on the public line | poke-env's reading DISCARDS it; the core carries the truth beside the reading (M1 rule R9) | the model cannot see who caused residual damage/healing on those lines → overlaps **E2** (per-cause HP accounting); decide at the next retrain whether to flip the reading to the truth, as a registered experiment (it changes the obs) | S, owner-gated, retrain-class |

### 1a. E12: MECHANIC COVERAGE, the long-term requirement (owner, 2026-09-24)

The owner: *"roar, baton pass, thief, spikes sack, etc. should all be covered in what we care about
long term."* This is not one fact but a COVERAGE requirement on the native event record (Rust core
M3's per-action/effect record list) and on whatever reshaped event block the next obs arch change
ships. Each mechanic below must be REPRESENTED FAITHFULLY, with attribution, rather than flattened.
Each gets a constructed-battle fixture that fails if it is flattened:

| mechanic | what must survive into the record |
|---|---|
| **Roar / Whirlwind (phazing)** | the switch was FORCED, not chosen; who forced it; the dragged-in mon is REVEALED; entry-hazard chip on the dragged mon attributed to Spikes, with the layer count. Phazing to rack up Spikes damage is a core gen3 plan |
| **Baton Pass** | WHAT was passed: each boost stage, Substitute and its HP, and the other passed volatiles (confusion, Focus Energy, Leech Seed, Curse, Ingrain, Mean Look trapping, Perish count, Lock-On); the receiver; a pass into Spikes |
| **Item transfer / removal**: Thief, Covet, Trick, Knock Off | the item moved or removed, FROM whom TO whom, and the REVEAL of both items (`item_transition` exists, E5; verify it carries direction and both sides) |
| **Sacking** (a deliberate sacrifice, including a "Spikes sack" and a hazard KO on entry) | the faint cause (move / hazard / recoil / residual / Destiny Bond / Perish / self-KO), the FREE switch that follows a faint, and the Spikes layers at that moment |
| **Pursuit on a switching target** | the hit landed BEFORE the switch, attributed to the switch |
| **Trade KOs**: Explosion / Self-Destruct, Destiny Bond, Perish Song | MULTIPLE faints in one window, each with its cause and order |
| **Called moves**: Metronome, Sleep Talk, Assist, Mirror Move, Nature Power | caller → called, both ids; labels must record the CALLED move where the intent is the call |
| **Rapid Spin** | the hazards cleared, and trapping/Leech Seed removed |
| **ACTION DENIAL** (owner, 2026-09-24): an action that was CHOSEN but never happened. **The dominant gen-3 case is the TURN CUT: in gen-3 singles ANY faint mid-turn cancels every remaining action for all active mons and skips to residuals** (Showdown `sim/battle.ts` `faintMessages()`: "in gen 3, fainting skips all moves and switches"). So a faster mon that faints to its OWN Explosion or to recoil (Double-Edge into a Blissey) still denies the opponent's move: Explosion denies a Dragon Dance even without a KO, and recoil suicide denies Softboiled. It can also be denied because its actor fainted first (outsped and KOed; Self-Destruct / Explosion or a recoil move such as Double-Edge KOing it first; a Destiny Bond trade), was blocked from acting (flinch, full paralysis, sleep, recharge, Taunt, Disable, Truant), or lost its target (a switch, Protect, Substitute) | a first-class DENIED record per actor: WHO was denied, WHY, and BY WHAT. **Information boundary:** the viewer knows its OWN denied choice, but only THAT the opponent was denied, never what the opponent chose. The step path knows both choices, so the per-side projection must drop the opponent's denied choice; a leak is a type error, not a convention. **Intent labels:** a denied opponent action has NO observed label, so it is MASKED, never recorded as "no action" or "switched" (a label GIGO if it is). **Entity model:** a per-entity "acted / denied(reason) / refused(reason)" state for the window, plus move ORDER, so the transformer can learn speed control and "KO it before it moves" as entity relations |
| **Wish / Substitute / Encore / Disable / Taunt / Focus Punch / charge and recharge turns** | the pending effect and its target; refusals with their reason |

**Status: DONE 2026-09-26** (`gen3_event_record_v2`, SHA_BATCH — the owner's observation-architecture
batch). The event row carries the native record's attribution (30 columns, a DENIED row type; schema in
`designs/ARCHITECTURE.md` §1.6). Represented, each with a constructed-battle fixture run through the
core and the Python path under slices T + O (`agents/battle/event_record_v2_fixture_test.py`): ACTION
DENIAL (fainted first; the TURN CUT by a self-KO and by recoil), trade KOs with cause and order
(Destiny Bond, Perish Song), phazing (forced entry, the phazer, the dragged mon, the Spikes chip and
layers), Baton Pass (passer, receiver, a pass into Spikes — the passed stages / volatiles ride the
receiver's active context, the entity-aligned home), Thief / Trick / Knock Off (direction and both
parties; the items themselves on the mons' item slots), sacks and the free replacement, Pursuit on a
switching target, called moves (caller → called). NOT carried as rows: Rapid Spin's cleared
Leech Seed / trap (volatile ends are not window rows; the active context shows the result), a Wish /
Substitute hit (unchanged from 22 columns), and a DRAG's cancellation of the dragged mon's queued
action (the native record does not model it either). Prior status: M3 built the native record to cover these and catalogued every case where the
frozen TurnDelta, the α/β intent labels and the current 22-column event window flatten or lose one,
with rates. That catalogue decides which columns the next obs arch change adds.

### 1b. The NEXT obs arch change batch (one retrain boundary, not several)

**LANDED 2026-09-26 as `gen3_event_record_v2` (SHA_BATCH)**: Mud / Water Sport, E4, the
parameter-free E10, E12 — obs 2501 → 2761, one ARCH_SIGNATURE bump. Batched so that existing checkpoints break once: **P0 Mud Sport / Water Sport slots**
(TECH_DEBT_BACKLOG (a)); **E10** (a state-dependent hidden-slot move posterior; recommended here
rather than after the cutover, because it does not depend on the Rust core); **E12's** event-block
reshape from M3's catalogue. Any other entry joins only on its own registered hypothesis (§0 rule 3).

### 1c. E10's shape: correlations LEARNED from what is seen, not a catalogue (owner, 2026-09-24)

The owner: *"Do we have to have a catalog? I would rather it learn correlations from what is seen."*
So E10 (and any joint set belief) is a LEARNED MIXTURE, not a hand-supplied set catalogue:

- Per opponent slot, K learned "set prototypes": mixture weights π_k(state) and per-component move,
  item and spread logits, each `Smogon prior ⊕ learned delta` (the cold start stays the Smogon
  marginal, and no legal move is ever zeroed, so a lure keeps its floor).
- **Reveals are EXACT evidence:** posterior π'_k ∝ π_k · Π_revealed p_k(m). One parallel pass, no
  sequential decoding. The policy reads the marginal Σ_k π'_k p_k(m); a sampler (search, when it
  returns) picks a k and then a coherent set.
- Trained on the SET-level likelihood of the true opponent set (the label we already have), which is
  what makes it learn "these moves come together".
- **Cost, expected small and unmeasured:** K × a marginal head per slot (for example K = 8), with no
  serial steps. Benchmark it before adoption (model forward at B = 1 CPU and batched GPU). An
  autoregressive decoder stays the fallback only if K prototypes provably miss real sets.
- ⚠️ **"Learned from what is seen" means learned from the OPPONENT TEAM DISTRIBUTION.** On the 719-team
  pool alone it learns the pool's sets: the memorization the 2026-09-24 calibration read measured.
  This entry therefore depends on the coverage-opponent decision (ladder-like teams in the training
  mix); tonight's belief win-rate A/B informs that decision.

### 1d. The TWO synergy levels, and the owner's rulings on them (2026-09-24)

- **Within-Pokémon (the SET):** moves, item, ability, nature/EVs and Hidden Power type co-vary.
  E10's prototypes span the WHOLE set, not moves alone. This level transfers across teams, and pool
  teams already teach reusable sets.
- **Team composition (the ROSTER):** which species appear together and which roles they split. Gen 3
  has no team preview, so species are hidden until they switch in. This is where pool memorization
  concentrates (the calibration read: the species head is −1.28 nats vs the Smogon prior off-pool).
  It needs breadth: coverage opponents, plus the Smogon teammate prior as the base.
- **They interact through ROLES** (a revealed Spiker changes what a teammate's set probably is). The
  entity transformer carries that: each slot's prototype weights are computed FROM its refined token,
  which has attended to its teammates.

**Owner rulings:**
1. **No hand-built role exclusivity.** "One Spiker per team" and the like must be LEARNED (through
   attention), never imposed as a constraint.
2. **Prototypes are SHARED ACROSS SPECIES** (role-like archetypes that transfer to rare Pokémon),
   **with enough capacity for per-mon nuance.** For example, a species-conditioned modulation of each
   shared prototype (species embedding → per-prototype delta, or FiLM), so Tyranitar's "setup
   sweeper" differs from Salamence's.
3. **Two meters, always reported separately:** ROSTER (species NLL for unrevealed slots, by number of
   teammates revealed) and SET (full-set NLL for a known species, by its own reveals). A gain on one
   must not hide a loss on the other.
   **WHEN a roster prediction is read matters (owner, 2026-09-24).** The pooled number weights every
   hidden slot at every decision, so long mid-game stages dominate. The ROSTER meter's HEADLINE is
   therefore PER REVEAL COUNT k, with **k = 3 (half the team seen) and k = 5 (the last mon) as the named
   endpoints**. Each stage is scored once per battle (at the first decision after the k-th reveal),
   with the decision-weighted pool as a secondary. Why it matters, from the 2026-09-24 read's own strata
   (species head minus Smogon prior, in nats; positive = the head is better):

   | k revealed | 1 | 2 | 3 | 4 | 5 |
   |---|---|---|---|---|---|
   | pool | +1.01 | +0.68 | +0.20 | +0.04 | −0.17 |
   | ladder | −0.85 | −0.84 | −1.34 | −1.67 | −1.70 |

   The head is WORST exactly where the prediction matters most (the last mon, off-pool), and even on
   the pool its edge fades as teammates are revealed. The Smogon teammate prior uses roster evidence
   better than the learned head does, so the learned edge looks like the pool's species BASE RATES
   rather than roster reasoning. (That is the orchestrator's reading of the strata, UNVERIFIED as a
   mechanism.) The POLICY reads the parameter-free Smogon species prior for species (the learned
   species head is a training-only readout), so this weakness reaches decisions only through the
   reinjected move posterior.
4. **The same synergies on OUR side are PLANNING, not inference** (we know our sets). Whether the
   policy and value price roles is probed by the tactics probe's ROLE family (§ measurements
   `tactics_probe_2026-09-24`).

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
