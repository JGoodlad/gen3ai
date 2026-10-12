# THE INFORMATION LEVERS: one spec for giving training, search and the model's inputs graded extra information

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-12): the SPEC, the LOG SOURCE and the MODEL-INPUT INJECTION are BUILT (`src/agents/info_levers/`); the other two consumers, the
gender pad and the audit suite are BUILDING (each section says which).** Owner, 2026-10-12: "I want a single dried
out implementation for giving both search and training additional information that would not generally be available
so that we can understand what the ceilings are and what is our constraint and how much value there would be in
improving determinization." It serves the hidden-information study ([`design_hidden_info_binding.md`](design_hidden_info_binding.md))
and the value-discrimination pilot ([`design_value_discrimination.md`](design_value_discrimination.md)).

---

## 0. Summary, in plain language

A real player does not know the opponent's hidden facts: which Pokémon it has not shown yet, their moves, items,
abilities and EV spreads, a shown Pokémon's exact HP, how many turns of sleep are left. To find out WHICH of those
facts limits us, and how much a better guess would be worth, we hand some of them to our system on purpose and
measure the difference. Before this doc, three separate pieces of code did that three different ways: the training
observation (`--oracle-reveal off|species|full`), the search world builder (which kept some facts true by accident),
and a hook the hidden-information study wrote into its measurement folder. They could not be compared, and one of
them leaked.

Now there is ONE description of "what extra information, how much of it, and whether it is true", the
**information-lever spec**, and every consumer reads it:

1. **The training observation**: the network is told the facts in its input, during training and its eval.
2. **The search world builder**: the guessed opponent teams search plays out in hold the facts.
3. **The model-input injection**: a frozen network's own belief about the facts is pushed toward them.

Every fact is a **lever** with a **mass** (how confident: 0.95 = "high mass but not 100 %", the owner's dose) and a
**source** (the truth; a confident WRONG but plausible fact, the placebo; or the honest belief's own best guess, an
honest sharpening). By default every consumer is **honest**: it is given nothing a real player would not have.

**How to read ceilings with it** (§11): the oracle (every lever, mass 1) is the information ceiling; honest is today;
the gap for one class of fact (reveal only it, or hide only it) is what improving our guesses about that class could
buy at most; the dose (mass 0.25 … 1) says how much a partly better guess buys; the placebo says whether it is the
content or just the confidence.

---

## 1. THE SPEC (BUILT: `agents.info_levers.spec`)

### 1.1 The taxonomy

| class | the hidden fact | parts |
|---|---|---|
| `species` | the identity of a never-seen mon (a seen mon's species is public) | unseen |
| `moves` | the move slots play has not revealed | seen, unseen |
| `item` | the held item, until play names it | seen, unseen |
| `ability` | the ability, until play names it (two-ability species) | seen, unseen |
| `spread` | nature, EVs, IVs (→ exact stats, Hidden Power type) | seen, unseen |
| `hp` | the exact HP inside the displayed 1 % bucket | seen |
| `counters` | hidden durations: sleep / confusion / Encore / Disable / lock turns | seen |

`seen` = the opponent mons the viewer has seen switch or be dragged in; `unseen` = the rest. A never-seen mon is at
full HP with no counters, so `hp` / `counters` have no unseen part. An **unseen lever applies where the consumer's mon
at that slot IS a true opponent mon** (its species is right: told by a `species` lever, or a draw / hypothesis that
happens to be right): the obs's tail slot of a told species, the world's never-seen slot holding a true species, the
injection's X5 hypothesis slot whose species is a true never-seen one.

### 1.2 A lever: (class, part, mass, source)

- **mass p ∈ (0, 1]**, a whole multiple of 1/10,000 (exact text, the same integer in Rust). The consumer's belief about
  the fact becomes **p·δ(target) + (1 − p)·(its honest belief)**. p = 1 is the oracle for that fact; a mass-0 lever is
  no lever (the honest belief) and is dropped from the canonical text.
- **source**: `truth` (the target is the true fact); `placebo` (an equally confident WRONG but plausible fact: drawn
  ∝ the honest prior over the candidates EXCLUDING the truth and everything play revealed; deterministic per
  decision / episode); `map` (the honest belief's own mode: an honest sharpening — no truth is read).

### 1.3 Text, hash, record

A spec is a **preset name** (§5) or a comma list of clauses `class[:scope][@mass][~source]` (defaults: the class's
widest scope, mass 1, `truth`). The **canonical text** — what the Rust core parses and the hash is taken over — puts
clauses in class order with the scope explicit (`both` when a class's seen and unseen parts are identical), a mass
below 1 as `@` + up to four decimals, a non-truth source as `~source`; the empty spec is `off`. The **label** is the
X32 level name (`off` / `species` / `full`) when the spec IS that level — so `model_config.json`'s `oracle_reveal`
records exactly what it recorded before — else the canonical text. **Hash** = the first 16 hex of
`sha256("info_reveal_v1:" + canonical)` (pinned for `off` / `species` / `full` / `F+M` by `spec_test.py`).
`InfoReveal.record()` is the block a run or a measurement records (label, canonical, hash, levers).

Examples: `full` = `species:unseen,moves:both,item:both,ability:both,spread:both`; the move dose at 0.9 =
`moves:seen@0.9`; the wrong-moves placebo = `moves:seen@0.9~placebo`; everything but the moves of seen mons =
`O-M` = `species:unseen,moves:unseen,item:both,ability:both,spread:both,hp:seen,counters:seen`.

---

## 2. THE ONE LOG SOURCE (BUILT: `agents.info_levers.revealed`)

What play has revealed is derived from the viewer's own PROTOCOL stream (the battle log it was sent), never from its
observation rows (the v1 world builder's row gate missed gender tokens, Focus Punch's tightening line and a Beat Up
hit: `design_hidden_info_binding.md` §16.4). Per opponent mon seen: its OWN moves named by play (a `|move|` line not
CALLED from another set — `[from] Assist` / `Snatch` / `Magic Coat` / `Metronome` / `Mirror Move` are not its own;
`Sleep Talk` / `Pursuit` / a locked move are — Focus Punch's `-singleturn`, the move a `cant` line says it chose;
never one used while transformed or copied by Mimic; never Struggle); its ORIGINAL item (`[from] item:` on its own
line, `-enditem`, `-activate|…|item:`, Recycle, or the item our mon received from it by Thief / Trick — before any
transfer); its ORIGINAL ability (an `-ability` announce, Trace on either side, `[from] ability:` where the line's form
makes the owner certain: Showdown's `heal` writes `[of]` = the attacker for Volt / Water Absorb, `damage` /
`trySetStatus` / weather write `[of]` = the holder).

**Conservative by design.** A fact the parser misses is only drawn by the search builder and then forced back to the
truth by the protocol gate (inefficient, never wrong); a fact it CLAIMED that play did not reveal would be kept true
(a leak). So an ambiguous owner reveals nothing and is counted (`Reveals.ambiguous`).

**One definition, two readers.** Python consumers (the builder, the injection, the audits) read this module; the
Rust encoder's reading parses the same stream for the observation. The LOG-CONSISTENCY audit (§10) checks they agree
on every decision of real battles.

---

## 3. THE CONSUMERS AND WHAT EACH HONOURS (`agents.info_levers.consumers`)

A lever a consumer cannot honour is **refused by name** before anything is built (`Consumer.validate`); a fact a
consumer can only ever give TRUE is its named **residual**, reported beside every honest spec it runs.

| fact | (a) training observation | (b) search world builder | (c) model-input injection |
|---|---|---|---|
| `species:unseen` | truth (a tail slot per told mon) | truth / placebo / map | refused (the X5 hypotheses are the network's own guesses) |
| `moves:seen` | truth | truth / placebo / map | truth / placebo / map (the typed move posterior) |
| `moves:unseen` | truth (needs `species:unseen`) | truth / placebo / map | truth (a hypothesis slot whose species is a true never-seen one) |
| `item:seen` | truth | truth / placebo / map | truth / placebo / map (the item posterior: the op's P(Choice Band)) |
| `item:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `ability:seen` / `:unseen` | truth (unseen needs `species:unseen`) | truth / placebo / map | refused (no ability belief input) |
| `spread:seen` | truth | truth / placebo / map | truth (the believed stats) |
| `spread:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `hp:seen` | refused (no exact-HP cell; a layout change) | truth (the world must reproduce the true exact HP) | refused |
| `counters:seen` | refused (no hidden-duration cell) | truth — **RESIDUAL**: always the truth's (§7.3) | refused |
| mass p | a per-(episode, side, mon, class) Bernoulli: the observation can only state a fact or not | a per-world draw from the mixture | the mixed marginals, written |

---

## 4. Status of each consumer

- **(a) The training observation** (§6): BUILDING. Today's `--oracle-reveal off|species|full` keeps its exact
  behaviour and bytes; the flag will accept any spec the OBS consumer honours.
- **(b) The search world builder** (§7): BUILDING. Honest by default; the pilot's builder becomes the `pilot` preset.
- **(c) The model-input injection** (§8): BUILT (`agents.info_levers.inject`), lifted from the study's `inject.py` into `src/` and driven by the spec.
- **The gender pad** (§9): BUILDING (an engine-side marker; the builder's residual gender-count leak).
- **The audit suite** (§10): BUILDING.

---

## 5. PRESETS, and how each study arm maps to one (`spec.PRESETS`)

| preset | canonical text | the arm it is |
|---|---|---|
| `off` = `F` | `off` | honest: production; the study's FULLY honest base F |
| `species` | `species:unseen` | X32 `--oracle-reveal species` (byte-identical observation); the study's F+Usp |
| `full` | `species:unseen,moves:both,item:both,ability:both,spread:both` | X32 `--oracle-reveal full` (byte-identical observation) |
| `oracle` = `O` | `full` + `hp:seen,counters:seen` | everything: the information ceiling (search: the TRUE battle) |
| `pilot` | `item:seen,ability:seen,spread:seen,hp:seen,counters:seen` | the value-discrimination pilot's world builder (v1 / v2): P |
| `F+M` / `F+U` / `F+I` / `F+A` / `F+S` | F plus the study class | §6.2 SUFFICIENCY ("reveal only X") |
| `O-M` / `O-U` / `O-I` / `O-A` / `O-S` | O minus the study class | §6.2 NECESSITY ("reveal all but X") |
| `D25` `D50` `D75` `D90` | `moves:seen@p` | §6.4 the move DOSE |
| `W50` `W90` `W100` | `moves:seen@p~placebo` | §6.4 the wrong-moves PLACEBO |
| `MAP90` `MAP100` | `moves:seen@p~map` | §6.4 the honest SHARPENING |
| `TRUE@p` / `PLACEBO@p` / `SELFMAP@p` / `TRUEHYP@p` | `moves:seen@p`, `…~placebo`, `…~map`, `moves:both@p` | §7.2 the MODEL-INPUT arms (injection) |
| `ki_moves` `ki_items` `ki_spreads` `ki_wrongmoves` | `moves:seen@0.95`, `item:seen@0.95`, `spread:seen@0.95`, `moves:seen@0.95~placebo` | §8.3 the knock-in levers at the study's mass 0.95 |

The study's five classes (§6.1 there) are `M` = `moves:seen`; `U` = `species:unseen` + every set class's unseen part;
`I` = `item:seen`; `A` = `ability:seen`; `S` = `spread:seen` + `hp:seen` (a seen mon's spread determines the exact HP
its replayed history leaves). `counters` belongs to no study class: only O tells it. The POOL / POOLSETS arms (a
different PRIOR, not extra information) and the hand evaluator (a different leaf) are not levers.

---

## 6. (a) THE TRAINING OBSERVATION and the DIAGNOSTIC-FORK consent (BUILDING)

**The flag.** `--oracle-reveal` accepts any spec the OBS consumer honours; `off` / `species` / `full` keep their exact
behaviour. Symmetric as today (each side's chain is told the other side's team at the run's spec; scripted bots read
the view, not the row); the in-loop eval core plays at the run's RECORDED spec; `main.h2h`'s per-side reveal takes
the recorded label. Recorded in `model_config.json` (`oracle_reveal` = the label) and `metadata.json` (the
`InfoReveal.record()` block).

**Mass in the observation.** The observation can only state a fact or not: a lever of mass p tells its fact for a
whole episode with probability p, decided per (episode seed, side, mon, class) by a hash, so the episode-averaged
belief is p·truth + (1 − p)·the network's own. A never-seen mon's tail slot exists only where its species is told.

**The fork rule (to build).** `--oracle-reveal` is RESUME-IMMUTABLE (a same-run resume that flips it is refused). The
knock-in forks need a reveal on a FORK of a non-reveal checkpoint, so a fork may change it only with an explicit,
recorded DIAGNOSTIC-FORK consent: never a silent flip, refused on a same-run resume and for a production lineage, and
stamped in the child's `metadata.json` lineage. The fork arm (`--fork-fraction > 0`, the collector's counterfactual
successor rows, encoded WITHOUT the reveal) stays refused with any non-`off` spec.

---

## 7. (b) THE SEARCH WORLD BUILDER, honest by default (BUILDING)

### 7.1 The leaks it fixes

The pilot's builder (v1 / v2, `main.value_search.worlds`) kept every SEEN opponent mon's item, ability and spread
TRUE (hence its exact HP), because they shape the observed history and a joint redraw would almost never pass the
gate. Honest by default, every unrevealed fact of a seen mon is SAMPLED from the Smogon prior and conditioned on the
log by the same byte-equal protocol + row gate.

### 7.2 How (the per-mon screen)

A seen mon's (item, ability, spread) draw is SCREENED one mon at a time: the candidate is replayed with every other
opponent mon at its true facts, and accepted only if the searched side's protocol and rows stay byte-equal; the
accepted per-mon draws are then combined and the whole world passes the full gate again. If each observable event
depends on one opponent mon's hidden facts (singles: true for damage, speed order, item and ability triggers), the
screen draws exactly from P(facts of mon k | everything observed), so the true background leaks nothing; an event
coupling two opponent mons' hidden facts is caught by the joint gate. The audit (§10) measures the leak instead of
assuming it away.

### 7.3 The residual: hidden counters

The world replays the prefix on the TRUE battle seed, so a duration rolled BEFORE the root (sleep, confusion, Encore,
a lock) is the truth's in every world (the study's A5 caveat). No honest world can hide it without a root-state
resample in the playout core; it is the builder's named residual (`counters`), reported with every honest spec.

---

## 8. (c) THE MODEL-INPUT INJECTION (BUILT: `agents.info_levers.inject`)

The study's hook (`measurements/hidden_info_binding_2026-10-11/inject.py`, its §7.1) lifted into `src/` at HEAD and
driven by the spec (`Injector(extractor)`; `inj.set(spec, decision_id, TrueTeam.of(opponent_packed_team))`; every
forward inside `inj.installed()` is injected). For a FROZEN network, offline (the study's model-input axis, probes);
nothing in training reads it.

- **Moves** (`moves:seen` truth / placebo / map, `moves:unseen` truth): it wraps the extractor INSTANCE's
  `_typed_hp_posterior` (the narrowest point every consumer of the move belief reads: the published
  `last_move_belief_logits`, the active's move group, the reinjection, the damage op, the X5 roster, the E4 seats,
  the edges and cells) and rewrites each targeted opponent slot's typed posterior q' = (1 − p)·σ(ℓ) + p·1[m ∈
  target]; the revealed moves, the bookkeeping typeless Hidden Power and (once revealed) all 16 typed Hidden Power
  channels keep their bytes; the clip never moves a channel against the injection; the Hidden Power presence and type
  posterior stay consistent. The placebo is 4 − r moves drawn ∝ the network's own belief excluding the truth and the
  pinned channels (deterministic per decision and slot); map is its own top 4 − r. An unseen lever applies on an X5
  hypothesis slot whose hypothesised species is a true never-seen one (the study's TRUE-HYP).
- **Item** (`item:seen`): a forward hook on `item_belief_head`: on a seen slot whose item the observation does not
  state, the posterior becomes (1 − p)·softmax + p·one-hot(target) (its consumers: the op's P(Choice Band) /
  P(Quick Claw)).
- **Spread** (`spread:seen`, truth): a forward hook on `spread_belief`: the believed stats become (1 − p)·believed +
  p·the true L100 stats (`priors.gen3_stat`, exact), and the head's token reinjection is recomputed from them. **Not
  covered (named):** the nature posterior the op marginalises under `--spread-belief-nature-marginalize`.
- **Teeth** (`inject_integration_test.py`, a production-config extractor on a real committed-bank battle): `off` /
  the empty spec bit-identical to the unhooked forward; every targeted true move ≥ its mass; pinned channels
  byte-equal; a placebo never a true move and deterministic per decision; map = the network's own top candidates; the
  item and spread levers reach their consumers exactly; refusals by name; a planted off-by-one target is caught.
- **Training-time injection** (the hidden-info study's §8 `--belief-oracle`, the label keys in the rollout) is NOT
  built. The knock-in levers are offered on the OBSERVATION channel instead (§6: the `ki_*` presets with the
  diagnostic-fork consent); which channel the study's forks take is the study's call.

---

## 9. THE GENDER PAD (BUILDING)

The engine draws one gender at team construction for every mon of a gender-ratio species whose packed set names
none, so a world needs at least as many such never-seen mons as the truth drew (m ≥ r): the gate conditions the
never-seen species on that count (accepted P(m = r) 0.67 vs the honest 0.57, `design_value_discrimination.md`
Amendment 8). The fix is a packed-set marker meaning "the gender is named, but consume the draw anyway", so any
never-seen mon can sit on a true draw slot. The default (marker-free) path stays Showdown's exactly.

---

## 10. THE AUDIT SUITE (BUILDING)

Against all three consumers: the planted-defect GIGO tests (the study's six plants plus the gender plants); `off`
byte-identical to production; the presets reproducing X32's `species` / `full` rows byte for byte; honest worlds'
per-class truth-match rate equal to the honest expectation; the log source agreeing with the Rust reading on every
decision of real battles. Deterministic (rule 8): fixed seeds and fixed battles, decisions away from their
thresholds. Cheap checks in the routine gate, the rest in the `slow` tier.

---

## 11. HOW TO READ CEILINGS WITH IT

For a decision-level score IMP (the study's: win probability per decision over the policy's pick, LEAF-K8):

- **The information ceiling** = IMP under `oracle`. **Today** = IMP under `off`. Their difference G is everything
  hidden information costs this searcher at this budget (part of it no honest player can recover: §2.2 item 6 there).
- **The value of improving determinization for class X** is bounded by the sufficiency gap IMP(`F+X`) − IMP(`F`) and
  the necessity gap IMP(`O`) − IMP(`O-X`); when the two disagree the classes interact.
- **How much a partly better belief buys** is the dose curve IMP(`moves:seen@p`) over p: a steep start says a small
  belief improvement pays; a curve flat until p ≈ 1 says only near-certainty pays.
- **Content or confidence?** IMP(`moves:seen@p`) − IMP(`moves:seen@p~placebo`) at the same p. And IMP(`…~map`) − IMP(`off`)
  is what an honest sharpening of today's belief buys.
- **For the network** (training observation, injection): the same arms read in strength (pp) or IMP of the policy's
  own pick; a lever the network is told but does not convert is a USABILITY limit, not an information one.

---

## 12. Open risks

- The per-mon screen's independence assumption (§7.2) is measured by the audit, not proven.
- The observation cannot state an exact HP, a hidden duration, a placebo or a map fact (§3): the knock-in forks'
  wrong-moves arm needs a placebo in the observation, which needs a prior sampler on the Rust side (not built).

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-12 | **This doc CREATED; the SPEC and the LOG SOURCE BUILT** (owner: "a single dried out implementation for giving both search and training additional information … so that we can understand what the ceilings are and what is our constraint and how much value there would be in improving determinization") | One declarative `InfoReveal` (seven classes × seen / unseen parts, a mass and a source per lever; presets for the X32 levels, the oracle, the pilot's builder and every study arm; canonical text + hash; per-consumer support tables that REFUSE what a consumer cannot honour and NAME its residual leaks); one protocol-derived source of what play revealed, conservative by design | three separate mechanisms (the status quo: incomparable, one leaked); a soft "known = p" cell in the observation (an input the network never sees in play; a Bernoulli keeps every row a possible observation); the observation rows as the reveal source (they miss protocol-only reveals: the study's A2) | `design_hidden_info_binding.md` §2.2, §16.4; `design_value_discrimination.md` Amendment 8 |
