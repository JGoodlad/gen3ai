# THE INFORMATION LEVERS: one spec for giving training, search and the model's inputs graded extra information

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-25): §13 points at the HUMAN-DATA switch (`--human-data`, PROPOSED, not built). Status (2026-10-12): the SPEC, the LOG SOURCE, the SEARCH WORLD BUILDER (honest by default), the MODEL-INPUT
INJECTION, the GENDER PAD and the AUDIT SUITE are BUILT; consumer (a), the TRAINING OBSERVATION's `--oracle-reveal
<spec>`, and its diagnostic-fork consent are BUILT (§6).** Owner, 2026-10-12: "I want a single dried
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

**The Hidden Power belongs to TWO classes, split (the blind replica's note, 2026-10-12).** In gen 3 its type is set
by the IVs, so `moves` and `spread` overlap there. The split every consumer follows: **`moves` tells THAT a set holds
a Hidden Power** (one untyped slot); **`spread` tells its TYPE** (the IVs that carry it). The search builder screens
the type with the spread (§7.2); the injection pushes the presence with the moves lever and the type with the spread
lever (§8); the observation writes a Hidden Power told without the spread BARE (§6). `full` tells both, so its bytes
are X32's.

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
| `species:unseen` | truth (a tail slot per told mon; BUILT) | truth / placebo / map | refused (the X5 hypotheses are the network's own guesses) |
| `moves:seen` | truth (WHICH moves; a Hidden Power written bare — its type is `spread`'s) | truth / placebo / map | truth / placebo / map (the typed move posterior) |
| `moves:unseen` | truth (needs `species:unseen`) | truth / placebo / map | truth (a hypothesis slot whose species is a true never-seen one) |
| `item:seen` | truth | truth / placebo / map | truth / placebo / map (the item posterior: the op's P(Choice Band)) |
| `item:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `ability:seen` / `:unseen` | truth (unseen needs `species:unseen`) | truth / placebo / map | refused (no ability belief input) |
| `spread:seen` | truth (nature, EVs, IVs, hence the Hidden Power type) | truth / placebo / map | truth (the believed stats) |
| `spread:unseen` | truth (needs `species:unseen`) | truth / placebo / map | refused |
| `hp:seen` | refused (no exact-HP cell; a layout change) | truth (the world must reproduce the true exact HP) | refused |
| `counters:seen` | refused (no hidden-duration cell) | truth — **RESIDUAL**: always the truth's (§7.3) | refused |
| mass p | a per-(episode, side, mon, class) Bernoulli: the observation can only state a fact or not (BUILT: `encoder::oracle::uniform_bp`) | a per-world draw from the mixture | the mixed marginals, written |

---

## 4. Status of each consumer

- **(a) The training observation** (§6): BUILT (2026-10-12). `--oracle-reveal` accepts any spec the OBS consumer
  honours; `off` / `species` / `full` (and their clause texts) keep their exact behaviour and bytes (pinned). The
  DIAGNOSTIC-FORK consent (§6) is BUILT. Its EVAL reads (`main.h2h --oracle-reveal-mode`, §6 "The eval reads of a
  knock-in checkpoint") are BUILT, the honest-play `withheld` read since 2026-10-14.
- **(b) The search world builder** (§7): BUILT — honest by default; the pilot's builder is the `pilot` preset.
- **(c) The model-input injection** (§8): BUILT (`agents.info_levers.inject`), lifted from the study's `inject.py` into `src/` and driven by the spec.
- **The gender pad** (§9): BUILT (the engine marker `98146ddb`, wired into the builder).
- **The audit suite** (§10): BUILT (`main.info_audit`); its read is §10.3.

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

## 6. (a) THE TRAINING OBSERVATION and the DIAGNOSTIC-FORK consent (BUILT)

**The flag (BUILT 2026-10-12).** `--oracle-reveal <spec>` accepts any spec the OBS consumer honours: a level (`off` /
`species` / `full`), a preset (`ki_moves` = `moves:seen@0.95`, `ki_items`, `ki_spreads`, `F+M`, `D50`, …) or a clause
list. The parser stores the spec's LABEL (`utils.rust_env.protocol.oracle_reveal_label` = `OBS.validate(spec).label`)
and refuses, BY NAME and at parse time, every lever OBS cannot honour (`hp`, `counters`, a `~placebo` / `~map` source —
so `ki_wrongmoves`, `W90`, `MAP90`, `oracle`, every `O-*` — and an unseen set-class part without `species:unseen`).
Symmetric as before (each side's chain is told the other side's team at the run's spec; scripted bots read the view,
not the row); the in-loop eval core and `rust_eval.offline` play at the run's RECORDED label; `main.h2h`'s per-side
reveal takes each checkpoint's recorded label. Recorded in `model_config.json` (`oracle_reveal` = the label, so a run
that recorded a level reads exactly as before) and in every checkpoint's `metadata.json` hparams (`oracle_reveal` + the
`InfoReveal.record()` block as `oracle_reveal_record`); `main.checkargs` and the launch banner print the label, the
canonical text and the hash. The resume gate (`check_oracle_reveal`) compares MEANING (`spec.same`): `full` and its
clause text are not a flip. `MODEL_CONFIG_VERSION` did NOT move: no field was added, the field's recorded values for
the three levels are unchanged, and a new label is written only by code that reads it (an older pinned commit refuses
it as "not one of" the levels rather than misreading it).

**The Rust side (BUILT).** `encoder::oracle::OracleSpec` = a mass in 1/10,000ths per (class, part), parsed STRICTLY
from a level name or the canonical clause text (`OracleSpec::parse` re-renders what it parsed and refuses any input
that is not its own canonical text; it refuses the same levers by name), rendered back as the label; the env spec key
`oracle_reveal` keeps its shape (a string, or a `[p1, p2]` pair of labels). A never-seen mon's tail slot is
`hypothesis_slot` when only its species is told and `told_slot` (an own mon's row holding exactly its told facts)
otherwise; a seen mon is overlaid with its told, unrevealed facts (`Oracle::overlay`: nothing play revealed is
overwritten); `slot_cells(told)` declares every cell for each of the sixteen told / untold set-class combinations and
`check_slot` throws on a hidden fact, with teeth per block. The label writers follow `Oracle::tail` (only TOLD tail
mons), so an unstated mon stays a belief target. Byte identity: `off_is_inert`, `species_bytes_are_pinned`,
`full_bytes_are_pinned` unchanged, and `the_level_clause_texts_reproduce_the_level_bytes`; the partial-spec differential
`a_partial_spec_tells_exactly_its_fired_levers` (`src/rust_env/tests/oracle_reveal_test.rs`; detail:
`designs/rust_sim/encoder.md` §11).

**Mass in the observation.** The observation can only state a fact or not: a lever of mass p tells its fact for a
whole episode with probability p, decided per (episode seed, side, mon, class) by a hash, so the episode-averaged
belief is p·truth + (1 − p)·the network's own. A never-seen mon's tail slot exists only where its species is told.
BUILT as ONE uniform draw per (episode seed words, side, the mon's packed index, class) (`uniform_bp`: splitmix64,
pinned by its count over a fixed draw set, `PINNED_HALF_MASS_FIRES`, and from Python by a mass-0.5 `species:unseen`
lever's count over a fixed corpus), compared with the mass of the part that applies — so a class told at equal seen /
unseen masses stays told when its mon is revealed; mass 1 needs no hash. **The Hidden Power TYPE is a
SPREAD fact, never a `moves` one** (the IVs determine it; the search builder follows the same split, §7.2): with the
spread told, a Hidden Power is written as the owner's TYPED move (an own mon's row, so `full` keeps X32's bytes); with
only `moves` told (`moves:seen`, `ki_moves`, …) it is the BARE `hiddenpower`, exactly as a Hidden Power play revealed is
written (no type; `hp_revealed` and the type probabilities as play left them) — chosen over refusing `moves` without
`spread` because it is byte-safe (`full` tells both) and keeps the knock-in lever usable on every team
(`encoder::oracle::told_move_id`; `the_hidden_power_type_is_a_spread_fact_never_a_moves_one`, and the partial-spec
differential asserts every Hidden Power told without the spread is bare). `hp_revealed` is 1 exactly when the spread is
told. A seen
mon that play revealed a move OUTSIDE its set gets only `4 − revealed` true moves, in set order (the four-slot bound).

**The fork rule (BUILT 2026-10-12: `main.train.diagnostic_fork`).** `--oracle-reveal` is RESUME-IMMUTABLE (a same-run
resume that flips it is refused). The knock-in forks need a reveal on a FORK of a non-reveal checkpoint, so a fork may
change it only with an explicit, recorded DIAGNOSTIC-FORK consent, `--diagnostic-fork-consent "<reason>"` (the reason
REQUIRED, non-empty):

- *What counts as a fork.* A `--model` that is NOT a checkpoint of the run directory being written —
  `main.train.fork_lr.is_same_run_checkpoint`, the ONE predicate the lineage block, the pool seeding, the idempotent-fork
  swap and `main.launcher --dry-run`'s `FORK of` / `RESTART of` line read. The trainer decides the consent the moment
  the run directory is known and BEFORE it is created (a refusal is `FATAL_CONFIG`, nothing left behind).
- *Refused.* Without the flag, every change in MEANING is refused exactly as before (`check_oracle_reveal`). With it: a
  same-run resume that flips the spec (consent or not); the consent with NO spec change (meaningless — except on a
  same-run restart of a run whose own lineage already records a consent, which is the launcher restarting it with its
  original argv); a consent without `--model`; and a PRODUCTION LINEAGE.
- *Production lineage, as built.* (1) A consented fork INTO a run directory the baselines registry names is refused (a
  registered PARENT is allowed: the knock-ins fork the adopted arm's own seed, and the child is the diagnostic, not the
  parent). (2) A run whose lineage carries the stamp — its own block or ANY ancestor's, because a plain fork of a
  consented run inherits its observation mode — can never be registered: `python -m main.baselines set` refuses it by
  name and `check` reports one already in the registry (a hand edit). (3) So it can never become the `production`
  entry, from which `designs/production_config.json` and hence `--arch production` are constructed. Rejected: refusing
  `--arch production` / `--arch endstate` on a consented fork (the arch surface is orthogonal to the observation mode;
  the knock-in argv types `--arch endstate`, which may become production), and refusing a registered parent.
- *Stamped.* The child's IMMUTABLE `metadata.json` lineage block gets `diagnostic_fork` = {schema, reason, date, the
  parent checkpoint, the parent's recorded spec and the child's spec as `InfoReveal.record()` blocks}; the loader's
  gate accepts exactly that change and no other (`check_oracle_reveal(…, diagnostic_fork=stamp)`). `main.lineage`,
  `main.sidecar_audit`, `main.checkargs` (which also prints the refusal as a launch verdict) and the launch banner show
  it.
- *The fork arm.* `--fork-fraction > 0` with a non-`off` spec stays refused (`combination_checks
  .oracle_reveal_vs_fork_arm`: its successor rows are encoded WITHOUT the reveal). A fork inherits `fork_fraction` from
  its parent's `model_config.json` like every unnamed flag; the knock-in parents `rb_es_end2_s2001` and
  `rb_es_prod_s2001` both record `fork_fraction` 0.0, so a reveal fork of either is NOT refused and neither the reveal
  arms nor the control need `--fork-fraction 0` typed (no recipe difference between them on this knob).

**The eval reads of a knock-in checkpoint (BUILT: `main.h2h`, `--oracle-reveal-mode`; `main/h2h/reveal.py`).** An
oracle-trained checkpoint is played by `main.h2h` per SIDE (each cell's two levels follow from the mode and the two
checkpoints' RECORDED levels, never typed). The consumers of a recorded spec, by mode:

| mode | what the oracle side is told at eval | what it reads | one oracle side per cell | eval protocol (`gen3_eval_protocol_v1_h2h…`) |
|---|---|---|---|---|
| `off` | nothing, either side | REFUSED for an oracle checkpoint (it would be fed observations it never trained on) | n/a | `_h2h` |
| `one_sided` | the oracle side: the other team at its OWN recorded spec; the other side as trained | the primary read: the oracle side plays exactly as trained (clairvoyance, the ceiling) | required | `_oracle_one_sided` |
| `both_sided` | the other side too, at the same spec | DESCRIPTIVE (out of distribution for that side) | required | `_oracle_both_sided` |
| `withheld` | NOTHING, either side, by declaration: the oracle side plays BELOW its recorded spec | the HONEST-PLAY read of a knock-in fork (`design_hidden_info_binding.md` §8.4): did learning WITH the information change its play WITHOUT it? | required (none: nothing to withhold; two: one stamp names one withheld level) | `_oracle_withheld` |

Rows never pool across modes (each is its own protocol and regime). Every row of a cell with an oracle side carries
`compute.oracle_reveal` = `{mode, player, opponent, oracle}`, a `withheld` row also `withheld_level` (the recorded spec
the side did not play at); `oracle` is the checkpoint's recorded spec (label, canonical text, `InfoReveal` hash) and
whether it is a DIAGNOSTIC fork (its nearest `--diagnostic-fork-consent` stamp: reason, date, the parent and child spec
hashes). The hash is derived from `model_config.json`'s recorded label: a knock-in fork written BEFORE
`gen3_reveal_provenance_v1` has a `metadata.json` `oracle_reveal_record` that reads the PARENT's `off` (the extractor
attribute it was rebuilt with), so `metadata.json` is never the source (the contract below).

**THE PROVENANCE CONTRACT (BUILT 2026-10-15, `gen3_reveal_provenance_v1`; GIGO FIX).** A run's reveal mode has ONE
authoritative source: the RESOLVED run config (`args.oracle_reveal`: the typed flag, else the checkpoint run's
`model_config.json`). `model_config.json` is written from it, and the env core is DECLARED from it
(`RustEnvDecl.oracle_reveal`, stamped into `metadata.json`'s `env_core.oracle_reveal`). `metadata.json` (`oracle_reveal`,
`oracle_reveal_record`, `env_core.oracle_reveal`, every checkpoint sidecar and `snapshot_history` entry) only COPIES it.
The extractor attribute is NOT a source: SB3 rebuilds the extractor from the PARENT zip's `policy_kwargs`, so a fork's
attribute read the parent's mode. The incident: every diagnostic knock-in fork recorded `oracle_reveal: off` /
`oracle_reveal_record: off` beside a `model_config.json` that said `moves:seen@0.95` (`model_build` re-applied
`belief_grad_mode` after a load, not `oracle_reveal`; `run_io` wrote the record from the attribute). **Provenance only,
verified:** the attribute is read by `run_io` alone (`grep` of every `.oracle_reveal` reader; the forward builds nothing
from it, and `reveal_provenance_test.py` executes that: re-applying it moves no weight and no output bit); training's
and eval's observations come from `RustEnvDecl` / the recorded config, never the attribute.

- *The class fix.* `agents.model.resume_reapply.SETTERS` is the ONE table of the registry's `resume_immutable` flags the
  loaded extractor carries as an attribute (`belief_grad_mode`, `oracle_reveal`), applied by `model_build` after EVERY
  `--model` load from the resolved config; a test fails the day a `resume_immutable` row has no setter. Deliberately the
  ATTRIBUTE only: the policy's constructor kwargs (`features_extractor_kwargs`) stay the parent's, because they are the
  forward-identity `forward_fingerprint` and `main.h2h`'s `EngineArch` toggles read, and a fork's forward IS its parent's
  (rewriting them would split a fork from its plain control into two slot groups and refuse the registered fork-vs-control
  cross). So a fork's zip still records `oracle_reveal: off` in its pickled kwargs; `model_config.json` is the record.
- *The record.* `metadata.json`'s `oracle_reveal` / `oracle_reveal_record` are taken from the env core's declaration
  (`run_io.run_reveal`), the extractor attribute only for a save with no collector (a fresh-checkpoint tool).
- *The guard.* `save_model_snapshot` and `record_checkpoint` raise the typed `oracle_reveal.RevealProvenanceError` (before
  a byte is written) unless metadata's `oracle_reveal`, `oracle_reveal_record.label` and `env_core.oracle_reveal` all equal
  `model_config.json`'s; the trainer's `run_io.require_reveal_provenance` runs the same check, plus the env declaration
  and the extractor attribute, before the FIRST snapshot of the resume and the fresh path and exits `FATAL_CONFIG`.
- *The readers.* `oracle_reveal.run_provenance(run_dir)` reads `model_config.json` first and flags a `metadata.json` that
  names another mode as a KNOWN-BAD record; `main.lineage` (`reveal` in `--json`, a `reveal mode` line + warning) and
  `main.sidecar_audit` (`reveal`, `n_reveal_mismatched`, a summary count) show it. `main.h2h` already read the recorded
  label from `model_config.json`. **Existing runs are NOT rewritten** (`models/` is read-only): the 11 knock-in forks written
  before the fix, `rb_hib_{items,moves}_s310{1,2,3}`, `rb_hib_species_s310{1,2}`, `rb_hib_spreads_s310{1,2,3}`, carry the
  wrong `oracle_reveal` / `oracle_reveal_record` in `metadata.json`, every sidecar and `snapshot_history` entry (the two
  controls `rb_hib_ctrl_*` are `off` and correct; no descendant of a knock-in fork exists in `models/`). Any fork launched
  under a commit older than the fix (a run is PINNED to its checkpoint's commit) is the same: read it through the readers.

---

## 7. (b) THE SEARCH WORLD BUILDER, honest by default (BUILT: `main.value_search.worlds`, `world_draws`, `world_screen`)

`WorldSampler(reveal=spec)` (default `off`: HONEST); `sample(log, at, side, J, key, screen_cache=…)`. `SearchConfig(oracle=
True)` stays the cheap way to play the TRUE battle; the `oracle` preset through the builder reproduces it exactly (the
audit's `oracle_identity`).

### 7.1 The leaks it fixed (before = the pilot's builder, the `pilot` preset; after = `off`; measured, §10.3)

The pilot's builder (v1 / v2) kept every SEEN opponent mon's item, ability and spread TRUE — hence its exact stats, its
Hidden Power type and the exact HP its history leaves — because they shape the observed history and a joint redraw
almost never passes the gate. Honest by default, each of them is DRAWN from the Smogon prior and conditioned on the log
by the same byte-equal protocol + row gate; what play revealed (`agents.info_levers.revealed`) stays the truth's.

Truth-match rate of each fact in the worlds built for 8 committed-bank decisions × 4 worlds (`python -m main.info_audit`; a revealed fact is the truth's under both and is not counted here):

| fact | the pilot's builder (`pilot`) | the honest builder (`off`) |
|---|---|---|
| a seen mon's unrevealed item | 1.000 (told, n 48) | 0.750 (drawn, n 48) |
| its unrevealed ability | 1.000 (told, n 96) | 1.000 (drawn, n 96) |
| its spread (nature + EVs + IVs, exactly) | 1.000 (told, n 124) | 0.032 (drawn, n 124) |
| its unrevealed move slots (all of them right) | 0.113 (drawn, n 124) | 0.097 (drawn, n 124) |
| a never-seen slot's species | 0.000 (drawn, n 68) | 0.000 (drawn, n 68) |
| a never-seen slot's item (when its species is right) | 0.044 (drawn, n 68) | 0.044 (drawn, n 68) |
| a seen mon's EXACT HP at the root equals the truth's | 124 / 124 | 70 / 124 |

Under `pilot` the seen mons' item, ability and spread are TOLD (the truth, every world); under `off` they are drawn and match the truth only as often as the log forces it (Leftovers on a mon whose healing showed; the one ability most species have) or the prior happens to.

### 7.2 What each fact of a world is

- **Revealed by play** (the ONE log source): the truth's. A reveal the source names that the true set does not hold is
  the parser's error: counted (`leak["reveal_mismatch"]`) and drawn instead.
- **A lever fires** (a categorical fact: with probability = its mass, decided once per world and mon; the MOVES lever is
  no coin — its mass is the presence mixture of every candidate move, §1.2): its target — the truth, a placebo
  (drawn ∝ the prior excluding the truth; for a seen mon's item / ability / spread, one the log allows), or the map
  (the prior's mode; for a seen mon's item / ability / spread, the most probable state the screen found the log to
  allow).
- **Otherwise drawn**, honestly: a NEVER-SEEN slot's species ∝ the Smogon teammate co-occurrence with the revealed
  species, then its whole set (moves by the conditional-Poisson presence draw, item, ability, spread); a SEEN mon's
  unrevealed moves by the same presence draw (`mixture.mix_presence` + `cp_sample`: p = 0 is the honest draw exactly,
  the study's §6.4 dose); a seen mon's item, ability and spread by the SCREEN (§7.3).
- **Hidden Power**: its TYPE and whether a set carries one are SPREAD facts (the IVs: they move the exact stats the log
  pins by a point). The screen draws them; the world's moves then hold a Hidden Power of exactly that type (or none);
  a revealed bare Hidden Power's type is drawn the same way, and a past choice of it is rendered as the world's typed
  move. A seen mon's moves that match the truth count modulo the Hidden Power's type. The state is (type, IV
  PATTERN): any of the type's 30 / 31 patterns (`packed.hp_iv_masks` = `agents.gen3_data.hidden_power.iv_patterns`; the canonical one, Showdown's validator fill, at `HP_CANON`
  0.8, the others sharing the rest), because pool sets use non-canonical ones (the blind replica's note) and a
  pattern moves the exact stats by a point, which the log pins. IVs below 30 are outside the support (§7.5).
- **Genders** (§9): the never-seen mons are placed so the turn-0 gender draws fall on exactly the true team's draw
  slots; with the engine's pad any of them may sit on a draw slot. The placement PERMUTES the drawn never-seen mons
  (whole packed entries), so each never-seen mon's provenance record MOVES WITH IT (`place_genders(src_of=…)`;
  `facts[i]["drawn_slot"]` names the slot it was drawn for). Until 2026-10-15 the records stayed at the draw slot: the
  WORLDS were right, the records of ~46 % of the honest arms' never-seen slots named another slot's mon (§7.6).
- **The never-seen SPECIES flags** are read at the slot the mon holds in the returned team: `match` = a TRUE
  never-seen species (SET-based: the drawn never-seen slots are exchangeable, the gender placement decides which is
  which), `slot_match` = the truth's species at this very slot. Before 2026-10-15 `match` compared a species with the
  truth of the slot it was DRAWN for (neither its slot nor the set): a 2-3x undercount of the set-based hit rate.
- **Legality**: a world breaking a gen3ou rule the true team does not break is rejected (`clause`); the never-seen
  draws and the genders are checked BEFORE the screen (cheap), each seen mon's set after its moves are drawn (a banned
  set — e.g. a Baton Passer drawn a Salac Berry — is redrawn up to `MOVE_REDRAWS` times), the whole team last.
- **Showdown's validator** (F2 of `design_world_sampler.md`, BUILT 2026-10-19: `main.value_search.world_legality`):
  every drawn SET and every returned world passes the pinned `TeamValidator('gen3ou')` — learnset, egg- and
  event-move COMBINATIONS, EVs, IVs, Hidden Power — which `team_legality` (clauses and bans) never checked; 5.2 % of
  the stored F worlds failed it. The method is REJECTION through the validator itself (never a hand-written table): a
  never-seen mon's whole set is redrawn until accepted (`world_draws.SET_REDRAWS`; its lever coins flipped ONCE per
  mon, so a lever's mass stays exact), a seen mon's unrevealed moves are redrawn after the screen (`MOVE_REDRAWS`), a
  never-seen set the gender placement made illegal and a world team the validator rejects are redrawn. The draw is
  then the prior CONDITIONED on legality, P(set | species, the fixed facts, legal); the species and the screened spread
  are not re-weighted by how often their sets are legal (that rate is an artefact of the independent per-slot draw,
  not evidence about the species). The validator is ONE persistent Node child per process
  (`utils/bridge/validate_stream.js`), memoised per set. **The TRUE-TEAM GUARD:** the truth must pass the same rule
  or the build RAISES `LegalityRuleError` before any draw (a defect, never an exclusion); the throwing guard
  re-validates every returned world (`world_guard.check_legal`, `IllegalWorldError`). Fails are counted:
  `illegal_set` (a rejected draw), `illegal_exhausted` (no legal draw in the redraw budget), `illegal_gender`,
  `illegal_team`. **Worlds built before this are not comparable with worlds built after.**
- **A stale Hidden Power TYPE field** (fixed with F2): three pool sets name their Hidden Power type in the packed
  `misc` field (Salamence `Flying`, Salamence `Grass`, Tyranitar `Bug`). A seen mon's world copied the true set and
  drew another spread, so it kept the truth's field beside another type's IVs: Showdown's validator rejects it (28 of
  F2's 581 illegal worlds), and Showdown's battle would play the FIELD's type where our Rust core plays the IVs'. The
  seen-mon entry now drops a field that disagrees with its IVs (`world_screen.clear_stale_hp_type`), unless the spread
  is TOLD true.
- **The playout tool opens it** (the blind replica's finding 3): a world that passed the gate is also opened by
  `rust_env_playout_open`, the search's own replay, before it is returned. The gate's `core_events` replay accepts
  some choice logs the playout tool refuses (a numeric `move k`; a Struggle the world does not force); such a world is
  rejected (`fails["playout_refused"]`, the refusal named), never handed to a search that cannot play it. The audit's
  `playout_open` re-checks every returned world; the planted numeric choice is gated green by `core_events` and
  refused here (`info_audit_integration_test.py`).

- **The CONSISTENCY GUARD** (`main.value_search.world_guard`, THROWING; §7.6): every returned world is checked slot ↔
  species ↔ facts ↔ the ENGINE's own view of the team (one `core_events` replay with no command: the opponent's first
  request, team order), and a disagreement raises `WorldConsistencyError` (a defect, never a `WorldError` exclusion).

### 7.3 The SCREEN: honest posterior draws of a seen mon's item, ability and spread (`world_screen`)

- **The honest prior.** Item and ability ∝ the species' Smogon usage. The spread (nature, EVs) ∝ (1 − ε)·J + ε·N·Π M:
  J the species' chaos spread table (every spread, not the top-25 cut, which covers only ~49 % of the pool's true
  spreads: the blind replica's measure), N its nature marginal, M each stat's EV
  marginal floored by a uniform over 0, 4, …, 252 (`EPS` 0.25, `FLOOR` 0.15), an EV total over 510 impossible. J alone
  is a coarse histogram: a pool spread Smogon never lists (a 16-HP-EV / 252-Def Vaporeon) has no consistent J
  candidate, and an honest player then believes the EVs that fit; the smooth term gives every legal spread mass. The
  Hidden Power: P(the set holds one) and P(type | one) from the moves prior.
- **One sampler per (seen mon, constraint)** — the constraint being the facts a lever told, the Hidden Power rule, an
  exact-HP requirement — serves every world of the decision, and, through `screen_cache`, every spec built at it: arms
  that differ only in other levers (the move dose, the placebo, the map) take the SAME seen-mon draws (common random
  numbers for their paired contrasts) and pay the screen once.
- **The screening world** holds every OTHER opponent mon at its true facts (with each observable event depending on one
  opponent mon's hidden facts — singles damage, speed order, item and ability triggers — the draw is then from P(facts
  of mon k | everything observed) and the true background leaks nothing); the combined world must pass the STRICT
  gate again (the `joint` count is the factorisation's measured residual).
- **The sampler**: (1) REJECTION — independent prior draws, each replayed; an accepted one is an exact posterior draw.
  (2) When the prior rarely hits the log, a CHAIN: a search phase climbs `divergence_score` (the number of leading
  protocol lines that match the truth's, plus a closeness of the first differing HP number) with moves DIRECTED by the
  first difference (`divergence_hint`: our mon's HP after their hit → their power, Atk / SpA, a Choice Band; their HP
  after ours → their bulk, HP / Def / SpD; a two-sided order → their Speed, signed by who acted first), restarts after
  `STALL` flat steps (from the best state shaken, or a fresh prior draw); once the log fits, Metropolis-Hastings on the
  prior × the gate (symmetric local moves at the prior ratio, independence draws at ratio 1), burnt in and thinned. No
  step reads the truth. Screening replays are LENIENT (`CHOOSEIF`: a recorded choice the diverged world no longer asks
  for is skipped, so a wrong candidate still yields the protocol its score reads); the world returned is STRICT.
  A sampler that never fits the log in `MAX_STEPS` makes the decision unbuildable (a `WorldError`: an EXCLUSION).
- **Legality inside the screen**: when a seen mon's moves are TOLD (a mass-1 truth lever), a drawn item or ability
  that breaks a clause with them, or that Showdown's validator rejects with them, is refused without a replay; never
  for drawn moves (the screening world holds the truth's unrevealed moves, so it would filter by them). A drawn
  seen mon's moves are held to the validator AFTER the screen (§7.2).

### 7.4 Cost and exclusions (measured, §10.3)

On 8 committed-bank decisions × 4 worlds, the box under a live training run (`nice -n 19`, three agent cores), one screen shared by every spec of a decision:

| spec | built | excluded | screening replays | seconds |
|---|---|---|---|---|
| `D50` | 8 | 0 | 32 | 4 |
| `F+M` | 8 | 0 | 1754 | 85 |
| `W90` | 8 | 0 | 134 | 8 |
| `off` | 8 | 0 | 3537 | 172 |
| `oracle` | 8 | 0 | 8 | 5 |
| `pilot` | 8 | 0 | 63 | 7 |

No decision was excluded.

A wider read under `off` alone (40 decisions × 2 worlds, turns 3–40, after the playout-tool check and the Hidden
Power IV patterns landed; `audit_off_40.json`): **39 built, 1 EXCLUDED** — a Tyranitar on a pool spread (184 HP / 104
Atk / 220 Spe, Adamant; its IVs inside the support) the chain did not fit in `MAX_STEPS` — every check PASS
(`playout_open`: 0 of 78 worlds refused; `gender_independence`: 0 gender-count rejections), 1,298 s contended.

The honest screen is the cost: a seen mon the prior fits is a few replays; one whose log pins a spread far from Smogon's table (a pool spread, a Choice Band at 0.5 % prior) takes the chain tens of steps × `SEARCH_WIDTH` replays. Specs that share the screen (`D50`, `W90`, …) cost almost nothing more.

### 7.5 The residuals (named)

- **Hidden counters** (`counters`, the builder's declared residual): the world replays the prefix on the TRUE battle
  seed, so a duration rolled BEFORE the root (sleep, confusion, Encore, a lock) is the truth's in every world (the
  study's A5 caveat). Hiding it needs a root-state resample in the playout core (not built).
- **The factorisation** of §7.3: measured by the `joint` rejections, never assumed.
- **Party-wide effects** (Beat Up's hit sizes read the party's base Attack, never-seen mons included): the log really
  does constrain the never-seen species there; the joint gate conditions on it by rejection, correct but slow.
- **A never-seen ratio'd mon on a true draw slot takes the truth's gender coin** (the draw's value, not its count):
  unobservable before the root and not a fact of the taxonomy.
- **The honest spread prior is a choice** (`EPS`, `FLOOR`): Smogon-derived (its joint table and marginals) with a
  uniform floor; it shapes the posterior only where the log leaves the spread loose.
- **Dice-silent facts under SEED REUSE** (the blind replica's finding 2; `SEARCH.partial`, reported in every world's
  `leak["partial"]`). The prefix is replayed on the TRUE seed, so a world must spend the truth's dice; a hidden fact
  that spends a die WITHOUT printing a line is then visible to the gate though no honest player sees it. Verified in
  `deps/pokemon-showdown`: King's Rock (a 10 % flinch roll added to most damaging moves, per hit), Focus Band (a
  1-in-10 roll on EVERY move damage its holder takes, before the lethal check), the contact abilities Static, Flame
  Body, Poison Point, Effect Spore, Cute Charm (a roll per contact hit taken), Shed Skin (a roll per statused turn),
  and a SPEED TIE (speedSort shuffles a tied group: a die only when a tie exists). Not one: Quick Claw (gen 3 rolls
  it battle-wide every turn). **Measured** (`main.info_audit`'s `seed_reuse`: COUNTERFACTUAL FLIPS of the true team,
  one hidden fact of one seen mon changed and gated exactly as a world is, against a passing unflipped control; 169
  committed-bank decisions, turns 3–40, 1,662 replays; `research_state/measurements/info_levers_2026-10-12/
  seed_reuse_200.json`):

  | fact | flips | the gate rejects | what that removes |
  |---|---|---|---|
  | a seen mon's Speed ±1 point (Spe EVs only: at ±1 only a tie changes anything) | 674 (decision, seen mon) | 34 (5.0 %), in 34 of 169 decisions (20.1 %): 6 both neighbours (the truth at a tie: its exact Speed pinned), 28 one (a neighbour would tie) | the honest posterior's mass on those Speed values |
  | a hidden item → King's Rock / Focus Band (truth not one; a neutral-item swap accepted) | 318 (159 mons) | 125 (39 %) | the prior's mass on them: 0.0075 summed, 4.7·10⁻⁵ per mon |
  | a hidden item that IS King's Rock / Focus Band → a neutral item | 0 | — | none of the bank's 1,386 pool mons (231 teams) holds one |
  | a hidden ability ↔ its dice-rolling alternative | 0 | — | no seen mon in the corpus had one unrevealed |

  So the item and ability legs are negligible on this pool; the SPEED-TIE leg is the real one: a fifth of decisions
  have a seen mon whose Speed the gate pins a point tighter than honest play could. Removing it needs a playout core
  that replays the prefix on fresh dice constrained to the printed outcomes (tier B of `design_world_sampler.md`
  §3.3.1, not built). The DAMAGE-ROLL leg of the same seed reuse is cured behind `WorldSampler(dice="rolls")` (tier A,
  BUILT 2026-10-19, default OFF: each damage roll's value any roll that prints what the searched side saw, the stream
  aligned, the world weighted by Π k / 16; exclusions 20 → 6 of 60 rebuilt states).
- **IVs outside the screen's support** (measured on the bank's 231 teams, 1,386 mons): of 609 Hidden Power sets, 146
  use the canonical 30 / 31 pattern, 253 another 30 / 31 pattern (both in the support), **210 (34 %) IVs below 30**
  (mostly a minimised Attack, e.g. 31 / 2 / 31 / 30 / 31 / 31 for Hidden Power Fire); of 777 other sets, 258 (33 %)
  are not all 31 (mostly 0 Attack IVs). The screen draws neither: where the log pins such a stat (a Rapid Spin, a
  confusion hit) no world fits and the decision is EXCLUDED (counted), never misbuilt; where it does not, the world
  holds a different but log-consistent value. **Owner's ruling (2026-10-12): no IV prior below 30** — a minimised
  Attack is taken as full Attack ("0 attack adds so little value. It only matters when a pokemon is confused and
  hurts itself in confusion or in a struggle end game"); those decisions stay excluded, counted.

### 7.6 The world CONSISTENCY GUARD (BUILT 2026-10-15, after a GIGO: `main.value_search.world_guard`)

**The defect.** The builder draws each never-seen mon FOR a slot, then the gender placement (§9) permutes the drawn
never-seen mons among the never-seen slots. The packed entries moved whole, so every WORLD was the team its text named
and every playout ran on it; but `Worlds.facts` stayed indexed by the DRAW slot, so a never-seen slot's record could
describe another mon, and the never-seen species `match` compared a species with the truth of the slot it was drawn
for. Measured on every stored Tier-1 world (125,176 worlds, 278,520 never-seen records): **128,822 (46 %) records at
another mon's slot, in 44,916 worlds (36 %) of 7,122 state × arm cells**; 0 in the arms that pin the never-seen
species to its slot (`F+U`, `O−M`) and in POOL (no builder records); 0 SEEN-mon records (seen mons never move); the
multiset of records equals the team's in every world (a pure permutation). The class: a per-slot record keyed by the
slot before a permutation — not species-vs-slot indexing, not formes (a gen3ou team never holds two mons of one `num`).

**The fix.** `place_genders(..., src_of=…)` reports the permutation (it spends no extra random number: the same worlds,
byte for byte, before and after on 24 decision × spec cells); the builder moves each record with its mon
(`drawn_slot` keeps where it was drawn) and reads the species flags at the returned slot (`match` set-based,
`slot_match` at this slot; §7.2).

**The guard (THROWING).** Before a `Worlds` is returned, ONE `core_events` replay of the J world teams with no command
gives the ENGINE's view of each (the opponent's first request: team order). Per slot: the record's slot and species,
the entry's species and the engine's are one; a seen mon sits in its true slot and is recorded seen; the engine's item,
ability, moves (as `packed.typed_moves` reads the entry, §7.7: a packed name variant such as `wisp` resolved; no move
twice) and EXACT stats (max HP included) are the entry's; every `match` flag (item, ability, spread, moves, the never-seen
species, `true_mon`) recomputed from the entry against the truth OF THAT MON equals the record's. A disagreement raises
`WorldConsistencyError` (never a `WorldError`: a defect is not an exclusion). Validated on 9,656 archived Tier-1
worlds (120 states, every builder arm): with the records re-paired by species, 0 false positives on species, items,
abilities, moves, stats and flags; the 1 world it flagged is a real defect (below). The audit suite reads the same
check as a verdict (`consistency`, §10.1), so a world from another builder or a tampered one is read too.

**What it found beyond the mis-pairing.** POOLSETS (the Tier-1 measurement's pool-set prior) built 7 worlds (7 of ~1,140
states) whose Gengar holds `wisp` AND `willowisp` — Will-O-Wisp twice: one pool team packs the move as the Showdown name
variant `wisp`, and `packed.typed_moves` did not resolve it. Fixed the same day: §7.7.

**A builder that records no provenance (POOL).** A world that IS a pool team carries no fact records, so it never reached
`WorldSampler.sample`'s guard. `world_guard.check_teams(root, teams, key)` is THE guard for such a builder: it derives the
records from the entries (`derived_facts`; the `match` flags then hold by construction and are NOT an independent check)
and runs the checks that stay independent — shape, slot ↔ species ↔ engine, a seen mon in its true slot, a never-seen
slot's species, entry ↔ engine (item, ability, moves, exact stats, no move twice). Validated on 120 archived POOL states
(960 worlds, both checkpoints): 0 false positives; and it flags all 7 defective POOLSETS worlds (§7.7). POOL's builder
(`measurements/hidden_info_tier1_2026-10-13/pool_worlds.py`, a frozen measurement script) is to call it on the worlds
`pool_sample` returns (`WG.check_teams(root, teams, key)` before `return PoolWorlds(...)`). **NOT WIRED (2026-10-15)**: the
script lives in a measurement folder the unit that built the guard was told not to edit; the hunk is in the ledger entry
of that day (GIGO FIX, packed tokens, FINDING 1). Any new builder with no records calls it from day one.

### 7.7 The packed-token reader (BUILT 2026-10-15, a GIGO class fix: `agents.info_levers.packed`)

**The defect.** A packed team may spell a move by one of Showdown's name variants (`wisp` for `willowisp`, `sd`, `hpice`,
… 44 in `data/pokemon/gen3_move_aliases.json`): the engine resolves it on its own, so the battle is the same. A reader that
compared the RAW token with an id from play (a protocol line, a prior, the log) saw another move. `typed_moves` was that
reader for the whole search / lever stack, so: a TRUE `wisp` slot was not recognised as revealed when Will-O-Wisp was
used (the choice attribution missed, the slot was redrawn and the gate conditioned it back), and POOLSETS' per-species
move table held `wisp` beside `willowisp`, so 7 worlds of the Tier-1 study ran Will-O-Wisp twice. Exactly ONE of the 719
pool teams spells a variant (team 555, Gengar `wisp`, from `data/teams/others/giraffe/dc99ab0afa2b1da9.txt`); no pool
team and no `data/` file spells a variant of a species, item, ability or any other move (the census, below).

**The fix (the class).** A packed token is not an id until it is resolved, and there is ONE reader:
`agents.gen3_data.moves.canonical_id` (the Showdown id, then the variant table — the same table the Rust dex resolves a
packed move with) behind `packed.move_id`; `packed.typed_moves` (variants resolved, a bare Hidden Power typed by the IVs),
`species_id`, `item_id`, `ability_id`, and `utils.showdown_id.to_id_str` as the one normaliser (the duplicate `toid` /
`_id` of `packed`, `revealed` and `world_screen` now delegate). Every read of a packed team's species / item / ability /
moves in `info_levers`, `value_search`, `info_audit` and `team_legality` goes through them. **Three throwing guards:**
`packed.require_known` (a species / item / ability / move the gen-3 data cannot resolve raises `PackedError`; called on
the TRUE team by `worlds.prepare_root` and `inject.TrueTeam.of` — an unknown item used to read as "no item", num 0), the
move draw refuses a prior table that names a move by a variant (`draw_moves`: it would draw the move twice), and
`world_guard` compares the reader with the ENGINE (it no longer re-resolves variants itself). The data carries a variant
table for MOVES only; Showdown has species (506 in `aliases.ts`), item (`cb`, `band`, `lefties`, `nmi`, `bj`, `bjuice`)
and ability (`intim`, `stag`, `vabs`, `wabs`) variants too, which the data does NOT carry and the Rust dex does not
resolve: a team that spells one raises at `require_known` rather than reading as another id.
**Resolved at READ time, not at load/pack time:** canonicalising team 555 in `utils.team_packing` would change the pool's
packed bytes (the pool's sha256 `4083c86bdffa…` that every Tier-1 POOL row records, and the raw-equality of the 600
registered opponent teams per checkpoint to a pool team that POOL's `true_indices` and the identity checks rest on) to
repair a spelling the engine already reads correctly.

**Impact on past reads** (stored Tier-1 archive, 2026-10-15; `designs/research_state/ledger.md`): the true-team branch
(an F / FN / O-* world built for a state whose TRUE opponent team spells a variant) touches **0 states** (0 of the
registered + supplement states of both checkpoints and 0 of the 20,712 `BANK_V1` decisions spell one on the opponent's
side): F, F+M, F+U, O−M, O−U, FN, D50, D90, W90, MAP100 and the oracle identity are unmoved. POOL: 4 worlds in 3 rows
hold team 555's `wisp` (a legitimate world, same game); no POOL candidate set changes (recomputed with the old and the
new reader on all 1,200 stored rows, 35 of which reveal Gengar's Will-O-Wisp: identical in every row); unmoved. POOLSETS: **7 worlds (7 of 9,112; 4 end2, 3 prod) held Will-O-Wisp
twice** and 2 more held `wisp` once; the reader also shifts the Gengar table (pool Gengar sets 182: Will-O-Wisp 118 → 119,
i.e. inclusion 0.648 → 0.654, and the 1-set `wisp` mass is gone), which touches the 2,208 POOLSETS worlds (728 rows) that
hold a Gengar. Exact leave-one-out of the 7 defective worlds (their state's scorers recomputed on the other 7 worlds): 6 of
the 7 states are scored (the 7th POOLSETS row was refused by the playout), an IMP changes in 3 of them, and the largest
change of the POOLSETS arm-mean IMP over any scorer is 3.3e-4 (LEAF-K8: +4e-5) over 1,138 scored states, against the
r_POOLSETS interval [−0.19, +0.13]: **stamped, not re-run** (a rebuild would re-draw only the Gengar-holding worlds).

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
  posterior stay consistent. A TRUE Hidden Power is no typed target (§1.1): the moves lever pushes its PRESENCE,
  presence' = (1 − p)·presence + p, and leaves the network's type posterior; a `spread:seen` lever pushes its TYPE,
  type' = (1 − p_s)·type + p_s·δ(true type); the 16 typed channels become presence'·type'. The placebo is 4 − r moves drawn ∝ the network's own belief excluding the truth and the
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

## 9. THE GENDER PAD (BUILT: the engine `98146ddb`, the builder's wiring)

The engine draws one gender at team construction for every mon of a gender-ratio species whose packed set names none,
so the pre-root dice depend on which never-seen slots draw. The v2 builder placed ratio'd never-seen mons on the true
draw slots, so a world needed at least as many as the truth drew (m ≥ r) and the gate conditioned the never-seen
species on that count (accepted P(m = r) 0.67 vs the honest 0.57; `design_value_discrimination.md` Amendment 8). The
engine's PAD — a packed gender `M*` / `F*` / `N*` keeps the named gender and still spends the one turn-0 draw
(`utils.team_packing.GENDER_DRAW_PAD`; `designs/rust_sim/module_map.md`, the `team.rs` row; the marker-free path is
Showdown's exactly, its Rust tests and differential / fuzz rungs green) — lets ANY never-seen mon sit on a true draw
slot. HEAD's `place_genders` uses it: no world is rejected on its gender count any more, so that conditioning is gone
by construction (the audit's `genders` check holds every world's draw slots to the truth's; `gender_independence`
fails any world rejected on its gender count, and its PLANT — the builder with the pad switched off, at a decision
where the pilot's builder needed it — must record such rejections: the blind replica's finding 1, seed reuse through
the turn-0 gender dice, is closed by the pad and caught without it). A padded team is a SEARCH
world only: it never reaches Node's validator or a live server (Showdown would read `M*` as unnamed).

---

## 10. THE AUDIT SUITE (BUILT: `main.info_audit`)

ONE suite over all three consumers; every check passes or fails DETERMINISTICALLY (fixed committed-bank battles,
fixed seeds; a statistical bound only where its planted defect sits far beyond it).

### 10.1 The checks

| consumer | check | what it catches | where |
|---|---|---|---|
| search | `template_identity` | the choice template rendered against the TRUE team reproduces the true prefix (a rewritten slot one off is invisible to a replay when the choice never executed) | `checks.py` |
| search | `oracle_identity` | the `oracle` preset builds exactly the true team and prefix | `checks.py` |
| search | `regate` | every returned world replays STRICTLY to the searched side's byte-equal rows and protocol (+ the seen mons' exact-HP match, descriptive) | `checks.py` |
| search | `consistency` | each slot's fact record names the mon the ENGINE built there and every match flag is that mon's (the builder's own throwing guard, §7.6, as a verdict) | `checks.py`, `value_search/world_guard.py` |
| search | `legality` | no world breaks a gen3ou rule the true team does not, and every world passes Showdown's validator (`showdown_invalid`, F2) | `checks.py`, `value_search/world_legality.py` |
| search | `genders` | the never-seen slots that draw (or pad) are exactly the true team's draw slots | `checks.py` |
| search | `gender_independence` | no world was rejected on its turn-0 gender-draw count (seed reuse through the gender dice; the pad closes it) | `checks.py` |
| search | `playout_open` | every returned world opens in the playout tool (`rust_env_playout_open`), not only in the gate's `core_events` replay | `checks.py` |
| search | `seed_reuse` | DESCRIPTIVE: the dice-silent residual's size by counterfactual flips (§7.5); fails only if its unflipped control does | `checks.py` |
| search | `provenance` | every fact is revealed / told / drawn exactly as the spec says (told without a lever, a mass-1 truth lever not the truth, a revealed fact not the truth, a placebo equal to the truth) | `checks.py` |
| search | `leak:moves:seen:drawn` | a seen mon's DRAWN moves match the truth no more often than their exact conditional-Poisson inclusion predicts (z ≤ 4) | `checks.Leakage` |
| search | `dose:moves:seen:told` | a TOLD move lever hits the truth exactly as often as its mixture says (\|z\| ≤ 4): the dose is the registered one, neither diluted nor sharpened | `checks.Leakage` |
| search | `leak:{item,ability}:seen:proposals` | the screen's independent proposals match the truth no more often than the prior's own mass on it (z ≤ 4) | `checks.Leakage` |
| log source | `log_consistency` | the Python source of what play revealed against the Rust reading the observation encodes, every decision: a fact the source claims that the reading does not show (the builder's leak direction) fails | `checks.py` |
| injection | reach / pins / placebo / map / item / spread / `off` bit-identity | §8's teeth | `agents/info_levers/inject_integration_test.py` |
| observation | `off` / `species` / `full` byte-identity, the declared-cell guard, label alignment, the mass Bernoulli | §6 | `rust_env/tests/oracle_reveal_test.rs`, `utils/rust_env/oracle_reveal_integration_test.py` |

**Planted defects (each must flip its check):** a template slot rendered one off (`template_identity`); a seen mon's
item swapped in a returned world (`regate`); a never-seen mon's gender flipped (`genders`); a world that copies a seen
mon's true moves but reports a draw (`leak:moves`, z > 6 against an honest z < 2.5); a screen whose item proposals peek
(`leak:item` proposals; the synthetic unit test, since a prior that already puts most mass on the true item gives the
real-battle version little power); two never-seen records swapped back to the pre-2026-10-15 pairing (`consistency`,
and the builder's guard raises); a source that claims a move play never named (`log_consistency`); an injection one
move channel off (the reach check); provenance violations of each kind (unit); the builder without the gender pad
(`gender_independence`); a numeric choice token `core_events` gates green (`playout_open`'s refusal); a control gate
that rejects everything (`seed_reuse`); an injection's Hidden Power type moved by the moves lever (§1.1; the
injection's `the_hidden_power_type_is_a_spread_fact` test); a never-seen set planted with an egg-move pair no father
passes on (`legality`'s `showdown_invalid`, F2). Plus the hidden-info study's six A1–A3
plants and its gender plants, which these generalise.

### 10.2 The tiers

- **Routine gate:** the unit checks (`checks_test.py`, `spec_test.py`, `revealed_test.py`, `mixture_test.py`,
  `worlds_test.py`) and two early-turn decisions under `oracle` / `pilot` / `off` / `F+M` with the teeth
  (`info_audit_integration_test.py`); the injection's teeth.
- **Slow tier:** twelve decisions under ten specs (`info_audit_slow_test.py`).
- **By hand:** `python -m main.info_audit --n N --worlds J --specs … --out report.json` (one JSON report: every
  verdict, the per-spec cost and exclusions, the per-class truth-match tables).

### 10.3 The read (2026-10-12)

8 committed-bank decisions × 4 worlds, specs `D50`, `F+M`, `W90`, `off`, `oracle`, `pilot`: **every check PASSES**.

| check | verdict | counts |
|---|---|---|
| `dose:moves:seen:told[D50]` | PASS | expected 299.16, slots 396, true_hits 290, z -1.12 |
| `dose:moves:seen:told[F+M]` | PASS | expected 396.0, slots 396, true_hits 396, z 0.0 |
| `dose:moves:seen:told[W90]` | PASS | expected 25.97, slots 396, true_hits 24, z -0.46 |
| `dose:moves:seen:told[oracle]` | PASS | expected 396.0, slots 396, true_hits 396, z 0.0 |
| `genders[D50]` | PASS | mismatches 0, worlds 32 |
| `genders[F+M]` | PASS | mismatches 0, worlds 32 |
| `genders[W90]` | PASS | mismatches 0, worlds 32 |
| `genders[off]` | PASS | mismatches 0, worlds 32 |
| `genders[oracle]` | PASS | mismatches 0, worlds 32 |
| `genders[pilot]` | PASS | mismatches 0, worlds 32 |
| `leak:ability:seen:proposals[D50]` | PASS | expected 162.99, proposals 172, true_hits 164, z 0.34 |
| `leak:ability:seen:proposals[F+M]` | PASS | expected 159.42, proposals 168, true_hits 159, z -0.15 |
| `leak:ability:seen:proposals[W90]` | PASS | expected 210.66, proposals 221, true_hits 210, z -0.21 |
| `leak:ability:seen:proposals[off]` | PASS | expected 162.99, proposals 172, true_hits 164, z 0.34 |
| `leak:item:seen:proposals[D50]` | PASS | expected 58.5, proposals 82, true_hits 58, z -0.12 |
| `leak:item:seen:proposals[F+M]` | PASS | expected 60.42, proposals 84, true_hits 59, z -0.35 |
| `leak:item:seen:proposals[W90]` | PASS | expected 66.04, proposals 99, true_hits 66, z -0.01 |
| `leak:item:seen:proposals[off]` | PASS | expected 58.5, proposals 82, true_hits 58, z -0.12 |
| `leak:moves:seen:drawn[off]` | PASS | expected 203.66, slots 396, true_hits 202, z -0.18 |
| `leak:moves:seen:drawn[pilot]` | PASS | expected 223.04, slots 396, true_hits 218, z -0.55 |
| `legality[D50]` | PASS | violations 0, worlds 32 |
| `legality[F+M]` | PASS | violations 0, worlds 32 |
| `legality[W90]` | PASS | violations 0, worlds 32 |
| `legality[off]` | PASS | violations 0, worlds 32 |
| `legality[oracle]` | PASS | violations 0, worlds 32 |
| `legality[pilot]` | PASS | violations 0, worlds 32 |
| `log_consistency` | PASS | decisions 98, items_agree 139, items_obs_only 0, items_python_only 0, mons 246, moves_agree 217, moves_obs_only 0, moves_python_only 0 |
| `oracle_identity` | PASS | mismatches 0, worlds 32 |
| `provenance[D50]` | PASS | facts 768, violations 0 |
| `provenance[F+M]` | PASS | facts 768, violations 0 |
| `provenance[W90]` | PASS | facts 768, violations 0 |
| `provenance[off]` | PASS | facts 768, violations 0 |
| `provenance[oracle]` | PASS | facts 768, violations 0 |
| `provenance[pilot]` | PASS | facts 768, violations 0 |
| `regate[D50]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[F+M]` | PASS | failures 0, seen_exact_hp_equal 69, seen_mons 124, worlds 32 |
| `regate[W90]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[off]` | PASS | failures 0, seen_exact_hp_equal 70, seen_mons 124, worlds 32 |
| `regate[oracle]` | PASS | failures 0, seen_exact_hp_equal 124, seen_mons 124, worlds 32 |
| `regate[pilot]` | PASS | failures 0, seen_exact_hp_equal 124, seen_mons 124, worlds 32 |
| `template_identity` | PASS | commands 184, mismatches 0 |

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

- **The screen's factorisation** (§7.3: each seen mon screened with the others at their true facts) is measured by
  the strict joint gate's rejections, not proven; a party-wide effect (Beat Up) is the known coupling.
- **The chain is an approximate posterior sampler.** Rejection draws are exact; a chain started by the directed search
  is Metropolis-Hastings after a short burn-in (`BURN` 6, `THIN` 2): a tightly pinned mon's draws may be correlated
  across worlds. No step reads the truth; the leak reads (§10) check the proposals and the drawn moves, and a spread
  that matches the truth EXACTLY stays rare (3.2 % in §10.3).
- **The proposal check's power** is low where the prior already puts most of its mass on the true item (Leftovers);
  the moves check is the strong one. Neither replaces the independent re-implementation the study registered (§13.1).
- **Cost.** The honest screen dominates: ≈ 28 s per decision at 4 worlds on this contended box (§7.4), the hardest
  decisions minutes. The study's F arm over 600 states × 2 checkpoints is hours of CPU, shared by every arm that keeps
  the seen mons' item / ability / spread hidden (`screen_cache`). A decision the search cannot fit in `MAX_STEPS` is an
  EXCLUSION, counted, never a silent leak.
- **The honest spread prior is a choice** (`EPS`, `FLOOR`, §7.3): Smogon-derived with a uniform floor.
- **Hidden counters** stay the builder's residual (§7.5) until a root-state resample exists.
- **Seed reuse** leaks dice-silent facts (§7.5, measured): speed ties in 20 % of decisions; King's Rock / Focus Band
  and the contact abilities negligible on this pool. Same cure as the counters (a fresh-dice prefix replay).
- **IVs below 30** (a third of pool sets) are outside the screen's support (§7.5), by the owner's ruling: an
  exclusion where the log pins the Attack (a confusion self-hit, a Struggle endgame), counted.
- The observation cannot state an exact HP, a hidden duration, a placebo or a map fact (§3): the knock-in forks'
  wrong-moves arm needs a placebo in the observation, which needs a prior sampler on the Rust side (not built).

---

## 13. THE HUMAN-DATA LEVERS (PROPOSED, NOT BUILT; REVISED DRAFT 2026-10-25)

The human-data switch `--human-data` is a second spec family beside the reveal spec, so one place answers "what did
this run get that plain self-play does not". It reveals no hidden fact of the battle being played: it changes the LOSS
or the POOL, never the observation, and never combines with `--oracle-reveal`. Its grammar, provenance, refusal rules
and the snapshot → rowset → model chain have ONE home, [`human_data_architecture.md`](human_data_architecture.md) §6
and §5; the plan is [`design_human_replay_lever.md`](design_human_replay_lever.md).

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-12 | **This doc CREATED; the SPEC and the LOG SOURCE BUILT** (owner: "a single dried out implementation for giving both search and training additional information … so that we can understand what the ceilings are and what is our constraint and how much value there would be in improving determinization") | One declarative `InfoReveal` (seven classes × seen / unseen parts, a mass and a source per lever; presets for the X32 levels, the oracle, the pilot's builder and every study arm; canonical text + hash; per-consumer support tables that REFUSE what a consumer cannot honour and NAME its residual leaks); one protocol-derived source of what play revealed, conservative by design | three separate mechanisms (the status quo: incomparable, one leaked); a soft "known = p" cell in the observation (an input the network never sees in play; a Bernoulli keeps every row a possible observation); the observation rows as the reveal source (they miss protocol-only reveals: the study's A2) | `design_hidden_info_binding.md` §2.2, §16.4; `design_value_discrimination.md` Amendment 8 |
| 2026-10-12 | **The SEARCH WORLD BUILDER made HONEST BY DEFAULT, the GENDER PAD wired, the AUDIT SUITE built** | `WorldSampler(reveal=…)`, default `off`: a seen mon's item, ability, spread (its IVs: the Hidden Power type and presence) and so its exact HP are DRAWN and screened against the log by a per-(seen mon, constraint) sampler — prior rejection, then a chain whose search is directed by the first differing line and scored on lenient replays, Metropolis-Hastings once the log fits — shared across a decision's worlds and specs; the pilot's builder is the `pilot` preset; any never-seen mon may sit on a true gender-draw slot (the engine pad); one audit suite (`main.info_audit`) over all three consumers, read on 8 committed-bank decisions × 6 specs: 41 checks PASS (the dose of the told move levers exact), no exclusion, a seen mon's item 1.000 → 0.750, exact spread 1.000 → 0.032, exact HP 124/124 → 70/124 | keeping the pilot's builder as the default (a search not honest by default measures the wrong thing whenever a caller forgets); joint rejection over every seen mon (hopeless: a log pins each attacker's stats to a window); a pure Smogon spread table as the prior (a pool spread outside it has no consistent candidate: the Vaporeon 16 HP / 252 Def case); starting the chain at the truth (a leak until it mixes); filtering a drawn item by the truth's unrevealed moves (a leak) | §7, §9, §10.3; `research_state/measurements/info_levers_2026-10-12/` |
| 2026-10-12 | **The blind replica's three findings answered** (orchestrator relay) | (1) seed reuse through the turn-0 GENDER dice: closed by the engine pad, a `gender_independence` check and its plant (the builder without the pad); (2) DICE-SILENT facts: seed reuse kept, the leak NAMED (`SEARCH.partial`) and MEASURED by counterfactual flips (speed ties 5.0 % of seen mons / 20 % of decisions; King's Rock, Focus Band, contact abilities negligible on this pool); (3) every world OPENED by the playout tool before it is returned, a `playout_open` check and a planted numeric choice. Plus: the Hidden Power's TYPE is a SPREAD fact, its presence a MOVES fact, in all three consumers (§1.1); the screen's Hidden Power state spans every 30 / 31 IV pattern | avoiding seed reuse (needs a playout core that replays on fresh dice constrained to the printed outcomes — not built); refusing `moves` without `spread` on Hidden Power sets (the observation writes it bare instead, byte-safe); an IV prior below 30 (no Smogon source: the owner's call) | §1.1, §7.2, §7.5, §9, §10.1; `seed_reuse_200.json` |
| 2026-10-12 | **(a) BUILT: `--oracle-reveal <spec>`** | The flag takes any spec OBS honours and records its LABEL; the Rust encoder's `OracleSpec` (mass per class × part, strict canonical parse, per-(episode, side, mon, class) Bernoulli by a pinned hash); `off` / `species` / `full` byte-identical (the three digests unchanged, the clause texts reproduce them); the resume gate compares meaning; `MODEL_CONFIG_VERSION` unchanged (no new field; the three levels' recorded values unchanged) | a soft "known = p" cell (a row the network never sees in play); per-DECISION draws (a fact flickering on and off within an episode is not a belief a player could hold); a lenient Rust parser (two spellings of one spec would be two env specs) | `src/rust_env/tests/oracle_reveal_test.rs`, `src/utils/rust_env/oracle_reveal_integration_test.py` |
| 2026-10-12 | **The DIAGNOSTIC-FORK consent BUILT** | `--diagnostic-fork-consent "<reason>"`: a genuine fork (the ONE `is_same_run_checkpoint` predicate) may change `--oracle-reveal` from its parent's recorded spec; refused on a same-run flip, with no change, without `--model`, and into a registry-named run; stamped in the child's immutable lineage block; a stamped lineage (own or ancestor) is refused by `main.baselines set` and flagged by `check` | an allow-change flag on the resume check (`--allow-belief-grad-mode-change`'s shape: would also allow a same-run flip); refusing a registered PARENT (the knock-ins fork the adopted arm itself); refusing `--arch production` / `endstate` on the fork (orthogonal to the observation mode) | `src/main/train/diagnostic_fork_test.py`; `main.checkargs` / `main.launcher --dry-run` on the §8.3 argv with `--oracle-reveal ki_moves` |
| 2026-10-14 | **`main.h2h --oracle-reveal-mode withheld` BUILT: the honest-play read of a knock-in fork** (Training Oct's spec for the §8.4 secondary cross) | A fourth mode: the oracle-trained checkpoint of a cell plays at `off` BY DECLARATION against a plain one; its own protocol `gen3_eval_protocol_v1_h2h_oracle_withheld` (in `PROTOCOL_OF_MODE`, `eval_ledger.schema.PROTOCOLS`, the design's closed-list table that the reader gate reads), never pooled; `(off, off)` accepted for an oracle cell ONLY under it (`off` still refuses an oracle checkpoint); the stamp `{mode: withheld, player: off, opponent: off, withheld_level: <recorded spec>}`; every oracle row (all three modes) gains an `oracle` provenance block (recorded spec, canonical text, reveal hash, diagnostic-fork consent). Oracle-vs-oracle REFUSED under it | no mode for it (a plain `off` over an oracle checkpoint is the silent dependence read the refusal exists to stop); loosening `off`; permitting a both-withheld cell (two dependence costs, one `withheld_level`; a cross of two knock-in forks is not a registered cell; cheap to add later as a deliberate stamp change); reading the hash off `metadata.json` (a fork's says `off`) | `src/main/h2h/reveal_test.py`, `play_test.py`, `play_reveal_integration_test.py` (withheld = `off`'s games bit for bit in either seat); CPU smoke `rb_hib_moves_s3101` vs `rb_hib_ctrl_s3101` |
| 2026-10-15 | **The REVEAL PROVENANCE contract BUILT (`gen3_reveal_provenance_v1`; GIGO FIX)**: the knock-in forks' `metadata.json` recorded `oracle_reveal: off` beside a `model_config.json` that said `moves:seen@0.95` | The resolved run config is the ONE source; `metadata.json` copies it from the env core's declaration (`env_core.oracle_reveal`); a post-load table re-applies every `resume_immutable` extractor attribute (`agents.model.resume_reapply`); `save_model_snapshot` / `record_checkpoint` raise `RevealProvenanceError` and the trainer exits `FATAL_CONFIG` on a record that disagrees; the readers (`oracle_reveal.run_provenance`, `main.lineage`, `main.sidecar_audit`) prefer `model_config.json` and flag the 11 existing knock-in forks KNOWN-BAD (files untouched). Impact verdict: PROVENANCE ONLY (the attribute is read by `run_io` alone; training and eval observations come from the resolved config) | rewriting the forks' `metadata.json` (`models/` is read-only; blocks are immutable); writing the true reveal into the policy's pickled `features_extractor_kwargs` (it is the forward-identity `forward_fingerprint` / `EngineArch` read: a fork would split from its plain control and the fork-vs-control h2h cell would be refused); reading the mode from the extractor and fixing only its re-apply (one witness, no cross-check) | `src/agents/model/reveal_provenance_test.py` (each assertion fails on revert), the CPU parent-then-fork smoke, `python -m main.lineage models/rb_hib_moves_s3101` |
| 2026-10-12 | **IVs below 30: taken as full Attack** (owner's ruling on the builder's finding 4: "0 attack adds so little value. It only matters when a pokemon is confused and hurts itself in confusion or in a struggle end game. You can toss those in as full attack.") | No new IV prior: the screen keeps every IV at 30 / 31 (the Hidden Power's patterns, else 31), so a minimised-Attack set is built and believed at full Attack; a decision whose log pins the Attack stays an EXCLUSION, counted and reported | an IV prior below 30 (no Smogon source; little value) | §7.5, §12; `world_screen.ivs_of` |
| 2026-10-15 | **The world CONSISTENCY GUARD BUILT, after a GIGO in `Worlds.facts` (§7.6)**: the gender placement permuted the drawn never-seen mons but their provenance records stayed at the slot each was DRAWN for (46 % of the never-seen records of the stored Tier-1 worlds named another slot's mon), and the never-seen species `match` compared a species with the truth of its draw slot | Each record moves with its mon (`place_genders(src_of=…)`, `drawn_slot`; the worlds byte-identical); the species flags read at the returned slot (`match` = a TRUE never-seen species, set-based; `slot_match` = the truth's species at this slot); every returned world checked slot ↔ species ↔ facts ↔ the ENGINE's view (item, ability, moves with aliases resolved and none twice, exact stats, every match flag recomputed) and a mismatch RAISES `WorldConsistencyError`; the audit's `consistency` verdict | keeping slot-based `match` (it measures neither the slot the mon holds nor whether the team holds a true species); matching records by species in each READER (the Tier-1 FN provenance does; every other reader would have to know); an exclusion instead of a raise (a record that is not its world is a defect) | `main/value_search/world_guard.py` (+ `_test.py`, `_integration_test.py`); impact: no world, playout or registered verdict changed; the Tier-1 Amendment-3 U hit-rate proxy and the FN species hit move (ledger 2026-10-15) |
| 2026-10-15 | **The PACKED-TOKEN READER BUILT, a GIGO class fix (§7.7)**: `packed.typed_moves` compared a packed team's RAW move token with ids from play, so the one pool team that spells Will-O-Wisp `wisp` (team 555) made a TRUE `wisp` slot unrecognised as revealed and POOLSETS' move table hold `wisp` beside `willowisp` (7 of its 9,112 Tier-1 worlds ran the move twice) | ONE reader: `gen3_data.moves.canonical_id` (the Showdown id through the variant table the Rust dex uses) behind `packed.move_id` / `typed_moves`, with `species_id` / `item_id` / `ability_id` and `to_id_str` as the one normaliser, used by every packed-field read of `info_levers`, `value_search`, `info_audit` and `team_legality`; THROWING guards: `packed.require_known` (an id the data cannot resolve, on the TRUE team in `prepare_root` and `TrueTeam.of`), `draw_moves` refuses a prior that names a move by a variant, `world_guard` compares the reader with the engine; `world_guard.check_teams` / `derived_facts` for a builder with no provenance records (POOL) | canonicalising at load / pack time in `utils.team_packing` (it changes the pool's packed bytes and sha256 that POOL's `true_indices` and the identity checks rest on, to repair a spelling the engine already reads); resolving variants in `world_guard` only (the reader stays wrong everywhere else); editing `data/` (rule 6: the team file stays as contributed) | `agents/info_levers/packed.py`, `agents/gen3_data/moves.py`, `main/value_search/{world_draws,worlds,world_guard,world_screen}.py` (+ `packed_test.py`, `inject_true_team_test.py`, `name_variant_integration_test.py`, `world_guard_test.py`, `worlds_test.py`, `moves_test.py`); impact: 0 states with a variant in the TRUE team (F / FN / O−* / D* / W90 / MAP100 unmoved), POOL unmoved, POOLSETS 7 worlds + a 1-set Gengar table shift, exact leave-one-out arm-mean IMP change ≤ 3.3e-4 (ledger 2026-10-15) |
| 2026-10-19 | **ONE Hidden Power impl** (owner: "Fix it so we only have 1 impl for the HP determinization") | `agents.gen3_data.hidden_power` is THE table (Showdown's validator fill, `fillStats(typechart.ts HPivs, 31)`) and THE formula (IVs → type / power); `packed.hp_type_of` / `hp_ivs` / `hp_iv_masks` delegate to it. The world screen's CANONICAL pattern (index 0, `HP_CANON` 0.8) is now Showdown's fill, not the most-31s pattern, which differed for Bug, Dragon, Fire, Flying, Grass, Ice, Psychic and Water; `world_draws`' never-seen Hidden Power spreads take the same fill. A world built after this commit can differ from one built before by a point of a stat on those 8 types | Keep the most-31s choice (rejected: a set a human builds through Showdown's teambuilder, and every pool set our teambuilder fills, carries the validator's fill, so it is the more likely pattern; and two tables had already drifted once, the Ghost row of `GEN3_HP_IVS`) | `hidden_power_test.py` (the table equals `typechart.ts`, the reverse formula on hand-worked spreads, every former copy IS the one table); `hidden_power_single_impl_test.py` (a grep guard over `src/` .py and .rs); the crate's `tests/hidden_power_fill_test.rs` (the Rust formula reads every `typechart.ts` fill back at 70 power). Runs pinned before this commit (X26 `9c033489`, the maturity queue `826bee89` / `198fa91e`) are unaffected |
| 2026-10-19 | **Showdown's VALIDATOR in the search world builder (F2 of `design_world_sampler.md`)**: 5.2 % of the stored honest worlds held sets no player can hold (egg / event move combinations) | §7.2: every drawn set and returned world passes the pinned `TeamValidator('gen3ou')` by REJECTION (`main.value_search.world_legality`; never-seen sets redrawn with their lever coins flipped once, seen mons' unrevealed moves redrawn after the screen, gender-made and team-level failures redrawn); the TRUE team must pass or `LegalityRuleError` raises; `world_guard.check_legal` and the audit's `legality` (`showdown_invalid`) re-check; a seen mon's stale Hidden Power type FIELD dropped (`world_screen.clear_stale_hp_type`). 316-state rebuild: 0 illegal worlds, no exclusion caused by the rule. Worlds before / after are not comparable | a derived per-species combination table (a second copy of the rules); re-weighting species or spreads by their sets' legality rate | `design_world_sampler.md` §6.1 / §7 F2; `measurements/world_legality_f2_2026-10-19/` |
| 2026-10-19 | **The damage-roll leg of seed reuse cured behind an option (F1 tier A of `design_world_sampler.md`)** | §7.5: `WorldSampler(dice="rolls")` (default `seed`) replays each damage roll on any value that prints what the searched side saw, the stream aligned, the world weighted; the speed-tie and silent-dice legs stay (tier B) | making it the default (changes every honest read); tier B now | `design_world_sampler.md` §3.3.1; `measurements/world_rolls_f1a_2026-10-19/` |
| 2026-10-25 | **§13 ADDED: the HUMAN-DATA levers registered as a second spec family (DESIGN, DRAFT; nothing built)** (owner's grant: human replays may train the policy, toggleable, off by default) | one flag `--human-data` with ONE clause (`bc` / `teams` / `kl` / `judge`) and a snapshot id, its own hash prefix `human_data_v1`, provenance as the reveal spec's | folding human data into `--oracle-reveal` (it reveals nothing about the battle played; mixing the two would make "honest" ambiguous); one flag per lever (the complexity budget) | `design_human_replay_lever.md` |
| 2026-10-25 | **§13 reduced to a pointer** (REVIEW C F10: the grammar, the hash prefix and the provenance rules lived in three docs and had already drifted on inheritance) | §13 keeps the boundary (no hidden fact; never with `--oracle-reveal`) and points at `human_data_architecture.md` §5–§6, which owns the grammar (`judge` dropped, rowset / model ids, fork and resume refuse without a typed clause) | restating the grammar here (a second home drifts) | `measurements/human_replay_quality_2026-10-25/RESPONSE.md` |
