# designs/ staleness census — 2026-09-08

**What this is.** A read-only assessment of the **explicit-only** `designs/` documents: the
top-level `designs/*.md`, the live-era `designs/ai_v12/` chapter, `designs/learning/`, and
`designs/references/`. Nothing here was edited — those documents change only when someone asks.
This census exists so a reader can tell, before opening one, whether it still describes the world.

**Scope note.** The always-current documents (`ARCHITECTURE.md`, every `CLAUDE.md`, the four
`designs/` trees that hold detail lifted out of a leaf, `research_state/UNDERSTANDING.md`) are NOT
assessed here — they are fixed in the same pass as the code that stales them. The append-only
histories (`CHANGELOG.md`, `research_state/ledger.md`) are never assessed for staleness at all:
recording what was believed at the time is their entire job.

**The bar used.** A doc is **MISLEADING** when a reader who trusts its own framing would act
wrongly — it presents itself as current, or instructs the reader to use it as current, and its
central claim is no longer true. A doc is **SUPERSEDED** when an always-current doc now owns its
subject. **PARTLY STALE** means the frame holds but named artifacts (flags, runs, numbers) have
moved. **HOLDS** means what it claims is still true, or it is explicitly dated history.

---

## Summary

| tree | docs | HOLDS | PARTLY STALE | MISLEADING | SUPERSEDED |
|---|---:|---:|---:|---:|---:|
| `designs/*.md` (top level, explicit-only) | 6 | 3 | 0 | 2 | 1 |
| `designs/ai_v12/` (the live chapter) | 7 | 2 | 2 | 2 | 1 |
| `designs/learning/` | 26 | 8 | 15 | 0 | 3 |
| `designs/references/` | 1 | 1 | 0 | 0 | 0 |

**The five a reader would be actively misled by, one line each:**

1. **`designs/model.md`** — calls itself "what the current model is" over a 2026-06-17 state table:
   wrong signature, wrong obs dim (3457 vs 2501), wrong config version (37 vs 113), wrong live run,
   wrong constants path — and calls `WinProbHead` a side readout "never in pi/vf" when it IS the
   critic.
2. **`designs/ai_v12/design_winprob_only_critic.md`** — the design **of record** for the live era
   opens "**Nothing here is built and nothing is sanctioned to run**", 68 M steps into the run it
   designed; its §1 "WHAT IS TRUE NOW" is the pre-migration config.
3. **`designs/ai_v12/launch_runbook.md`** — "the thing the training session executes from", whose
   three arms never launched, whose `$CLEAN` block contradicts the shipped era in six tokens, and
   which names the DEAD `ai_v12_01` as the live arm.
4. **`designs/learning/conditioning_architectures.md`** — present tense throughout about the
   z_arch/FiLM subsystem deleted at v78, with **no correction banner** (its two sibling notes have
   one).
5. **`designs/design_pathologies.md`** — instructs "review this before every retrain" over an ai_v4
   fixed-bot register from May.

**One live test disagrees with the shipped era.** `src/main/launch_runbook_test.py:128` asserts
`--draw-penalty -1.0` in the runbook's `$CLEAN`, while both shipped ai_v12 runs and
`designs/baselines.json`'s `production.config_overrides` pin **0.0**. The test is green because it
pins the *runbook*, which is itself stale — a gate holding a stale doc in place. Backlogged.

**A governance gap the census surfaced.** `src/claude_md_freshness_gate_test.py` validates paths
and `--flags` only inside `CLAUDE.md` files. Nothing checks `designs/ai_v12/*.md` or
`designs/learning/*.md`, which is why roughly ten deleted flags and two dangling `CLAUDE.md`
section pointers survive in them. Extending the gate's scan to those two trees is a small change
and would close the whole class; it is owner-gated because it would fail the tree today.

---

## 1. Top-level `designs/*.md`

| doc | lines | last touched | verdict | reason |
|---|---:|---|---|---|
| `ARCHITECTURE.md` | 1,279 | 2026-09-07 | *(always-current — not assessed)* | states the model as it is now; one stale parenthetical fixed in this pass (`MODEL_CONFIG_VERSION` 109 → 113) |
| `CHANGELOG.md` | 8,326 | 2026-09-07 | *(append-only — not assessed)* | history by construction |
| `CLAUDE.md` | 244 | 2026-09-07 | *(always-current — not assessed)* | the version map; live chapter `ai_v12` agrees with `ARCHITECTURE.md` |
| `deleted_flags.md` | 77 | 2026-09-06 | **HOLDS** | gate-enforced: `claude_md_freshness_gate_test.py` fails an uncited row and fails a row whose flag came back |
| `flag_registry.md` | 183 | 2026-09-06 | **HOLDS** | generated from `src/agents/model/flag_registry.py` and pinned by `flag_registry_test.py` — cannot drift silently |
| `production_config.README.md` | 130 | 2026-09-06 | **HOLDS** | provenance for the live mirror; run, signature and the 13 changed critic rows all match `production_config.json` |
| `model.md` | 274 | 2026-08-04 | 🚨 **SUPERSEDED / MISLEADING** | see §1.1 |
| `design_pathologies.md` | 230 | 2026-06-06 | 🚨 **MISLEADING** | see §1.2 |
| `pubval_deletion_decision.md` | 121 | 2026-08-15 | 🚨 **MISLEADING** | see §1.3 |

### 1.1 `model.md` — a running log that stopped running

It opens **"A curated, dated journal … This is the single narrative timeline: what the current
model is"**, and its state table is headed **"Current state (2026-06-17)"** — nearly three months
old and eight `MODEL_CONFIG_VERSION` bumps behind. Every headline figure in that table is wrong now:

| the doc says | true now | evidence |
|---|---|---|
| `` `ARCH_SIGNATURE` `` = `gen3_wish_wired_v1` | `gen3_critic_route_wave_v1` | `agents/model/model_version/constants.py` |
| obs vector **3457-dim** | **2501-dim** | `ARCHITECTURE.md` §obs; `Gen3ObservationEncoder.dimension` |
| `` `MODEL_CONFIG_VERSION` `` **37** | **113** | `agents/model/model_version/constants.py` |
| "Live training run `ai_v6_09_dmg_reattend_N_0617`" | `ai_v12_02_winprob_critic` | `designs/CLAUDE.md` version map, `ARCHITECTURE.md` |
| "The architecture-constant SoT is the module-level constants in `features_extractor.py`; `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` in `model_version.py`" | `arch_constants.py`, and `model_version/constants.py` | root `CLAUDE.md`; both files exist at the new paths |

Its architecture diagram still shows the ai_v6 phase chain (`DamageOperator` → `damage-reattend` →
`ProjectionAssembler`) and describes `WinProbHead` as a **side readout "never in pi/vf"** — under
`--critic winprob` the win-prob head **is** the critic (`ARCHITECTURE.md` §"WHICH readout is the
critic"). That single sentence would send a reader in exactly the wrong direction.

**Why it is the worst of the three.** `ARCHITECTURE.md` now holds "what is true now",
`CHANGELOG.md` holds "how it got here", and `research_state/UNDERSTANDING.md` holds "what we
believe now" — `model.md` was written to be all three before any of them existed. It duplicates
their subject wholesale and loses every race. Owner call: retire it with a one-line pointer at the
three, or resume it.

### 1.2 `design_pathologies.md` — a live instruction pointed at a dead run

The doc instructs, in its second paragraph: **"Review this before every retrain"**, and **"For each
fix marked *implemented*, check the 'expect to see' prediction against the new run."** An agent
that obeys it today reviews a register whose entire evidence base is
`models/run_20260531_182804` — arch **ai_v4**, `gen3_trapping_signals_v1`, obs **3321**, a
**fixed-bot** eval pool from before self-play existed. Its open items (P-0 "needs self-play", P-2
under-switching, P-3 stall loops) were the agenda of the ai_v5/ai_v6 eras and have since been
answered, reframed or closed by named ledger work; nothing in the file says so.

The findings themselves are good history and should not be deleted. What misleads is the standing
instruction and the absence of a header saying which era the register belongs to.

### 1.3 `pubval_deletion_decision.md` — a decision package for a decision that was taken

Its first line is **"Status: NOT DELETED. Prepared for owner decision, deliberately kept out of the
v78 batch."** V_pub was subsequently deleted: `designs/deleted_flags.md` records `--pubval-mode`
and `--pubval-coef` removed at **v88 `gen3_dead_flag_purge_v1`** (CHANGELOG L4078), together with
`agents.training.pubval`, `PubValHead`, `_pubval_loss` and `data/gen3_pubval.json`. No parser in
the tree defines either flag today. The document reads as an open question; the question is closed.

It stays valuable as the argument that produced the decision — it needs one line at the top saying
so, which is an owner-gated edit.

### 1.4 The historical `ai_vN/` chapters

`designs/CLAUDE.md` is the version map and marks each chapter's status (live / open / history), so
the folders are not unlabelled in practice. But only `designs/ai_v3/` carries a marker **inside**
the folder (`README.md`, the frozen ai_v3 digraph); `ai_v1`, `ai_v4`–`ai_v9`, `ai_v10`, `ai_v11`
have neither a `CLAUDE.md` nor a `README.md`. A reader who arrives at
`designs/ai_v6/design_*.md` from a search result — which is how these files are usually reached —
gets no signal that they are reading a closed era. A one-paragraph `README.md` per closed chapter
would fix the whole class; it is a docs-only change and it is owner-gated.

---

## 2. `designs/ai_v12/` — the live chapter

`designs/ai_v12/` is the **live** chapter, which is what makes its staleness expensive: these are
the documents an agent opens to find out what the running experiment is.

| doc | lines | last touched | verdict | reason |
|---|---:|---|---|---|
| `design_winprob_only_critic.md` | 1,437 | 2026-09-06 | 🚨 **MISLEADING** (header) · body PARTLY STALE | banner says nothing is built; the design shipped and is 68 M/75 M steps in |
| `design_winprob_behavior_coupling.md` | 716 | 2026-08-29 | **PARTLY STALE** | its Route 1 surface is REFUSED under the live critic; the production `--win-prob-coef` it states is wrong |
| `launch_runbook.md` | 724 | 2026-09-06 | 🚨 **SUPERSEDED** as an execution doc | its three arms never launched; it names the dead arm as live |
| `probe_risk_modulation_capstone.md` | 82 | 2026-08-30 | **HOLDS** | a deliberately frozen pre-registration carrying its own baseline |
| `promotion_dry_run_demo.md` | 111 | 2026-08-30 | **HOLDS** | self-labelled DEMO and dated; one dead absolute worktree path |
| `team_slate_40.md` | 430 | 2026-08-30 | **PARTLY STALE** | self-declares its baselines would move; the fleet campaign it feeds was overtaken by the pivot |
| `todo.md` | 27 | 2026-08-30 | 🚨 **MISLEADING** | names the wrong design of record and says nothing runs yet |

**The sentences.**

`design_winprob_only_critic.md` L3 — *"**[STATE 2026-09-06]** **DESIGN + GAP AUDIT. Nothing here is
built and nothing is sanctioned to run.**"* The document contradicts itself 500 lines later
("🟢 **THE FROZEN RUNG IS BUILT**", "**DECIDED AND BUILT**"), and `CHANGELOG.md` carries both
`gen3_winprob_critic_mode_v1` (v109) and `gen3_frozen_phi_actor_only_v1` (v110).

Same file, §1.1, under the heading *"WHAT IS TRUE NOW"* — *"`victory_value` **30.0** ·
`draw_penalty` **−35.0** … `hand_shaping` **true**"*. Now `1.0` / `0.0` / `false`
(`production_config.json`, and the run's own `model_config.json`). Its `value_tail_weight` "0.3,
ACTIVE" is `0.0`; its "Production **`shaping` at coefficient 0.05**" is `win_prob_coef: 1.0`, and
`ARCHITECTURE.md` §3.4 says the flag "is refused as a separate weight".

Same file, §5.1 — *"⇒ FRESH WEIGHTS. The signature bump forbids a warm start"*. **No signature bump
happened.** `ARCH_SIGNATURE` is still `gen3_critic_route_wave_v1` and `MIGRATION_FLOOR` is still 96;
`ARCHITECTURE.md` §3.4 states it outright: "It carries **no `ARCH_SIGNATURE` bump**". The run was
fresh, but not for the reason the design gives — and a reader who takes that reasoning forward will
mis-plan the next era's warm start. Its §5.3 deletion list also names ten flags as deleted that all
still exist in a parser; §3.7 of the same document carries the owner amendment "Refused, not
deleted" and §5.3 was never brought into line.

`launch_runbook.md` L5 — *"This document is the thing the training session executes from."* It is
not: the executed command is `design_winprob_only_critic.md` §5.4's. Its `$CLEAN` block still reads
`--draw-penalty -1.0 --win-prob-mode read_only --win-prob-coef 0.05 --value-dist-mode shaping
--value-dist-bins 51 …`, of which the last six tokens contradict the shipped era, and its arm
directories (`cw1_sparse`, `cw2_self_phi`, `cw3_frozen_phi`) do not exist while the document gives
runnable commands against them.

`todo.md` L7 — *"Plan of record: `design_winprob_behavior_coupling.md`"*. Both `designs/CLAUDE.md`
and `UNDERSTANDING.md` name `design_winprob_only_critic.md`.

---

## 3. `designs/learning/`

The learning notes are **conceptual and largely durable**, and the census bar for them is narrow:
a note is stale only where a *Gen3AI-specific* claim (a flag, a run, a number, a "we do X") has
moved. Judged that way, eight hold, fifteen are partly stale, three are superseded, and none is
merely wrong about the concept it teaches. They are also consistently **additive** — they explain
*why* and point at `ARCHITECTURE.md` / `UNDERSTANDING.md` for *what* — so none duplicates an
always-current doc.

**Three are SUPERSEDED**, all by the same event — the z_arch / FiLM family's deletion at v78:

| doc | the sentence | true now |
|---|---|---|
| `self_discovered_archetype_latent.md` | "**Status (2026-07-17): v1 is BUILT as `gen3_zarch_film_v1` (v44, `--zarch-film heads`)**" | `--zarch-film` is in `deleted_flags.md` (v78); `--zarch-dim` / `--zarch-recon-coef` / `--zarch-vicreg-coef` are in no parser; the `src/agents/model/CLAUDE.md` section it points at no longer exists; `UNDERSTANDING.md` §2.8 files per-team FiLM as REFUTED |
| `conditioning_architectures.md` | "we already have the surgical version (`--film-grad-accum-steps`) plus the metric that says when to use it (`film/noise_scale_ratio`)" | none of it exists. **This is the worst of the three**: its two sibling notes carry a status-correction banner and this one reads as a live description |
| `regularization_and_noise_in_ppo.md` | "**Deterministic representation regularizers** — PopArt (`--use-popart` …), the z_arch VICReg variance floor + recon BCE" | `use_popart` is `false` in production and refused under the live critic; both zarch coefficients are deleted |

**The fifteen PARTLY STALE notes** cluster into three causes, and the fix for each is one dated
line, not a rewrite:

- **The 2026-09-06 critic pivot** (7 notes: `popart_value_scale_and_currencies`,
  `credit_assignment_and_value_errors`, `objective_richness_and_representation`,
  `marginalization_and_uncertainty`, `pbs_value_functions_and_search`, `temperature_mixing_and_risk`,
  `distillation_flywheel_lessons`) — each is present-tense about PopArt, `ValueDistHead` or the
  ±30/−35 shaped terminal, all of which the live critic retired. `pbs_value_functions_and_search.md`
  goes furthest and *recommends enabling* the distributional aux loss, which is now refused.
- **A shipped thing still described as unbuilt** (4 notes) —
  `shortcut_learning_and_feature_delivery.md` "Add the tail bound (E5) — **STILL UNSHIPPED**"
  (shipped at v57; `entity_tail_seats: true`); `generalist_specialist_amortization_gap.md`
  "teacher-side fold-back is not built" (it is the entire 2026-08/09 fold campaign);
  `on_policy_self_distillation.md` "today it distills toward a single `A*`"
  (`--search-teacher-mode winprob_oneply` shipped 2026-08-29); `exercises_and_reading.md` calls the
  public-info value head "the next real build" (killed at v88).
- **Numbers that moved** (4 notes) — the largest is `entity_tokens_biases_pointers.md` (2,153
  lines): "all **fifteen** of our edge families" (17 now, `ARCHITECTURE.md` §5) and obs "2925"
  (**2501**), plus four model_config *fields* named as if they were CLI flags. Also
  `latent_belief_metrics_and_collapse.md`, which calls a run that died 2.5 months ago "the **live**
  run", and `amortization_gap_and_conditioning.md`, whose research verdict was corrected in place
  (the right pattern) but whose zarch flag names went with the family.

**One note's summary contradicts its own body.** `negative_transfer_and_shared_functions.md` says in
its TL;DR that the architecture hypothesis is *"pre-registered and **under test**"* and that the
architecture is a *"**live suspect**"*, while its own §3 says the claim "is dead as stated" and
`UNDERSTANDING.md` §2.8 records both probes as run: content locality **REFUTED with the sign
REVERSED**, sharing kernel **NOT DETECTED**. A reader who stops at the TL;DR — which is what a
TL;DR is for — gets the opposite of the finding.

---

## 4. `designs/references/`

| doc | verdict | reason |
|---|---|---|
| `wang2024_pokemon_rl.pdf` | **HOLDS** | the Wang 2024 MIT MEng thesis, the project's foundational reference. Cited by `README.md` and by the memory index; the path both give resolves |

---

## 5. What this census could not check

- **Prose judgement at scale.** The ai_v12 chapter is ~240 KB of design argument; the load-bearing
  sentences were sampled (headers, status/verdict sections, flag and run tables), not read line by
  line. A wrong *argument* buried mid-document would not be caught here — only a wrong *fact*.
- **Claims about the future.** A design doc that proposes an experiment cannot be stale for not
  having run it. Those are marked HOLDS.
- **The always-current corpus**, deliberately — it is verified by the gates and by the same-pass
  rule, not by a census.
- **`designs/ai_v12/design_winprob_only_critic.md` §1** was read in full only for its status and
  config claims; the four 2026-09-06 audit findings it carries and `ARCHITECTURE.md` does not
  (PopArt's POP never touching the phase-B critic, `value_tail_weight` inert, `value_net` frozen by
  omission, the critic loss mis-tagged `aux` in the noise-scale groups) were not re-verified against
  the code — they are the reason that section reads *stale* rather than *redundant*.
- **`designs/learning/entity_tokens_biases_pointers.md`** (2,153 lines) was sampled, not read
  whole; a later section may already correct the 15-families / 2925-dim items flagged above.
