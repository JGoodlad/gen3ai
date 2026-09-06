# `gen3_dex_ids_split_v1` — the second cut of `damage_tables.py`, and its equivalence evidence

**2026-09-06.** `damage_tables.py` **1,226 → 930 lines**, under the file-size gate's 1,000-line
TARGET. The second and final pass of the cut `gen3_belief_tables_split_v1` began the day before
(1,433 → 1,226).

## What moved, and why a THIRD module

Pass one moved the belief PRIORS a head fuses with (spread, nature+EV, Hidden-Power type, item) into
`belief_tables.py` and left the note that a second cut — the MOVE prior and the team-composition
SPECIES prior, ~350 lines — would take the file under 1,000 but "needs dex-identity constants
relocated into a belief module — a judgement call, left."

That judgement is what `dex_ids.py` resolves. The two remaining builders read `HIDDEN_POWER_NUM`,
`_belief_num`, `_hp_typed_nums` and `build_species_usage_prior`, and **the op's physics reads them
too**: `build_damage_buffers` registers `SPECIES_USAGE_PRIOR` for the OUTGOING kernel's unrevealed
columns, and the typed-HP candidate expansion reads the Hidden-Power nums. Moving them into
`belief_tables` would have put dex-identity facts the physics depends on inside a belief module —
the layering would still have been acyclic, but it would have said something false about ownership.
A neutral bottom layer says the true thing: these are facts about *which num is this thing, and how
often is it seen*, and neither half owns them.

| module | lines | holds |
|---|---|---|
| `damage_tables.py` | 1226 → **930** | the damage/type/stat buffers the `DamageOperator` physics reads. Also the re-export HUB |
| `belief_tables.py` | 291 → **560** | + the MOVE prior (`build_move_prior_logits`, `sanitize_historical_move_floor`, the `_PRIOR_FLOOR` / `_ILLEGAL_PROB` / `_MIN_PRIOR_FLOOR` triple) and the SPECIES co-occurrence prior (`build_species_cooccur_prior`, `SPECIES_CLAUSE_LOGIT`, its two floors and the lift clamp) |
| `dex_ids.py` | **117** (new) | `HIDDEN_POWER_NUM`, `_belief_num`, `_hp_typed_nums`, `_USAGE_PRIOR_FLOOR`, `build_species_usage_prior` |

The layering, and it only ever points down:

```
damage_tables  →  belief_tables  →  dex_ids          (and damage_tables → dex_ids)
```

## The evidence

`equivalence_probe.py` in this directory, run with `--baseline c255851d` (the commit before the
cut). It materialises the baseline with `git archive`, builds both arms in their own subprocess
under a fixed torch seed, and compares by sha256 over raw tensor bytes.

```
baseline tree: HEAD -> /tmp/dex_ids_baseline_itz86v5o

=== 1. executable-AST identity of every moved name (docstrings stripped) ===
  == HIDDEN_POWER_NUM                   agents.model.dex_ids
  == _belief_num                        agents.model.dex_ids
  == _hp_typed_nums                     agents.model.dex_ids
  == _USAGE_PRIOR_FLOOR                 agents.model.dex_ids
  == build_species_usage_prior          agents.model.dex_ids
  == _PRIOR_FLOOR                       agents.model.belief_tables
  == _ILLEGAL_PROB                      agents.model.belief_tables
  == _MIN_PRIOR_FLOOR                   agents.model.belief_tables
  == sanitize_historical_move_floor     agents.model.belief_tables
  == build_move_prior_logits            agents.model.belief_tables
  == _SPECIES_PRIOR_FLOOR               agents.model.belief_tables
  == _SPECIES_CLAUSE_PROB               agents.model.belief_tables
  == SPECIES_CLAUSE_LOGIT               agents.model.belief_tables
  == _COOCCUR_LIFT_CLAMP                agents.model.belief_tables
  == build_species_cooccur_prior        agents.model.belief_tables

=== 2. the production-config extractor, seeded, baseline vs this tree ===
  == state_dict: 236 entries, 0 differing
  == buffers: 80 entries, 0 differing
  == module tree: 178 modules
  digest over this tree's state_dict+buffers:
      dcc06635aed8bcde6dd61c0b5f2b5076c4ac1f84910c1d25293a40fb129ff30b

VERDICT: IDENTICAL
```

All 15 moved definitions are executable-AST identical (each gains exactly one origin line in its
docstring, which the comparison strips). The built model is bit-identical on every `state_dict`
entry and every registered buffer.

**Why the `state_dict` half is a guard rather than a claim.** Every relocated table is registered
`persistent=False` by its owning head — data-derived from `data/`, recomputable, never a saved
weight — so it contributes ZERO `state_dict` keys and a relocation *cannot* move a key that does not
exist. The comparison is there to catch the case where that stops being true, and
`belief_tables_test.py` pins it permanently on the live module tree.

## What is a permanent test, and what is this file

A refactor's equivalence is a **one-time measurement**; the INVARIANT is what earns a permanent
test. `belief_tables_test.py` therefore holds only what the split can break — the re-export surface,
the one-way layering (an AST scan over a declared `_LAYERS` tuple, plus a cold-import test in a
fresh interpreter, plus a no-name-defined-twice scan), the `persistent=False` contract, and
bit-for-bit identity between each head's registered buffer and a fresh constructor call through both
import paths. Both new layering guards were **verified failing on a deliberate violation**: an
import-time up-edge produces a real `ImportError: ... partially initialized module ... (most likely
due to a circular import)` at collection, and a *deferred* (function-level) up-edge — the case an
import cannot catch — is named by the AST scan with its file line.

The per-table SEMANTICS stay beside the head each prior feeds (`move_prior_fusion_test.py`,
`species_prior_fusion_test.py`, `damage_tables_test.py`, `spread_belief_test.py`,
`hp_type_belief_test.py`, `item_belief_test.py`). All of those reach their subject through the
`damage_tables` hub, so neither round touched them.

## Gates

`mypy` (in scope), `ruff` (`F,E9`), the file-size ratchet, `delivery_graph --check`,
`build_arch_viewer --check`, and the routine suite `pytest src/ -m "not slow and not e2e"` — all
green. `damage_tables.py` is off the 1,000-2,000 census entirely.
