# The flag registry — the rules behind the five surfaces

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The flag registry — one declaration, five surfaces (`flag_registry.py`)

**Add a model-relevant toggle by adding a `ModelFlag` row to `agents/model/flag_registry.py`, then
following where the tests send you.** That file is the single declaration of every extractor
architecture toggle and of the five hand-synced places each one has to appear:

| # | surface | what it buys | how it is kept honest |
|---|---|---|---|
| 1 | the `argparse` entry in `main.train_rl_agent`, **defaulting to `None`** | a human can SET it, and leaves the sentinel `_resolve` needs | **validated** (both halves) |
| 2 | the `_resolve("name", default)` line beside it | a **flagless** resume INHERITS it | **validated** |
| 3 | `extractor_arch.ARCH_ARG_KEYS` / `_DERIVED` / `FROZEN_ARCH_KWARGS` | it reaches the extractor | **generated** |
| 4 | `snapshot.current_model_version()`'s keyword (via `_run_arch_toggles` → `arch_toggles_from_args`) | an eval/self-play WORKER rebuilds the SAME gate | **generated** + validated |
| 5 | the `ModelVersion` dataclass field | it is RECORDED and version-GATED | **validated** |

`flag_registry_test.py` fails with a message **naming the missing site**, which is the whole point:
every historical failure in this class was silent. A toggle in `ARCH_ARG_KEYS` but not on
`ModelVersion` means a resume version-checks against an architecture it does not build; one with an
argparse entry but no `_resolve` line means a flagless resume reverts it to OFF. The test earned its
keep on the first run — it found three rows whose flag name is not `--<field>`: `--damage-topk`
writes `damage_topk_k`, and the `--damage-matrices` MODE flag desugars into both
`damage_matrices_*` bools. Those name their flag with `cli_name=` rather than being exempted.

🚨 **Surface 1 is TWO claims, and for a long time only one was gated.** `_resolve(name, default)`
fires on `getattr(args, name) is None`, so an argparse entry defaulting to anything else makes its
own `_resolve` line **dead code** — while `test_cli_flags_have_a_resolve_line` keeps passing, because
the line is *present*. `test_cli_flags_argparse_default_is_none` closes the REACHABILITY half, and it
found **five live flags** in exactly that state (2026-08-22): `value_threat_inject` (ON in the gen-17
production config) and `opp_intent_coef` (which the structural `opp_intent` is DERIVED from) — either
of which would have made a flagless resume of PRODUCTION FATAL at `check_compatible` — plus
`cf_evidential` / `cf_twin_heads` / `cf_shadow_critic`. Use `default=None` with `action=BoolFlag` for
a bool, so `--no-<flag>` can still turn one off explicitly on a resume, and let `_resolve` supply the
OFF value for a fresh run. **That gate asserts against the BUILT parser, not the source text** — a
default can be an expression, so only the constructed object knows what it is.

**Read `designs/flag_registry.md`** for the current table (generated; `--check` is the gate).
### The three TIERS — a flag can lose its CLI entry without losing explicitness

A flag plays three independent roles — **SELECT** (choose it at launch), **RECORD** (write it into
`model_config.json`), **GATE** (refuse a mismatched resume). Only SELECT needs argparse; RECORD and
GATE live in `ModelVersion` and are reached whether or not argparse ever heard of the toggle.

| tier | argparse | `_resolve` | recorded + gated | reachable for an experiment |
|---|---|---|---|---|
| `cli` | yes | yes | yes | via the flag |
| `config_only` | **no** | **no** | yes | via the extractor **constructor kwarg** |
| `constructor_only` | no | no | no | via the constructor only (`pair_reduce`'s `reduce_how`) |

**A `config_only` toggle is FROZEN at its registry `default` for every CLI-launched run** — that is
the only value the CLI can now produce, so the default must be the value production actually wants.
Demote a toggle when it is *settled*: same value in every run, no live experiment. The extractor's
own constructor default is deliberately left alone, so the OFF baseline stays constructible for a
test or a probe; only the launch surface shrinks. `config_only_pattern_test.py` pins the contract
end to end (recorded in a fresh `model_config.json` · rejected on a mismatched resume · no argparse
entry). The one config_only survivor: `attend_unrevealed_opponents` (frozen **ON** — a hard
prerequisite of the whole belief stack since v16). The other two v78 demotions
(`value_active_readout`, `damage_matrices_outgoing_all`) were frozen OFF and are **deleted
outright at v88** (`gen3_dead_flag_purge_v1`): the fields, gates and forwards are gone, and the
migration refuses a checkpoint that recorded either ON.

### The four CLASSES — which gate a mismatch gets

| class | a mismatch means | gate |
|---|---|---|
| `structural` | weights and/or the trained forward differ | `check_compatible` — runs on **every** load |
| `resume_immutable` | the forward is bit-identical; only TRAINING differs | a dedicated `check_*`, **resume path only** |
| `training_coef` | a loss weight moved | none; recorded for provenance **and `_resolve`-inherited** |
| `runtime` | a perf knob moved | none; never recorded, never inherited on resume |

Getting this wrong hurts in **both** directions, so both are asserted: a `structural` toggle with no
`check_compatible` compare lets a resume silently flip the architecture, and a `resume_immutable`
toggle *inside* `check_compatible` makes a run FATAL while loading its own pool snapshots (that gate
runs on frozen eval/pool/distill opponents too, whose forward is identical regardless).


### Dependencies — `requires=`, and why both directions are enforced

A flag's DEPENDENCIES used to live only as ~30 hand-written `raise ValueError` lines inside
`Gen3FeaturesExtractor.__init__` ("intent_threshold requires opp_intent", "value_entity_pool_full
requires value_entity_pool", …). Nothing outside that function knew them, so `checkargs` could not
warn about an unsatisfiable command, the generated table could not show the graph, and answering
"what is the minimum config that turns X on?" meant reading the constructor.

`ModelFlag.requires` is that data — the flags that must be **enabled** for this one to be, where
enabled is `flag_registry.is_enabled` (`False` / `0` / `'off'` / `'none'` are OFF; note it is *not*
`bool()`, because a mode string's OFF state is the truthy `'off'`). `requirement_closure(name)`
gives the transitive set. **24 of 44** toggles declare one.

`flag_requires_test.py` enforces it in **both** directions, because a declaration nothing checks is
a comment and a check nothing declares is invisible:

| direction | what it does | what it catches |
|---|---|---|
| forward, positive | build with the flag on + **only** its declared closure | an INCOMPLETE `requires` — the ctor refuses a config the registry called complete |
| forward, negative | build with the closure minus **one** declared dep | a dependency that stopped being enforced |
| reverse | AST-scan `__init__` for every `raise` guarded on ≥2 registry flags | a hand-written coupling the registry never learned |

The positive control found a real omission on its first run (`value_dist_mode` also needs
`value_dist_vmax > value_dist_vmin`, enforced inside `ValueDistHead`), which is why the test carries
an explicit `_VALUE_RELATIONS` table rather than a silent fixup.

**What stays bespoke, and how narrow the carve-out is.** `requires` says only "must be enabled", so
a per-VALUE dependency has no flag-level form. Only `edge_bias_families` is exempt
(`BESPOKE_COUPLINGS` in the test, with the reason): its 17 family letters each carry their own
requirement and `h` carries none, so no statement about the flag is true. `damage_op` is NOT exempt
— it declares `move_belief_mode` (the weaker truth) while the constructor keeps the stronger
`in {revealed, both}`, because a weaker truth in the registry beats a blank. A stale exemption fails
too: `test_every_bespoke_coupling_still_exists` refuses an entry that matches no live raise.

**Downstream:** `python -m main.checkargs` reads the graph, so a recorded command that enables a
flag while explicitly disabling one of its dependencies is reported offline instead of crashing the
child ~40 s into a launch. It only fires on an EXPLICIT negation — an omitted dependency is inherited
from the checkpoint's config on resume, so absence carries no information.
