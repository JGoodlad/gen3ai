# The STUB-VACUITY census — every patch target in `src/**/*_test.py`, 2026-09-07

**Instrument:** `src/stub_vacuity_scan.py` (`gen3_stub_vacuity_gate_v1`), pure AST, run in report
mode from the worktree `stubgate-0907` at `60075ea9`. **Cost: 3.6 s** for the whole tree.

**Recorded BEFORE any fix**, so the number is the number the tree actually carried rather than the
number left after the sweep. The fixes and the outcome changes they produced are in the ledger
entry of the same date.

---

## 1. What was counted

A **patch site** is one place a test installs a stub over a module global. Four spellings, all
resolved to the same `(module, attribute)` pair:

| spelling | sites |
|---|---:|
| `monkeypatch.setattr` / `monkeypatch.delattr` | 284 |
| `patch("…")` (bare name) | 79 |
| `patch.object(mod, "…")` | 29 |
| `<assignment>` — `mod.name = stub`, the hand-rolled save/restore | 10 |
| `mock.patch("…")` | 5 |
| **total** | **407** |

The `<assignment>` shape is in scope because it is where two of the four `ccd08003` sites actually
were: a hand-rolled `mod.name = stub` inside a `try`/`finally` is exactly as vacuous as a
`monkeypatch.setattr` at the same target, and — unlike `setattr` — it is **silent at runtime even
when the attribute does not exist**, because assigning simply creates it. A gate that understood
only `monkeypatch.setattr` would have caught one of that commit's three patch-target sites and
called the other two clean.

Sites live in **70 test files**; **56** of those carry at least one JUDGED (non-skipped) site.

## 2. The verdicts

| verdict | sites | meaning |
|---|---:|---|
| `ok` | 267 | the stub can reach the code under test |
| `skipped` | 139 | out of scope, with the reason recorded (never silently dropped) |
| `missing` | **0** | the module named has no such attribute under any binding |
| `unread` | **1** | the module HAS the name but never reads it, and no consumer reaches it either |

### How the 267 `ok` sites clear

| evidence | sites |
|---|---:|
| the patched module LOADS the name itself (an `ast.Name` in `Load` context, or `__all__`, or a `getattr`-family string) | 223 |
| the module never reads it, but a **non-test consumer** reaches it as `<module>.<attr>` — the legitimate definition-site patch | 44 |

That second row is the one a naive scan gets wrong. `main.launcher.ipc` defines `emit` and never
calls it, yet patching `ipc.emit` works perfectly because its consumers hold the MODULE and resolve
the attribute at call time. 44 of 407 sites are that pattern, so a gate without the consumer search
would report a 44-finding false-positive storm and be excluded within the week.

### The 139 `skipped`, by reason

| reason | sites |
|---|---:|
| the target is not a module of ours — stdlib (`sys.argv`, `os`, `time`, `subprocess`), third party (`torch`, `sb3_contrib`, `poke_env`), or a local object | 97 |
| a CLASS attribute, not a module global (`patch.object(SomeClass, "method")`) | 34 |
| no literal target (the name is computed) | 5 |
| the target object is an expression, not a name | 3 |

The class-attribute skip is **safe by construction, not a gap**: patching `SomeClass.method`
mutates the one class object, so a consumer that did `from x import SomeClass` sees the stub
anyway. There is no vacuity shape to find there.

## 3. The one finding

`src/main/launcher/dry_run_test.py:83` — shape (1), **patched at the definition site while the
consumer imported the name directly.**

```python
monkeypatch.setattr(
    child_mod, "_launch_child",
    lambda *a, **k: pytest.fail("--dry-run must never spawn a child"))
```

`main/launcher/run.py:42` takes `_launch_child` by a **module-level** `from … import
_launch_child`, so it holds its own reference and `run.py:459` calls that. Setting the attribute on
`main.launcher.child` cannot reach it. The consumer search confirms: **zero** non-test modules
reach `main.launcher.child._launch_child` as a qualified attribute.

This is a **booby trap that is not armed**. The `isolated` fixture's docstring says "every
effectful launcher entry point booby-trapped so a future edit that reaches one FAILS", and the two
worktree traps beside it (`_create_run_worktree`, `_prune_stale_launcher_worktrees`) are correctly
installed on BOTH `wt` and `launcher_run`. Only the child-spawn trap — the most consequential of
the three, since tripping it means `--dry-run` launched a real training child — was installed on
one module and read from another. Nineteen `--dry-run` tests carry it.

## 4. What the instrument would have caught

Planting the three pre-`ccd08003` patch spellings against current source (worktree, then removed):

```
410 patch sites  |  missing=3  ok=267  skipped=139  unread=1

=== MISSING (3) ===
  …/zz_plant_test.py:3  monkeypatch.setattr(ppo_mod, 'shared_trunk_parameters')
                        -> agents.training.instrumented_ppo.ppo.shared_trunk_parameters
  …/zz_plant_test.py:4  <assignment>(ppo_mod.live_gauge_metrics = …)
                        -> agents.training.instrumented_ppo.ppo.live_gauge_metrics
  …/zz_plant_test.py:5  <assignment>(ppo_mod.advantage_density_metrics = …)
                        -> agents.training.instrumented_ppo.ppo.advantage_density_metrics
```

All three, as the STRONGER shape: after the decomposition moved those names out of `ppo.py`, the
module does not bind them at all. Two of the three are assignments, which raise nothing at runtime.

## 5. Precision — what the scan does NOT count as a read

An earlier draft counted every `ast.Attribute`'s `.attr` and every string constant in the module as
a read. That is conservative in the safe direction (it can only suppress a finding, never invent
one) but blind where it matters: `self.foo` is not a module global, and a docstring naming a
function is not a call to it, so a module whose own prose mentions its symbol cleared every patch
aimed at it. **Measured: exactly 2 of 407 sites rested on that looseness**
(`utils.team_loader.TeamLoader` from `snapshot_ladder_test.py:61`,
`obs_materializer.materialize_branches` from `search_dividend/search_test.py:435`) — and **both are
cleared on their own merits by the consumer search**, each reached by a DEFERRED (in-function)
import in its consumer, which re-executes after the stub is installed. Precision cost nothing, so
the loose reads were dropped; the consumer search is the real safety net.

## 6. Reading this number honestly

**One finding in 407 sites is a clean tree, and it is clean partly because `ccd08003` had already
swept the file family that motivated this row.** The value of the census is therefore not the count
— it is that the count is now MEASURED every routine run instead of discovered the next time a
module is decomposed. The vacuity shapes here are created by a MOVE, not by writing a bad test:
every one of the four `ccd08003` sites was correct on the day it was written, and became vacuous
when the symbol it named moved to another module. `reward_manager.py` (1,990 lines, the next
decomposition on the backlog) is patched at 32 sites from `reward_tracker_test.py` alone — the
largest single concentration in the tree, and the reason to have the gate standing before that cut
rather than after it.
