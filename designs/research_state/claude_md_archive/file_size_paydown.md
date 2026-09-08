# The file-size ratchet — the paydown record

Lifted verbatim from the root `CLAUDE.md` § *The FILE-SIZE ratchet* on **2026-09-07**. The gate's
rules stayed in `CLAUDE.md`; this is the CLOSED history — the five original allowlist entries, what
each became, and the dates. Both lists have been empty since 2026-08-23.

---

### The FILE-SIZE ratchet (`src/file_size_gate_test.py`) — the third static gate

**Under 1,000 lines is the TARGET for a core source file; under 2,000 is the STRONG bound.** The
gate encodes that asymmetry rather than flattening it: over 2,000 **hard-fails**, while the
1,000-2,000 band fails nobody and is instead **reported** (run with `-s` for the census — 16 files
on 2026-08-23, and the pool the next decomposition should come from). A target that fails the build
is a bound; a target nothing ever prints is a wish. Same scope as ruff (`src/agents src/main src/utils`,
`poke_env`/`rust_sim` excluded), unmarked, **0.04 s**, opt out with `GEN3AI_SKIP_SIZE_GATE=1`.

**Test files are EXEMPT when they exercise a single subject** — a fuzz test, or an `X_test.py` with
a source sibling named `X` (also `X/` as a package, and `X_Y_test.py` ↔ `X/Y.py`). A long test that
pins one module is that module's specification and splitting it scatters the spec; a test that
sprawls across six subsystems is the same liability as an oversized source file and takes the same
2,000 bound. The criterion is the **NAME**, deliberately, because an import graph classifies almost
every test as cross-cutting (the same measurement that killed inferred tier markers). When it
misfires, rename the test to match its subject or split it — do not park it in the allowlist.

**🎉 THE ALLOWLIST IS EMPTY — every source file in the tree is under the 2,000-line bound**
(2026-08-23). Both lists are now empty, and that is a MEASUREMENT rather than a claim: one meta-test
walks the real tree and fails if either list and the set of oversized files disagree in *either*
direction, so an empty list means there is nothing to list. The gate has stopped being a debt
schedule and is now a guard on a clean state.

The source list landed with five entries and was paid off in five passes, each one **deleting** its
entry rather than lowering it:

| file | lines | became | date |
|---|---|---|---|
| `main/train_rl_agent.py` | 4574 | `main/train/` package | 08-22 |
| `main/prober/engine.py` | 3058 | `main/prober/engine/` package | 08-23 |
| `main/prober/session.py` | 2573 | `main/prober/session/` package | 08-23 |
| `agents/training/instrumented_ppo.py` | 2152 | `agents/training/instrumented_ppo/` package | 08-23 |
| `agents/model/features_extractor.py` | 2280 | a base-class CHAIN behind the same hub → **277** | 08-23 |

(`model_version.py` sat at exactly 2,000 — one line from tripping a gate it had never been listed on
— and became `agents/model/model_version/` on 08-23 too.)

**A new entry is not a legal move**, and there is nowhere to park an oversized file: decompose it.
The last one shows the second shape a decomposition can take — `features_extractor.py` was already a
re-export hub over the per-phase modules split out on 2026-08-16, so the FORWARD PATH itself came out
into `extractor_build` / `extractor_api` / `extractor_forward` (+ `extractor_stashes`, `projection`)
as base classes the hub's `Gen3FeaturesExtractor` inherits — not a package, because the sibling
convention was already established there and inheritance is what keeps every `state_dict` key and
every `inspect.signature(...__init__)` reader byte-identical. A listed file may shrink freely but
**fails if it grows ≥10%** past its recorded count, and **fails when it drops back under 2,000
without being removed** — the list may only shrink, because a stale entry misleads every reader after
it (the ruff handoff list and the c-family lesson both). **Keeping it empty is always-welcome
piecemeal work** — no design doc, no coordination; the 1,000-2,000 census is where the next one comes
from.
