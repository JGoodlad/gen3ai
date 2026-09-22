"""Serializing ONE extractor across THREADS — `gen3_extractor_forward_guard_v1`.

🚨 **`Gen3FeaturesExtractor` is not thread-safe, and nothing said so.** Its whole per-forward
contract is instance state: `forward_internal` opens with `self.stash = ExtractorStashes()`
("ONE reset, no stash can go stale"), `DamageOperator.forward` opens with
`self.stash = OpStashes()`, and consumers read those attributes back LATER in the same
forward (`CLSPool` reads `damage_op.last_reduced_extra`; `RLPlayer` reads
`last_win_prob_logits` and the α publication AFTER `forward` has returned). Every one of those
reads assumes no other forward ran in between — true of training, where each env worker is its
own PROCESS, and false the moment two threads share one model object.

**They do share one.** `main.search_dividend`'s mirror runs the searched side's
`SearchEngine.choose` in a `run_in_executor` worker — it must, because the materializer drives a
replay through POKE_LOOP and would deadlock against itself — while the UNSEARCHED side commits
its own live choice on POKE_LOOP, on the same `model`. `PlayoffRunner._live_rollout` says so in
its own docstring. Torch releases the GIL inside its kernels, so the two forwards interleave.

**MEASURED 2026-09-22**, two threads forwarding one real extractor at B=1 and B=9, 2,400
forwards: **1,063 failures in seven distinct classes**, the most common an `IndexError` and the
named one exactly the crash that killed the playoff repro —
`ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)`, i.e. a B=1 forward reading the
B=9 forward's stashed threat rows. ⚠️ **The crash is the lucky case.** The two colliding forwards
only raise when their BATCH SIZES differ; the mirror opponent's live decision and a playoff
rollout player's decision are both B=1, and there the same race silently swaps one decision's
belief logits, threat rows and P(win) for another's, with nothing anywhere saying so.

**The guard is OPT-IN and NULL by default**, which is deliberate on two counts: training pays
nothing (it has no threads to protect against, and a lock in the hot forward would be pure cost),
and `--compile-trainer` / `--compile-extractor` see the identical graph they see today — the
default path takes a branch with no context manager in it at all, rather than a `with` over a
no-op object that dynamo would have to reason about.

**Stored OUTSIDE the module, in a `WeakKeyDictionary`.** A `threading.RLock` set as an attribute
on an `nn.Module` is unpicklable and undeepcopyable, so it would turn a `copy.deepcopy(policy)`
(SB3 does this) into a `TypeError` at a distance from anything that asked for a lock. The
registry keys on object identity and holds no strong reference, so an extractor that goes out of
scope takes its entry with it.
"""
from __future__ import annotations

import threading
from typing import Any, Optional
from weakref import WeakKeyDictionary

#: extractor → the lock serializing its forwards. Absent (the default) means "single-threaded,
#: guard nothing". Guarded by its own lock because two threads may install concurrently.
_GUARDS: "WeakKeyDictionary[Any, threading.RLock]" = WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def forward_guard_for(extractor: Any) -> Optional["threading.RLock"]:
    """The lock serializing ``extractor``'s forwards, or ``None`` when none was installed.

    ``None`` rather than a null context object so the caller can branch on it — see the module
    docstring on why the default path must contain no context manager.
    """
    return _GUARDS.get(extractor)


def install_forward_guard(extractor: Any) -> "threading.RLock":
    """Install (idempotently) and return the lock serializing ``extractor``'s forwards.

    RE-ENTRANT on purpose. The unit that must be atomic is not the forward — it is the forward
    PLUS the per-forward stash reads that belong to it (``RLPlayer`` reads the win-prob logits and
    the α publication after ``forward`` returns, and both are clobbered by the next forward from
    any thread). A caller holds the guard across that whole span, and ``forward``'s own acquire
    inside it must then be free.

    Called ONCE where the second thread is created — not inside the extractor — so that a tree
    which never threads never learns about locks.
    """
    with _REGISTRY_LOCK:
        existing = _GUARDS.get(extractor)
        if existing is not None:
            return existing
        fresh = threading.RLock()
        _GUARDS[extractor] = fresh
        return fresh


def install_model_forward_guard(model: Any) -> Optional["threading.RLock"]:
    """Guard ``model``'s features extractor, if it has one. Returns the lock, or ``None``.

    ``None`` for a model with no extractor (a scripted opponent, a test double) rather than a
    raise: the caller's job is "make this model safe to share", and a model with no per-forward
    stash is already safe.
    """
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    return install_forward_guard(extractor) if extractor is not None else None
