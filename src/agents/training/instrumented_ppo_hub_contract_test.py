"""The CONTRACT the `agents.training.instrumented_ppo` decomposition rests on.

On 2026-08-23 `instrumented_ppo.py` (2,152 lines — the last entry on the size ratchet's
grandfathered list) became a PACKAGE whose `__init__.py` is a pure re-export hub. Five things
make that safe rather than merely tidier, and none of them is self-enforcing:

**1. The hub still exports everything it used to.** `from agents.training.instrumented_ppo import
<name>` has to keep resolving — for `main/train/model_build.py`, `agents/model/snapshot.py`, and
a dozen test modules. The list below is the module-level *definitions* of the file as it stood at
the commit before the split, recovered by AST, plus the six `belief_bank` constants it
deliberately re-exported (`_SPREAD_LOSS_SCALE` is imported from here by
`spread_belief_loss_test.py`, so those bindings are public surface, not private imports).

⚠️ The other pure import BINDINGS are deliberately not pinned (`np`, `th`, `spaces`, `F`,
`explained_variance`, `rank_probe`, the four `grad_balance` metric functions, `_belief_bank`).
Nothing in the tree reads them off this module — verified by grep at the split — and pinning them
would freeze one module's private imports as another's public surface.

**2. `InstrumentedMaskablePPO` is assembled from MIXINS, so it has a BASE LIST**, and a base list
can lose an entry without any import failing: the class would still construct and still train,
just without one family of loss terms. `MaskablePPO` must also stay LAST, or `super()` calls
inside a mixin stop reaching upstream.

**3. `train()` and its FOLD SEQUENCE stay in ONE module.** The order in which the terms are added
to `loss` is a contract (see `ppo.py`), and the property that matters about it — no flag
combination reorders these — is only visible while it is straight-line source. This test pins
that `train()` is defined in `ppo.py` and still carries every `+` instrumentation marker.

**4. No submodule imports its own hub**, and **5. every submodule stands up on its own.**

Milliseconds — attribute lookups and an AST walk. No marker; runs in the fast inner loop.
"""
import ast
import importlib
import inspect
import pathlib

import pytest
from sb3_contrib import MaskablePPO

import agents.training.instrumented_ppo as hub

_PKG = "agents.training.instrumented_ppo"
_DIR = pathlib.Path(hub.__file__).parent

# Every module-level name `src/agents/training/instrumented_ppo.py` DEFINED at ea5cd98 (the commit
# before the decomposition), recovered by AST, PLUS the six `belief_bank` constants it re-exported
# under an explicit `# noqa: F401 (re-exports)` for older call sites.
_PRE_SPLIT = (
    "CfForward", "InstrumentedMaskablePPO", "_EXPECTED_UPSTREAM_TRAIN_HASH",
    "_NOISE_SCALE_EMA_DECAY", "_VALUE_TAIL_FRAC", "_WIN_CONTESTED_TAU",
    "_verify_upstream_unchanged",
    # the declared re-exports
    "_EV_LOSS_SCALE", "_EV_LOSS_WEIGHT", "_LATENT_STD_TARGET", "_LATENT_VICREG_WEIGHT",
    "_NATURE_CE_WEIGHT", "_SPREAD_LOSS_SCALE",
)

_BASES = ("PpoHyperparameters", "NoiseScaleDiagnostics", "DistillTerms", "ValueTerms", "AuxTerms",
          "CapacityTerms")


@pytest.mark.parametrize("name", _PRE_SPLIT)
def test_the_hub_still_exports_every_pre_split_name(name):
    assert hasattr(hub, name), (
        f"`{_PKG}.{name}` no longer resolves. The hub's whole job is that the decomposition "
        f"changed no import path — re-export it from whichever module now owns it, or delete "
        f"this entry deliberately if the name is genuinely gone."
    )


def test_the_ppo_class_carries_every_term_family():
    """A mixin dropped from the base list silently removes a whole family of loss terms."""
    mro = [c.__name__ for c in hub.InstrumentedMaskablePPO.__mro__]
    for base in _BASES:
        assert base in mro, (
            f"`{base}` is gone from `InstrumentedMaskablePPO`'s bases (MRO is {mro}). The class "
            f"would still construct and still train — just without that family of loss terms.")
    for method in ("_searchteacher_loss", "_opd_loss", "_distill_loss", "_value_distill_mse",
                   "_value_feat_distill", "_win_prob_loss", "_value_dist_loss",
                   "_value_loss_from_se", "_td_aux_term", "_belief_aux_loss",
                   "_move_belief_loss", "_spread_belief_loss", "_nature_ev_belief_loss",
                   "_hp_type_belief_loss", "_move_belief_latent_loss", "_cf_winprob_term",
                   "_cf_evidential_term", "_cf_twin_terms", "_cf_shadow_term",
                   "_noise_scale_estimate", "_global_grad_sq", "_emit_noise_scale_warnings",
                   "_capacity", "_capacity_snapshot_features", "_capacity_observe",
                   "_capacity_finish",
                   "_excluded_save_params", "collect_rollouts", "train"):
        assert callable(getattr(hub.InstrumentedMaskablePPO, method, None)), (
            f"`InstrumentedMaskablePPO.{method}` is gone — a mixin dropped out of the base list "
            f"in `instrumented_ppo/ppo.py`.")


class _ExcludeProbe(hub.InstrumentedMaskablePPO):
    """Just enough of an instance to call `_excluded_save_params` without building a real PPO."""
    def __init__(self):    # noqa: D107 - deliberately does NOT call MaskablePPO.__init__
        pass


def test_maskable_ppo_stays_last_in_the_mro():
    """The mixins must sit BEFORE `MaskablePPO`, or `super()` inside one stops reaching upstream.

    `_excluded_save_params` is the live case: it appends to `super()._excluded_save_params()`, and
    that `super()` resolves through the instance MRO. Put a mixin after `MaskablePPO` and the
    checkpoint silently starts pickling a `threading.Lock`.
    """
    mro = hub.InstrumentedMaskablePPO.__mro__
    assert mro[0] is hub.InstrumentedMaskablePPO
    mixin_positions = [mro.index(c) for c in mro if c.__name__ in _BASES]
    assert mixin_positions and max(mixin_positions) < mro.index(MaskablePPO), (
        f"a mixin sits after MaskablePPO in the MRO: {[c.__name__ for c in mro]}")
    assert issubclass(hub.InstrumentedMaskablePPO, MaskablePPO)
    excluded = _ExcludeProbe()._excluded_save_params()
    for name in ("_correction_buffer", "_distill_teachers", "_cf_buffer", "_capacity_state",
                 "rollout_buffer"):
        assert name in excluded, (
            f"{name!r} left `_excluded_save_params` — the last entry can only come from "
            f"upstream, so its absence means the `super()` chain no longer reaches MaskablePPO.")


def test_the_fold_sequence_stays_in_one_module():
    """`train()` is not split, on purpose. This pins that it is still whole and still in `ppo.py`.

    The ordering of the `loss = loss + <term>` lines is a contract — in particular every term that
    READS an extractor stash must be folded before the counterfactual block, which CLOBBERS the
    minibatch's stashes. That property is checkable by reading only while the sequence is
    straight-line source in one file.
    """
    src = inspect.getsource(hub.InstrumentedMaskablePPO.train)
    assert inspect.getfile(hub.InstrumentedMaskablePPO.train) == str(_DIR / "ppo.py")
    assert len(src.splitlines()) > 1000, "train() has been carved up — see this test's docstring"
    for marker in ("+INSTRUMENTATION", "+GRAD-ACCUM", "+PopArt", "+TD-AUX", "+CF-WINPROB",
                   "+DISTILL", "+SEARCH-TEACHER", "+OPD", "+NOISE-SCALE", "+WIN-PROB",
                   "+VALUE-DIST", "+BELIEF", "+CAPACITY"):
        assert marker in src, f"the `{marker}` block left `train()`"
    # The stash-clobber ordering, read off the source: every stash-reading fold precedes the
    # counterfactual block. `_td_aux_term` is the documented last one before it.
    assert src.index("self._td_aux_term(popart)") < src.index("self._cf_winprob_term("), (
        "the counterfactual fold now runs BEFORE the TD-aux fold. The CF forward CLOBBERS the "
        "minibatch's extractor stashes, so every term that reads one must be folded first.")


def test_no_submodule_imports_its_own_hub():
    offenders = []
    for path in sorted(_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            elif isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            if mod == _PKG:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"submodule(s) import the `{_PKG}` hub back: {offenders}. The hub imports `ppo`, which "
        f"imports every mixin — a submodule importing the hub back closes that cycle, and its "
        f"symptom is an AttributeError on a name that plainly exists."
    )


def test_every_submodule_imports_on_its_own():
    for path in sorted(_DIR.glob("*.py")):
        if path.name != "__init__.py":
            importlib.import_module(f"{_PKG}.{path.stem}")


def test_the_hub_re_exports_from_every_module():
    """Every `.py` beside the hub must be reachable from it — directly, or through `ppo`."""
    modules = {p.stem for p in _DIR.glob("*.py") if p.stem != "__init__"}
    hub_src = (_DIR / "__init__.py").read_text(encoding="utf-8")
    ppo_src = (_DIR / "ppo.py").read_text(encoding="utf-8")
    missing = sorted(m for m in modules
                     if f"{_PKG}.{m} import" not in hub_src and f"{_PKG}.{m} import" not in ppo_src)
    assert not missing, f"nothing in the package reaches: {missing}"
