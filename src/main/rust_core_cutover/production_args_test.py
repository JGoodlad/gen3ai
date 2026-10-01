"""`production_args()` IS a real fresh `--arch production` launch — on every mirror key, the recipe included.

The cutover harness (slice N, the parity suites, the learner golden, the M5 throughput arms) builds its
"production" envs and learners from `rust_core_cutover.envs.production_args()`. It used to rebuild the
surface by hand — `apply_production_arch` plus a `hasattr` copy of every top-level
`designs/production_config.json` key — and that filter silently skipped the nested `recipe` block
(K10(a), `dc4232d4`), so the harness ran 32 envs x 4096 x 5 epochs at ent 0.02 with no self-play and no
grad accumulation while a real launch ran `recipe.fresh`. It now runs the launch's own
`resolve_config`; this file holds the two together.

Every key is compared on BOTH sides against the MIRROR, so either side dropping a key fails: a key the
real launch stops resolving (a renamed dest) leaves the resolved set and trips the declared
`MODEL_DERIVED` set; a key `production_args` stops carrying is a missing attribute or a wrong value.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any, Dict, List

import pytest

#: Mirror keys a real launch does NOT resolve onto its namespace: computed by the model build from the layout and the arch
#: kwargs and recorded in `model_config.json`, never set on a namespace. Declared, so a key that
#: LEAVES the dest set (a renamed or deleted flag) fails here instead of silently going uncompared.
MODEL_DERIVED = frozenset({
    "ability_embedding_dim", "active_context_dim", "arch_signature", "attend_unrevealed_opponents",
    "config_version", "item_embedding_dim", "max_abilities", "max_items", "max_moves", "max_species",
    "max_types", "move_embedding_dim", "move_net_hidden", "net_arch", "opp_belief_slots", "opp_intent",
    "opp_intent_grad_mode", "projection_dim", "role_encoder_hidden", "role_token_size",
    "species_embedding_dim", "total_dim", "type_embedding_dim",
})

#: The fresh argv, spelled here rather than read from `envs.PRODUCTION_ARGV`: the reference must not
#: move when the thing under test does.
_FRESH_ARGV = ["--steps", "1", "--arch", "production"]


def _real_fresh_launch() -> argparse.Namespace:
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    parser = build_parser()
    ns = parser.parse_args(list(_FRESH_ARGV))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(ns, parser)
    return ns


def _mirror() -> Dict[str, Any]:
    """Every value the mirror states for a fresh launch: the top-level fields + `recipe.fresh`'s rows
    (the two agree wherever they overlap — `recipe_surface.recipe_blocks` refuses otherwise)."""
    from agents.training.baselines import production_config
    from main.train.recipe_surface import production_recipe

    out = dict(production_config())
    out.update(production_recipe())
    return out


def _disagreements(ns: argparse.Namespace, mirror: Dict[str, Any], *, dests: frozenset) -> List[str]:
    from main.train.recipe_surface import _agree

    bad = []
    for k in sorted(mirror):
        if k in MODEL_DERIVED:
            continue
        if k not in dests:
            bad.append(f"{k}: not resolved by a real launch and not declared MODEL_DERIVED")
        elif not hasattr(ns, k):
            bad.append(f"{k}: MISSING from the namespace (mirror {mirror[k]!r})")
        elif not _agree(getattr(ns, k), mirror[k]):
            bad.append(f"{k}: {getattr(ns, k)!r} vs mirror {mirror[k]!r}")
    return bad


@pytest.fixture(scope="module")
def real():
    return _real_fresh_launch()


@pytest.fixture(scope="module")
def dests(real) -> frozenset:
    """The keys a real fresh launch RESOLVES onto its namespace — parser dests plus what the desugars
    write (`--damage-matrices` -> `damage_matrices_incoming` / `_outgoing`)."""
    return frozenset(vars(real))


def test_the_declared_model_derived_keys_are_exactly_the_mirror_keys_no_launch_resolves(dests):
    mirror = _mirror()
    non_dest = frozenset(k for k in mirror if k not in dests)
    assert non_dest == MODEL_DERIVED, (
        f"mirror keys a real launch does not resolve changed: new {sorted(non_dest - MODEL_DERIVED)}, "
        f"gone {sorted(MODEL_DERIVED - non_dest)} — a renamed/deleted flag drops a key from the comparison")


def test_a_real_fresh_arch_production_launch_resolves_every_mirror_key(real, dests):
    """The reference itself: if THIS fails, the launch (not the harness) has drifted from the mirror."""
    bad = _disagreements(real, _mirror(), dests=dests)
    assert not bad, "a real fresh --arch production launch disagrees with the mirror:\n  " + "\n  ".join(bad)


def test_production_args_agrees_with_a_real_fresh_launch_on_every_mirror_key(real, dests):
    from main.rust_core_cutover.envs import production_args

    mirror = _mirror()
    pa = production_args()
    bad = _disagreements(pa, mirror, dests=dests)
    assert not bad, "production_args() disagrees with the mirror:\n  " + "\n  ".join(bad)
    from main.train.recipe_surface import _agree
    split = sorted(k for k in mirror if k in dests and not _agree(getattr(pa, k), getattr(real, k)))
    assert not split, f"production_args() and a real fresh launch disagree on {split}"
    # the recipe is applied by the launch's own resolver, so it is stamped like a launch's
    assert pa.recipe_source == real.recipe_source and pa.arch_source == real.arch_source
    assert pa.recipe_source and pa.recipe_source.startswith("production_config@")


def test_every_recipe_fresh_key_is_compared():
    """`recipe.fresh` holds its rows plus `kl_controller` — constants of the KL callback, not flags,
    pinned against the callbacks' own defaults by `recipe_surface_test`. Nothing else escapes."""
    from main.train.recipe_surface import KL_KEY, production_recipe, recipe_blocks

    fresh, _fork = recipe_blocks()
    assert set(fresh) - set(production_recipe()) == {KL_KEY}


def test_production_args_returns_an_independent_namespace_per_call():
    from main.rust_core_cutover.envs import production_args

    a = production_args()
    a.self_play, a.n_envs = False, 1
    b = production_args()
    assert b.self_play is True and b.n_envs != 1


def test_the_comparison_catches_a_dropped_key_and_the_old_hasattr_builder(real, dests):
    """The comparator has teeth: a namespace that lost a key fails, and so does the pre-fix builder
    (the `hasattr` copy of the top-level mirror, which never saw the nested `recipe` block)."""
    import copy

    from agents.training.baselines import production_config
    from main.train.arch_surface import apply_production_arch
    from main.train.config import desugar_umbrella_flags
    from main.train.parser import build_parser

    mirror = _mirror()
    dropped = copy.deepcopy(real)
    del dropped.grad_accum_steps
    assert any(b.startswith("grad_accum_steps: MISSING") for b in _disagreements(dropped, mirror, dests=dests))

    old = build_parser().parse_args(["--steps", "1"])
    apply_production_arch(old)
    for k, v in production_config().items():
        if hasattr(old, k):
            setattr(old, k, v)
    with contextlib.redirect_stdout(io.StringIO()):
        desugar_umbrella_flags(old)
    flagged = {b.split(":")[0] for b in _disagreements(old, mirror, dests=dests)}
    assert {"n_envs", "batch_size", "n_epochs", "grad_accum_steps", "ent_coef", "self_play"} <= flagged
