"""gen3_policy_activation_pin_v1 — the SB3 tower's nonlinearity must be OURS, not a default.

Until 2026-08-16 `train_rl_agent.py` built `policy_kwargs` with `net_arch` but no `activation_fn`,
so the `[512, 512]` actor/critic tower ran on `MaskableActorCriticPolicy`'s signature default
(`nn.Tanh`). Nothing was wrong with the VALUE — the problem is that no gate in this tree could see
it:

  * `ModelVersion.from_layout_and_policy_kwargs` records `net_arch` and has **no activation field**,
    so the tower's shape is version-checked while its nonlinearity is not;
  * an activation swap is retrain-class but weight-shape-NEUTRAL, so `check_compatible` cannot
    catch it either — an old checkpoint loads clean into a network computing a different function;
  * therefore an sb3-contrib upgrade that changed that default would silently rewrite four layers
    of the live policy, and the only symptom would be a generation that trains differently for no
    recorded reason.

These tests pin all three halves of the fix: the constant's VALUE, that BOTH construction sites
pass it, and that the kwarg is actually WIRED into the tower rather than swallowed and ignored.

Same family as `src/main/default_port_test.py` — a test whose whole job is to catch a drift between
two defaults that no other gate compares.
"""

from __future__ import annotations

import ast
import inspect

import torch as th

from agents.model.arch_constants import NET_ARCH
from agents.model.policy import POLICY_ACTIVATION_FN


from main.train import entry_source_files


def test_the_pinned_activation_is_tanh() -> None:
    """The pin must reproduce what the unpinned default gave, or landing it changed the model.

    This is the assertion that makes the pin behaviour-neutral. If someone edits
    POLICY_ACTIVATION_FN, this test fails and forces the ARCH_SIGNATURE conversation the
    model-versioning playbook requires — which is the entire point, since nothing downstream
    (check_compatible, model_config.json) can detect the change on its own.
    """
    assert POLICY_ACTIVATION_FN is th.nn.Tanh, (
        f"POLICY_ACTIVATION_FN is {POLICY_ACTIVATION_FN!r}, not nn.Tanh. Changing the policy "
        "tower's activation is an ARCHITECTURE change that no version check can see: bump "
        "ARCH_SIGNATURE deliberately (src/agents/model/CLAUDE.md) before updating this test."
    )


def _policy_kwargs_dicts() -> list[tuple[str, ast.Dict]]:
    """Every `policy_kwargs` dict literal in the ENTRY POINT, found by AST not by grep.

    Identified by the `features_extractor_class` key — the one entry that is unique to a
    policy_kwargs dict. There are two: the resume path's `_load_policy_kwargs` and the fresh-run
    path's `policy_kwargs`. Both moved into `main/train/model_build.py` with the 2026-08-22
    decomposition, so this walks the whole entry point (hub + phase modules) rather than one file
    — reading a single path would have left the gate finding zero dicts and passing on a
    `len(...) == 2` it could no longer satisfy.
    """
    out = []
    for path in entry_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if "features_extractor_class" in keys:
                out.append((f"{path.name}:{node.lineno}", node))
    return out


def test_both_policy_kwargs_sites_pass_activation_fn() -> None:
    """A pin on ONE of the two construction paths is a pin that drifts.

    Fails if either dict drops the key — the literal revert of this change.
    """
    dicts = _policy_kwargs_dicts()
    assert len(dicts) == 2, (
        f"expected exactly 2 policy_kwargs dicts in the training entry point (resume + fresh), "
        f"found {len(dicts)} at {[where for where, _ in dicts]}. A new construction site must "
        "also pass activation_fn — update this test deliberately."
    )
    for where, node in dicts:
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        assert "activation_fn" in keys, (
            f"the policy_kwargs dict at {where} does not pass "
            "'activation_fn' — the tower would silently fall back to sb3-contrib's signature "
            "default (gen3_policy_activation_pin_v1)."
        )


def test_activation_fn_is_a_real_policy_parameter() -> None:
    """The kwarg must be CONSUMED, not swallowed into **kwargs and ignored.

    Guards the failure mode where a pin looks applied in the diff but does nothing.
    """
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

    params = inspect.signature(MaskableMultiInputActorCriticPolicy.__init__).parameters
    assert "activation_fn" in params, (
        "MaskableMultiInputActorCriticPolicy no longer takes activation_fn — the pin in "
        "train_rl_agent.py's policy_kwargs is now inert and the tower's nonlinearity is "
        "whatever the new API decides."
    )


def test_the_pin_actually_builds_a_tanh_tower() -> None:
    """End of the chain: our constant + our net_arch really do produce Tanh in both branches.

    Built directly from SB3's MlpExtractor (no env, no observation space, milliseconds) so the
    wiring claim is demonstrated rather than asserted.
    """
    from stable_baselines3.common.torch_layers import MlpExtractor

    extractor = MlpExtractor(
        feature_dim=8, net_arch=list(NET_ARCH),
        activation_fn=POLICY_ACTIVATION_FN, device="cpu",
    )
    acts = [m for m in extractor.modules() if isinstance(m, POLICY_ACTIVATION_FN)]
    # NET_ARCH = [512, 512] -> one activation per hidden layer, per branch (policy + value).
    expected = 2 * len(NET_ARCH)
    assert len(acts) == expected, (
        f"expected {expected} {POLICY_ACTIVATION_FN.__name__} modules in the tower "
        f"(len(NET_ARCH)={len(NET_ARCH)} x 2 branches), found {len(acts)}"
    )
    assert not [m for m in extractor.modules() if isinstance(m, th.nn.ReLU)], (
        "the tower contains ReLU modules — activation_fn did not govern its construction"
    )
