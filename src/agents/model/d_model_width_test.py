"""gen3_d_model_width_v1 (config v169, `--d-model N`; MF-5's scale bet, EXPERIMENT_BACKLOG §2.0): the per-run trunk width.

What this file pins, each assertion failing on the revert it names:

* THE DEFAULT IS THE PRE-WIDTH NETWORK: a production build's state-dict signature (keys x shapes x dtypes) and its
  element count equal the ones measured at 2551befed (the commit before the width existed); an explicit
  ``d_model=128`` builds the same bytes and the same forward as the kwarg left out (the K9 learner golden pins the
  forward / update numerics beside this).
* EVERYTHING FOLLOWS THE WIDTH: at a width that collides with no fixed constant (192), no weight of a production, a
  legacy all-routes, a trunk-readout, a deep per-kind-FFN, a width-branch or an all-ride-along build has a 128 axis,
  and each runs a forward and a backward through a real `MaskablePPO`-built policy.
* THE RECORD AND THE GATES: `model_config.json` records `d_model` (and the token width / role encoder that follow it);
  `check_compatible` refuses a mismatch naming `--d-model`; a v168 config migrates to 128; the trunk FFN below the
  width's base is refused by the extractor and by the combination check; a fresh launch records the width and a
  flagless resume inherits it; a save / load round trip keeps it.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import (D_MODEL, check_d_model, ffn_base, role_encoder_hidden, static_dyn_hidden,
                                         trunk_pointer_hidden)
from agents.model.model_version import ModelVersion, ModelVersionError
from utils.torch_state_guard import single_thread_build

#: A width no fixed constant equals (128 / 256 are the default token / FFN widths, 64 / 32 / 96 / 512 fixed widths),
#: so a stray axis of the default width is a module that did not follow `d_model`.
WIDE = 192

#: Measured at 2551befed (the commit before the width existed) on `fresh_checkpoint.build_fresh_model(0)`: the sha256
#: of the sorted ``(key, shape, dtype)`` list, and the element count. A deliberate weight-shape change re-pins both.
PRE_WIDTH_SIGNATURE_SHA = "b07442f96c687e52f8be8deec448029967d59d387e999e36bf00a777300f07e3"
PRE_WIDTH_ELEMENTS = 2_609_183


def _signature_sha(sd: Dict[str, torch.Tensor]) -> str:
    rows = sorted((k, tuple(v.shape), str(v.dtype)) for k, v in sd.items())
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _bytes_sha(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _production(**over: Any) -> Any:
    from main.train.production_args import production_args

    args = production_args()
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _model(args: Any = None, seed: int = 0) -> Any:
    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(seed, args=args)
    return model


def _rows(n: int = 8) -> np.ndarray:
    from agents.training.golden_obs_core import core_rows

    return np.stack(core_rows()[:n]).astype(np.float32)


def _obs(policy: Any, rows: np.ndarray) -> Dict[str, Any]:
    from agents.model.extra_obs_keys import zero_extra_obs

    obs: Dict[str, Any] = {"observation": torch.as_tensor(rows),
                           "action_mask": torch.ones(rows.shape[0], 11, dtype=torch.int8)}
    obs.update(zero_extra_obs(policy.features_extractor, batch=rows.shape[0]))
    return obs


@pytest.fixture(scope="module")
def rows() -> np.ndarray:
    return _rows()


# ── the width's own rules ────────────────────────────────────────────────────────────────────────────────────────
def test_the_width_functions_return_the_constants_they_replaced_at_the_default() -> None:
    from agents.model.arch_constants import (ROLE_ENCODER_HIDDEN, ROLE_TOKEN_SIZE, STATIC_DYN_HIDDEN,
                                             TRANSFORMER_FFN_DIM, TRUNK_POINTER_HIDDEN)
    assert D_MODEL == ROLE_TOKEN_SIZE == 128
    assert ffn_base(D_MODEL) == TRANSFORMER_FFN_DIM
    assert role_encoder_hidden(D_MODEL) == list(ROLE_ENCODER_HIDDEN)
    assert static_dyn_hidden(D_MODEL) == STATIC_DYN_HIDDEN
    assert trunk_pointer_hidden(D_MODEL) == TRUNK_POINTER_HIDDEN
    assert (ffn_base(256), role_encoder_hidden(256), static_dyn_hidden(256), trunk_pointer_hidden(256)) == \
        (512, [512, 256], 256, 256)


@pytest.mark.parametrize("bad", [96, 100, 130, 0, -128, True, 128.5, "wide"])
def test_check_d_model_refuses_what_is_not_a_width(bad: Any) -> None:
    with pytest.raises(ValueError, match="d_model must be"):
        check_d_model(bad)


@pytest.mark.parametrize("good", [128, 160, 192, 256, 512, "256"])
def test_check_d_model_accepts_a_width(good: Any) -> None:
    assert check_d_model(good) == int(good)


# ── the default is the pre-width network ─────────────────────────────────────────────────────────────────────────
def test_the_default_build_is_the_pre_width_network(rows: np.ndarray) -> None:
    """Revert any module's width source to a wrong default (or move a construction) and the signature, the bytes or
    the forward moves."""
    torch.set_num_threads(1)
    plain = _model()
    explicit = _model(_production(d_model=D_MODEL))
    sd, sd_x = plain.policy.state_dict(), explicit.policy.state_dict()
    assert _signature_sha(sd) == PRE_WIDTH_SIGNATURE_SHA
    assert sum(p.numel() for p in plain.policy.parameters()) == PRE_WIDTH_ELEMENTS
    assert _bytes_sha(sd) == _bytes_sha(sd_x)
    assert plain.policy.features_extractor.d_model == explicit.policy.features_extractor.d_model == D_MODEL
    for m in (plain, explicit):
        m.policy.eval()
    with torch.no_grad():
        a = plain.policy.get_distribution(_obs(plain.policy, rows)).distribution.logits
        b = explicit.policy.get_distribution(_obs(explicit.policy, rows)).distribution.logits
        va = plain.policy.predict_values(_obs(plain.policy, rows))
        vb = explicit.policy.predict_values(_obs(explicit.policy, rows))
    assert torch.equal(a, b) and torch.equal(va, vb)


# ── everything follows the width ─────────────────────────────────────────────────────────────────────────────────
#: name -> the production overrides that select it (each at d_model = WIDE, the trunk FFN at its base unless named).
WIDE_ARMS: Dict[str, Dict[str, Any]] = {
    "production": {},
    "trunk_readout": {"policy_readout": "trunk"},
    "deep_per_kind_ffn": {"trunk_layers": 4, "trunk_ffn_by_kind": "on"},
    "width_branch": {"trunk_ffn_dim": ffn_base(WIDE) + 64},
    "ridealong_all": {"ridealong_ensemble": 2, "ridealong_rnd": True, "ridealong_adv": 2, "ridealong_opp": 2,
                      "ridealong_rnd_variants": "fast,decay,small,feat"},
    "belief_pool": {"opp_belief_cls_k": 3},
}


def _wide_policy(over: Dict[str, Any]) -> Any:
    kw = {"d_model": WIDE, "trunk_ffn_dim": ffn_base(WIDE)}
    kw.update(over)
    return _model(_production(**kw)).policy


def _assert_follows(policy: Any, rows: np.ndarray) -> None:
    stray = [(k, tuple(v.shape)) for k, v in policy.state_dict().items() if D_MODEL in tuple(v.shape)]
    assert not stray, f"weights still {D_MODEL} wide at d_model={WIDE} (a module that ignores d_model): {stray}"
    fe = policy.features_extractor
    assert fe.d_model == WIDE and fe.role_token_size == WIDE and fe.vf_features_dim == WIDE
    policy.train()
    obs = _obs(policy, rows)
    logits = policy.get_distribution(obs).distribution.logits
    values = policy.predict_values(obs)
    assert values.shape == (rows.shape[0], 1)
    assert torch.isfinite(logits[obs["action_mask"].bool()]).all() and torch.isfinite(values).all()
    (logits.logsumexp(-1).mean() + values.mean()).backward()
    grads = [p.grad for p in policy.features_extractor.team_transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("arm", sorted(WIDE_ARMS))
def test_a_wide_build_follows_the_width_and_trains(arm: str, rows: np.ndarray) -> None:
    torch.set_num_threads(1)
    _assert_follows(_wide_policy(WIDE_ARMS[arm]), rows)
    if arm == "ridealong_all":
        heads = _wide_policy(WIDE_ARMS[arm]).ridealong
        assert heads is not None and heads.ensemble.in_dim == WIDE


def test_a_wide_legacy_all_routes_extractor_follows_the_width() -> None:
    """The legacy token encoding (no production run uses it; a test or a probe builds it) at the wide width."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.value_route_gradient_test import _ALL_ROUTES_ON
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    with single_thread_build():
        fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, d_model=WIDE,
                                   trunk_ffn_dim=ffn_base(WIDE), **dict(_ALL_ROUTES_ON, history_events=True))
    stray = [(k, tuple(v.shape)) for k, v in fe.state_dict().items() if D_MODEL in tuple(v.shape)]
    assert not stray, stray
    obs = {"observation": torch.rand(2, layout["total_dim"], generator=torch.Generator().manual_seed(5))}
    pi, vf = fe(obs)
    assert vf.shape == (2, WIDE) and torch.isfinite(pi).all()


def test_the_trunk_ffn_below_the_widths_base_is_refused_by_the_extractor() -> None:
    with pytest.raises(ValueError, match=f"--trunk-ffn-dim {ffn_base(256)}"):
        _model(_production(d_model=256))                # the production mirror's trunk_ffn_dim 256 < 512


# ── the record and the gates ─────────────────────────────────────────────────────────────────────────────────────
def _version_at(width: int) -> ModelVersion:
    from main.train.derived_toggle_resume_test import _resolve, _saved_version

    return _saved_version(_resolve(["--arch", "production", "--steps", "1000", "--d-model", str(width),
                                    "--trunk-ffn-dim", str(ffn_base(width)), "--allow-nonproduction-arch"]))


def test_the_record_carries_the_width_and_what_follows_it() -> None:
    v = _version_at(256)
    assert (v.d_model, v.role_token_size, v.role_encoder_hidden, v.trunk_ffn_dim) == (256, 256, [512, 256], 512)
    base = _version_at(D_MODEL)
    assert (base.d_model, base.role_token_size, base.role_encoder_hidden, base.trunk_ffn_dim) == \
        (128, 128, [256, 128], 256)


def test_check_compatible_refuses_a_width_mismatch_naming_the_flag() -> None:
    wide, base = _version_at(256), _version_at(D_MODEL)
    wide.check_compatible(dataclasses.replace(wide))
    with pytest.raises(ModelVersionError, match="(?s)d_model mismatch.*--d-model"):
        # only the field differs: the dedicated gate, not the weight-field list, must name it
        dataclasses.replace(base, d_model=256).check_compatible(base)
    with pytest.raises(ModelVersionError):
        wide.check_compatible(base)


def test_a_pre_width_config_migrates_to_its_token_width() -> None:
    from agents.model.model_version.migrations import _migrate_config

    data = dataclasses.asdict(_version_at(D_MODEL))
    data.pop("d_model")
    data["config_version"] = 168
    out = _migrate_config(dict(data))
    assert out["d_model"] == 128 and out["config_version"] >= 169


def test_a_fresh_launch_records_the_width_and_a_flagless_resume_inherits_it(tmp_path) -> None:
    from main.train.derived_toggle_resume_test import _resolve, _run_dir
    from main.train.recipe_testkit import recorded_cli_args

    _run, ckpt = _run_dir(tmp_path, _version_at(256), cli_args=recorded_cli_args())
    resumed = _resolve(["--model", ckpt, "--steps", "2000"])
    assert resumed.d_model == 256 and resumed.trunk_ffn_dim == 512


def test_the_combination_check_names_the_base() -> None:
    import types

    from main.train.combination_checks import BY_NAME

    chk = BY_NAME["trunk_ffn_dim_below_the_width_base"]
    a = types.SimpleNamespace(d_model=256, trunk_ffn_dim=256)
    assert chk.predicate(a) and "--trunk-ffn-dim 512" in chk.message(a)
    assert not chk.predicate(types.SimpleNamespace(d_model=256, trunk_ffn_dim=512))
    assert not chk.predicate(types.SimpleNamespace(d_model=None, trunk_ffn_dim=None))


def test_a_wide_checkpoint_round_trips_at_its_width(tmp_path, rows: np.ndarray) -> None:
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    model = _model(_production(d_model=WIDE, trunk_ffn_dim=ffn_base(WIDE)))
    model.save(str(tmp_path / "wide"))
    back = MaskablePPO.load(str(tmp_path / "wide.zip"), device="cpu")
    assert back.policy.features_extractor.d_model == WIDE
    assert _bytes_sha(back.policy.state_dict()) == _bytes_sha(model.policy.state_dict())
    for m in (model, back):
        m.policy.eval()
    with torch.no_grad():
        assert torch.equal(model.policy.predict_values(_obs(model.policy, rows)),
                           back.policy.predict_values(_obs(back.policy, rows)))


def test_the_slot_fingerprint_drops_the_default_width() -> None:
    """T2 groups slots by (state signature, forward fingerprint): a default-width checkpoint that does not record the
    kwarg and one that records 128 must share a fingerprint, so every pre-width snapshot still loads into a slot."""
    from agents.inference.service.slots import canonical_extractor_kwargs
    from agents.model.features_extractor import Gen3FeaturesExtractor

    assert "d_model" not in canonical_extractor_kwargs(Gen3FeaturesExtractor, {"d_model": 128})
    assert canonical_extractor_kwargs(Gen3FeaturesExtractor, {"d_model": 256}) == {"d_model": 256}


@pytest.mark.slow  # a CPU service startup gates every slot x bucket (~10 s beside a live run)
def test_t2_serves_a_wide_policy_and_refuses_a_default_width_one_in_its_group() -> None:
    """The inference tier (T2) has no width of its own: a wide group starts, gates and serves its slots at the
    policy's own decision (eager CPU; the CUDA backends are UNVERIFIED at this width), and a default-width
    checkpoint is refused by the state signature (its own group, `rust_rollout.build.slot_groups`, serves it)."""
    from agents.inference.service import InferenceService, ServiceSpec, SlotArchMismatch, SlotGroupSpec
    from agents.inference.service.decision import policy_reference
    from agents.inference.service.parity import fixture_rows
    from agents.model.parity_probe import perturb_

    torch.set_num_threads(1)
    wide = [perturb_(_model(_production(d_model=WIDE, trunk_ffn_dim=ffn_base(WIDE)), seed=s).policy.eval(),
                     seed=1000 + s) for s in (0, 1)]
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("wide", 2, wide[0]),), device="cpu", backend="eager",
                                       buckets=(2, 8))).startup()
    svc.load(1, wide[1], "wide-1")
    obs, mask = fixture_rows(wide[0].observation_space["observation"].shape[0], 8)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    ref_logp, ref_value = policy_reference(wide[1], o, m)
    t = svc.submit(1, obs, mask)
    svc.flush()
    got = t.result()
    fin = torch.isfinite(ref_logp)
    assert float((got.logp - ref_logp)[fin].abs().max()) < 1e-5
    assert float((got.value - ref_value).abs().max()) < 1e-5
    with pytest.raises(SlotArchMismatch, match="signature"):
        svc.load(0, _model().policy.eval(), "default-width")
