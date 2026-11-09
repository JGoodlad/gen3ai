"""X34's launch premise: the T-ENT toy NETWORK is the same network at HEAD as at its pin (`3899277a`).

X34 (`designs/endstate/design_compute_efficiency.md` §4) continues `rb_toy_ent_s4002` from its 100M cap at HEAD, not
at its pin (it must carry T36's canary fix `5236eb22` and the memory fix `9c033489`). The toy's argv typed 17 legacy
OFF flags that the deletion units U1 / U3 removed; at HEAD they are dropped (`adopted_levers_test.TOY_ARGV`), which
is sound only because every deleted lever is VACUOUS without `damage_op` (`adopted_levers` D-LD2). These two tests
pin that the network did not move:

1. **Structure.** HEAD's toy argv builds exactly the checkpoint's `state_dict` key set and shapes (316 keys, 3,121,590
   elements, recorded from `models/rb_toy_ent_s4002/final_model.zip`). Pure in-process; no archive needed.
2. **Function.** The real checkpoint, loaded strictly into that HEAD build, gives the SAME logits and values on 64
   real Rust-core observation rows as the PIN's own build gave (recorded at `3899277a` on a `git archive` export;
   bit-identical there on 2026-10-17), with each DECLARED input-semantics change since the pin undone (v167's nature
   truncation in the static tokens' stat input; v168's opponent HP stat prior, the operator's max-HP belief).
   Skipped where the run archive is absent.

The evidence, the argvs and the script that recorded both fixtures:
`designs/research_state/measurements/x34_toy_deep_stage_2026-10-17/`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile

import numpy as np
import pytest

from utils.paths import main_models_dir, repo_path

STAGE = repo_path("designs", "research_state", "measurements", "x34_toy_deep_stage_2026-10-17")
MANIFEST = STAGE / "toy_s4002_state_dict_manifest.json"
FORWARD_REF = STAGE / "toy_s4002_forward_ref.npz"
RUN = "rb_toy_ent_s4002"

#: fp32 on CPU: the same weights and inputs through the same ops agree to the last bit on one torch build (measured:
#: max |d| = 0.0); 1e-5 absolute admits a kernel-order change between torch builds and nothing else — a wrong
#: network (a missing layer, another input route) moves the logits by O(0.1) or more.
ATOL = 1e-5


def _toy_model():
    from agents.model.adopted_levers_test import _toy_args
    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(0, args=copy.deepcopy(_toy_args()))
    return model


def _manifest():
    return json.loads(MANIFEST.read_text())


def test_the_head_toy_argv_builds_the_s4002_checkpoints_exact_structure():
    man = _manifest()
    built = {k: list(v.shape) for k, v in _toy_model().policy.state_dict().items()}
    want = man["state_dict"]
    assert sorted(set(built) - set(want)) == [] and sorted(set(want) - set(built)) == [], (
        f"HEAD's toy argv no longer builds rb_toy_ent_s4002's key set: only HEAD {sorted(set(built) - set(want))}, "
        f"only the checkpoint {sorted(set(want) - set(built))}. X34 continues that checkpoint at HEAD — a deletion "
        f"unit or an argv change moved the toy network.")
    moved = {k: (built[k], want[k]) for k in want if built[k] != want[k]}
    assert not moved, f"shapes moved (HEAD, checkpoint): {moved}"
    assert len(built) == man["n_keys"] == 316
    assert sum(int(np.prod(s)) for s in built.values()) == man["numel"] == 3_121_590


def _forward(zp, ref):
    import torch

    from utils.torch_state_guard import torch_globals

    with torch_globals(num_threads=1):
        model = _toy_model()
        with zipfile.ZipFile(zp) as z, z.open("policy.pth") as f:
            sd = torch.load(f, map_location="cpu", weights_only=True)
        model.policy.load_state_dict(sd, strict=True)
        policy = model.policy
        policy.set_training_mode(False)
        mask = ref["mask"]
        with torch.no_grad():
            obs = {"observation": torch.as_tensor(ref["obs"]), "action_mask": torch.as_tensor(mask.astype(np.int8))}
            logits = policy.get_distribution(obs, action_masks=mask).distribution.logits.numpy()
            values = policy.predict_values(obs).numpy()
    assert logits.shape == ref["logits"].shape == (64, 11) and values.shape == ref["values"].shape == (64, 1)
    assert np.array_equal(np.isfinite(logits), np.isfinite(ref["logits"]))
    fin = np.isfinite(logits)
    return float(np.abs(logits[fin] - ref["logits"][fin]).max()), float(np.abs(values - ref["values"]).max())


def _v167_static_stat_prior(n_species: int):
    """`belief_tables.build_static_stat_prior` as the pin spelled its HP column (through config v167): the mean and
    std of ``2·base + 141 + tr(EV / 4)`` over the TOP-25 spread cut (`gen3_data.priors.spreads`), a species with no
    record reading the 0-EV value with half the 0-to-252-EV span as its std. v168 (`gen3_their_max_hp_belief_v1`)
    reads the operator's max-HP belief there instead (`their_max_hp_support`: every spread, a uniform fallback)."""
    from agents import gen3_data
    from agents.model.belief_tables import build_static_stat_prior

    out = build_static_stat_prior(n_species)

    def hp_stat(base: int, ev: int) -> float:
        return 1.0 if base == 1 else float(2 * base + 31 + ev // 4 + 110)

    for sid in gen3_data.species.base_form_ids():
        sd = gen3_data.species.get(sid)
        if not (0 <= sd.num < n_species):
            continue
        base = int(sd.base_stats.get("hp", 0))
        m1 = m2 = wsum = 0.0
        for _nature, evs, w in gen3_data.priors.spreads(sid):
            val = hp_stat(base, int(evs[0]))
            m1 += w * val
            m2 += w * val * val
            wsum += float(w)
        if wsum <= 0.0:
            neutral, maxed = hp_stat(base, 0), hp_stat(base, 252)
            out[sd.num, 0, 0] = neutral
            out[sd.num, 0, 1] = max(1.0, (maxed - neutral) / 2.0)
        else:
            mean = m1 / wsum
            out[sd.num, 0, 0] = mean
            out[sd.num, 0, 1] = max(1.0, max(0.0, m2 / wsum - mean * mean) ** 0.5)
    return out


def test_the_s4002_checkpoint_forwards_at_head_as_at_its_pin(monkeypatch):
    """The NETWORK is the pin's: with the two DECLARED input-semantics changes since the pin undone, HEAD reproduces
    the pin's logits and values. They are v167's `gen3_nature_floor_v1` (the static tokens' stat input truncates the
    nature as `statModify` does) and v168's `gen3_their_max_hp_belief_v1` (the static tokens' opponent HP stat prior
    is the operator's max-HP belief, every Smogon spread, where the pin read the top-25 cut). Both reach the toy
    (measured 2026-11-09 on these rows, max |d logits| / |d value|: the nature truncation alone 1.5e-3 / 5.6e-5, the HP
    prior alone 8.4e-3 / 5.3e-5, both 7.9e-3 / 6.6e-5; both undone 0.0 / 0.0): a toy fork run at HEAD reads Showdown's
    integer stats and the one max-HP belief, not the pin's inputs. A new difference fails here until it is explained
    and declared the same way."""
    import agents.model.damage_kinds as dk
    import agents.model.static_tokens as st

    root = main_models_dir()
    zp = root / RUN / "final_model.zip" if root is not None else None
    if zp is None or not zp.exists():
        pytest.skip(f"no archived {RUN}/final_model.zip on this box")
    man = _manifest()
    assert hashlib.sha256(zp.read_bytes()).hexdigest() == man["source_sha256"], (
        f"{zp} is not the file the fixtures were recorded from (models/ is read-only: who rewrote it?)")
    ref = np.load(FORWARD_REF)
    d_head = _forward(zp, ref)
    with monkeypatch.context() as only_nature:
        only_nature.setattr(dk, "gen3_nature_stat", lambda raw, nature_mult: raw * nature_mult)   # the v166 spelling
        d_hp = _forward(zp, ref)                              # what is left is the HP prior's change alone
    with monkeypatch.context() as only_hp:
        only_hp.setattr(st, "build_static_stat_prior", _v167_static_stat_prior)
        d_nature = _forward(zp, ref)                          # what is left is the nature truncation alone
    assert max(d_head) > ATOL and max(d_hp) > ATOL and max(d_nature) > ATOL, (
        f"a declared change no longer reaches the toy (HEAD {d_head}, the HP prior alone {d_hp}, the nature "
        f"truncation alone {d_nature}): drop its undo below")
    monkeypatch.setattr(dk, "gen3_nature_stat", lambda raw, nature_mult: raw * nature_mult)
    monkeypatch.setattr(st, "build_static_stat_prior", _v167_static_stat_prior)
    d_logits, d_values = _forward(zp, ref)
    assert d_logits <= ATOL and d_values <= ATOL, (
        f"rb_toy_ent_s4002 at HEAD no longer computes what it computed at its pin 3899277a: max |d logits| "
        f"{d_logits:.3g}, max |d value| {d_values:.3g} (bar {ATOL}), beyond the declared gen3_nature_floor_v1 and "
        f"gen3_their_max_hp_belief_v1 changes. A toy fork at HEAD must not run until the difference is explained "
        f"(re-run the stage dir's identity.py at both checkouts).")
