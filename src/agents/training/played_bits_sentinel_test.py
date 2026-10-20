"""THE ROUTINE PLAYED-BITS SENTINEL — a few seeded decisions of the production network on the Rust core, digested.

WHY (2026-10-19). Since the ship-gate re-tier, the only goldens that notice a change to ENGINE-FACING DATA or to
MODEL NUMERICS moving the bits the agent plays are ``slow`` (``src/main/h2h/play_reveal_integration_test.py``'s
``OFF_BITS``, 24 games, ~45 min): T62 (``bd3126e0``, X5's tau solved in float64) shipped on a targeted set and left
that file red on main until a bisect. This is the CHEAP routine twin (< 8 s beside a live run): it cannot replace
the reveal file (a handful of decisions, not 24 games), it only makes such a change FAIL the ship gate the day it
lands, so the agent re-records here and runs the reveal file (``designs/ops/testing.md``).

WHAT IS PLAYED. The smallest PRODUCTION-architecture network (``production_args()``: it runs X5's tau solve on
every forward — the T62 path), seeded fresh + the parity probe's seeded perturbation (so its logits are not
near-uniform), CPU, eager, ONE torch thread, against ITSELF (``self_play``: both sides' decisions) on the Rust core
(``utils.rust_env.fixture_battles.play_rows``: seeded end to end, one thread, envs stepped in index order), greedy
over the mask, :data:`N_GAMES` games cut at :data:`TURN_LIMIT` turns. The teams are FIXED pool teams that carry a
Hidden Power Ghost mon with no IVs line — the teambuilder's fill decides their IVs (the 2026-10-19 Ghost-row fix,
``f2052657``, changed exactly those) — packed by the TRAINING teambuilder's own parse / fill / join.

TWO DIGESTS, so a failure says which moved:
* :data:`PACKED_DIGEST` — sha256 over the packed strings of the :data:`N_PLAYED_TEAMS` played teams (the first
  pool teams, in the loader's order, carrying an unfilled HP Ghost mon): the engine-facing team data — the
  teambuilder fill, the packing, the HP IV table, the pool files themselves;
* :data:`PLAYED_DIGEST` — sha256 over every decision in order: side, chosen action, and the 11 fp32 masked
  log-prob bits (the model numerics: a dtype, a solve, a reduction order).

RE-RECORD PROCEDURE (only for an INTENDED move — never to make a red go away unexplained):
  1. Name the cause: the commit, and the one change in it that moves the bits (bisect if unsure: revert ONLY that
     change and show the old digest reproduces, as T62's record did).
  2. ``pytest src/agents/training/played_bits_sentinel_test.py -q`` — the failure line prints the new digest(s).
  3. Paste them below, and append a line to the RE-RECORDED log: the date, the cause (commit + what moved), the
     old digest's first 8 hex chars, and how it was verified.
  4. Run the slow reveal file too (``pytest src/main/h2h/play_reveal_integration_test.py -m slow``) and re-record its
     ``OFF_BITS`` / ``OFF_OUTCOME_ALL`` for the same cause, or say why it did not move.

RECORDED 2026-10-19 on ``803a13e3`` (+ this file): the production surface of that commit (T62's
``gen3_tau_fp64_solve_v1`` in), torch 2.8 CPU, 17 decisions, reproduced twice. VERIFIED to move, each alone:
``hypothesis_set.TAU_SOLVE_DTYPE`` back to float32 (T62's one semantic change) moves PLAYED_DIGEST only (to
f1deb42d…); the Hidden Power Ghost IV row back to 30 Atk / 30 SpD (``f2052657`` reverted) moves BOTH (79a2ae96… /
09582cc2…). The Node team validation is NOT in the pack here (it judges legality, never a byte: same digest with it).
RE-RECORDED log (newest last):
  2026-10-20 — FP32_ALT (``gen3_tau_fp32_alt_v1``, config v160): the default tau solve moved from T62's float64 naive
  sum to the fp32 split residual (tau moves by <= 2 fp32 ulp). PLAYED_DIGEST only (was 6b2517a4…), PACKED_DIGEST
  unchanged. Verified: with every hypothesis builder forced to ``fp64`` the old 6b2517a4… reproduces exactly (and the
  K9 learner golden's T62 record with it), so the solve switch is the sole cause.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

PACKED_DIGEST = "94d96d5746b1a8e5402d940ba649787dbb5097e57f750ccc9abd2e970b6cf5f4"
PLAYED_DIGEST = "41a3593909a5325610edb6d068013e0be014ada5edd69a2233e113c6e5b0b8f7"

N_GAMES = 2
TURN_LIMIT = 4
SEED = 6217
MODEL_SEED = 31
PERTURB_SEED = 3100
N_PLAYED_TEAMS = 4


def _hp_ghost_unfilled(paste: str) -> bool:
    """A pool paste with a Hidden Power Ghost mon whose IVs the teambuilder FILLS (no ``IVs:`` line in that mon)."""
    for mon in paste.strip().split("\n\n"):
        low = mon.lower()
        if "hidden power [ghost]" in low and "ivs:" not in low:
            return True
    return False


def _packed_ghost_teams():
    """The training teambuilder's own pack (``Gen3Teambuilder``: parse, the gen-3 Hidden Power IV fill, join),
    without its Node validation pass (~1 s of subprocess; it judges legality and never changes a packed byte)."""
    from utils.gen3_utils import fix_gen3_hp_ivs
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder as TB

    raw = [t for t in team_list("pool") if _hp_ghost_unfilled(t)][:N_PLAYED_TEAMS]
    assert len(raw) == N_PLAYED_TEAMS, f"only {len(raw)} pool teams carry an unfilled HP Ghost mon"
    return [TB.join_team(fix_gen3_hp_ivs(TB.parse_showdown_team(t))) for t in raw]


def _model():
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    model, _, _ = build_fresh_model(MODEL_SEED, args=production_args())
    perturb_(model.policy, seed=PERTURB_SEED, scale=PERTURB_SCALE)
    model.policy.eval()
    return model


def test_the_production_policy_plays_the_recorded_bits():
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.training.rust_eval.offline import declared_torch_state
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built
    from utils.rust_env.fixture_battles import play_rows

    packed = _packed_ghost_teams()
    packed_digest = hashlib.sha256("\n".join(packed).encode()).hexdigest()
    ensure_built("selfcheck")
    lib = ffi.load(ffi.default_path("selfcheck"), nan_poison=True)
    h = hashlib.sha256()
    n = {"decisions": 0}
    with declared_torch_state(1):
        model = _model()
        fx = model.policy.features_extractor

        def greedy(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
            mb = torch.tensor(mask)
            ob = {"observation": torch.tensor(obs), "action_mask": mb}
            ob.update(zero_extra_obs(fx, batch=len(mb), device=torch.device("cpu")))
            with torch.no_grad():
                logits = model.policy.get_distribution(ob).distribution.logits.float()
            logp = torch.log_softmax(logits.masked_fill(~mb, -1e9), dim=-1).numpy().astype(np.float32)
            acts = np.where(mask, logp, -np.inf).argmax(axis=1)
            for k in range(len(acts)):
                h.update(struct.pack("<I", int(acts[k])))
                h.update(logp[k].tobytes())
            n["decisions"] += len(acts)
            return acts

        games = play_rows(N_GAMES, seed=SEED, teams=packed, turn_limit=TURN_LIMIT,
                          n_envs=N_GAMES, policy=greedy, self_play=True, lib=lib)
    played_digest = h.hexdigest()
    assert len(games) == N_GAMES and n["decisions"] >= 8, (len(games), n)   # the precondition: real decisions
    got = {"PACKED_DIGEST": packed_digest, "PLAYED_DIGEST": played_digest}
    want = {"PACKED_DIGEST": PACKED_DIGEST, "PLAYED_DIGEST": PLAYED_DIGEST}
    assert got == want, (
        f"PLAYED BITS MOVED ({n['decisions']} decisions): {got} (recorded {want}). A change to engine-facing data "
        "(PACKED_DIGEST: the team fill / packing) or to model numerics (PLAYED_DIGEST) moved what the agent plays. "
        "If INTENDED, follow the RE-RECORD PROCEDURE in this file's docstring (cause, paste, log line, and the slow "
        "reveal file's OFF_BITS); if not, it is a regression.")
