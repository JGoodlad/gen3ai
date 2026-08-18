"""Pin the E2 active-context injection (gen3_entity_rehome_v1): each side's boosts+volatiles
block lands on its ACTIVE mon's role-encoder row — and ONLY there (bench rows read zeros).

The injection slice sits right after `pokemon_enriched` in the role-encoder input concat
(`[pokemon_enriched | active_ctx_inject | context_broadcasted |
struggle_from_prev]`), so its columns are located from the tail: the last
`active_ctx_dim + global_ctx` columns start it. A hook on the role encoder's first
Linear captures the real input — the test would fail if the scatter indexed the wrong slot,
the wrong side, or the concat order moved without this file being updated.
"""
import numpy as np
import pytest
import torch
import gymnasium as gym

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.constants import (
    TEAM_SIZE, POKEMON_FULL_DIM, POKEMON_ACTIVE_OFFSET, ACTIVE_CONTEXT_DIM,
)
from agents.model.features_extractor import Gen3FeaturesExtractor


@pytest.fixture(scope="module")
def fe_and_layout():
    enc = Gen3ObservationEncoder(load_mappings())
    layout = enc.get_layout()
    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (enc.dimension,), np.float32),
        "action_mask": gym.spaces.Box(0, 1, (11,), np.float32),
    })
    return Gen3FeaturesExtractor(space, layout=layout), layout


def test_active_ctx_lands_on_the_active_rows_only(fe_and_layout):
    fe, layout = fe_and_layout
    sl_ctx_start = 2 * TEAM_SIZE * POKEMON_FULL_DIM
    obs = torch.zeros(1, layout["total_dim"])
    # Make OUR slot 2 and THEIR slot 4 the actives.
    obs[0, 2 * POKEMON_FULL_DIM + POKEMON_ACTIVE_OFFSET] = 1.0
    obs[0, (TEAM_SIZE + 4) * POKEMON_FULL_DIM + POKEMON_ACTIVE_OFFSET] = 1.0
    # Distinct, recognizable ctx blocks per side.
    our_ctx = torch.linspace(0.1, 0.9, ACTIVE_CONTEXT_DIM)
    opp_ctx = torch.linspace(-0.9, -0.1, ACTIVE_CONTEXT_DIM)
    obs[0, sl_ctx_start: sl_ctx_start + ACTIVE_CONTEXT_DIM] = our_ctx
    obs[0, sl_ctx_start + ACTIVE_CONTEXT_DIM: sl_ctx_start + 2 * ACTIVE_CONTEXT_DIM] = opp_ctx

    captured = {}

    def _pre(_m, args):
        captured["x"] = args[0].detach().clone()
        return None

    h = fe.pokemon_encoder.role_encoder[0].register_forward_pre_hook(_pre)
    try:
        with torch.no_grad():
            fe({"observation": obs, "action_mask": torch.ones(1, 11)})
    finally:
        h.remove()

    x = captured["x"].reshape(1, 2 * TEAM_SIZE, -1)[0]          # [12, role_input_dim]
    gl = layout["global_layout"]
    tail_after_inject = (
        # gen3_deadline_clock_v1: the clock is 3 scalars, not 1 — read the WIDTH from the layout
        # (this literal used to be a hardcoded `1`, which silently mis-sliced the window by 2 the
        # moment the group grew; the extractor's matching literals were removed in the same pass).
        gl["clock"]["dim"] + gl["weather"]["dim"] + layout["reactive_layout"]["fainted"]["dim"]
        + gl["hazards"]["dim"] + gl["screens"]["dim"]           # global_context
        # gen3_frame_deletion_v1: `+ 1 + 1` for switch_validity + struggle_from_prev used to
        # sit here. Both were PREV-TURN features fed by the deleted action-mask block, so the
        # role input now ENDS at global_context. Same trap the note above records — a literal
        # encoding a width that lives elsewhere — which is why this presented as "the E2
        # injection broke" when the injection was untouched (measured: tail_after = 22 = the
        # global_ctx sum, injection exactly where it belongs).
    )
    inj_end = x.shape[1] - tail_after_inject
    inj = x[:, inj_end - ACTIVE_CONTEXT_DIM: inj_end]            # the injection slice [12, 58]

    assert torch.allclose(inj[2], our_ctx), "OUR active row must carry our ctx block"
    assert torch.allclose(inj[TEAM_SIZE + 4], opp_ctx), "THEIR active row must carry opp ctx"
    bench = [i for i in range(2 * TEAM_SIZE) if i not in (2, TEAM_SIZE + 4)]
    assert inj[bench].abs().max() == 0.0, "bench rows must read zero ctx"
