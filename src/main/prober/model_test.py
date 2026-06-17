"""Tests for ProbeModel (the torch boundary).

Most of the engine is exercised through a FakeProbeModel (no torch); these pin the few real
ProbeModel attribute reads a fake can't catch — specifically WHERE each forward stash lives, which
is invisible to a fake that returns the decoded view directly. (The DamageOperator stashes on the OP
submodule, not the extractor — a read of the wrong object silently returned None and hid the entire
op view in the prober, with nothing to catch it until this test.)
"""

import numpy as np
import torch

from main.prober.model import ObsOffsets, ProbeModel

_OFF = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5,
                  turn_history_offset=0, turn_history_dim=0)


class _Op:
    """Stand-in DamageOperator submodule: it stashes ``last_raw_block`` on ITSELF (as the real op does)."""
    def __init__(self, row, outgoing=False):
        self.outgoing = outgoing
        self.last_raw_block = torch.as_tensor(row).unsqueeze(0)


class _Ext:
    """Extractor carrying the op submodule but NO ``last_raw_block`` of its own (the bug read it here)."""
    def __init__(self, op):
        self.damage_op = op


class _Pol:
    def __init__(self, ext):
        self.features_extractor = ext

    def extract_features(self, obs):   # no-op — the stash is pre-set, no real forward needed
        return None


def test_damage_op_view_reads_op_submodule_stash():
    """REGRESSION: the DamageOperator stashes ``last_raw_block`` on the OP submodule, not the extractor.
    ``damage_op_view`` must read ``op.last_raw_block`` — reading ``extractor.last_raw_block`` (the old bug)
    always returned None and silently hid the ENTIRE op view (incoming + outgoing damage) in the prober."""
    width = 6 * 12 + 6 + 10 + 13          # incoming + effect + incoming_secondary + choice_band (outgoing off)
    row = (np.arange(width, dtype=np.float32) + 1.0) / width
    pm = ProbeModel(policy=_Pol(_Ext(_Op(row, outgoing=False))), offsets=_OFF)
    out = pm.damage_op_view(np.zeros(8, dtype=np.float32), np.ones(11, dtype=np.int8))
    assert out is not None                                  # was None before the fix (read the wrong object)
    assert len(out["incoming"]) == 6 and out["outgoing"] is None
    assert set(out["incoming"][0]["phys"]) == {"low", "high", "crit", "pko", "acc"}


def test_damage_op_view_none_when_no_op():
    """No DamageOperator submodule (``--damage-op`` off) → None, cleanly (before any forward)."""
    class _ExtNoOp:
        damage_op = None

    pm = ProbeModel(policy=_Pol(_ExtNoOp()), offsets=_OFF)
    assert pm.damage_op_view(np.zeros(8, dtype=np.float32), np.ones(11, dtype=np.int8)) is None
