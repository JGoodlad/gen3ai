"""
Shared harness for the feature-coverage suite.

Every test in this folder answers the same two-part question for one battle
edge case:

  1. CAPTURE  — is the edge case encoded into the observation vector at the
                expected offset (right one-hot bit / scalar / embedded id)?
  2. NETWORK  — does that encoded signal actually flow through the *real*
                `Gen3FeaturesExtractor` and move its policy/value output,
                rather than being silently dropped (e.g. by `embed_delta_slot`
                or a key-padding mask)?

The suite is deterministic and server-free: it drives the real
`TurnDeltaEncoder` and the real `Gen3FeaturesExtractor` (live layout + mappings)
on hand-built `TurnDelta`s / observation regions. Real-battle *capture* of these
same signals is covered by the bridge-backed fuzz tests under
`training/poke_env_gaps/`; this suite closes the last hop those don't exercise —
that the captured dims reach the network.

Helpers
-------
`feature_model()`      -> (model, layout, mappings), built once (extractor init
                          runs a dummy forward, so reuse it).
`td_encoder()`         -> the shared TurnDeltaEncoder.
`make_delta(**f)`      -> a TurnDelta.empty() with fields overridden.
`anchor_delta(**f)`    -> like make_delta but with a default move set so the
                          history slot is non-empty (NOT treated as padding),
                          letting you isolate a single field against a baseline.
`encode_delta(d)`      -> [TURN_DELTA_DIM] encoded slot vector.
`obs_zero(layout)`     -> [1, total_dim] zero observation.
`obs_with_delta(...)`  -> zero obs with `delta` in a history slot.
`history_slot_offset`  -> absolute obs offset of a history slot.
`set_region(...)`      -> write raw values into a static obs region by offset.
`forward(model, obs)`  -> (pi, vf) under no_grad.
`policy_value_changed` -> (pi_changed, vf_changed) bools between two obs.
`assert_reaches_network(model, base, variant, msg)` -> asserts BOTH heads move.
`read_block` / `onehot_idx` / `multihot_idxs` -> vector inspection helpers.
"""
import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.turn_delta_encoder import TurnDeltaEncoder, TURN_DELTA_DIM
from agents.training.turn_delta import TurnDelta

_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model / encoder (built once — extractor __init__ runs a dummy forward pass)
# ---------------------------------------------------------------------------

def feature_model():
    """Return (model, layout, mappings). Cached; safe to call per-test."""
    if "model" not in _CACHE:
        mappings = load_mappings()
        encoder = Gen3ObservationEncoder(mappings)
        layout = encoder.get_layout()
        space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(layout["total_dim"],), dtype=np.float32
        )
        # gen3_frame_deletion_v1: `history_events=True`. The lag frames these probes used to
        # write into were read unconditionally; the H-B event window that replaced them is read
        # ONLY by the flag-gated event-seat encoder, so with the flag off no Linear touches the
        # block and every NETWORK-half probe would compare two identical forwards. Building the
        # fixture the way the run is configured is what keeps those halves meaningful.
        model = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings,
                                      history_events=True)
        model.eval()
        _CACHE["model"] = (model, layout, mappings)
    return _CACHE["model"]


def td_encoder() -> TurnDeltaEncoder:
    if "td" not in _CACHE:
        m = load_mappings()
        _CACHE["td"] = TurnDeltaEncoder(m.get("moves", {}), m.get("species", {}))
    return _CACHE["td"]


# ---------------------------------------------------------------------------
# TurnDelta construction
# ---------------------------------------------------------------------------

def make_delta(**fields) -> TurnDelta:
    """A `TurnDelta.empty()` with the given fields overridden."""
    return dataclasses.replace(TurnDelta.empty(), **fields)


def anchor_delta(**fields) -> TurnDelta:
    """A non-empty TurnDelta whose history slot is guaranteed present (unmasked).

    Defaults `our_move_id="tackle"` + a real `our_prev_active` so the encoded
    slot is non-zero even before your edge field is added — so a baseline and a
    variant that differ in ONE field are both 'present' history, and the output
    delta isolates that field (not slot presence). Override any default freely."""
    base = dict(our_move_id="tackle", our_prev_active="snorlax")
    base.update(fields)
    return make_delta(**base)


def encode_delta(delta: TurnDelta) -> np.ndarray:
    """Encode a TurnDelta to its [TURN_DELTA_DIM] slot vector (float32)."""
    vec = td_encoder().encode(delta)
    assert vec.shape == (TURN_DELTA_DIM,), vec.shape
    return vec


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------

def obs_zero(layout) -> torch.Tensor:
    return torch.zeros(1, layout["total_dim"], dtype=torch.float32)


def obs_with_event(layout, slot_from_recent: int = 0, **cols) -> torch.Tensor:
    """Zero observation with ONE H-B event record written into the event window.

    gen3_frame_deletion_v1 REPLACED `obs_with_delta` / `history_slot_offset`. Those wrote an
    encoded `TurnDelta` into a lag frame; the frames are deleted, so a fact now reaches the
    network as an EVENT ROW instead. `slot_from_recent=0` is the most-recent row — the window is
    most-recent-LAST, so row index `EVENT_WINDOW_N - 1 - slot_from_recent`.

    `cols` are written by NAME so a caller never hardcodes a column index (the manifest rule the
    obs layout is built on). `valid` defaults to 1: a row with valid=0 is PAD and is key-masked
    out of attention entirely, so forgetting it would make every probe compare two identically
    masked observations and pass vacuously.
    """
    from agents.observation.constants import (
        OFFSET_EVENT_WINDOW, EVENT_WINDOW_N, EVENT_TOKEN_DIM,
    )
    assert 0 <= slot_from_recent < EVENT_WINDOW_N
    obs = obs_zero(layout)
    row = EVENT_WINDOW_N - 1 - slot_from_recent
    off = OFFSET_EVENT_WINDOW + row * EVENT_TOKEN_DIM
    named = {
        "type_id": 0, "actor": 1, "actor_side": 2, "target": 3, "move_num": 4, "magnitude": 5,
        "hit": 6, "miss": 7, "fail": 8, "crit": 9,
        "eff_neutral": 10, "eff_super": 11, "eff_resist": 12, "eff_immune": 13,
        "we_first": 14, "status_id": 15, "turns_ago": 16, "forced_window": 17, "valid": 18,
        "cant_id": 19,
    }
    cols.setdefault("valid", 1.0)
    for k, v in cols.items():
        assert k in named, f"unknown event column {k!r} (have: {sorted(named)})"
        obs[0, off + named[k]] = float(v)
    return obs


def set_region(obs: torch.Tensor, abs_offset: int, values) -> torch.Tensor:
    """Write `values` into obs[0, abs_offset:abs_offset+len]. Returns obs."""
    t = torch.as_tensor(np.asarray(values, dtype=np.float32))
    obs[0, abs_offset:abs_offset + t.numel()] = t.reshape(-1)
    return obs


# ---------------------------------------------------------------------------
# Forward / assertions
# ---------------------------------------------------------------------------

def forward(model, obs: torch.Tensor):
    """Return (pi_features, vf_features) under no_grad."""
    with torch.no_grad():
        return model({"observation": obs})


def policy_value_changed(model, obs_a: torch.Tensor, obs_b: torch.Tensor) -> Tuple[bool, bool]:
    pa, va = forward(model, obs_a)
    pb, vb = forward(model, obs_b)
    return (not torch.allclose(pa, pb)), (not torch.allclose(va, vb))


def assert_reaches_network(model, base: torch.Tensor, variant: torch.Tensor, msg: str = ""):
    """Assert the difference between `base` and `variant` moves BOTH heads.

    The transformer body is shared and history/static signals feed both the
    policy and value pools, so a live signal should move both. If a specific
    edge case legitimately only touches one head, use `policy_value_changed`
    and assert the relevant one instead."""
    pi_changed, vf_changed = policy_value_changed(model, base, variant)
    assert pi_changed, f"policy head did not move — signal not reaching network: {msg}"
    assert vf_changed, f"value head did not move — signal not reaching network: {msg}"


def assert_event_reaches_network(model, layout, base_cols: dict, variant_cols: dict,
                                 msg: str = ""):
    """Place two EVENT ROWS in the most-recent window slot and assert the output differs.

    gen3_frame_deletion_v1 successor to `assert_delta_reaches_network`. Both rows carry the same
    columns except the one under test, so the comparison isolates that fact — the same discipline
    `anchor_delta` enforced for the lag frames.

    ⚠️ Requires the model to be built with `history_events=True`: the event window's raw ids are
    consumed ONLY by the flag-gated event-seat encoder, so with the flag off NO Linear reads the
    block and every probe here would compare two identical forwards and pass while testing
    nothing. That is the exact vacuous-pass this helper must not permit, so it asserts the flag."""
    assert getattr(model, "history_events", None) is not None, (
        "assert_event_reaches_network needs history_events=True — with the event seats off "
        "nothing reads the window and the probe would pass vacuously")
    base = obs_with_event(layout, **base_cols)
    variant = obs_with_event(layout, **variant_cols)
    assert_reaches_network(model, base, variant, msg)


# ---------------------------------------------------------------------------
# Vector inspection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TurnDelta -> event-window translation (gen3_frame_deletion_v1)
# ---------------------------------------------------------------------------
# The frames are deleted, so a fact that used to reach the network as a TurnDelta lag slot now
# reaches it as an H-B EVENT ROW. Rather than rewrite ~40 probe call sites into column literals,
# the probes keep saying what they mean (`anchor_delta(our_move_crit=True)`) and this ONE table
# routes it to the new delivery. That makes the table the substitution artifact: which
# TurnDelta facts have an event home, stated once, in code, where it can be tested.
#
# UNMAPPED fields RAISE rather than silently produce an identical row for base and variant —
# which would make the probe pass while testing nothing. That is the whole hazard of a
# translation layer and it is why the default is loud.

_UNMAPPED = {
    # `attempted_switch_to` / `our_attempted_switch_spec`: the refused switch's TARGET. The event
    # window records the rejection (EVENT_T_SWITCH_REJECTED) but not whom it aimed at, and cannot:
    # `Gen3Battle.record_choice_rejected` recovers the target from the ACTION INDEX at fold time,
    # and this window folds from events alone. See ARCHITECTURE.md §1.6.
    "attempted_switch_to",
    "attempted_switch_rejected_target",
}


def delta_to_event_cols(delta) -> dict:
    """Map a `TurnDelta` to the event-window columns that carry the same fact.

    Returns the `cols` dict `obs_with_event` takes. Raises on a field with no event home, naming
    it — a coverage gap must be enumerable, not silently vacuous."""
    from agents.observation.constants import (
        EVENT_T_MOVE, EVENT_T_SWITCH_IN, EVENT_T_FAINT, EVENT_T_STATUS_APPLIED,
        EVENT_T_STATUS_CURED, EVENT_T_BOOST, EVENT_T_CANT,
    )
    from agents.observation.gen3_effects import cant_reason_id
    from agents import gen3_data

    def _mv(mid):
        m = gen3_data.moves.get(mid) if mid else None
        return float(m.num) if m is not None else 0.0

    def _sp(name):
        s_ = gen3_data.species.get(name) if name else None
        return float(s_.num) if s_ is not None else 0.0

    for f in _UNMAPPED:
        if getattr(delta, f, None):
            raise AssertionError(
                f"TurnDelta.{f} has NO event-window column — it cannot reach the network at all "
                f"since gen3_frame_deletion_v1. A probe asserting it 'reaches the network' would "
                f"be asserting something false; see ARCHITECTURE.md \u00a71.6.")

    cols: dict = {"valid": 1.0}

    # --- CANT: the column added FOR the deletion ---
    if getattr(delta, "our_cant_reason", None) or getattr(delta, "opp_cant_reason", None):
        ours = bool(getattr(delta, "our_cant_reason", None))
        cols.update(type_id=EVENT_T_CANT, actor_side=(1.0 if ours else -1.0), fail=1.0,
                    cant_id=cant_reason_id(delta.our_cant_reason if ours else delta.opp_cant_reason))
        return cols

    # --- STATUS ---
    for fld, t in (("our_status_applied", EVENT_T_STATUS_APPLIED),
                   ("opp_status_applied", EVENT_T_STATUS_APPLIED),
                   ("our_status_cured", EVENT_T_STATUS_CURED),
                   ("opp_status_cured", EVENT_T_STATUS_CURED)):
        v = getattr(delta, fld, None)
        if v is None:
            continue
        # The NAME, never `.value` — a Status enum's value is an int, so `str(value)` is a digit,
        # falls out of the lookup, and every status collapses to the same id. That made PAR and
        # SLP translate identically; the `b != v` guard in assert_delta_reaches_network caught it,
        # which is exactly why that guard is there.
        name = getattr(v, "name", str(v).split(".")[-1]).lower()
        order = ["", "brn", "frz", "par", "psn", "slp", "tox"]
        cols.update(type_id=t, actor_side=(1.0 if fld.startswith("our") else -1.0),
                    status_id=float(order.index(name)) if name in order else 1.0)
        return cols

    # --- BOOST ---
    def _boost_total(x):
        """Sum a boost delta whatever shape it arrives in. Explicitly NOT `if x:` — these are
        numpy arrays, and a bare truth test raises "truth value of an array is ambiguous"."""
        if x is None:
            return 0.0
        if isinstance(x, dict):
            return float(sum(x.values()))
        arr = np.asarray(x, dtype=np.float32)
        return float(arr.sum())

    our_b = _boost_total(getattr(delta, "our_boost_delta", None))
    opp_b = _boost_total(getattr(delta, "opp_boost_delta", None))
    if our_b or opp_b:
        cols.update(type_id=EVENT_T_BOOST,
                    magnitude=(our_b if our_b else opp_b) / 6.0,
                    actor_side=1.0 if our_b else -1.0)
        return cols

    # --- FAINT ---
    if getattr(delta, "our_fainted", None) or getattr(delta, "opp_fainted", None):
        cols.update(type_id=EVENT_T_FAINT,
                    actor_side=1.0 if getattr(delta, "our_fainted", None) else -1.0)
        return cols

    # --- SWITCH-IN, LAST and only when nothing move-shaped is set -------------------------------
    # ORDER IS LOAD-BEARING. `anchor_delta` sets `our_prev_active="snorlax"` on EVERY delta as its
    # slot-presence anchor, so a branch that fires on it swallows whatever field the probe is
    # actually varying — 35 probes collapsed to the identical SWITCH_IN row before this moved.
    # It is the fallback, not the first match, and the `b != v` guard is what surfaced the bug.
    _move_shaped = any(getattr(delta, f, None) is not None for f in (
        "our_move_id", "opp_move_id", "our_attempted_move_id", "our_move_outcome",
        "opp_move_outcome", "our_move_crit", "opp_move_crit", "our_effectiveness",
        "we_moved_first"))
    if not _move_shaped:
        for fld, side in (("our_prev_active", 1.0), ("opp_prev_active", -1.0)):
            v = getattr(delta, fld, None)
            if v:
                cols.update(type_id=EVENT_T_SWITCH_IN, actor=_sp(v), actor_side=side)
                if getattr(delta, "phase_is_forced_switch", None):
                    cols["forced_window"] = 1.0
                return cols

    # --- MOVE (the default: outcome / crit / effectiveness / order / known) ---
    ours = getattr(delta, "our_move_id", None) or getattr(delta, "our_attempted_move_id", None)
    theirs = getattr(delta, "opp_move_id", None)
    side = 1.0 if (ours or not theirs) else -1.0
    cols.update(type_id=EVENT_T_MOVE, actor_side=side, move_num=_mv(ours or theirs))
    out = (getattr(delta, "our_move_outcome", None) if side > 0
           else getattr(delta, "opp_move_outcome", None))
    cols["hit"] = 1.0 if out == "hit" else 0.0
    cols["miss"] = 1.0 if out == "miss" else 0.0
    cols["fail"] = 1.0 if out == "fail" else 0.0
    crit = (getattr(delta, "our_move_crit", None) if side > 0
            else getattr(delta, "opp_move_crit", None))
    cols["crit"] = 1.0 if crit else 0.0
    eff = getattr(delta, "our_effectiveness", None)
    if eff is not None:
        key = {0.0: "eff_immune", 0.25: "eff_resist", 0.5: "eff_resist",
               1.0: "eff_neutral", 2.0: "eff_super", 4.0: "eff_super"}.get(float(eff))
        if key:
            cols[key] = 1.0
    wf = getattr(delta, "we_moved_first", None)
    cols["we_first"] = 1.0 if wf else 0.0
    if getattr(delta, "phase_is_forced_switch", None):
        cols["forced_window"] = 1.0
    # `opp_move_known=False` means the move was never observed — an unobserved move produces NO
    # event row at all, which is the honest translation (absence, not a zeroed id).
    if theirs is not None and getattr(delta, "opp_move_known", True) is False:
        cols["move_num"] = 0.0
        cols["valid"] = 0.0
    return cols


def obs_with_delta(layout, delta, slot_from_recent: int = 0) -> torch.Tensor:
    """A `TurnDelta` as an observation. gen3_frame_deletion_v1: it lands in the EVENT WINDOW now,
    not a lag frame — the name and signature are kept so the probes that build an obs directly
    (rather than through `assert_delta_reaches_network`) read the same as before."""
    return obs_with_event(layout, slot_from_recent, **delta_to_event_cols(delta))


def assert_delta_reaches_network(model, layout, base_delta, variant_delta, msg: str = ""):
    """Two deltas -> two EVENT ROWS -> assert the network output differs.

    gen3_frame_deletion_v1: the NAME and the call sites are unchanged, the DELIVERY is not. The
    probes still express a fact as a TurnDelta (readable, and the CAPTURE half of every probe
    still checks the real `TurnDeltaEncoder`); `delta_to_event_cols` routes it to the block that
    actually reaches the network now.

    Guards against the two ways a translation layer goes quietly wrong:
      * the model must have the event seats built, or nothing reads the window;
      * the two translated rows must DIFFER, or the probe compares a row with itself and passes
        while testing nothing — the exact failure this indirection could otherwise hide."""
    assert getattr(model, "history_events", None) is not None, (
        "the feature-coverage fixture must build history_events=True — with the event seats off "
        "nothing reads the window and every probe here would pass vacuously")
    b = delta_to_event_cols(base_delta)
    v = delta_to_event_cols(variant_delta)
    assert b != v, (
        f"the two deltas translate to the SAME event row {b} — this probe would pass without "
        f"testing anything. The fact under test has no distinguishing event column: {msg}")
    assert_reaches_network(model, obs_with_event(layout, **b), obs_with_event(layout, **v), msg)


def read_block(vec, offset: int, dim: int) -> np.ndarray:
    return np.asarray(vec)[offset:offset + dim]


def onehot_idx(vec, offset: int, dim: int) -> Optional[int]:
    """Index of the set bit in vec[offset:offset+dim], or None if all < 0.5."""
    block = read_block(vec, offset, dim)
    if block.max() <= 0.5:
        return None
    return int(np.argmax(block))


def multihot_idxs(vec, offset: int, dim: int) -> Sequence[int]:
    """Indices set (>0.5) in vec[offset:offset+dim]."""
    block = read_block(vec, offset, dim)
    return [i for i in range(dim) if block[i] > 0.5]
