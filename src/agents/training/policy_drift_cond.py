"""POLICY DRIFT meter — CONDITIONAL action-class rates (the class's rate WHEN IT WAS LEGAL).

The overall action-mix share ("hazard 2%" of all choice states) mixes two things: does the policy
LIKE Spikes, and how often is Spikes even available? This module reports each class as a rate
CONDITIONAL ON ELIGIBILITY instead — among the probe states where the class was legal, how often the
greedy choice is that class — plus the mean probability MASS on the class there (it shows a lean
before the argmax flips).

Two eligibility grains:
  * ELIGIBLE — at least one LEGAL action of the class in the state's mask. Every class.
  * USEFUL   — at least one legal action of the class that would NOT fail, read from facts in the
    probe state's observation (via the obs layout — never a hardcoded index). Defined for three
    classes only; each rule verified in deps/pokemon-showdown (gen3 inherits data/moves.ts unless
    data/mods/gen3|gen4 overrides it):
      - hazard (Spikes): the opponent side has < 3 layers (`spikes.condition.onSideRestart` returns
        false at layers >= 3);
      - recovery: our active's HP < 100% (`heal()` fails at full HP; Rest's `onTry` fails at full HP);
        Rest also fails when already asleep (`onTry`: status === 'slp'); Swallow needs a Stockpile
        (`onTry`: `!!source.volatiles['stockpile']`); Wish is useful at FULL HP (it heals whoever
        is in the slot next turn) but fails while one is already pending on the slot
        (`side.addSlotCondition` returns false for a present condition without `onRestart`);
      - setup: at least one stat the move raises is below +6 (`getCappedBoost` → a +0 boost at +6);
        Belly Drum additionally fails at HP <= 50% (`bellydrum.onHit`).
    The universe is states with >= 2 legal actions: a forced state carries no preference.

The DELTA vs a reference is PAIRED — eligibility is a property of the STATE, so current and
reference are measured on the same states — and its 95% interval is a paired bootstrap over those
states (seeded, so a recompute is bit-identical). Each rate carries a 95% Wilson interval.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from agents.action.constants import MOVE_END, MOVE_START, SWITCH_END

# the classes reported conditionally ("other" = struggle / an unknown move is not a strategy class)
COND_CLASSES: Tuple[str, ...] = ("switch", "attack", "status", "setup", "hazard", "recovery",
                                 "phazing", "self_ko")
USEFUL_CLASSES: Tuple[str, ...] = ("setup", "hazard", "recovery")
_CLS_IDX = {c: i for i, c in enumerate(COND_CLASSES)}
OTHER = -2
ILLEGAL = -1

# ActiveContextEncoder.encode writes the 7 boost stages in THIS order, [positive, negative] / 6 each.
# Pinned by policy_drift_cond_test against the encoder itself.
BOOST_STATS: Tuple[str, ...] = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
MAX_STAGE = 6
# setup moves whose boost is not in the dex's `self_boosts` map (Showdown keys it on a callback /
# a volatile / a type-conditional onModifyMove): the stats each one raises
_SETUP_STATS_OVERRIDE = {"bellydrum": ("atk",), "curse": ("atk", "def"),    # curse: non-Ghost user
                         "defensecurl": ("def",), "minimize": ("evasion",)}

BOOT_B = 4000
BOOT_SEED = 20260926
Z95 = 1.959963984540054


# ── dex tables ────────────────────────────────────────────────────────────────────────────────
def move_ids_by_num() -> Dict[int, str]:
    from agents.gen3_data import moves
    out: Dict[int, str] = {}
    for mid in moves.raw():
        md = moves.get(mid)
        if md is not None and md.num > 0:
            out[md.num] = md.id
    return out


def setup_stats(move_id: str) -> Tuple[str, ...]:
    """The stats a SETUP-class move raises (empty only for a move that is not setup)."""
    if move_id in _SETUP_STATS_OVERRIDE:
        return _SETUP_STATS_OVERRIDE[move_id]
    from agents.gen3_data import moves
    md = moves.get(move_id)
    return tuple(s for s, v in (md.self_boosts if md is not None else ()) if v > 0)


# ── state facts from the observation (layout-driven) ──────────────────────────────────────────
def state_facts(obs: np.ndarray, layout: dict) -> Dict[str, np.ndarray]:
    """Per-state facts the USEFUL rules read, each through the obs layout:
    opp_spikes [N] int, hp [N] (our active's fraction; 1.0 when no active is flagged),
    boosts {stat: [N] int}, asleep / stockpile / wish_pending [N] bool."""
    from agents.observation import constants as C
    from agents.observation.active_context import ActiveContextEncoder
    from agents.observation.gen3_effects import VOLATILE_SLOTS
    from agents.observation.pokemon import _STATUS_STR_IDX

    obs = np.asarray(obs, np.float32)
    n = len(obs)
    parts = layout["parts"]
    # our team: [6, slot] — the active flag's offset is the constant the encoder writes it at
    ot = parts["our_team"]
    slots, width = ot["reshape"]
    team = obs[:, ot["start"]: ot["end"]].reshape(n, slots, width)
    act = team[:, :, C.POKEMON_ACTIVE_OFFSET] > 0.5
    has_active = act.any(-1)
    ai = act.argmax(-1)
    active = team[np.arange(n), ai]                                     # [N, width]
    pl = layout["pokemon"]
    hp = np.where(has_active, active[:, pl["hp"]["offset"]], 1.0)
    cond = pl["condition"]["offset"]
    asleep = has_active & (active[:, cond + _STATUS_STR_IDX["slp"]] > 0.5)
    # our active's context: parts["context"] is [2, dim], ours first
    cx = parts["context"]
    ctx = obs[:, cx["start"]: cx["end"]].reshape(n, *cx["reshape"])[:, 0]
    acl = ActiveContextEncoder().get_layout()
    b0 = acl["boosts"]["offset"]
    boosts = {s: np.rint((ctx[:, b0 + 2 * i] - ctx[:, b0 + 2 * i + 1]) * MAX_STAGE).astype(np.int64)
              for i, s in enumerate(BOOST_STATS)}
    stockpile = ctx[:, acl["volatiles"]["offset"] + VOLATILE_SLOTS.index("stockpile")] > 0
    # global env: hazards = [ours, opp] Spikes layers / MAX_SPIKES
    g0 = parts["global"]["start"] + layout["global_layout"]["hazards"]["offset"]
    opp_spikes = np.rint(obs[:, g0 + 1] * C.MAX_SPIKES).astype(np.int64)
    r0 = parts["reactive"]["start"] + layout["reactive_layout"]["wish_floating_our"]["offset"]
    wish_pending = obs[:, r0] > 0
    return {"opp_spikes": opp_spikes, "hp": hp.astype(np.float64), "boosts": boosts,
            "asleep": asleep, "stockpile": stockpile, "wish_pending": wish_pending}


def move_is_useful(move_id: str, cls: str, i: int, facts: Dict[str, np.ndarray]) -> bool:
    """Would this legal move do something in state ``i``? Only hazard/recovery/setup have a rule;
    every other class is useful whenever legal."""
    hp = float(facts["hp"][i])
    if cls == "hazard":
        return int(facts["opp_spikes"][i]) < 3
    if cls == "recovery":
        if move_id == "wish":
            return not bool(facts["wish_pending"][i])
        if move_id == "rest":
            return hp < 1.0 and not bool(facts["asleep"][i])
        if move_id == "swallow":
            return hp < 1.0 and bool(facts["stockpile"][i])
        return hp < 1.0
    if cls == "setup":
        stats = setup_stats(move_id)
        if not stats:
            return True                   # unknown boost map: never silently ruled useless
        if move_id == "bellydrum" and hp <= 0.5:
            return False
        return any(int(facts["boosts"][s][i]) < MAX_STAGE for s in stats)
    return True


def action_tables(mask: np.ndarray, move_nums: np.ndarray, table: Dict[int, str],
                  ids: Optional[Dict[int, str]] = None,
                  facts: Optional[Dict[str, np.ndarray]] = None):
    """→ (cls_act [N, A] int — COND_CLASSES index, ILLEGAL, or OTHER;
          useful_act [N, A] bool, or None when no facts are available)."""
    n, a = mask.shape
    cls_act = np.full((n, a), ILLEGAL, np.int64)
    useful = np.zeros((n, a), bool) if facts is not None else None
    legal = mask > 0.5
    for i in range(n):
        for j in np.flatnonzero(legal[i]):
            if j < SWITCH_END:
                c = "switch"
                mid = ""
            elif MOVE_START <= j < MOVE_END:
                num = int(move_nums[i, j - MOVE_START])
                c = table.get(num, "other")
                mid = (ids or {}).get(num, "")
            else:
                c, mid = "other", ""
            cls_act[i, j] = _CLS_IDX.get(c, OTHER)
            if useful is not None:
                useful[i, j] = move_is_useful(mid, c, i, facts) if c in USEFUL_CLASSES else True
    return cls_act, useful


# ── statistics ────────────────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = Z95) -> Optional[Tuple[float, float]]:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, mid - half)), float(min(1.0, mid + half)))


def paired_delta_ci(x_cur: np.ndarray, x_ref: np.ndarray, b: int = BOOT_B,
                    seed: int = BOOT_SEED) -> Optional[Tuple[float, float]]:
    """95% percentile interval of mean(x_cur − x_ref) under a bootstrap over STATES. For 0/1
    indicators the per-state difference is −1/0/+1, so a resample is a multinomial over those
    three counts — exact, and cheap at any n."""
    d = np.asarray(x_cur, np.int64) - np.asarray(x_ref, np.int64)
    n = len(d)
    if n == 0:
        return None
    probs = np.array([(d == 1).sum(), (d == -1).sum(), (d == 0).sum()], np.float64) / n
    draws = np.random.default_rng(seed).multinomial(n, probs, size=b)
    means = (draws[:, 0] - draws[:, 1]) / n
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _legal_norm(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q = np.where(mask > 0.5, p, 0.0).astype(np.float64)
    s = q.sum(-1, keepdims=True)
    return np.divide(q, s, out=np.zeros_like(q), where=s > 0)


def _grain(sel: np.ndarray, g_is: np.ndarray, mass: np.ndarray,
           refs: Dict[str, Optional[Tuple[int, np.ndarray, np.ndarray]]]) -> dict:
    n = int(sel.sum())
    k = int(g_is[sel].sum())
    out = {"n": n, "k": k, "rate": (k / n) if n else None, "ci": wilson(k, n),
           "mass": float(mass[sel].mean()) if n else None, "refs": {}}
    for name, r in refs.items():
        if r is None:
            out["refs"][name] = None
            continue
        step, g_ref, m_ref = r
        kr = int(g_ref[sel].sum())
        out["refs"][name] = {
            "step": int(step), "rate": (kr / n) if n else None,
            "delta": ((k - kr) / n) if n else None,
            "ci": paired_delta_ci(g_is[sel], g_ref[sel]),
            "mass": float(m_ref[sel].mean()) if n else None,
            "mass_delta": float((mass[sel] - m_ref[sel]).mean()) if n else None,
        }
    return out


def _per_class(p: np.ndarray, mask: np.ndarray, cls_act: np.ndarray, ci: int):
    g = np.where(mask > 0.5, p, -1.0).argmax(-1)          # == policy_drift.greedy_actions
    g_is = cls_act[np.arange(len(g)), g] == ci
    mass = (_legal_norm(p, mask) * (cls_act == ci)).sum(-1)
    return g_is, mass


def cond_block(p_cur: np.ndarray, ref_probs: Dict[str, Optional[Tuple[int, np.ndarray]]],
               mask: np.ndarray, cls_act: np.ndarray, useful_act: Optional[np.ndarray]) -> dict:
    """The per-snapshot conditional block. ``ref_probs`` maps a reference name → (step, probs) or
    None. Pure and deterministic: the same inputs give the same block, byte for byte."""
    universe = (mask > 0.5).sum(-1) >= 2
    classes = {}
    for c in COND_CLASSES:
        ci = _CLS_IDX[c]
        of_c = cls_act == ci
        elig = universe & of_c.any(-1)
        g_is, mass = _per_class(p_cur, mask, cls_act, ci)
        refs = {}
        for name, r in ref_probs.items():
            refs[name] = None if r is None else (r[0], *_per_class(r[1], mask, cls_act, ci))
        entry = {"elig": _grain(elig, g_is, mass, refs), "useful": None}
        if c in USEFUL_CLASSES and useful_act is not None:
            entry["useful"] = _grain(universe & (of_c & useful_act).any(-1), g_is, mass, refs)
        classes[c] = entry
    return {"universe": ">=2 legal actions", "n_universe": int(universe.sum()),
            "useful_available": useful_act is not None, "classes": classes}


def primary_grain(entry: dict) -> Tuple[str, dict]:
    """USEFUL eligibility where it is defined and was computed, else plain eligibility."""
    if entry.get("useful") is not None:
        return "useful", entry["useful"]
    return "elig", entry["elig"]


def interval_clear(ci) -> bool:
    return ci is not None and (ci[0] > 0 or ci[1] < 0)
