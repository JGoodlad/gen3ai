"""Public-information value (V_pub) — the human-replay-calibrated position evaluator.

`gen3_pubval_aux_v1` (design: `designs/ai_v8/design_public_info_value.md`). The measured limiter is
the value function: it is blind to defensive/positional value (defensive AUC ≈ 0.50), so the
advantage is ~0 on positional decisions and the policy never learns a game plan. V_pub is the
value-INDEPENDENT exogenous signal that breaks that bootstrap: `P(win | PUBLIC board)`, calibrated
OFFLINE on the human gen3ou replay corpus (164k rated ladder games), then evaluated LIVE per
decision as a dense per-step aux TARGET for a shared-trunk head (`PubValHead`, the WinProbHead
pattern) — so the trunk is pressured to represent the human-outcome-calibrated positional read of
the board (hazards/status/attrition value priced by human play), turn by turn, instead of only the
terminal outcome. It NEVER touches GAE/bootstrapping (it is V^human, not V^π) and reads PUBLIC
state only (leak-free by construction — the POC's turn-1 AUC ≈ 0.51 is the guard).

This module is the SINGLE SOURCE of the feature definition. Both consumers use it:
  * OFFLINE — `pubval_calibration.py` parses replay logs into `PubSide` rows (`parse_replay_log`)
    and trains the logistic on `features(...)`, writing the frozen artifact `data/gen3_pubval.json`.
  * LIVE — `Gen3Env` folds the current board into the SAME `PubSide` rows via `pub_side_from_live`
    (over the vetted `LiveView` read-model) and evaluates the frozen `PubValModel` on the SAME
    `features(...)`. Feature parity is therefore structural, and guarded end-to-end by
    `pubval_parity_fuzz_test.py` (live rows == parser rows over real bridge battles).

The 17 features are the POC's validated "crude aggregates" (test AUC 0.734, calibrated, clean
leakage; richer identity features overfit — see the design doc): material diffs (alive / active-HP /
known-HP), positional diffs (spikes / status-count / active-boosts / revealed-count), per-side
absolutes, a saturating turn clock, and a 5-way weather one-hot.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

# The frozen artifact (written by `python -m agents.training.pubval_calibration`). Lives in data/
# like `gen3_bot_elo_anchors.json` — a derived, committed calibration artifact with provenance.
DEFAULT_PUBVAL_PATH = os.path.join("data", "gen3_pubval.json")

# Weather one-hot axis (lowercased protocol names; None → 'none'). Gen3's four weathers.
PUBVAL_WEATHERS = ("none", "sandstorm", "raindance", "sunnyday", "hail")

PUBVAL_FEATURE_NAMES = (
    "alive_diff", "active_hp_diff", "known_hp_diff",
    "spikes_diff", "statusc_diff", "boost_diff", "revealed_diff",
    "my_alive", "their_alive", "my_active_hp", "their_active_hp",
    "turn_clock",
) + tuple(f"weather_{w}" for w in PUBVAL_WEATHERS)

PUBVAL_N_FEATURES = len(PUBVAL_FEATURE_NAMES)  # 17


@dataclass(frozen=True)
class PubSide:
    """One side's PUBLIC aggregates at a decision point — the parser and the live path both emit
    exactly this (the parity contract). All fields are spectator-visible."""
    alive: int          # 6 − fainted (unrevealed mons are alive by definition)
    active_hp: float    # active mon's HP fraction (0.0 when fainted/none)
    known_hp: float     # Σ HP fraction over REVEALED, non-fainted mons (unrevealed contribute 0)
    revealed: int       # how many of the 6 have been seen on the field
    spikes: int         # Spikes layers on THIS side's field (0–3)
    statusc: int        # statused (non-fainted) mons — a faint clears the count
    boost: float        # Σ of the ACTIVE mon's stat stages (reset on switch; 0 when none active)


def features(me: PubSide, them: PubSide, turn: int, weather: "str | None") -> np.ndarray:
    """The 17-dim V_pub feature vector from MY perspective — THE single feature definition
    (order == PUBVAL_FEATURE_NAMES; the calibration artifact records it for a load-time guard)."""
    w = (weather or "none").lower()
    vec = [
        float(me.alive - them.alive),
        me.active_hp - them.active_hp,
        me.known_hp - them.known_hp,
        float(me.spikes - them.spikes),
        float(me.statusc - them.statusc),
        me.boost - them.boost,
        float(me.revealed - them.revealed),
        float(me.alive), float(them.alive),
        me.active_hp, them.active_hp,
        min(turn, 50) / 50.0,
    ] + [1.0 if w == name else 0.0 for name in PUBVAL_WEATHERS]
    return np.asarray(vec, dtype=np.float32)


# ── live side (Gen3Env) ────────────────────────────────────────────────────────

def pub_side_from_live(side) -> PubSide:
    """Fold a `LiveSide` (the vetted current-board read-model) into the SAME `PubSide` the replay
    parser emits. Semantics parity notes: `known_hp`/`revealed` filter on `m.revealed` (our own
    side's LiveSide carries all 6 mons — the opponent only sees the revealed ones); `statusc`
    counts LIVING statused mons ('fnt' is a faint marker, not a status — the parser clears status
    on faint for the same reason); `boost` sums the ACTIVE mon's stages (both sides' boosts are
    public; gen3 resets them on switch, which poke-env and the parser both track)."""
    active = side.active
    return PubSide(
        alive=6 - sum(1 for m in side.mons if m.fainted),
        active_hp=float(active.hp_fraction) if active is not None else 0.0,
        known_hp=float(sum(m.hp_fraction for m in side.mons if m.revealed and not m.fainted)),
        revealed=sum(1 for m in side.mons if m.revealed),
        spikes=int(side.side_conditions.get("spikes", 0)),
        statusc=sum(1 for m in side.mons
                    if not m.fainted and m.status is not None and m.status != "fnt"),
        boost=float(sum(active.boosts.values())) if active is not None else 0.0,
    )


# ── frozen artifact ────────────────────────────────────────────────────────────

class PubValModel:
    """The frozen replay-calibrated logistic: `P(win) = σ(((f − mu) / sd) · w + b)`.

    Loaded once from `data/gen3_pubval.json` (fail-loud if missing/stale — the run must not
    silently train the aux head toward garbage). `meta` carries provenance (n_games, held-out AUC,
    date, git hash) so the artifact is auditable."""

    def __init__(self, mu, sd, w, b: float, feature_names, meta: "dict | None" = None):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.b = float(b)
        self.feature_names = tuple(feature_names)
        self.meta = dict(meta or {})
        if self.feature_names != PUBVAL_FEATURE_NAMES:
            raise ValueError(
                "pubval artifact feature_names do not match this code's PUBVAL_FEATURE_NAMES — "
                "the artifact is stale; regenerate with `python -m agents.training.pubval_calibration`. "
                f"artifact={self.feature_names} code={PUBVAL_FEATURE_NAMES}")
        if not (self.mu.shape == self.sd.shape == self.w.shape == (PUBVAL_N_FEATURES,)):
            raise ValueError(f"pubval artifact has wrong shapes: mu{self.mu.shape} sd{self.sd.shape} "
                             f"w{self.w.shape}, expected ({PUBVAL_N_FEATURES},)")

    def predict(self, feats: np.ndarray) -> float:
        """P(win) ∈ (0, 1) for one 17-dim feature vector."""
        z = ((np.asarray(feats, dtype=np.float64) - self.mu) / self.sd) @ self.w + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_json(self) -> dict:
        return {"feature_names": list(self.feature_names), "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "w": self.w.tolist(), "b": self.b, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "PubValModel":
        return cls(mu=d["mu"], sd=d["sd"], w=d["w"], b=d["b"],
                   feature_names=d["feature_names"], meta=d.get("meta"))

    @classmethod
    def load(cls, path: str = DEFAULT_PUBVAL_PATH) -> "PubValModel":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"pubval artifact not found at {path!r} — the public-replay value aux "
                "(--pubval-mode) needs the calibrated logistic. Generate it once with:\n"
                "  python -m agents.training.pubval_calibration")
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


# ── replay side (the calibration corpus parser) ────────────────────────────────
# An improved version of the POC parser (tmp/pubval_experiment.py): same public-state fold, plus
# the protocol handlers the POC skipped whose absence would break LIVE parity — a faint clears the
# status count, |-cureteam| (Heal Bell/Aromatherapy), |-setboost| (Belly Drum), |-clearallboost|
# (Haze), |-copyboost| (Psych Up). Per-line try/except: a malformed line degrades one game, never
# the corpus pass.

_NICK_RE = re.compile(r"^(p[12])[a-z]?: ?(.*)$")


def _who(field: str) -> "tuple[str, str] | None":
    """'p1a: Zapdos' → ('p1', 'Zapdos')."""
    m = _NICK_RE.match(field.strip())
    return (m.group(1), m.group(2)) if m else None


def _parse_hp(field: str) -> float:
    """'123/300' → fraction; 'X/100 par' → X/100; '0 fnt' → 0. Own-side lines carry exact
    numerators, spectator/opponent lines are /100 — both divide the same way."""
    tok = field.strip().split()[0] if field.strip() else "0"
    if "fnt" in field or tok == "0":
        return 0.0
    if "/" in tok:
        a, b = tok.split("/")[:2]
        try:
            return max(0.0, min(1.0, float(a) / float(b)))
        except (ValueError, ZeroDivisionError):
            return 1.0
    return 1.0


def _new_side() -> dict:
    return {"faints": 0, "active": None, "mons": {}, "spikes": 0, "boosts": {}}


def _side_row(s: dict) -> PubSide:
    mons = s["mons"]
    act = mons.get(s["active"]) or {}
    return PubSide(
        alive=6 - s["faints"],
        active_hp=float(act.get("hp", 1.0)) if not act.get("fainted") else 0.0,
        known_hp=float(sum(m["hp"] for m in mons.values() if not m.get("fainted"))),
        revealed=len(mons),
        spikes=int(s["spikes"]),
        statusc=sum(1 for m in mons.values() if m.get("status") and not m.get("fainted")),
        boost=float(sum(s["boosts"].values())),
    )


def parse_replay_log(text: str):
    """Parse one Showdown protocol log into per-turn public snapshots.

    Returns ``(positions, winner, ratings, is_rated)`` where ``positions`` is a list of
    ``(turn, PubSide_p1, PubSide_p2, weather)`` snapshotted at each ``|turn|`` boundary (the same
    board state a player sees at the start-of-turn decision), ``winner`` ∈ {0 (p1), 1 (p2), None},
    and ``ratings`` = (p1, p2) ints (0 = unknown). Works on both spectator replay logs (the corpus)
    and a player's one-sided stream (the parity fuzz) — the grammar is identical; only HP precision
    differs (/100 vs exact), which the shared ``_parse_hp`` division absorbs."""
    S = {"p1": _new_side(), "p2": _new_side()}
    names = {"p1": None, "p2": None}
    ratings = {"p1": 0, "p2": 0}
    weather = "none"
    positions: list = []
    winner_name = None
    is_rated = False

    for ln in text.splitlines():
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        try:
            if tag == "player" and len(parts) >= 4:
                p = parts[2]
                if p in names:
                    names[p] = parts[3].strip()
                    if len(parts) >= 6 and parts[5].strip().isdigit():
                        ratings[p] = int(parts[5].strip())
            elif tag == "rated":
                is_rated = True
            elif tag in ("switch", "drag") and len(parts) >= 5:
                pk = _who(parts[2])
                if pk is None:
                    continue
                p, nick = pk
                S[p]["active"] = nick
                S[p]["boosts"] = {}                     # gen3: boosts reset on switch
                S[p]["mons"].setdefault(nick, {"hp": 1.0, "fainted": False, "status": None})
                S[p]["mons"][nick]["hp"] = _parse_hp(parts[4])
                S[p]["mons"][nick]["fainted"] = False
            elif tag in ("-damage", "-heal", "-sethp") and len(parts) >= 4:
                pk = _who(parts[2])
                if pk and pk[1] in S[pk[0]]["mons"]:
                    S[pk[0]]["mons"][pk[1]]["hp"] = _parse_hp(parts[3])
            elif tag == "faint":
                pk = _who(parts[2])
                if pk is None:
                    continue
                p, nick = pk
                S[p]["faints"] += 1
                if nick in S[p]["mons"]:
                    S[p]["mons"][nick].update(fainted=True, hp=0.0, status=None)  # faint ends status
            elif tag == "-status" and len(parts) >= 4:
                pk = _who(parts[2])
                if pk and pk[1] in S[pk[0]]["mons"]:
                    S[pk[0]]["mons"][pk[1]]["status"] = parts[3].strip()
            elif tag == "-curestatus":
                pk = _who(parts[2])
                if pk and pk[1] in S[pk[0]]["mons"]:
                    S[pk[0]]["mons"][pk[1]]["status"] = None
            elif tag == "-cureteam":                      # Heal Bell / Aromatherapy
                pk = _who(parts[2])
                if pk:
                    for m in S[pk[0]]["mons"].values():
                        m["status"] = None
            elif tag in ("-boost", "-unboost") and len(parts) >= 5:
                pk = _who(parts[2])
                if pk is None:
                    continue
                d = S[pk[0]]["boosts"]
                amt = int(parts[4])
                d[parts[3]] = d.get(parts[3], 0) + (amt if tag == "-boost" else -amt)
            elif tag == "-setboost" and len(parts) >= 5:  # Belly Drum
                pk = _who(parts[2])
                if pk:
                    S[pk[0]]["boosts"][parts[3]] = int(parts[4])
            elif tag == "-clearallboost":                 # Haze — both sides
                S["p1"]["boosts"] = {}
                S["p2"]["boosts"] = {}
            elif tag == "-clearboost":
                pk = _who(parts[2])
                if pk:
                    S[pk[0]]["boosts"] = {}
            elif tag == "-copyboost" and len(parts) >= 4:  # Psych Up: SOURCE copies TARGET
                src, tgt = _who(parts[2]), _who(parts[3])
                if src and tgt:
                    S[src[0]]["boosts"] = dict(S[tgt[0]]["boosts"])
            elif tag == "-sidestart" and "Spikes" in ln:
                pk = parts[2].strip()[:2]
                if pk in S:
                    S[pk]["spikes"] = min(3, S[pk]["spikes"] + 1)
            elif tag == "-sideend" and "Spikes" in ln:     # Rapid Spin
                pk = parts[2].strip()[:2]
                if pk in S:
                    S[pk]["spikes"] = 0
            elif tag == "-weather" and len(parts) >= 3:
                weather = parts[2].strip() or "none"
            elif tag == "turn":
                positions.append((int(parts[2]), _side_row(S["p1"]), _side_row(S["p2"]), weather))
            elif tag == "win":
                winner_name = parts[2].strip()
        except (ValueError, IndexError, KeyError):
            continue

    winner = None
    if winner_name is not None:
        if winner_name == names["p1"]:
            winner = 0
        elif winner_name == names["p2"]:
            winner = 1
    return positions, winner, (ratings["p1"], ratings["p2"]), is_rated
