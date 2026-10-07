"""STATE-LEVEL RND novelty reads (`gen3_ridealong_rnd_states_v1`) — does the ride-along RND input
(the raw observation) pick out unseen STATES, not only unseen teams?

:mod:`.rnd_choice`'s read (meter (v)) is a TEAM-level proxy confounded with time (train = the early N0
eval cycles, positives = held-out rows whose team never occurs in train). This module asks the
state-level questions, for BOTH variants, with the heads' own ``RndNovelty`` shapes:

* **obs-RND** over the 2761-dim observation;
* **feature-RND** over the 128-dim detached ``value_pooled`` of ONE fixed checkpoint (N0 final,
  :data:`FEATURE_CKPT_REL`), so there is no representation drift inside this read.

Every predictor: Adam at the ride-along rate (3e-4), minibatch 256, fixed seed, the input
normalisation fitted on its own training rows, read at 3 / 10 / 30 epochs (headline 10). Every
AUROC / Spearman carries a 95 % battle-clustered percentile bootstrap (B = 1000, fixed seed;
:mod:`.boot`).

The SPLIT (:func:`split_within_cells`): battles 70 / 30 WITHIN each (source × opponent class) cell —
the same teams, the same opponent mix, the same eras on both sides — fixed seed.

(1) :func:`read_seen_vs_heldout` — AUROC(held-out rows vs train rows). Generalisation reads ~0.5+,
    memorisation reads high.
(2) :func:`read_visitation` — a coarse STATE KEY decoded from the observation through
    ``Gen3ObservationEncoder.get_layout()`` and the ``constants`` offsets (:func:`state_layout`,
    :func:`decode_state`; cross-checked against the bank's own ``turn`` / ``our_alive`` /
    ``opp_alive`` stamps); per held-out row, its key's count among the TRAIN rows; RND error by
    decile of log(1 + count), the Spearman ρ, monotonicity, the count-0 rows.
(3) :func:`read_off_distribution` — bot vs self-play states with predictors trained on ONE class;
    late-game rare configurations; off-pool teams (:func:`pool_membership` — the bank holds none).
(4) :func:`read_starvation` — the SUCCESSOR states of Lane S's STARVED near-best moves (π < 1 %,
    truth gap ≤ ε) against the successor of the policy's argmax, on N0 final's decisive truth turns,
    re-derived with the X4 pre-read's one-ply machinery (``main.policy_spectrum.qhat.one_ply``, S = 1,
    the opponent's open root answered by N0 final's greedy choice — the pre-read's variant M).

Everything heavy is durable and resumable: the re-encoded bank, the N0 forward, every predictor's
snapshots and the successor chunks are cached; each read writes its own JSON and a rerun skips a read
whose JSON exists.

    python -m main.ridealong_read.rnd_states --out <dir> [--cache <dir>] [--archive <dir>] [--threads 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "gen3_ridealong_rnd_states_v1"
FEATURE_CKPT_REL = "ai_v14_01_base/final_model.zip"
TRUTH_LABEL = "N0final75M"
TRAIN_FRAC = 0.7
SPLIT_SEED = 20260930
EPOCHS_READ = (3, 10, 30)
HEADLINE_EPOCHS = 10
VARIANTS = ("obs", "feat")
#: predictor kinds: trained on every TRAIN row, or on one opponent class's TRAIN rows only (read (3))
KINDS: Dict[str, Optional[str]] = {"all": None, "bot": "bot", "pool": "pool_snapshot"}
#: turn bucket upper edges (inclusive): 1–3, 4–10, 11–20, 21–35, 36–60, 61+
TURN_EDGES = (3, 10, 20, 35, 60)
#: active HP buckets: 0 = fainted / empty, then (0, .25], (.25, .5], (.5, .75], (.75, 1]
HP_EDGES = (0.25, 0.5, 0.75)
LATE_TURN_Q = 0.95
ENDGAME_ALIVE = 2
SUCC_CHUNK = 50


# ---------------------------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------------------------

def split_within_cells(decisions: Sequence[dict], frac: float = TRAIN_FRAC,
                       seed: int = SPLIT_SEED) -> np.ndarray:
    """Boolean TRAIN mask in bank order: battles drawn ``frac`` / (1 − ``frac``) WITHIN each
    (source, opp_class) cell, round(frac · n) battles per cell to train. REFUSES a battle in two
    cells (the cell is a property of the battle)."""
    cell_of: Dict[str, Tuple[str, str]] = {}
    for d in decisions:
        c = (d["source"], d["opp_class"])
        if cell_of.setdefault(d["battle"], c) != c:
            raise ValueError(f"battle {d['battle']} sits in two cells ({cell_of[d['battle']]} and {c})")
    cells: Dict[Tuple[str, str], List[str]] = {}
    for b, c in cell_of.items():
        cells.setdefault(c, []).append(b)
    rng = np.random.default_rng(seed)
    train_b: set = set()
    for c in sorted(cells):
        bs = sorted(cells[c])
        k = int(round(frac * len(bs)))
        train_b.update(bs[i] for i in rng.permutation(len(bs))[:k])
    return np.array([d["battle"] in train_b for d in decisions], dtype=bool)


# ---------------------------------------------------------------------------------------------
# the state key, decoded from the observation through the encoder's layout
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StateLayout:
    our_start: int
    opp_start: int
    team_size: int
    slot_dim: int
    active_off: int        # within a slot
    species_off: int       # within a slot (the species id scalar)
    hp_off: int            # within a slot (HP fraction)
    weather_off: int       # absolute: the weather one-hot (index 0 = none)
    weather_onehot: int
    hazards_off: int       # absolute: [our spikes / MAX, their spikes / MAX]
    clock_off: int         # absolute: [log(1 + turn) / log(1 + MAX_TURNS), …]
    log_max_turns: float
    total_dim: int


def state_layout() -> StateLayout:
    """Every offset from ``Gen3ObservationEncoder.get_layout()`` (and the slot constants the
    encoder's own ``describe_vector`` uses). No literal index."""
    from agents.observation import constants as C
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    parts, pk, glay = lay["parts"], lay["pokemon"], lay["global_layout"]
    team_size, slot_dim = parts["our_team"]["reshape"]
    g0 = int(parts["global"]["start"])
    return StateLayout(
        our_start=int(parts["our_team"]["start"]), opp_start=int(parts["opp_team"]["start"]),
        team_size=int(team_size), slot_dim=int(slot_dim), active_off=int(C.POKEMON_ACTIVE_OFFSET),
        species_off=int(pk["species"]["offset"]) + int(pk["species"]["layout"]["species_id"]["offset"]),
        hp_off=int(pk["hp"]["offset"]),
        weather_off=g0 + int(glay["weather"]["offset"]), weather_onehot=int(C.WEATHER_ONEHOT_DIM),
        hazards_off=g0 + int(glay["hazards"]["offset"]), clock_off=g0 + int(glay["clock"]["offset"]),
        log_max_turns=math.log(1 + C.MAX_TURNS), total_dim=int(lay["total_dim"]))


def decode_state(rows: np.ndarray, L: StateLayout) -> Dict[str, np.ndarray]:
    """Per row: both actives' species id and HP fraction, the turn, the weather index, the spikes
    flags and each side's alive count (our side: present slots with HP > 0; theirs: the team size
    minus REVEALED fainted slots, since an unrevealed opponent is alive)."""
    rows = np.asarray(rows)
    if rows.shape[1] != L.total_dim:
        raise ValueError(f"rows are {rows.shape[1]}-dim, the layout says {L.total_dim}")
    out: Dict[str, np.ndarray] = {}
    for side, start in (("our", L.our_start), ("opp", L.opp_start)):
        slots = rows[:, start:start + L.team_size * L.slot_dim].reshape(len(rows), L.team_size, L.slot_dim)
        active = slots[:, :, L.active_off] > 0.5
        species = np.rint(slots[:, :, L.species_off]).astype(np.int64)
        hp = slots[:, :, L.hp_off].astype(np.float64)
        has = active.any(1)
        idx = active.argmax(1)
        r = np.arange(len(rows))
        out[f"{side}_species"] = np.where(has, species[r, idx], -1)
        out[f"{side}_hp"] = np.where(has, hp[r, idx], 0.0)
        present = species > 0
        fainted = present & (hp <= 0.0)
        out[f"{side}_alive"] = (present & (hp > 0.0)).sum(1) if side == "our" else L.team_size - fainted.sum(1)
    out["turn"] = np.rint(np.exp(rows[:, L.clock_off].astype(np.float64) * L.log_max_turns) - 1).astype(np.int64)
    out["weather"] = rows[:, L.weather_off:L.weather_off + L.weather_onehot].argmax(1)
    out["our_spikes"] = rows[:, L.hazards_off] > 0
    out["opp_spikes"] = rows[:, L.hazards_off + 1] > 0
    return out


def turn_bucket(turn: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(TURN_EDGES), np.asarray(turn), side="left")


def hp_bucket(hp: np.ndarray) -> np.ndarray:
    hp = np.asarray(hp, dtype=np.float64)
    return np.where(hp <= 0.0, 0, 1 + np.searchsorted(np.asarray(HP_EDGES), hp, side="left"))


def state_keys(dec: Mapping[str, np.ndarray]) -> np.ndarray:
    """The coarse STATE KEY per row, as a string: (our active species, opp active species, turn
    bucket, our-active HP bucket, opp-active HP bucket, weather on, our spikes, their spikes)."""
    cols = [dec["our_species"], dec["opp_species"], turn_bucket(dec["turn"]), hp_bucket(dec["our_hp"]),
            hp_bucket(dec["opp_hp"]), (dec["weather"] > 0).astype(int), dec["our_spikes"].astype(int),
            dec["opp_spikes"].astype(int)]
    return np.array(["|".join(str(int(c[i])) for c in cols) for i in range(len(cols[0]))])


def train_counts(keys: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Per row, how often its key occurs among the TRAIN rows."""
    uniq, inv = np.unique(keys, return_inverse=True)
    c = np.bincount(inv[train], minlength=len(uniq))
    return c[inv]


def decode_check(dec: Mapping[str, np.ndarray], decisions: Sequence[dict]) -> dict:
    """Agreement of the decoded turn / alive counts with the bank's own stamps (the teeth on the
    layout decode)."""
    out = {}
    for k in ("turn", "our_alive", "opp_alive"):
        ref = np.array([d[k] for d in decisions])
        out[k] = {"agree": int((dec[k] == ref).sum()), "rows": int(len(ref))}
    out["active_found"] = {"our": int((dec["our_species"] > 0).sum()), "opp": int((dec["opp_species"] > 0).sum()),
                           "rows": int(len(dec["turn"]))}
    return out


# ---------------------------------------------------------------------------------------------
# rank statistics with battle-clustered intervals
# ---------------------------------------------------------------------------------------------

def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def au(score: np.ndarray, label: np.ndarray, clusters: np.ndarray) -> dict:
    from agents.training.instrumented_ppo.ridealong_terms import rank_auroc
    from main.ridealong_read.meters import cluster_boot_auroc

    score, label = np.asarray(score, dtype=np.float64), np.asarray(label, dtype=bool)
    a = rank_auroc(score, label)
    return {"auroc": _r(a), "ci": cluster_boot_auroc(score, label, np.asarray(clusters)) if a is not None else None,
            "n_pos": int(label.sum()), "n_neg": int((~label).sum()),
            "mean_err_pos": _r(score[label].mean(), 6) if label.any() else None,
            "mean_err_neg": _r(score[~label].mean(), 6) if (~label).any() else None}


def spearman_ci(x: np.ndarray, err: np.ndarray, clusters: np.ndarray) -> dict:
    from agents.training.instrumented_ppo.ridealong_terms import spearman
    from main.ridealong_read.boot import boot_rank_stats, ci, n_clusters

    x, err = np.asarray(x, dtype=np.float64), np.asarray(err, dtype=np.float64)
    rho = spearman(x, err)
    out = {"rho": _r(rho), "ci": None, "n": int(len(x))}
    if rho is not None and n_clusters(clusters) >= 2:
        out["ci"] = ci(boot_rank_stats(x, np.asarray(clusters), err=err)["spearman"])
    return out


def _jitter(n: int, seed: int = SPLIT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).random(n) * 1e-9


def decile_profile(x: np.ndarray, err: np.ndarray) -> dict:
    """Mean ``err`` in each of 10 equal-count deciles of ``x`` (ties broken by a fixed jitter), the
    x range of each decile, and how many of the 9 adjacent steps DECREASE."""
    x, err = np.asarray(x, dtype=np.float64), np.asarray(err, dtype=np.float64)
    if len(x) < 10:
        return {"mean_err": None, "x_range": None, "decreasing_steps": None}
    order = np.argsort(x + _jitter(len(x)), kind="stable")
    parts = np.array_split(order, 10)
    m = [float(err[p].mean()) for p in parts]
    return {"mean_err": [_r(v, 6) for v in m], "x_range": [[_r(x[p].min()), _r(x[p].max())] for p in parts],
            "decreasing_steps": int(sum(m[i + 1] < m[i] for i in range(9)))}


def count_bin_profile(count: np.ndarray, err: np.ndarray) -> dict:
    """Tie-free bins of the raw count: 0, 1, 2–3, 4–7, 8–15, … (powers of two); mean error and n
    per non-empty bin, and how many adjacent non-empty steps decrease."""
    count = np.asarray(count)
    b = np.where(count == 0, 0, 1 + np.floor(np.log2(np.maximum(count, 1))).astype(int))
    bins = []
    for k in sorted(set(b.tolist())):
        sel = b == k
        lo = 0 if k == 0 else 2 ** (k - 1)
        hi = 0 if k == 0 else 2 ** k - 1
        bins.append({"count": [lo, hi], "n": int(sel.sum()), "mean_err": _r(err[sel].mean(), 6)})
    m = [x["mean_err"] for x in bins]
    return {"bins": bins, "decreasing_steps": int(sum(m[i + 1] < m[i] for i in range(len(m) - 1))),
            "steps": max(0, len(m) - 1)}


# ---------------------------------------------------------------------------------------------
# predictors: trained once, snapshots cached, scored anywhere
# ---------------------------------------------------------------------------------------------

def train_snapshots(x_train: np.ndarray, epochs_read: Sequence[int] = EPOCHS_READ) -> dict:
    """``{"snap": {ep: state_dict}, "train_loss": {ep: loss}, "in_dim": D, "n_train": N}`` —
    :func:`rnd_choice.train_and_read` (Adam 3e-4, minibatch 256, fixed seed, normalisation fitted on
    ``x_train``) with a snapshot taken at each read epoch."""
    import copy

    from main.ridealong_read import rnd_choice as R

    res = R.train_and_read(x_train, lambda rnd: copy.deepcopy(rnd.state_dict()), epochs_read=epochs_read)
    res.pop("_rnd")
    return {"snap": {int(k): v for k, v in res.items() if k.isdigit()}, "train_loss": res["train_loss"],
            "in_dim": int(x_train.shape[1]), "n_train": int(len(x_train))}


def score(snapshot: dict, in_dim: int, x: np.ndarray) -> np.ndarray:
    from main.ridealong_read import rnd_choice as R

    rnd = R.make_rnd(in_dim)
    rnd.load_state_dict(snapshot)
    rnd.eval()
    return R.rnd_errors(rnd, x)


# ---------------------------------------------------------------------------------------------
# reads (1)–(3)
# ---------------------------------------------------------------------------------------------

def read_seen_vs_heldout(err: Mapping[int, np.ndarray], train: np.ndarray, battles: np.ndarray) -> dict:
    """(1) per epoch count: AUROC(held-out rows > train rows) on the RND error."""
    return {str(ep): au(e, ~train, battles) for ep, e in err.items()}


def read_visitation(err: Mapping[int, np.ndarray], count: np.ndarray, train: np.ndarray,
                    battles: np.ndarray, turn: Optional[np.ndarray] = None) -> dict:
    """(2) on HELD-OUT rows: RND error vs log(1 + the key's TRAIN count). With ``turn``, the
    count-0 AUROC is also read WITHIN turn quintiles (rare keys are mostly late-game keys, so this
    asks whether the count carries anything the clock does not)."""
    from main.ridealong_read.meters import cluster_boot_conditional_auroc, conditional_auroc

    ho = ~train
    lc = np.log1p(count[ho].astype(np.float64))
    out: dict = {"heldout_rows": int(ho.sum()), "heldout_count0": int((count[ho] == 0).sum()),
                 "heldout_count_quantiles": {q: int(np.quantile(count[ho], float(q)))
                                             for q in ("0.1", "0.25", "0.5", "0.75", "0.9")}}
    for ep, e in err.items():
        eh = e[ho]
        out[str(ep)] = {"spearman_err_vs_log1p_count": spearman_ci(lc, eh, battles[ho]),
                        "deciles_of_log1p_count": decile_profile(lc, eh),
                        "count_bins": count_bin_profile(count[ho], eh),
                        "auroc_count0_vs_seen": au(eh, count[ho] == 0, battles[ho])}
        if turn is not None:
            lab, ref = count[ho] == 0, turn[ho].astype(np.float64)
            out[str(ep)]["auroc_count0_vs_seen_within_turn_quintiles"] = {
                "auroc": _r(conditional_auroc(eh, lab, ref)),
                "ci": cluster_boot_conditional_auroc(eh, lab, ref, battles[ho])}
    return out


def read_off_distribution(errs: Mapping[str, Mapping[int, np.ndarray]], train: np.ndarray,
                          decisions: Sequence[dict], battles: np.ndarray) -> dict:
    """(3) on HELD-OUT rows: (a) bot vs self-play with a predictor trained on ONE class (the
    all-class predictor's read of the same contrast is the reference); (b) late-game rare
    configurations, scored by the all-class predictor."""
    ho = ~train
    oc = np.array([d["opp_class"] for d in decisions])
    turn = np.array([d["turn"] for d in decisions])
    our_a = np.array([d["our_alive"] for d in decisions])
    opp_a = np.array([d["opp_alive"] for d in decisions])
    p95 = float(np.quantile(turn[train], LATE_TURN_Q))
    late = turn > p95
    both_low = (our_a <= ENDGAME_ALIVE) & (opp_a <= ENDGAME_ALIVE)
    bp = ho & np.isin(oc, ["bot", "pool_snapshot"])
    out: dict = {"turn_p95_train": p95,
                 "definitions": {"late_turn": f"turn > {p95:g} (the TRAIN rows' p95)",
                                 "both_sides_le2": f"our_alive <= {ENDGAME_ALIVE} and opp_alive <= {ENDGAME_ALIVE}",
                                 "union": "late_turn or both_sides_le2",
                                 "bank_stamps": "turn / our_alive / opp_alive are the bank's own stamps"}}
    for ep in sorted(errs["all"]):
        blk: dict = {}
        is_pool = oc[bp] == "pool_snapshot"
        blk["class"] = {
            "trained_on_bot__selfplay_vs_bot": au(errs["bot"][ep][bp], is_pool, battles[bp]),
            "trained_on_pool__bot_vs_selfplay": au(errs["pool"][ep][bp], ~is_pool, battles[bp]),
            "reference_trained_on_all__selfplay_vs_bot": au(errs["all"][ep][bp], is_pool, battles[bp])}
        ex = ho & np.isin(oc, ["bot", "exploiter"])
        blk["class"]["trained_on_bot__exploiter_vs_bot"] = au(errs["bot"][ep][ex], oc[ex] == "exploiter", battles[ex])
        e = errs["all"][ep]
        blk["late_game"] = {name: au(e[ho], m[ho], battles[ho]) for name, m in
                            (("late_turn", late), ("both_sides_le2", both_low), ("union", late | both_low))}
        out[str(ep)] = blk
    return out


def pool_membership(bank: Any) -> dict:
    """Is any bank team (either side) OFF the training pool (every team file under ``data/teams``,
    the ``default_biased`` trainee builder's full pool)? And are the untaught-meter teams off it?
    Teams are matched on their (species, moveset) set."""
    from agents.training.untaught_meter import DEFAULT_TEAMS_MANIFEST, load_team_manifest
    from utils.paths import repo_path
    from utils.team_packing import ConstantTeambuilder

    def fp(packed: str) -> frozenset:
        return frozenset((m.split("|")[0].lower(), m.split("|")[4]) for m in packed.split("]") if m)

    pool: Dict[frozenset, str] = {}
    unparsed = []
    root = repo_path("data", "teams")
    for p in sorted(root.rglob("*.txt")):
        try:
            pool[fp(ConstantTeambuilder(p.read_text()).yield_team())] = str(p.relative_to(root))
        except Exception:                                    # noqa: BLE001 - counted, never dropped
            unparsed.append(str(p.relative_to(root)))
    off = {"banked": 0, "opponent": 0}
    for b in bank.battles:
        me, them = (b.p1, b.p2) if b.banked_side == "p1" else (b.p2, b.p1)
        off["banked"] += fp(me["team"]) not in pool
        off["opponent"] += fp(them["team"]) not in pool
    untaught = load_team_manifest(str(DEFAULT_TEAMS_MANIFEST))
    in_pool = sum(Path(t.path).resolve().is_relative_to(root.resolve()) for t in untaught)
    return {"pool_files_parsed": len(pool), "pool_files_unparsed": unparsed, "battles": len(bank.battles),
            "battles_with_off_pool_team": off, "untaught_meter_teams": len(untaught),
            "untaught_meter_teams_inside_data_teams": int(in_pool),
            "verdict": ("NO off-pool states: every bank team on both sides is a training-pool team, and the "
                        "untaught-meter teams are pool files themselves (untaught = not TAUGHT by a teacher, "
                        "not unseen by the trainee)")
            if off["banked"] == 0 and off["opponent"] == 0 and in_pool == len(untaught) else "off-pool teams found"}


# ---------------------------------------------------------------------------------------------
# (4) successors of starved near-best moves vs the played (argmax) move
# ---------------------------------------------------------------------------------------------

def forward_pooled_logits(model: Any, rows: np.ndarray, masks: np.ndarray, batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """``(value_pooled [N, D], logits [N, 11])`` — one extractor forward per minibatch."""
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs

    pol = model.policy
    fe = pol.features_extractor
    pooled, logits = [], []
    for i in range(0, len(rows), batch):
        mb = torch.tensor(np.asarray(masks[i:i + batch]).astype(np.float32))
        ob = {"observation": torch.from_numpy(np.ascontiguousarray(rows[i:i + batch], dtype=np.float32)),
              "action_mask": mb}
        ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
        with torch.no_grad():
            pi_f, _ = pol.extract_features(ob)
            pooled.append(fe.last_value_pooled.numpy().copy())
            lat = pol.mlp_extractor.forward_actor(pi_f)
            logits.append(pol._get_action_dist_from_latent(lat).distribution.logits.numpy().copy())
    return np.concatenate(pooled, 0), np.concatenate(logits, 0)


def starved_turns(bank: Any, truth_rows: Sequence[dict], logits: np.ndarray, masks: np.ndarray) -> List[dict]:
    """N0 final's DECISIVE truth turns, each with its legal actions classified: ``argmax`` (the
    policy's), ``starved_near`` (π < 1 %, truth gap ≤ ε), ``starved_far`` (π < 1 %, gap > ε), ``fed``
    (π ≥ 1 %, not the argmax)."""
    from agents.training.instrumented_ppo.ridealong_terms import STARVED_PI
    from main.policy_spectrum.qhat import greedy_of
    from main.policy_spectrum.spectrum import masked_probs
    from main.ridealong_read.meters import truth_turns

    id_to_row = {d["id"]: i for i, d in enumerate(bank.decisions)}
    battle_of = {d["id"]: d["battle"] for d in bank.decisions}
    pi = masked_probs(logits, masks)
    am = greedy_of(np.concatenate([logits, np.zeros((len(logits), 1), np.float32)], 1), masks)
    out = []
    for T in truth_turns(truth_rows, id_to_row, battle_of):
        if not T.decisive:
            continue
        p = pi[T.row, T.acts]
        cls = {}
        for j, a in enumerate(T.acts.tolist()):
            if a == int(am[T.row]):
                cls[a] = "argmax"
            elif p[j] < STARVED_PI:
                cls[a] = "starved_near" if bool(T.near[j]) else "starved_far"
            else:
                cls[a] = "fed"
        out.append({"id": T.id, "battle": T.battle, "row": T.row, "argmax": int(am[T.row]), "cls": cls})
    return out


def derive_successors(bank: Any, turns: Sequence[dict], model: Any, lib_path: Path, chunk_dir: Path,
                      log: Callable[[str], None] = print) -> None:
    """Every legal action's ONE-PLY successor (S = 1: the truth's first seed), the opponent's open
    root decision and any later opponent-only decision answered by ``model``'s greedy choice (the
    X4 pre-read's variant M). One durable ``.npz`` per chunk of :data:`SUCC_CHUNK` turns; a chunk on
    disk is skipped."""
    from main.policy_spectrum.qhat import QhatError, greedy_of, logits_and_values, one_ply
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    chunk_dir.mkdir(parents=True, exist_ok=True)
    lib = F.load(Path(lib_path))
    dec_by_id = {d["id"]: d for d in bank.decisions}

    def greedy(rows: np.ndarray, masks: np.ndarray, j: np.ndarray) -> np.ndarray:
        return greedy_of(logits_and_values(model, rows, masks), masks)

    t0 = time.time()
    with SearchCore(lib=lib) as core:
        for c0 in range(0, len(turns), SUCC_CHUNK):
            path = chunk_dir / f"chunk_{c0 // SUCC_CHUNK:04d}.npz"
            if path.exists():
                continue
            ids, acts, rows, msk, ended, other_open = [], [], [], [], [], []
            refused: List[dict] = []
            for T in turns[c0:c0 + SUCC_CHUNK]:
                d = dec_by_id[T["id"]]
                b = bank.battles[bank.battle_index[d["battle"]]]
                at = cmd_index(b, d["side"], d["n"])
                try:
                    r = one_ply(core, to_log(b), at, d["side"], turn_seeds(T["id"], 1), greedy)
                except (SuccessorsError, QhatError) as exc:
                    refused.append({"id": T["id"], "error": str(exc)[:300]})
                    continue
                banked = {str(k): v for k, v in sorted((int(a), t) for a, t in d["tokens"].items())}
                if r["tokens"] != banked:
                    refused.append({"id": T["id"], "error": "root tokens differ from the bank's"})
                    continue
                for br in r["branches"]:
                    ids.append(T["id"])
                    acts.append(br["action"])
                    has = br["row"] is not None
                    rows.append(br["row"] if has else np.zeros(core.obs_dim, np.float32))
                    msk.append(br["mask"].astype(bool) if has else np.zeros(11, bool))
                    ended.append(not has)
                    other_open.append(r["other_open"])
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, ids=np.array(ids), acts=np.array(acts, np.int32),
                                rows=np.array(rows, np.float32).reshape(-1, core.obs_dim),
                                masks=np.array(msk, bool).reshape(-1, 11), ended=np.array(ended, bool),
                                other_open=np.array(other_open, bool), refused=np.array(json.dumps(refused)))
            tmp.rename(path)
            log(f"[rnd_states] successors: {min(c0 + SUCC_CHUNK, len(turns))}/{len(turns)} turns "
                f"({time.time() - t0:.0f} s, {len(refused)} refused in this chunk)")


def load_successors(chunk_dir: Path) -> dict:
    parts = [np.load(p) for p in sorted(chunk_dir.glob("chunk_*[0-9].npz"))]
    if not parts:
        raise FileNotFoundError(f"no successor chunks under {chunk_dir}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "acts", "rows", "masks", "ended", "other_open")}
    out["refused"] = [r for p in parts for r in json.loads(str(p["refused"]))]
    return out


def read_starvation(turns: Sequence[dict], succ: Mapping[str, Any], err: np.ndarray,
                    heldout_battles: Optional[set] = None, scale: float = 1e3,
                    log_ratio: bool = True) -> dict:
    """On turns holding ≥ 1 STARVED near-best action whose successor and the argmax's successor are
    both captured (a game-ending branch has no successor row): AUROC(starved-near successors >
    argmax successors), the within-turn paired difference mean err(starved near) − err(argmax)
    (× ``scale``: RND errors are ~1e-3, and the interval rounds to 4 decimals) and its paired LOG
    RATIO (scale-free, comparable across variants; ``log_ratio``), and the share of turns where it is
    positive (battle-clustered intervals); the same against the starved-FAR (π < 1 %, gap > ε) and
    FED (π ≥ 1 %) actions of the same turns as controls, and starved-near − starved-far paired.
    ``heldout_battles`` restricts to turns of those battles."""
    from main.policy_spectrum.qhat_report import Boot

    by: Dict[Tuple[str, int], int] = {}
    for i, (tid, a) in enumerate(zip(succ["ids"].tolist(), succ["acts"].tolist())):
        if not succ["ended"][i]:
            by[(tid, int(a))] = i
    err = np.asarray(err, dtype=np.float64)
    rows: Dict[str, List[dict]] = {"starved_near": [], "starved_far": [], "fed": []}
    n_turns = 0
    ended = {"starved_near": 0, "argmax": 0}
    pairs: List[Tuple[str, float, float]] = []
    for T in turns:
        if heldout_battles is not None and T["battle"] not in heldout_battles:
            continue
        if "starved_near" not in T["cls"].values():
            continue
        n_turns += 1
        grp = {g: [by.get((T["id"], a)) for a, c in T["cls"].items() if c == g] for g in rows}
        ended["starved_near"] += sum(j is None for j in grp["starved_near"])
        ia = by.get((T["id"], T["argmax"]))
        if ia is None:
            ended["argmax"] += 1
            continue
        live = {g: [j for j in js if j is not None] for g, js in grp.items()}
        for g, js in live.items():
            if js:
                rows[g].append({"battle": T["battle"], "pos": [float(err[j]) for j in js], "neg": float(err[ia])})
        if live["starved_near"] and live["starved_far"]:
            a_, b_ = float(np.mean(err[live["starved_near"]])), float(np.mean(err[live["starved_far"]]))
            pairs.append((T["battle"], a_, b_))
    out: dict = {"starved_turns": n_turns, "ended_branches": ended, "paired_diff_scale": scale}
    for g, rr in rows.items():
        if not rr:
            out[f"{g}_vs_argmax"] = {"turns": 0}
            continue
        sc = np.array([v for r in rr for v in r["pos"]] + [r["neg"] for r in rr])
        lab = np.r_[np.ones(sum(len(r["pos"]) for r in rr), bool), np.zeros(len(rr), bool)]
        cl = np.array([r["battle"] for r in rr for _ in r["pos"]] + [r["battle"] for r in rr])
        diffs = np.array([np.mean(r["pos"]) - r["neg"] for r in rr])
        bcl = [r["battle"] for r in rr]
        B = Boot(bcl)
        blk = {"turns": len(rr), "auroc": au(sc, lab, cl), "paired_diff": B.stat(diffs * scale, bcl),
               "share_turns_diff_positive": B.stat((diffs > 0).astype(float), bcl)}
        if log_ratio:
            blk["paired_log_ratio"] = B.stat(np.array([np.log(np.mean(r["pos"]) / r["neg"]) for r in rr]), bcl)
        out[f"{g}_vs_argmax"] = blk
    if pairs:
        B = Boot([p[0] for p in pairs])
        bcl = [p[0] for p in pairs]
        blk = {"turns": len(pairs), "paired_diff": B.stat(np.array([(p[1] - p[2]) * scale for p in pairs]), bcl)}
        if log_ratio:
            blk["paired_log_ratio"] = B.stat(np.array([np.log(p[1] / p[2]) for p in pairs]), bcl)
        out["starved_near_minus_starved_far"] = blk
    return out


def within_turn_share(ids: np.ndarray, score: np.ndarray) -> Optional[float]:
    """The share of ``score``'s variance across successors that lies WITHIN a turn (between the
    actions of one root) rather than between turns. Near 0 = the score mostly tells roots apart."""
    ids, score = np.asarray(ids), np.asarray(score, dtype=np.float64)
    if len(score) < 2 or score.var() == 0:
        return None
    _, inv = np.unique(ids, return_inverse=True)
    mean = np.bincount(inv, weights=score) / np.bincount(inv)
    return _r(float(((score - mean[inv]) ** 2).mean() / score.var()))


# ---------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------

def _write(out: Path, name: str, obj: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{name}.json.tmp"
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.rename(out / f"{name}.json")


def _sha_rows(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def main(argv: Optional[List[str]] = None) -> int:
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, reencode, refuse_under_models
    from main.ridealong_read import reader as RD
    from main.ridealong_read.boot import METHOD as BOOT_METHOD
    from utils.paths import main_models_dir, repo_root

    ap = argparse.ArgumentParser(prog="python -m main.ridealong_read.rnd_states")
    ap.add_argument("--out", required=True, help="the committed JSONs")
    ap.add_argument("--cache", required=True, help="re-encoded rows, N0 forwards, predictor snapshots")
    ap.add_argument("--archive", required=True, help="per-row arrays and successor chunks (never committed)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2, help="re-encode workers")
    ap.add_argument("--reads", default="1,2,3,4")
    ap.add_argument("--lib", default=None, help="the rust env cdylib (default: this checkout's release build)")
    a = ap.parse_args(argv)
    out, cache, arch = Path(a.out), Path(a.cache), Path(a.archive)
    for p in (out, cache, arch):
        refuse_under_models(p)
        p.mkdir(parents=True, exist_ok=True)
    reads = {int(x) for x in a.reads.split(",") if x}
    log = lambda m: print(m, flush=True)                      # noqa: E731

    models = main_models_dir()
    if models is None:
        sys.exit("[rnd_states] no models/ archive on this box")
    ckpt = models / FEATURE_CKPT_REL
    RD.check_checkpoint_arg(ckpt)

    bank = load_bank(RD.bank_dir())
    dec = bank.decisions
    battles = np.array([d["battle"] for d in dec])
    bsha = bank.manifest["content_sha256"]
    bank_npz = cache / f"bank_rows_{bsha[:12]}.npz"
    if bank_npz.exists():
        z = np.load(bank_npz)
        rows, masks, gate = z["rows"], z["masks"], json.loads(str(z["gate"]))
    else:
        t = time.time()
        rows, masks, gate = reencode(bank, workers=a.workers)
        gate = {k: v for k, v in gate.items() if k != "encoder"} | {"encoder_sha256": gate["encoder"]["core_events_sha256"]}
        np.savez(bank_npz, rows=rows, masks=masks, gate=np.array(json.dumps(gate)))
        log(f"[rnd_states] re-encoded {len(rows)} rows in {time.time() - t:.0f} s")
    if not gate["obs_as_recorded"]:
        sys.exit(f"[rnd_states] REFUSED: the re-encoding is not the recording ({gate})")

    fwd_npz = cache / f"n0final_fwd_{bsha[:12]}.npz"
    model = None
    if not fwd_npz.exists():
        with inference_globals(a.threads):
            model = load_checkpoint(ckpt)
            pooled, logits = forward_pooled_logits(model, rows, masks)
        np.savez(fwd_npz, pooled=pooled, logits=logits)
        log("[rnd_states] N0 final forward cached")
    z = np.load(fwd_npz)
    pooled, logits = z["pooled"], z["logits"]
    X = {"obs": rows, "feat": pooled}

    train = split_within_cells(dec)
    oc = np.array([d["opp_class"] for d in dec])
    errs: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {v: {} for v in VARIANTS}
    snaps: Dict[str, Dict[str, dict]] = {v: {} for v in VARIANTS}
    import torch

    with inference_globals(a.threads):
        for v in VARIANTS:
            for kind, cls in KINDS.items():
                pth = cache / f"pred_{v}_{kind}_{bsha[:12]}.pt"
                if pth.exists():
                    s = torch.load(pth, weights_only=False)
                else:
                    t = time.time()
                    m = train if cls is None else train & (oc == cls)
                    s = train_snapshots(X[v][m])
                    torch.save(s, pth)
                    log(f"[rnd_states] trained {v}-RND on {kind} ({s['n_train']} rows, {time.time() - t:.0f} s)")
                snaps[v][kind] = s
                errs[v][kind] = {ep: score(sd, s["in_dim"], X[v]) for ep, sd in s["snap"].items()}
    np.savez_compressed(arch / "bank_rnd_errors.npz", ids_sha256=np.array(RD.ids_sha(bank)), train=train,
                        **{f"{v}_{k}_{ep}": e for v in VARIANTS for k in KINDS for ep, e in errs[v][k].items()})

    meta = {"schema": SCHEMA, "reader_commit": _commit(), "bootstrap": BOOT_METHOD,
            "bank": {"content_sha256": bsha, "decisions": len(dec), "battles": len(bank.battles)},
            "reencode": gate, "feature_checkpoint": {"path": str(ckpt), "sha256": RD.file_sha256(ckpt)},
            "split": {"rule": f"battles {TRAIN_FRAC:.0%} / {1 - TRAIN_FRAC:.0%} WITHIN each (source, opp_class) "
                              f"cell, seed {SPLIT_SEED}", "train_rows": int(train.sum()),
                      "heldout_rows": int((~train).sum()),
                      "train_battles": len(set(battles[train].tolist())),
                      "heldout_battles": len(set(battles[~train].tolist()))},
            "training": {v: {k: {"n_train": snaps[v][k]["n_train"], "train_loss": snaps[v][k]["train_loss"]}
                             for k in KINDS} for v in VARIANTS},
            "predictor": "agents.model.ridealong_heads.RndNovelty shapes; Adam 3e-4, minibatch 256, seed "
                         "20260930; normalisation fitted on the predictor's own training rows",
            "epochs_read": list(EPOCHS_READ), "headline_epochs": HEADLINE_EPOCHS}

    if 1 in reads and not (out / "read1_seen_vs_heldout.json").exists():
        _write(out, "read1_seen_vs_heldout", dict(meta, **{v: read_seen_vs_heldout(errs[v]["all"], train, battles)
                                                           for v in VARIANTS}))
        log("[rnd_states] read (1) written")
    L = state_layout()
    ds = decode_state(rows, L)
    keys = state_keys(ds)
    count = train_counts(keys, train)
    turn_arr = np.array([d["turn"] for d in dec])
    if 2 in reads and not (out / "read2_visitation.json").exists():
        np.savez_compressed(arch / "state_keys.npz", keys=keys, count=count)
        body = dict(meta, layout=L.__dict__, decode_check=decode_check(ds, dec),
                    key={"fields": ["our active species", "opp active species",
                                    f"turn bucket (upper edges {list(TURN_EDGES)})",
                                    f"our-active HP bucket (0 = fainted, edges {list(HP_EDGES)})",
                                    "opp-active HP bucket", "weather on", "our spikes", "their spikes"],
                         "distinct_train": int(len(set(keys[train].tolist()))),
                         "distinct_all": int(len(set(keys.tolist())))},
                    **{v: read_visitation(errs[v]["all"], count, train, battles, turn=turn_arr) for v in VARIANTS})
        _write(out, "read2_visitation", body)
        log("[rnd_states] read (2) written")
    if 3 in reads and not (out / "read3_off_distribution.json").exists():
        body = dict(meta, off_pool=pool_membership(bank),
                    **{v: read_off_distribution(errs[v], train, dec, battles) for v in VARIANTS})
        _write(out, "read3_off_distribution", body)
        log("[rnd_states] read (3) written")
    if 4 in reads and not (out / "read4_starvation.json").exists():
        from main.policy_spectrum.truth import load_rows

        tfile, owner = RD.TRUTH_CONTINUATIONS[TRUTH_LABEL]
        if owner != FEATURE_CKPT_REL:
            sys.exit(f"[rnd_states] the {TRUTH_LABEL} truth belongs to {owner}, not {FEATURE_CKPT_REL}")
        turns = starved_turns(bank, load_rows(RD.truth_dir() / tfile), logits, masks)
        chunk_dir = arch / "successors_N0final_S1"
        lib = Path(a.lib) if a.lib else repo_root() / "src" / "rust_env" / "target" / "release" / "libpokesim_env.so"
        with inference_globals(a.threads):
            if model is None:
                model = load_checkpoint(ckpt)
            derive_successors(bank, turns, model, lib, chunk_dir, log=log)
            succ = load_successors(chunk_dir)
            if succ["rows"].shape[1] != rows.shape[1]:
                sys.exit(f"[rnd_states] successor rows are {succ['rows'].shape[1]}-dim, the bank's {rows.shape[1]}")
            sp_npz = cache / f"succ_pooled_{_sha_rows(succ['rows'])}.npz"
            if sp_npz.exists():
                sp = np.load(sp_npz)["pooled"]
            else:
                ok = ~succ["ended"]
                sp = np.zeros((len(succ["rows"]), pooled.shape[1]), np.float32)
                sp[ok] = forward_pooled_logits(model, succ["rows"][ok], succ["masks"][ok])[0]
                np.savez(sp_npz, pooled=sp)
            SX = {"obs": succ["rows"], "feat": sp}
            serr = {v: {ep: score(sd, snaps[v]["all"]["in_dim"], SX[v]) for ep, sd in snaps[v]["all"]["snap"].items()}
                    for v in VARIANTS}
        ho_b = set(battles[~train].tolist())
        cls_n: Dict[str, int] = {}
        for T in turns:
            for c in T["cls"].values():
                cls_n[c] = cls_n.get(c, 0) + 1
        body = dict(meta, truth={"continuation": TRUTH_LABEL, "file": tfile,
                                 "decisive_turns": len(turns),
                                 "turns_with_starved_near": sum("starved_near" in T["cls"].values() for T in turns),
                                 "actions_by_class": cls_n},
                    successors={"branches": int(len(succ["ids"])), "ended": int(succ["ended"].sum()),
                                "root_opponent_open": int(len({i for i, o in zip(succ["ids"].tolist(), succ["other_open"]) if o})),
                                "refused_turns": succ["refused"],
                                "rule": "one_ply per decisive turn, every legal action, S = 1 (truth seed 0), the "
                                        "opponent's open root + later opponent-only decisions answered by N0 final's "
                                        "greedy (variant M); scored by the predictors trained on every TRAIN row"})
        ok = ~succ["ended"]
        tc: Dict[str, int] = {}
        for k in keys[train].tolist():
            tc[k] = tc.get(k, 0) + 1
        skeys = np.full(len(ok), "", dtype=object)
        skeys[ok] = state_keys(decode_state(succ["rows"][ok], L))
        scount = np.array([tc.get(k, 0) for k in skeys.tolist()])
        rarity = -np.log1p(scount.astype(np.float64))
        by_cls: Dict[str, List[int]] = {}
        cls_of = {T["id"]: T["cls"] for T in turns}
        for i, (tid, act) in enumerate(zip(succ["ids"].tolist(), succ["acts"].tolist())):
            if ok[i]:
                by_cls.setdefault(cls_of[tid].get(int(act), "?"), []).append(int(scount[i]))
        body["key_count_reference"] = {
            "rule": "the read-(2) STATE KEY of each successor, counted among the TRAIN rows; score = "
                    "-log(1 + count) (higher = rarer); paired_diff in log(1 + count) units, negated",
            "count0_share_by_class": {c: _r(float(np.mean(np.array(v) == 0))) for c, v in sorted(by_cls.items())},
            "median_count_by_class": {c: float(np.median(v)) for c, v in sorted(by_cls.items())},
            "all_turns": read_starvation(turns, succ, rarity, scale=1.0, log_ratio=False),
            "heldout_battle_turns": read_starvation(turns, succ, rarity, heldout_battles=ho_b, scale=1.0,
                                                    log_ratio=False)}
        for v in VARIANTS:
            body[v] = {}
            for ep, e in serr[v].items():
                eb = errs[v]["all"][ep]
                body[v][str(ep)] = {
                    "within_turn_variance_share": within_turn_share(succ["ids"][ok], e[ok]),
                    "all_turns": read_starvation(turns, succ, e),
                    "heldout_battle_turns": read_starvation(turns, succ, e, heldout_battles=ho_b),
                    "reference_mean_err": {"bank_train_rows": _r(eb[train].mean(), 6),
                                           "bank_heldout_rows": _r(eb[~train].mean(), 6),
                                           "successors_all": _r(e[~succ["ended"]].mean(), 6)}}
        _write(out, "read4_starvation", body)
        np.savez_compressed(arch / "successor_rnd_errors.npz",
                            **{f"{v}_{ep}": e for v in VARIANTS for ep, e in serr[v].items()})
        log("[rnd_states] read (4) written")
    log("[rnd_states] done")
    return 0


def _commit() -> str:
    from main.ridealong_read.__main__ import _commit as c, _dirty

    return c() + ("+dirty" if _dirty() else "")


if __name__ == "__main__":
    sys.exit(main())
