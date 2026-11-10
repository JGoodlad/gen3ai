"""ONE checkpoint on a bank: CPU forwards, twice per row — the checkpoint's own belief (λ = 0) and the Smogon prior
alone (λ = 1 of ``agents.model.belief_prior_mix``, bit-identical to the three deltas zeroed in the weights) — then
:mod:`.metrics` per unit, summed per panel TEAM.

Both forwards read the same rows, so the prior column is the prior THIS checkpoint falls back to (its own tables),
on exactly the decisions its posterior is scored on. A λ column is the two columns blended in closed form
(:mod:`.metrics`): what a FLOOR of λ on the prior does to this belief. For the species leg that is exactly the model
run at λ; for moves and items it is not (their heads read tokens downstream of the mixed species presence, so the
model run at λ has a different delta), and the column never claims to be.

The observation is sliced by the extractor's own layout (never a typed index), and a bank whose observation width
is not the checkpoint's is refused. A checkpoint trained under an oracle reveal is refused (the bank's rows carry
no reveal). A checkpoint recorded at an older config than the bank's rows were encoded at is scored and STAMPED
(``obs_gap``): every observation-semantics row between the two is a small input shift the checkpoint never trained
on, the same badge every other read at head carries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.offpool_belief import SCORE_SCHEMA, OffPoolBeliefError
from main.offpool_belief import metrics as M
from main.offpool_belief import stats as S
from main.offpool_belief.bank import Bank

CHUNK = 64


def load_checkpoint(zip_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Inference load on the CPU through the foreign-opponent loader (architecture signature checked, historical
    kwargs sanitised). Returns ``(model, {"path", "sha256", "config_version"})``. Sets no torch global: call it, and
    every forward, inside ``utils.torch_state_guard.torch_globals``."""
    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    from utils.hashing import sha256_file

    p = Path(zip_path)
    if p.is_dir() or p.suffix != ".zip":
        raise OffPoolBeliefError(f"{p}: name the checkpoint .zip (a bare run directory resolves to a file that moves)")
    refuse_if_revealed(str(p), tool="main.offpool_belief", reason="The bank's rows are encoded without the reveal.")
    model, version = load_foreign_opponent(str(p), current_model_version(load_mappings()), device="cpu")
    model.policy.set_training_mode(False)
    fe = model.policy.features_extractor
    for leg in ("hypothesis_builder", "move_belief", "item_belief_head"):
        if getattr(fe, leg, None) is None:
            raise OffPoolBeliefError(f"{p}: the extractor has no {leg}; the meter reads all three belief legs")
    return model, {"path": str(p), "sha256": sha256_file(p), "config_version": _recorded_config_version(p)}


def _recorded_config_version(zip_path: Path) -> Optional[int]:
    """The ``config_version`` the checkpoint's run recorded (``model_config.json`` beside the zip or one level up)."""
    for d in (zip_path.parent, zip_path.parent.parent):
        f = d / "model_config.json"
        if f.is_file():
            try:
                v = json.loads(f.read_text()).get("config_version")
            except ValueError:
                return None
            return int(v) if v is not None else None
    return None


def obs_gap(recorded: Optional[int], bank_obs_semantics: int) -> Dict[str, Any]:
    """The observation-semantics rows between a checkpoint's recorded config and the bank's encoding."""
    from agents.model.model_version import constants as MV

    if recorded is None:
        return {"recorded_config_version": None, "bank_obs_semantics_version": int(bank_obs_semantics),
                "rows_between": None, "note": "the checkpoint records no config version: the gap is unknown"}
    rows = [int(r[0]) for r in MV.OBS_SEMANTICS_HISTORY if int(recorded) < int(r[0]) <= int(bank_obs_semantics)]
    return {"recorded_config_version": int(recorded), "bank_obs_semantics_version": int(bank_obs_semantics),
            "rows_between": rows}


@dataclass(frozen=True)
class ObsIndex:
    """Where the opponent block's cells sit, read from the extractor's layout."""

    opp_start: int
    slot_dim: int
    n_slots: int
    species: int
    move_ids: Tuple[int, ...]
    move_known: Tuple[int, ...]
    item_known: int
    total_dim: int


def obs_index(layout: Mapping[str, Any]) -> ObsIndex:
    part = layout["parts"]["opp_team"]
    n_slots, slot_dim = (int(x) for x in part["reshape"])
    pk = layout["pokemon"]
    sp = pk["species"]["offset"] + pk["species"]["layout"]["species_id"]["offset"]
    mv = pk["moves"]
    sl = mv["layout"]["slot_layout"]
    ids = tuple(int(mv["offset"] + s["offset"] + sl["id"]["offset"]) for s in mv["layout"]["slots"])
    known = tuple(int(mv["offset"] + s["offset"] + sl["known"]["offset"]) for s in mv["layout"]["slots"])
    it = pk["items"]
    return ObsIndex(opp_start=int(part["start"]), slot_dim=slot_dim, n_slots=n_slots, species=int(sp), move_ids=ids,
                    move_known=known, item_known=int(it["offset"] + it["layout"]["known"]["offset"]),
                    total_dim=int(layout["total_dim"]))


def _forward(model: Any, obs: np.ndarray, mask: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    """One no-grad forward at belief-prior-mix ``lam`` (0 = the posterior, 1 = the prior); the three legs' outputs."""
    import torch as th

    from agents.model import belief_prior_mix as BPM
    from agents.model.extra_obs_keys import zero_extra_obs

    fe = model.policy.features_extractor
    mb = th.as_tensor(mask, dtype=th.float32)
    ob = {"observation": th.as_tensor(obs, dtype=th.float32), "action_mask": mb}
    ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
    BPM.set_belief_prior_mix(model, lam)
    try:
        with th.no_grad():
            model.policy.get_distribution(ob)
            sp = fe.last_hypothesis.species
            return {"species_log_pi": sp.log_pi.double().numpy().copy(),
                    "species_cand": (sp.cand & (sp.live | sp.full).unsqueeze(-1)).numpy().copy(),
                    "move_logits": fe.last_move_belief_logits.double().numpy().copy(),
                    "item_logp": th.log_softmax(fe.last_item_logits.double(), dim=-1).numpy().copy()}
    finally:
        BPM.set_belief_prior_mix(model, 0.0)


class TeamSums:
    """Per-team sums of every metric and variant, and per-team unit and exclusion counts."""

    def __init__(self, n_teams: int, variants: Sequence[str]):
        self.n_teams = n_teams
        self.variants = tuple(variants)
        self.sums = {m: {v: np.zeros(n_teams) for v in self.variants} for m in M.METRICS}
        self.units = {f: np.zeros(n_teams, dtype=np.int64) for f in M.FAMILIES}
        self.excluded: Dict[str, np.ndarray] = {}

    def add(self, family: str, units: M.Units, team_of_row: np.ndarray) -> None:
        teams = team_of_row[units.row]
        np.add.at(self.units[family], teams, 1)
        for metric, by in units.values.items():
            for v, x in by.items():
                np.add.at(self.sums[metric][v], teams, x)
        for reason, rows in units.excluded.items():
            col = self.excluded.setdefault(reason, np.zeros(self.n_teams, dtype=np.int64))
            np.add.at(col, team_of_row[rows], 1)


def score_rows(model: Any, rows: Mapping[str, np.ndarray], n_teams: int, *, mixes: Sequence[float] = (),
               chunk: int = CHUNK, log: Any = None) -> TeamSums:
    """Score every row of ``rows`` (a bank's arrays) and return the per-team sums."""
    from agents.model.hypothesis_set import TYPED_HP_NUMS
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

    fe = model.policy.features_extractor
    ix = obs_index(fe.pokemon_encoder.layout)
    obs, mask = rows["obs"], rows["mask"]
    if obs.shape[1] != ix.total_dim:
        raise OffPoolBeliefError(f"the bank's observation is {obs.shape[1]} wide, the checkpoint reads "
                                 f"{ix.total_dim}: rebuild the bank at the checkpoint's observation")
    hp_family = frozenset((int(HIDDEN_POWER_MOVE_NUM), *(int(x) for x in TYPED_HP_NUMS)))
    legal_table = (fe.move_belief.move_prior_probs > 2.0 * _illegal_prob()).numpy()       # [S, M] learnable moves
    variants = [M.POST, M.PRIOR] + [M.mix_name(x) for x in mixes]
    out = TeamSums(n_teams, variants)
    n = len(obs)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        post, prior = _forward(model, obs[sl], mask[sl], 0.0), _forward(model, obs[sl], mask[sl], 1.0)
        if not np.array_equal(post["species_cand"], prior["species_cand"]):
            raise OffPoolBeliefError("the posterior and the prior forward disagree on the species candidate set")
        opp = obs[sl][:, ix.opp_start:ix.opp_start + ix.n_slots * ix.slot_dim].reshape(-1, ix.n_slots, ix.slot_dim)
        species = np.rint(opp[:, :, ix.species]).astype(np.int64)
        shown_ids = np.rint(opp[:, :, list(ix.move_ids)]).astype(np.int64)
        shown_flag = opp[:, :, list(ix.move_known)]
        sp = {M.POST: post["species_log_pi"], M.PRIOR: prior["species_log_pi"]}
        mv = {M.POST: post["move_logits"], M.PRIOR: prior["move_logits"]}
        it = {M.POST: post["item_logp"], M.PRIOR: prior["item_logp"]}
        for x in mixes:
            name = M.mix_name(x)
            sp[name] = M.mix_log_prob(sp[M.POST], sp[M.PRIOR], x)
            mv[name] = M.mix_bernoulli_logits(mv[M.POST], mv[M.PRIOR], x)
            it[name] = M.mix_log_prob(it[M.POST], it[M.PRIOR], x)
        team = rows["team"][sl]
        out.add("species", M.species_units(sp, post["species_cand"], rows["belief_species"][sl]), team)
        legal = legal_table[np.clip(species, 0, legal_table.shape[0] - 1)]
        out.add("moves", M.move_units(mv, legal, rows["known_moves"][sl], shown_ids, shown_flag, species,
                                      hp_family), team)
        out.add("items", M.item_units(it, rows["item_label"][sl], rows["item_mask"][sl], opp[:, :, ix.item_known]),
                team)
        if log is not None and (a // chunk) % 20 == 0:
            log(f"  rows {sl.stop}/{n}")
    return out


def _illegal_prob() -> float:
    from agents.model import belief_tables as BT

    return float(BT._ILLEGAL_PROB)


def summarize(ts: TeamSums) -> Dict[str, Any]:
    """Per metric: every variant's team-clustered mean and interval, and the paired ``variant − prior`` gaps."""
    idx = S.boot_index(ts.n_teams)
    out: Dict[str, Any] = {}
    for metric in M.METRICS:
        n = ts.units[M.FAMILY_OF[metric]]
        tm = {v: S.team_means(ts.sums[metric][v], n) for v in ts.variants}
        cell: Dict[str, Any] = {v: S.mean_ci(tm[v], idx) for v in ts.variants}
        for v in ts.variants:
            if v != M.PRIOR:
                cell[f"{v}_minus_prior"] = S.paired_diff(tm[v], tm[M.PRIOR], idx)
        cell["units"] = int(n.sum())
        out[metric] = cell
    return out


def panel_block(bank: Bank, ts: TeamSums) -> Dict[str, Any]:
    m = bank.manifest
    return {
        "panel": bank.panel, "kind": "OFF-POOL" if m["panel_held_out"] else "IN-POOL reference",
        "held_out": bool(m["panel_held_out"]), "panel_content_digest": m["panel_content_digest"],
        "bank": {"path": str(bank.path), "rows_sha256": m["rows_sha256"], "n_rows": m["n_rows"],
                 "n_games": m["n_games"], "games_per_team": m["games_per_team"], "seed": m["seed"],
                 "pilot_sha256": m["pilot"]["sha256"], "code": m["code"]},
        "team_labels": list(m["team_labels"]), "variants": list(ts.variants),
        "units": {f: ts.units[f].tolist() for f in M.FAMILIES},
        "sums": {metric: {v: ts.sums[metric][v].tolist() for v in ts.variants} for metric in M.METRICS},
        "excluded": {k: v.tolist() for k, v in sorted(ts.excluded.items())},
        "summary": summarize(ts),
    }


def score_checkpoint(zip_path: str, label: str, banks: Sequence[Bank], *, mixes: Sequence[float] = (),
                     threads: int = 1, log: Any = print) -> Dict[str, Any]:
    """Score one checkpoint on every bank; the returned document is the score file (:data:`SCORE_SCHEMA`)."""
    from main.offpool_belief.bank import code_stamp
    from utils.torch_state_guard import torch_globals

    if not banks:
        raise OffPoolBeliefError("no bank to score on")
    panels: Dict[str, Any] = {}
    with torch_globals(num_threads=int(threads)):
        model, ckpt = load_checkpoint(zip_path)
        for bank in banks:
            if bank.panel in panels:
                raise OffPoolBeliefError(f"two banks of panel {bank.panel!r}: one score file holds one bank a panel")
            log(f"[score {label}] {bank.panel}: {bank.manifest['n_rows']} rows")
            ts = score_rows(model, bank.rows, bank.n_teams, mixes=mixes, log=log)
            block = panel_block(bank, ts)
            block["obs_gap"] = obs_gap(ckpt["config_version"], bank.manifest["code"]["obs_semantics_version"])
            panels[bank.panel] = block
    return {"schema": SCORE_SCHEMA, "label": label, "checkpoint": ckpt, "mixes": [float(x) for x in mixes],
            "code": code_stamp(), "torch_threads": int(threads),
            "bootstrap": {"draws": S.DEFAULT_DRAWS, "seed": S.DEFAULT_SEED, "unit": "team"},
            "panels": panels}


def load_score(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise OffPoolBeliefError(f"{path} is not a score file: {e}") from e
    if doc.get("schema") != SCORE_SCHEMA:
        raise OffPoolBeliefError(f"{path}: schema {doc.get('schema')!r}, this meter reads {SCORE_SCHEMA}")
    return doc


def team_metric(block: Mapping[str, Any], metric: str, variant: str) -> np.ndarray:
    """One score-file panel block's per-team mean of ``metric`` for ``variant`` (NaN = no unit)."""
    return S.team_means(block["sums"][metric][variant], block["units"][M.FAMILY_OF[metric]])


def arm_team_metric(blocks: Sequence[Mapping[str, Any]], metric: str, variant: str) -> np.ndarray:
    """An ARM's per-team value: the mean over its checkpoints of each checkpoint's team mean. Every checkpoint must
    be scored on the same bank (rows sha256), and a team must have units in all of them or in none."""
    if not blocks:
        raise OffPoolBeliefError("an arm needs at least one score")
    if len({b["bank"]["rows_sha256"] for b in blocks}) != 1:
        raise OffPoolBeliefError("an arm's checkpoints were scored on different banks: their rows differ")
    cols = np.stack([team_metric(b, metric, variant) for b in blocks])
    return cols.mean(axis=0)


