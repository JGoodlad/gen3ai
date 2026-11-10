"""THE ROW BANK: the fixed decision rows every checkpoint is scored on.

One pilot checkpoint plays GREEDY against itself on the Rust env core (one core thread, envs answered in index
order). Seat 1 pilots the panel's team ``g mod n`` in game ``g``; seat 2 pilots a ``home72`` team. The bank keeps seat
2's decision rows — the seat that FACES the panel team — with the core's privileged labels.

**A game is a pure function of ``(seed, g)``** (:func:`game_schedule`): its ``home72`` team and its battle seed do
not depend on which env runs it or on when another game ends, so the same call rebuilds the same bank, and
``games_per_team`` games land on EVERY panel team (the cluster the statistics resample). Envs that outlive the
schedule play filler games whose rows are dropped.

A bank directory holds ``rows.npz`` and ``manifest.json``; the manifest names everything a number depends on (the
panel's content digest and the registry's sha256, the pilot's sha256, the schedule, the commit and observation
semantics the rows were encoded at, the rows file's sha256). :func:`load_bank` re-checks the rows file and the
panel against the LIVE registry before a row is read.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.offpool_belief import BANK_SCHEMA, HOME_PANEL, OffPoolBeliefError
from main.offpool_belief import panels as PN

ROWS_FILE = "rows.npz"
MANIFEST_FILE = "manifest.json"
DEFAULT_GAMES_PER_TEAM = 2
DEFAULT_SEED = 20261109
DEFAULT_ENVS = 24
#: The stall-forfeit threshold of the bank's games (the frozen P5 probe's).
TURN_LIMIT = 250
#: The row arrays a bank holds, in file order.
ROW_KEYS = ("obs", "mask", "game", "team", "home", "turn", "winner", "belief_species", "belief_moves",
            "known_moves", "item_label", "item_mask")
_LABEL_KEYS = ("belief_species", "belief_moves", "known_moves", "item_label", "item_mask")
#: The seat whose rows the bank keeps: seat 2 (index 1), which FACES the panel team seat 1 pilots.
FACING_SIDE = 1


@dataclass(frozen=True)
class Game:
    g: int                       # the game's index, 0 .. n_games - 1
    team: int                    # the panel team seat 1 pilots
    home: int                    # the home72 team seat 2 pilots
    seed: Tuple[int, int, int, int]


def game_schedule(n_teams: int, n_home: int, games_per_team: int, seed: int, *, same_panel: bool) -> List[Game]:
    """The bank's games: ``games_per_team`` passes over the panel in order, each game's home team and battle seed
    drawn from its OWN generator ``default_rng([seed, g])``. ``same_panel`` (the panel IS home72): a draw equal to
    the game's own team moves to the next team, so no game is a mirror."""
    if n_teams < 1 or n_home < 1 or games_per_team < 1:
        raise ValueError("a bank needs at least one panel team, one home team and one game per team")
    if same_panel and n_home < 2:
        raise ValueError("a same-panel bank needs at least two teams (no mirror games)")
    out = []
    for g in range(n_teams * games_per_team):
        rng = np.random.default_rng([int(seed), g])
        i = g % n_teams
        j = int(rng.integers(0, n_home))
        if same_panel and j == i:
            j = (j + 1) % n_home
        s = tuple(int(x) for x in rng.integers(0, 65536, 4))
        out.append(Game(g=g, team=i, home=j, seed=(s[0], s[1], s[2], s[3])))
    return out


Act = Callable[[np.ndarray, np.ndarray], np.ndarray]


def greedy_actor(model: Any) -> Act:
    """``(obs [B,D], mask [B,A] bool) -> actions [B]``: the pilot's argmax over its masked policy, no gradient."""
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs

    fe = model.policy.features_extractor

    def act(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mb = th.as_tensor(mask, dtype=th.float32)
        ob = {"observation": th.as_tensor(obs, dtype=th.float32), "action_mask": mb}
        ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
        with th.no_grad():
            logits = model.policy.get_distribution(ob).distribution.logits
            return (logits + (mb - 1.0) * 1e9).argmax(dim=-1).numpy()
    return act


def play(packed_panel: Sequence[str], packed_home: Sequence[str], schedule: Sequence[Game], act: Act, *,
         lib: Any, n_envs: int = DEFAULT_ENVS, turn_limit: Optional[int] = TURN_LIMIT,
         progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, np.ndarray]:
    """Play ``schedule`` and return the facing seat's rows (:data:`ROW_KEYS`), ordered by game then decision.

    ``packed_panel`` / ``packed_home`` are packed teams; ``act`` answers both seats. A quarantined battle raises: the
    bank is every scheduled game or nothing."""
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P

    n_panel = len(packed_panel)
    n_envs = max(1, min(int(n_envs), len(schedule)))
    spec = P.spec_json(n=n_envs, threads=1, teams=list(packed_panel) + list(packed_home), names=("opba", "opbb"),
                       turn_limit=turn_limit, refusal_budget=4, bank_dir=None, labels=("belief", "item"),
                       opponents=[{"kind": "external"}])
    core = ffi.FfiCore(spec, lib=lib)
    c = core.cols
    nxt = [0]
    staged: Dict[int, Optional[Game]] = {}

    def stage(envs: Sequence[int]) -> None:
        for e in envs:
            real = nxt[0] < len(schedule)
            gm = schedule[nxt[0]] if real else schedule[nxt[0] % len(schedule)]   # a filler replays a real spec
            c["ep_team"][e] = [gm.team, n_panel + gm.home]
            c["ep_seed"][e] = list(gm.seed)
            c["ep_opp"][e] = 0
            staged[e] = gm if real else None
            nxt[0] += 1

    per_game: Dict[int, Dict[str, List[Any]]] = {}
    buf: Dict[int, List[Dict[str, Any]]] = {e: [] for e in range(n_envs)}
    done = 0
    try:
        stage(range(n_envs))
        cur = dict(staged)                       # the episodes RESET starts
        core.reset()
        stage(range(n_envs))                     # staged one ahead: what each env's auto-reset starts next
        while done < len(schedule):
            pend = [(e, s) for e in range(n_envs) for s in (0, 1) if c["need"][e, s]]
            if pend:
                obs = np.stack([np.array(c["obs"][e, s], dtype=np.float32) for e, s in pend])
                mask = np.stack([np.array(c["mask"][e, s]).astype(bool) for e, s in pend])
                acts = np.asarray(act(obs, mask))
                for k, (e, s) in enumerate(pend):
                    a = int(acts[k])
                    if not mask[k][a]:
                        raise OffPoolBeliefError(f"the pilot chose an illegal action {a} (env {e}, seat {s + 1})")
                    if s == FACING_SIDE and cur[e] is not None:
                        row = {"obs": obs[k], "mask": mask[k], "turn": int(c["turn"][e])}
                        row.update({key: np.array(c[key][e, s]) for key in _LABEL_KEYS})
                        buf[e].append(row)
                    c["action"][e, s] = a
            core.step()
            fin = core.finished()
            for e in range(n_envs):
                if c["refused"][e]:
                    raise OffPoolBeliefError(f"env {e}: a battle was quarantined (refused); a bank is every "
                                             "scheduled game or nothing")
            for f in fin:
                e = int(f["env"])
                rows, buf[e] = buf[e], []
                gm = cur[e]
                if gm is not None:
                    per_game[gm.g] = {"rows": rows, "winner": [int(f["winner"])]}
                    done += 1
                    if progress is not None:
                        progress(done, len(schedule))
                cur[e] = staged[e]               # the env auto-reset into its staged episode
            if fin:
                stage([int(f["env"]) for f in fin])
    finally:
        core.close()
    out: Dict[str, List[Any]] = {k: [] for k in ROW_KEYS}
    for gm in schedule:
        got = per_game[gm.g]
        for row in got["rows"]:
            for key in ("obs", "mask", "turn", *_LABEL_KEYS):
                out[key].append(row[key])
            out["game"].append(gm.g)
            out["team"].append(gm.team)
            out["home"].append(gm.home)
            out["winner"].append(got["winner"][0])
    if not out["obs"]:
        raise OffPoolBeliefError("the bank's games produced no decision row for the facing seat")
    return {"obs": np.asarray(out["obs"], dtype=np.float32), "mask": np.asarray(out["mask"], dtype=bool),
            "game": np.asarray(out["game"], dtype=np.int32), "team": np.asarray(out["team"], dtype=np.int32),
            "home": np.asarray(out["home"], dtype=np.int32), "turn": np.asarray(out["turn"], dtype=np.int32),
            "winner": np.asarray(out["winner"], dtype=np.int8),
            "belief_species": np.asarray(out["belief_species"], dtype=np.int32),
            "belief_moves": np.asarray(out["belief_moves"], dtype=np.int32),
            "known_moves": np.asarray(out["known_moves"], dtype=np.int32),
            "item_label": np.asarray(out["item_label"], dtype=np.int32),
            "item_mask": np.asarray(out["item_mask"], dtype=np.float32)}


def pack(texts: Sequence[str]) -> List[str]:
    from utils.teambuilder import Gen3Teambuilder

    packed = list(Gen3Teambuilder(list(texts)).packed_teams)
    if len(packed) != len(texts):
        raise OffPoolBeliefError(f"{len(texts) - len(packed)} of {len(texts)} teams failed to pack")
    return packed


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return str(get_git_hash())
    except Exception:                                    # noqa: BLE001 - a bank still names what it can
        return "unknown"


def code_stamp() -> Dict[str, Any]:
    from agents.model.model_version import constants as MV

    return {"commit": _commit(), "model_config_version": int(MV.MODEL_CONFIG_VERSION),
            "obs_semantics_version": int(MV.OBS_SEMANTICS_VERSION), "arch_signature": str(MV.ARCH_SIGNATURE)}


def build(panel: str, pilot_zip: str, out_dir: Path, *, pilot_name: Optional[str] = None,
          games_per_team: int = DEFAULT_GAMES_PER_TEAM, seed: int = DEFAULT_SEED, n_envs: int = DEFAULT_ENVS,
          threads: int = 1, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Build the bank of ``panel`` under ``out_dir`` (created; an existing bank there is refused) and return its
    manifest. The Rust env core is built incrementally first, so a stale library is never played."""
    from main.offpool_belief.score import load_checkpoint
    from utils.hashing import sha256_file
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built
    from utils.torch_state_guard import torch_globals

    out_dir = Path(out_dir)
    PN.refuse_under_models(out_dir, "the bank directory")
    PN.refuse_inside_repo(out_dir, "the bank directory")
    if (out_dir / MANIFEST_FILE).exists() or (out_dir / ROWS_FILE).exists():
        raise OffPoolBeliefError(f"{out_dir} already holds a bank: a bank is never overwritten, name a new directory")
    ref = PN.panel_ref(panel)
    home = PN.panel_ref(HOME_PANEL)
    schedule = game_schedule(len(ref.teams), len(home.teams), games_per_team, seed, same_panel=panel == HOME_PANEL)
    _child, lib_path = ensure_built("release", emit=log)
    t0 = time.time()
    with torch_globals(num_threads=int(threads)):
        model, ckpt = load_checkpoint(pilot_zip)

        def progress(done: int, total: int) -> None:
            if done % 24 == 0 or done == total:
                log(f"[bank {panel}] {done}/{total} games, {time.time() - t0:.0f} s")
        rows = play(pack([t for _l, t in ref.teams]), pack([t for _l, t in home.teams]), schedule,
                    greedy_actor(model), lib=ffi.load(str(lib_path)), n_envs=n_envs, progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / ROWS_FILE, **{k: rows[k] for k in ROW_KEYS})
    games = len(schedule)
    manifest = {
        "schema": BANK_SCHEMA, "panel": panel, "panel_content_digest": ref.content_digest,
        "panel_held_out": ref.held_out, "held_out_material": True,
        "use": "scoring only: this bank states a registered panel's teams in the clear and is never a training input",
        "registry_sha256": PN.registry_sha256(), "team_labels": [l for l, _t in ref.teams],
        "home_panel": HOME_PANEL, "home_content_digest": home.content_digest,
        "pilot": {"name": pilot_name, **ckpt}, "policy": "greedy (argmax of the masked policy), both seats",
        "facing_side": FACING_SIDE + 1, "games_per_team": int(games_per_team), "seed": int(seed),
        "n_envs": int(n_envs), "torch_threads": int(threads), "turn_limit": TURN_LIMIT, "n_games": games,
        "n_rows": int(len(rows["obs"])),
        "obs_dim": int(rows["obs"].shape[1]),
        "p1_win_rate": float(np.mean([w == 1 for w in _per_game(rows, "winner")])),
        "rows_sha256": sha256_file(out_dir / ROWS_FILE), "env_core_sha256": sha256_file(lib_path),
        "code": code_stamp(), "built_s": round(time.time() - t0, 1),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"[bank {panel}] {games} games, {manifest['n_rows']} rows -> {out_dir}")
    return manifest


def _per_game(rows: Dict[str, np.ndarray], key: str) -> List[int]:
    _g, first = np.unique(rows["game"], return_index=True)
    return [int(rows[key][i]) for i in first]


@dataclass
class Bank:
    manifest: Dict[str, Any]
    rows: Dict[str, np.ndarray]
    path: Path

    @property
    def panel(self) -> str:
        return str(self.manifest["panel"])

    @property
    def n_teams(self) -> int:
        return len(self.manifest["team_labels"])


def load_bank(path: Path, *, verify_registry: bool = True) -> Bank:
    """Read a bank, refusing one whose rows file or panel moved since the manifest was written. With
    ``verify_registry`` the panel is re-resolved against the LIVE registry (sealed / retired / digest)."""
    from utils.hashing import sha256_file

    path = Path(path)
    try:
        manifest = json.loads((path / MANIFEST_FILE).read_text())
    except (OSError, ValueError) as e:
        raise OffPoolBeliefError(f"{path} is not a bank (no readable {MANIFEST_FILE}): {e}") from e
    if manifest.get("schema") != BANK_SCHEMA:
        raise OffPoolBeliefError(f"{path}: bank schema {manifest.get('schema')!r}, this meter reads {BANK_SCHEMA}")
    got = sha256_file(path / ROWS_FILE)
    if got != manifest["rows_sha256"]:
        raise OffPoolBeliefError(f"{path}: {ROWS_FILE} sha256 {got[:12]} is not the manifest's "
                                 f"{manifest['rows_sha256'][:12]}")
    if verify_registry:
        ref = PN.panel_ref(manifest["panel"])
        if ref.content_digest != manifest["panel_content_digest"] or ref.held_out != manifest["panel_held_out"]:
            raise OffPoolBeliefError(f"{path}: panel {manifest['panel']!r} is no longer what the bank recorded "
                                     "(content digest or held-out status changed in the registry)")
        if [l for l, _t in ref.teams] != manifest["team_labels"]:
            raise OffPoolBeliefError(f"{path}: panel {manifest['panel']!r} lists its teams in another order now")
    with np.load(path / ROWS_FILE) as z:
        rows = {k: z[k] for k in ROW_KEYS}
    n = len(rows["obs"])
    if n != manifest["n_rows"] or any(len(v) != n for v in rows.values()):
        raise OffPoolBeliefError(f"{path}: the row arrays do not all hold the manifest's {manifest['n_rows']} rows")
    return Bank(manifest=manifest, rows=rows, path=path)
