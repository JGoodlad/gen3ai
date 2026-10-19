"""``python -m main.team_strength`` — the TEAM-STRENGTH RATING meter (TSR). See the package docs.

    python -m main.team_strength register --spec SPEC.json --out REG.json        # freeze every input (no game)
    python -m main.team_strength install-kakuna --reg REG.json --state DIR       # Metamon sets + per-set configs
    python -m main.team_strength run --reg REG.json --state DIR --lane x26|kakuna [--stop-at ISO] [--tree PIN]
    python -m main.team_strength status --reg REG.json --state DIR
    python -m main.team_strength fit --reg REG.json --state DIR --out RESULTS.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from main.anchors.config import load_config
from main.team_strength import fit as F
from main.team_strength import kakuna_lane as KL
from main.team_strength import panel as PN
from main.team_strength import plan as PLN
from main.team_strength import register as RG
from main.team_strength import report as RP
from main.team_strength import x26_lane as XL

def default_metamon_teams() -> Path:
    """Metamon's team cache, from the one anchors config (``designs/ops/anchors.json`` / ``GEN3AI_METAMON_CACHE_DIR``)."""
    return load_config().metamon.cache_dir / "teams"


def expected_digest(paths: Sequence[Path]) -> str:
    hs = sorted(hashlib.sha256(Path(p).read_text().strip().encode()).hexdigest() for p in paths)
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def kakuna_digests(reg: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"ref": digest, "<set>": digest}`` of the files each Metamon set must hold (from the registered sources)."""
    out = {"ref": expected_digest(list(KL.ref_files(reg).values()))}
    for s in reg["pilots"]["kakuna"]["subset_sets"]:
        out[s] = expected_digest(list(KL.subset_files(reg, s).values()))
    return out


def config_path(state: Path, set_name: str) -> Path:
    return Path(state) / "kakuna" / "configs" / f"anchors_tsr_{set_name}.json"


def cmd_register(a: argparse.Namespace) -> int:
    spec = json.loads(Path(a.spec).read_text())
    reg = RG.build(spec)
    out = Path(a.out)
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("digest") != reg["digest"]:
            raise SystemExit(f"{out} exists with another digest ({old.get('digest', '')[:12]}): a registration is "
                             "frozen — write a new one")
        print(f"[tsr] {out} already registered (digest {reg['digest'][:12]})")
        return 0
    out.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    units = XL.x26_units(reg)
    print(f"[tsr] registered {out}: digest {reg['digest']}; reference {len(reg['reference']['members'])} teams; "
          f"pilot A plan {json.dumps(PLN.summary(units))}")
    return 0


def cmd_install(a: argparse.Namespace) -> int:
    reg = RG.load(Path(a.reg))
    root = Path(a.metamon_teams)
    got = KL.install(reg, root)
    want = kakuna_digests(reg)
    k = reg["pilots"]["kakuna"]
    if got[k["ref_set"]] != want["ref"] or any(got[k["subset_sets"][s]] != want[s] for s in k["subset_sets"]):
        raise SystemExit(f"[tsr] installed digests {got} differ from the registered sources {want}")
    template = json.loads(RG.resolve_dir(k["config_template"]).read_text())
    for s in k["subset_sets"]:
        p = config_path(Path(a.state), s)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(KL.config_for(reg, template, s, root), indent=1) + "\n"
        if p.exists() and p.read_text() != text:
            raise SystemExit(f"{p} exists with other content")
        p.write_text(text)
    print(f"[tsr] Metamon sets installed: {json.dumps(got, indent=1)}")
    return 0


def _port_free(port: int) -> bool:
    """Can a server bind ``port``? SO_REUSEADDR, as a server sets it: the previous unit's closed connections linger in
    TIME_WAIT for up to a minute, and a bind without it failed right after the first unit (2026-10-18)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_port_free(port: int, timeout_s: float = 120.0, poll_s: float = 5.0) -> bool:
    """True once ``port`` is bindable, False if it is still busy after ``timeout_s`` (the previous unit's server may
    take a few seconds to exit)."""
    t_end = time.monotonic() + timeout_s
    while True:
        if _port_free(port):
            return True
        if time.monotonic() >= t_end:
            return False
        time.sleep(poll_s)


def kakuna_plan(reg: Mapping[str, Any]) -> List[Any]:
    k = reg["pilots"]["kakuna"]
    return PLN.kakuna_units(sorted(k["subset_sets"]), int(k["units_per_set"]), int(k["seed"]))


def run_kakuna(reg: Mapping[str, Any], state: Path, tree: Path, py: str, stop_at: Optional[float],
               teams_root: Path) -> Dict[str, Any]:
    from main.anchors.session import team_set_digest

    k = reg["pilots"]["kakuna"]
    port = int(k["port"])
    if not 9500 <= port <= 9599:
        raise SystemExit(f"port {port} is outside the anchors' 9500-9599")
    want = kakuna_digests(reg)
    fails: Dict[str, int] = {}
    for unit, s, seed in kakuna_plan(reg):
        while not KL.is_done(state, unit) and fails.get(unit, 0) < 2:
            if stop_at is not None and time.time() > stop_at:
                return {"stopped": "stop_at", "fails": fails}
            XL.check_halt()
            PN.load_registered(reg)
            if team_set_digest(KL.metamon_set_dir(teams_root, k["ref_set"])) != want["ref"] or \
                    team_set_digest(KL.metamon_set_dir(teams_root, k["subset_sets"][s])) != want[s]:
                raise SystemExit(f"[tsr] a Metamon set moved since install ({s})")
            if not wait_port_free(port):
                raise SystemExit(f"[tsr] port {port} is busy — never another process's server")
            KL.run_unit(py=py, tree=tree, state=state, unit=unit, config=config_path(state, s), seed=seed,
                        port=port)
            if not KL.is_done(state, unit):
                fails[unit] = fails.get(unit, 0) + 1
    return {"stopped": None, "fails": fails}


def cmd_run(a: argparse.Namespace) -> int:
    reg = RG.load(Path(a.reg))
    state = Path(a.state)
    stop_at = dt.datetime.fromisoformat(a.stop_at).timestamp() if a.stop_at else None
    if a.lane == "x26":
        from main.h2h.play import Compute

        c = dict(reg["pilots"]["x26"].get("compute") or {})       # the REGISTERED compute block, when there is one
        comp = Compute(device=c.get("device", "cpu"), backend=c.get("backend", "eager"),
                       n_envs=int(c.get("n_envs", a.n_envs)), threads=int(c.get("threads", a.threads)),
                       torch_threads=int(c.get("torch_threads", a.torch_threads)), front=c.get("front", a.front),
                       profile=c.get("profile", a.profile))
        res = XL.run(reg, state, compute=comp, stop_at=stop_at, max_units=a.max_units)
    else:
        from utils.paths import repo_root

        tree = Path(a.tree) if a.tree else repo_root()
        res = run_kakuna(reg, state, tree, a.python, stop_at, Path(a.metamon_teams))
    done = state / f"lane_{a.lane}.done"
    done.write_text(json.dumps({"t": time.time(), **res}) + "\n")
    print(f"[tsr] lane {a.lane}: {res}")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    reg = RG.load(Path(a.reg))
    state = Path(a.state)
    xu = XL.x26_units(reg)
    xd = sum((state / "x26" / "units" / f"{u.name}.json").exists() for u in xu)
    ku = kakuna_plan(reg)
    kd = sum(KL.is_done(state, u) for u, _, _ in ku)
    print(f"x26: {xd}/{len(xu)} units; kakuna: {kd}/{len(ku)} units")
    return 0


#: How pilot-B games are CLUSTERED for the bootstrap. ``matchup`` (the default): the games of one unit with the same
#: (side A team, side B team) are ONE cluster — the anchors' two role halves draw the SAME team sequence from one
#: ``--team-seed`` (measured on the first TSR unit, 2026-10-18: game ``10 + i`` replays game ``i``'s two teams with the
#: seats swapped on another battle seed), so they are a seat-swapped pair, not two independent draws. ``game``: each
#: game its own cluster (PREREG §3 as written, before that was seen).
KAKUNA_CLUSTERS = ("matchup", "game")


def load_kakuna(reg: Mapping[str, Any], state: Path, showdown_sim: Path, cluster: str = "matchup") -> Dict[str, Any]:
    """Every COUNTING pilot-B unit's games (``F.Game``, from side A = the subset team) and every unit's verdict."""
    from main.anchors.results import unit_counts

    if cluster not in KAKUNA_CLUSTERS:
        raise ValueError(f"cluster {cluster!r} not in {KAKUNA_CLUSTERS}")
    if "kakuna" not in reg["pilots"]:
        return {"games": [], "verdicts": [], "self_games": 0}
    k = reg["pilots"]["kakuna"]
    want = kakuna_digests(reg)
    sets = PN.load_registered(reg)
    texts = PN.text_of(sets)
    ref_ids = {m["team_id"]: m["team_id"] for m in reg["reference"]["members"]}
    games, verdicts, self_games = [], [], 0
    for unit, s, _seed in kakuna_plan(reg):
        udir = KL.unit_dir(state, unit)
        sp = udir / "summary.json"
        if not sp.exists():
            continue
        summ = json.loads(sp.read_text())
        uc = unit_counts(summ, games=KL.GAMES, declared_digests=(want[s], want["ref"]))
        reasons = list(uc.reasons)
        side_a = {t: t for t in k["subsets"][s]}
        rows, why = KL.parse_unit(udir, side_a=side_a, side_b=ref_ids, texts=texts, showdown_sim=showdown_sim)
        reasons += why
        verdicts.append({"unit": unit, "set": s, "counts": not reasons, "reasons": reasons})
        if reasons:
            continue
        for r in rows:
            if r["a"] == r["b"]:
                self_games += 1
                continue
            cl = f"{unit}:{r['a']}:{r['b']}" if cluster == "matchup" else f"{unit}:{r['tag']}"
            games.append(F.Game(a=r["a"], b=r["b"], s=float(r["s"]), owner=r["a"], cluster=cl))
    return {"games": games, "verdicts": verdicts, "self_games": self_games}


def compute(reg: Mapping[str, Any], state: Path, kakuna_cluster: str = "matchup") -> Any:
    """``(results, {pilot: Rating})`` — both pilots' fits, set means, comparisons and agreement (PREREG §3-§4)."""
    from utils.paths import repo_path

    fz = reg["fit"]
    refs = [m["team_id"] for m in reg["reference"]["members"]]
    out: Dict[str, Any] = {"schema": RP.SCHEMA, "registration_digest": reg["digest"], "pilots": {}}
    ratings: Dict[str, F.Rating] = {}
    xg = XL.load_games(reg, state)
    if xg:
        ratings["x26"] = F.rate(xg, prior_sd=float(fz["prior_sd"]), anchor=refs, B=int(fz["B"]), seed=int(fz["seed"]))
    kk = load_kakuna(reg, state, repo_path("deps", "pokemon-showdown", "dist", "sim"), cluster=kakuna_cluster)
    out["kakuna_cluster"] = kakuna_cluster
    if kk["games"]:
        ratings["kakuna"] = F.rate(kk["games"], prior_sd=float(fz["prior_sd"]), anchor=refs, B=int(fz["B"]),
                                   seed=int(fz["seed"]) + 1)
    out["kakuna_units"] = kk["verdicts"]
    out["kakuna_self_games_dropped"] = kk["self_games"]
    margin = float(fz["competitive_margin_elo"])
    for pilot, rt in ratings.items():
        if pilot == "x26":
            members = {s: sorted(set(b["team_ids"].values())) for s, b in reg["sets"].items()}
        else:
            members = {s: sorted(set(v) & set(rt.ids)) for s, v in reg["pilots"]["kakuna"]["subsets"].items()}
        members = {s: v for s, v in members.items() if v}
        means = F.set_means(rt, members, seed=int(fz["seed"]) + 10)
        blk: Dict[str, Any] = {"n_games": int(sum(rt.games.values()) // 2), "sets": {
            s: {k2: v for k2, v in means[s].items()} for s in members}}
        comps = {}
        for x, y in fz["comparisons"]:
            if x in members and y in members:
                d = F.diff(means, x, y)
                comps[f"{x}-{y}"] = {**d, **RP.competitive_class(d["ci"], margin)}
        blk["comparisons"] = comps
        blk["teams"] = RP.rating_table(rt, reg["membership"])
        out["pilots"][pilot] = blk
    if "x26" in ratings and "kakuna" in ratings:
        common = sorted(set(ratings["kakuna"].ids) - set(refs))
        common = [t for t in common if t in ratings["x26"].ids]
        ag = F.agreement(ratings["x26"], ratings["kakuna"], common, seed=int(fz["seed"]) + 20)
        ag["class"] = RP.agreement_class(ag["ci"])
        out["agreement"] = ag
    return out, ratings


def cmd_fit(a: argparse.Namespace) -> int:
    reg = RG.load(Path(a.reg))
    out, ratings = compute(reg, Path(a.state))
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    if a.ratings_npz:
        import numpy as np

        np.savez_compressed(a.ratings_npz, **{f"{p}_reps": rt.reps for p, rt in ratings.items()},
                            **{f"{p}_r": rt.r for p, rt in ratings.items()},
                            **{f"{p}_ids": np.array(rt.ids) for p, rt in ratings.items()})
    print(f"[tsr] wrote {a.out}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.team_strength", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    i = sub.add_parser("install-kakuna")
    i.add_argument("--reg", required=True)
    i.add_argument("--state", required=True)
    i.add_argument("--metamon-teams", default=str(default_metamon_teams()))
    u = sub.add_parser("run")
    u.add_argument("--reg", required=True)
    u.add_argument("--state", required=True)
    u.add_argument("--lane", required=True, choices=("x26", "kakuna"))
    u.add_argument("--stop-at", default=None, help="ISO time: start no unit after it")
    u.add_argument("--max-units", type=int, default=None)
    # the compute flags apply only when the registration carries no `pilots.x26.compute` block
    u.add_argument("--n-envs", type=int, default=48)
    u.add_argument("--threads", type=int, default=3)
    u.add_argument("--torch-threads", type=int, default=3)
    u.add_argument("--front", default="ffi", choices=("proc", "ffi"))
    u.add_argument("--profile", default="release", choices=("release", "selfcheck"))
    u.add_argument("--tree", default=None, help="kakuna: the pinned checkout ROOT every unit runs from")
    u.add_argument("--python", default=sys.executable)
    u.add_argument("--metamon-teams", default=str(default_metamon_teams()))
    s = sub.add_parser("status")
    s.add_argument("--reg", required=True)
    s.add_argument("--state", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--reg", required=True)
    f.add_argument("--state", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--ratings-npz", default=None)
    return p


def run(argv: Optional[Sequence[str]] = None) -> int:
    a = parser().parse_args(argv)
    return {"register": cmd_register, "install-kakuna": cmd_install, "run": cmd_run, "status": cmd_status,
            "fit": cmd_fit}[a.cmd](a)
