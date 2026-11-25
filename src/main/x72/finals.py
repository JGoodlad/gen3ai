"""X72's FINALS: the archive copies every cross and read plays (PREREG §3), and the banked round-0 finals.

After every round the chain copies each lineage's ``final_model.zip``, ``final_model.magnet.pt``,
``model_config.json`` and ``metadata.json`` to ``<archive>/x72_swap_plateau/finals/r<k>/rb_x72_swapplat_s734N/``
(``<archive>`` = ``utils.paths.gen3ai_archive_dir()``) and records their sha256 in ``finals.json`` beside them: the
next round overwrites the run dir's final, so nothing of X72 ever plays a run dir's live file.

* :func:`archive` is that copy step. It hashes each source file before and after the copy and the copy itself (a
  file that moved under the copy is a refusal), never overwrites a different existing copy, and is a no-op on a
  round already archived with the same bytes. The PREREG names ``finals.json`` and not its fields; the schema
  (:data:`SCHEMA`) is this module's, and :func:`load_final` reads nothing else.
* :func:`load_final` is the ONLY way a tool takes a final. Round 0 is the banked swap final of the lineage, in the
  run archive, at its REGISTERED sha256 (:data:`SWAP`, pinned by test to ``read_spec_kakuna.json``). Round k >= 1 is
  the archive copy: every file re-hashed against ``finals.json``, then :func:`record_problems`.
* :func:`record_problems` is the part of PREREG §11's VOID list a COPY'S OWN RECORD can show: the training pin on
  every ``pin_history`` entry, the lever (``ent_coef`` 0.0 in ``metadata.json`` and its ``cli_args``,
  ``magnet_kl_coef`` 0.05 with ``magnet_refresh_steps`` 2,000,000 in ``model_config.json``, the magnet sidecar), the
  right parent (``lineage.fork_parent.sha256`` = the lineage's banked swap final), the right round (the step is at or
  past the round's target and less than one round past it). What only the RUN DIR shows (the exit code, the newest
  checkpoint, K9(b), the launcher log's pin, the pool-seed check, ``main.dose``, ``magnet/refreshes``) stays the
  chain's own check after each launch (PREREG §16 step 5).

    python -m main.x72.finals archive --round K     # the chain's copy step, after round K's three launches pass §11
    python -m main.x72.finals verify --round K      # re-hash and re-judge the round's three copies; exit 3 on a problem

STDLIB ONLY (it sits under the cross's plan, which must not open a CUDA context).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.hashing import sha256_file
from utils.paths import gen3ai_archive_dir, main_models_dir

PREREG = "designs/research_state/measurements/x72_swap_plateau_2026-11-25/PREREG.md"
#: the three lineages' seeds, in lineage order [PREREG §3]
SEEDS = (7341, 7342, 7343)
RUN_FMT = "rb_x72_swapplat_s{seed}"
#: lineage seed -> (the banked swap run it forks, the REGISTERED sha256 of that run's final_model.zip)
#: [read_spec_kakuna.json models.s0_s732N.ref]
SWAP: Dict[int, tuple] = {
    7341: ("rb_x35_ce3dswap_s7321", "c068bb1a0df29afe3ebdaba4e15452c42df76020f02e1d48871ba68216100cb7"),
    7342: ("rb_x35_ce3dswap_s7322", "f1306f8dc3c62a4660db1fccc1e2ab9c96a08bc7895304160cb989329fc8aaff"),
    7343: ("rb_x35_ce3dswap_s7323", "7eee05c7cc06502284f8373001d47975c0f2c2cd0430e224a2b590ecf7c647dd"),
}
#: the training pin of every launch of every round [PREREG §3]
TRAIN_PIN = "9e97cad5709da21a1f9d4d8846096c87eb17b1d0"
#: round 1's start = the banked swap forks' target; a round's target is this + 20M a round [PREREG §3's table]
ROUND0_TARGET = 300_000_170
ROUND_STEPS = 20_000_000
MAX_ROUNDS = 4
#: the four files a copy holds [PREREG §3]
FILES = ("final_model.zip", "final_model.magnet.pt", "model_config.json", "metadata.json")
MANIFEST = "finals.json"
SCHEMA = "x72_finals_v1"
#: the lever as the record must show it [PREREG §11]
ENT_COEF = 0.0
MAGNET_KL_COEF = 0.05
MAGNET_REFRESH_STEPS = 2_000_000


class FinalsError(ValueError):
    """A final that is missing, moved, or not what the registration says it is: a refusal, never a guess."""


@dataclass(frozen=True)
class Final:
    round: int
    seed: int
    run: str          # the run the file is a final of (round 0: the banked swap run)
    path: str         # the final_model.zip a cross or read plays
    sha256: str
    step: int
    source: str       # "banked" (round 0, the run archive) or "archive" (an archive copy)


def run_name(seed: int) -> str:
    if seed not in SEEDS:
        raise FinalsError(f"seed {seed} is not an X72 lineage ({list(SEEDS)})")
    return RUN_FMT.format(seed=seed)


def check_round(k: int, *, lo: int = 0) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not lo <= k <= MAX_ROUNDS:
        raise FinalsError(f"round {k!r} is not one of {lo}..{MAX_ROUNDS}")
    return k


def target(k: int) -> int:
    """The registered target step of round ``k`` (round 0: the banked swap forks' own target)."""
    return ROUND0_TARGET + check_round(k) * ROUND_STEPS


def archive_root() -> Path:
    return gen3ai_archive_dir() / "x72_swap_plateau" / "finals"


def copy_dir(k: int, seed: int, root: Optional[Path] = None) -> Path:
    return Path(root or archive_root()) / f"r{check_round(k, lo=1)}" / run_name(seed)


def _models(models_dir: Optional[Path]) -> Path:
    m = Path(models_dir) if models_dir is not None else main_models_dir()
    if m is None or not Path(m).is_dir():
        raise FinalsError("no run archive (models/) to read from: set $GEN3AI_MODELS_DIR or pass --models-dir")
    return Path(m)


def _json(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalsError(f"{path}: unreadable ({exc})") from exc
    if not isinstance(doc, dict):
        raise FinalsError(f"{path}: not a JSON object")
    return doc


def _step(md: Dict[str, Any], where: Path) -> int:
    s = md.get("num_timesteps")
    if isinstance(s, bool) or not isinstance(s, int):
        raise FinalsError(f"{where}: metadata.json holds no integer num_timesteps ({s!r})")
    return s


def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def record_problems(d: Path, k: int, seed: int) -> List[str]:
    """Why the four files in ``d`` are NOT a round-``k`` final of lineage ``seed`` by their own record (module
    docstring; PREREG §11). ``[]`` = they are."""
    check_round(k, lo=1)
    run, d = run_name(seed), Path(d)
    missing = [f for f in FILES if not (d / f).is_file() or (d / f).stat().st_size == 0]
    if missing:
        return [f"{run} r{k}: missing or empty {', '.join(missing)} in {d}"]
    try:
        md, cfg = _json(d / "metadata.json"), _json(d / "model_config.json")
        step = _step(md, d)
    except FinalsError as exc:
        return [str(exc)]
    probs: List[str] = []
    lo = target(k)
    if not lo <= step < lo + ROUND_STEPS:
        probs.append(f"{run} r{k}: final at {step:,}, not in [{lo:,}, {lo + ROUND_STEPS:,}) (the round's target and "
                     f"one round past it)")
    hist = md.get("pin_history")
    pins = [e.get("git_hash") if isinstance(e, dict) else None for e in hist] if isinstance(hist, list) else []
    if not pins or any(p != TRAIN_PIN for p in pins):
        probs.append(f"{run} r{k}: pin_history names {pins or 'nothing'}, not only the training pin {TRAIN_PIN[:8]}")
    cli = _dict(md.get("cli_args"))
    for name, got in (("metadata.json ent_coef", md.get("ent_coef")), ("cli_args.ent_coef", cli.get("ent_coef"))):
        if isinstance(got, bool) or not isinstance(got, (int, float)) or float(got) != ENT_COEF:
            probs.append(f"{run} r{k}: {name} is {got!r}, not {ENT_COEF} (half of the lever)")
    if cfg.get("magnet_kl_coef") != MAGNET_KL_COEF or cfg.get("magnet_refresh_steps") != MAGNET_REFRESH_STEPS:
        probs.append(f"{run} r{k}: model_config.json magnet_kl_coef {cfg.get('magnet_kl_coef')!r} / "
                     f"magnet_refresh_steps {cfg.get('magnet_refresh_steps')!r}, not {MAGNET_KL_COEF} / "
                     f"{MAGNET_REFRESH_STEPS:,} (half of the lever)")
    parent = _dict(_dict(md.get("lineage")).get("fork_parent"))
    if parent.get("sha256") != SWAP[seed][1]:
        probs.append(f"{run} r{k}: lineage.fork_parent.sha256 is {str(parent.get('sha256'))[:12]}, not the banked "
                     f"final of {SWAP[seed][0]} ({SWAP[seed][1][:12]}): the wrong parent")
    for name, want in (("seed", seed), ("run_name", run)):
        if cli.get(name) is not None and cli.get(name) != want:
            probs.append(f"{run} r{k}: cli_args.{name} is {cli.get(name)!r}, not {want!r}")
    return probs


def _hashes(d: Path) -> Dict[str, Dict[str, Any]]:
    return {f: {"sha256": sha256_file(d / f), "bytes": (d / f).stat().st_size} for f in FILES}


def load_final(k: int, seed: int, *, root: Optional[Path] = None, models_dir: Optional[Path] = None) -> Final:
    """The final of lineage ``seed`` after round ``k``, VERIFIED (module docstring), or :class:`FinalsError`."""
    check_round(k)
    run = run_name(seed)
    if k == 0:
        swap_run, want = SWAP[seed]
        d = _models(models_dir) / swap_run
        z = d / "final_model.zip"
        if not z.is_file():
            raise FinalsError(f"{swap_run}: no final_model.zip under {d.parent}")
        got = sha256_file(z)
        if got != want:
            raise FinalsError(f"{swap_run}: final_model.zip has sha256 {got[:12]}, not the registered {want[:12]} "
                              f"(the banked file moved)")
        return Final(0, seed, swap_run, str(z), got, _step(_json(d / "metadata.json"), d), "banked")
    d = copy_dir(k, seed, root)
    if not (d / MANIFEST).is_file():
        raise FinalsError(f"{run} r{k}: no {MANIFEST} in {d} (the round is not archived: "
                          f"python -m main.x72.finals archive --round {k})")
    man = _json(d / MANIFEST)
    ident = {"schema": SCHEMA, "round": k, "seed": seed, "run": run}
    off = {f: man.get(f) for f, v in ident.items() if man.get(f) != v}
    if off:
        raise FinalsError(f"{d / MANIFEST}: names {off}, not {ident}")
    missing = [f for f in FILES if not (d / f).is_file()]
    if missing:
        raise FinalsError(f"{run} r{k}: {', '.join(missing)} missing from {d}")
    files = _dict(man.get("files"))
    now = _hashes(d)
    moved = [f for f in FILES if not isinstance(files.get(f), dict) or files[f].get("sha256") != now[f]["sha256"]]
    if moved:
        raise FinalsError(f"{run} r{k}: {', '.join(moved)} in {d} no longer match(es) {MANIFEST}'s sha256")
    probs = record_problems(d, k, seed)
    step = _step(_json(d / "metadata.json"), d)
    if man.get("step") != step:
        probs.append(f"{run} r{k}: {MANIFEST} records step {man.get('step')!r}, metadata.json {step:,}")
    if probs:
        raise FinalsError("; ".join(probs))
    return Final(k, seed, run, str(d / "final_model.zip"), now["final_model.zip"]["sha256"], step, "archive")


def load_round(k: int, *, root: Optional[Path] = None, models_dir: Optional[Path] = None) -> List[Final]:
    """The three finals of round ``k`` in lineage order; every problem of every lineage in ONE refusal."""
    out, probs = [], []
    for seed in SEEDS:
        try:
            out.append(load_final(k, seed, root=root, models_dir=models_dir))
        except FinalsError as exc:
            probs.append(str(exc))
    if probs:
        raise FinalsError("; ".join(probs))
    return out


def _copy_one(src: Path, dst: Path, want: str) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(src, tmp)
        got = sha256_file(tmp)
        if got != want or sha256_file(src) != want:
            raise FinalsError(f"{src}: changed while it was copied (sha256 {want[:12]} before, {got[:12]} in the "
                              f"copy): is the run still writing? nothing was archived for this file")
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def archive(k: int, *, root: Optional[Path] = None, models_dir: Optional[Path] = None,
            now: Optional[_dt.datetime] = None) -> List[Final]:
    """Copy round ``k``'s three finals from their run dirs to the archive and record their sha256 (PREREG §3; §16
    step 6). The run dirs are only read. Every lineage's source is judged (:func:`record_problems`) BEFORE any file
    is copied; a copy that exists with other bytes is a refusal, one with the same bytes is left as it is."""
    check_round(k, lo=1)
    models = _models(models_dir)
    plans, probs = [], []
    for seed in SEEDS:
        src, dst = models / run_name(seed), copy_dir(k, seed, root)
        p = record_problems(src, k, seed)
        if p:
            probs += p
            continue
        before = _hashes(src)
        if (dst / MANIFEST).is_file():
            had = _json(dst / MANIFEST).get("files")
            if had != before:
                probs.append(f"{run_name(seed)} r{k}: {dst} already holds a copy with OTHER bytes than {src} "
                             f"(an archive copy is never overwritten)")
                continue
        plans.append((seed, src, dst, before))
    if probs:
        raise FinalsError("; ".join(probs))
    stamp = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat(timespec="seconds")
    for seed, src, dst, before in plans:
        if (dst / MANIFEST).is_file():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            _copy_one(src / f, dst / f, before[f]["sha256"])
        man = {"schema": SCHEMA, "round": k, "seed": seed, "run": run_name(seed),
               "step": _step(_json(dst / "metadata.json"), dst), "source_dir": str(src), "archived_at": stamp,
               "files": before}
        tmp = dst / f".{MANIFEST}.tmp-{os.getpid()}"
        tmp.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, dst / MANIFEST)
    return load_round(k, root=root, models_dir=models)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x72.finals", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("archive", "copy round K's three finals to the archive and record their sha256"),
                       ("verify", "re-hash and re-judge round K's three finals (0 = the banked swap finals)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--round", type=int, required=True)
        p.add_argument("--archive-root", default=None, help="default: <gen3ai archive>/x72_swap_plateau/finals")
        p.add_argument("--models-dir", default=None, help="default: the run archive (models/)")
    a = ap.parse_args(argv)
    root = Path(a.archive_root) if a.archive_root else None
    models = Path(a.models_dir) if a.models_dir else None
    try:
        got = (archive if a.cmd == "archive" else load_round)(a.round, root=root, models_dir=models)
    except FinalsError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(json.dumps([asdict(f) for f in got], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
