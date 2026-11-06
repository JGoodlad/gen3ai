r"""B3, THE EXPORT AT THE PLAY PIN (PREREG §3): the picked student as P\*'s checkpoint with the student's policy
weights, checked bit for bit at the commit that will play it.

    python -m main.x65.export write --fold-dir <dir>/s_h --init <P* zip> --pin-root <worktree at 0c76b521e>
    python -m main.x65.export smoke --fold-dir <dir>/s_h --pin-root <worktree at 0c76b521e> --teamset <read panel>

``write``:

* the zip is P\*'s zip with ``policy.pth`` replaced by the picked checkpoint's policy state dict
  (``main.x50_r1.hom.export``'s form: ``data``, ``policy.optimizer.pth`` and ``pytorch_variables.pth`` are P\*'s,
  unchanged), beside a copy of P\*'s run's ``model_config.json`` (the loader's lookup) and of P\*'s checkpoint
  sidecar with an ``x65_student`` block naming the student;
* THE TRAINER'S LOGITS: P\* loaded at this checkout (``policy_spectrum.reader.load_checkpoint``, the trainer's own
  load) with the picked state dict, ``main.x50_r1.hom.policy_logits_t`` on the 64-row compile-parity fixture
  (``agents.model.compile_parity_fixture``), CPU, fp32, one thread;
* THE PIN'S LOGITS: a child at the play pin (``PYTHONPATH=<pin-root>/src``) loads the written zip by the BARE loader
  the play path uses (``main.anchors.core_side.load_policy(..., "bare")``) and computes the same function on the
  pin's own fixture (its bytes must be this checkout's), plus the play path's masked action logits
  (``policy.get_distribution``, ``CorePolicy.decide``'s arithmetic). Both must equal this checkout's, bit for bit
  (``uint32`` views), and the pin's loader must report ``bare``: anything else is VOID (PREREG §10);
* ``export.json`` records the zip's sha256, the picked checkpoint, the fixture's sha256 and the pin's commit.

``smoke`` plays ONE game of the student against a local Kakuna peer at the pin (one ``main.anchors`` per-game unit,
``main.x35.kakuna.unit_argv``'s argv, unit seed :data:`SMOKE_SEED`, outside every registered deal range), under the
heavy slots; it records only that the game finished. It is the last step of B3 and runs only after a student exists.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from main.x65 import student as ST

SCHEMA = "gen3_x65_export_v1"
PLAY_PIN = "0c76b521e"
SMOKE_SEED = 900
THREADS = 1

#: the child at the pin: bare load, the trainer's function and the play path's, on the pin's fixture
PIN_CHILD = r"""
import hashlib, json, subprocess, sys
import numpy as np
import torch as th
zp, out = sys.argv[1], sys.argv[2]
th.set_num_threads(1)
from agents.model.compile_parity_fixture import FIXTURE_PATH
from main.anchors.core_side import OurSideState, load_policy
from main.x50_r1.hom import policy_logits_t
from utils.paths import repo_path
st = OurSideState()
pol = load_policy(zp, "cpu", "bare", st).policy
pol.eval()
z = np.load(FIXTURE_PATH)
obs = np.asarray(z["obs"], np.float32)
mask = np.asarray(z["action_mask"], bool)
fx = hashlib.sha256(open(FIXTURE_PATH, "rb").read()).hexdigest()
with th.no_grad():
    x = th.from_numpy(obs)
    raw = policy_logits_t(pol, x, th.from_numpy(mask.astype(np.float32))).float().numpy()
    mi = th.from_numpy(mask.astype(np.int8))
    d = pol.get_distribution({"observation": x, "action_mask": mi})
    play = (d.distribution.logits + (mi - 1.0) * 1e9).float().numpy()
np.savez(out, raw=raw, play=play)
head = subprocess.run(["git", "-C", str(repo_path()), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
print(json.dumps({"loader": st.model_loader, "fixture_sha256": fx, "commit": head.strip()}))
"""


class ExportError(RuntimeError):
    """An export that does not reproduce the trained student (VOID, PREREG §10)."""


def write_zip(init: Path, state_dict: dict[str, Any], dst: Path) -> None:
    """P\\*'s zip with ``policy.pth`` replaced (every other member byte for byte P\\*'s)."""
    import torch as th

    buf = io.BytesIO()
    th.save(state_dict, buf)
    tmp = dst.with_suffix(".tmp")
    with zipfile.ZipFile(init) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        names = [i.filename for i in zi.infolist()]
        if names.count("policy.pth") != 1:
            raise ExportError(f"{init}: {names.count('policy.pth')} policy.pth members")
        for info in zi.infolist():
            zo.writestr(info, buf.getvalue() if info.filename == "policy.pth" else zi.read(info.filename))
    os.replace(tmp, dst)


def fixture() -> dict[str, Any]:
    from agents.model.compile_parity_fixture import FIXTURE_PATH
    from main.x65.bank import sha256_file

    with np.load(FIXTURE_PATH) as z:
        obs = np.asarray(z["obs"], np.float32)
        mask = np.asarray(z["action_mask"], bool)
    if obs.shape[0] != 64:
        raise ExportError(f"the compile-parity fixture holds {obs.shape[0]} rows, the registration says 64")
    return {"obs": obs, "mask": mask, "sha256": sha256_file(FIXTURE_PATH)}


def trainer_logits(init: Path, state_dict: dict[str, Any], fx: dict[str, Any]) -> dict[str, np.ndarray]:
    """The trainer's function (its own load, the picked weights) on the fixture: CPU, fp32, one thread."""
    import torch as th

    from main.policy_spectrum.reader import inference_globals, load_checkpoint
    from main.x50_r1.hom import policy_logits_t

    with inference_globals(THREADS):
        model = load_checkpoint(init)
        pol = model.policy
        pol.load_state_dict(state_dict)
        pol.eval()
        with th.no_grad():
            x = th.from_numpy(fx["obs"])
            m = th.from_numpy(fx["mask"].astype(np.float32))
            raw = policy_logits_t(pol, x, m).float().numpy()
            mi = th.from_numpy(fx["mask"].astype(np.int8))
            d = pol.get_distribution({"observation": x, "action_mask": mi})
            play = (d.distribution.logits + (mi - 1.0) * 1e9).float().numpy()
    return {"raw": raw, "play": play}


def pin_logits(zp: Path, pin_root: Path, out: Path, py: str = sys.executable) -> dict[str, Any]:
    """The child at the pin (:data:`PIN_CHILD`); returns its report and logits."""
    env = dict(os.environ, PYTHONPATH=str(pin_root / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    r = subprocess.run([py, "-c", PIN_CHILD, str(zp), str(out)], cwd=str(pin_root), env=env, capture_output=True,
                       text=True, timeout=1200, check=False)
    if r.returncode != 0:
        raise ExportError(f"the pin's bare load failed: {r.stderr.strip()[-2000:]}")
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    with np.load(out) as z:
        rep["raw"], rep["play"] = z["raw"], z["play"]
    return rep


def bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype == np.float32 and np.array_equal(a.view(np.uint32),
                                                                                       b.view(np.uint32))


def verify(trainer: dict[str, np.ndarray], pin: dict[str, Any], fx_sha: str) -> list[str]:
    """Every reason the pin does not reproduce the trainer (empty = bit-identical)."""
    bad = []
    if pin.get("loader") != "bare":
        bad.append(f"the pin loaded the zip through {pin.get('loader')!r}, not the bare loader")
    if pin.get("fixture_sha256") != fx_sha:
        bad.append("the pin's compile-parity fixture is not this checkout's")
    for k in ("raw", "play"):
        if not bit_equal(np.asarray(trainer[k]), np.asarray(pin[k])):
            bad.append(f"the {k} logits differ from the trainer's")
    return bad


def run_config(init: Path) -> Path:
    from main.policy_spectrum.reader import _find_config

    return _find_config(init)


def write(fold_dir: Path, init: Path, pin_root: Path, init_sha256: str = ST.PSTAR_SHA256,
          pin: str = PLAY_PIN) -> dict[str, Any]:
    """The export (module docstring); raises :class:`ExportError` (VOID) unless the pin reproduces the trainer."""
    import torch as th

    from main.x65.bank import sha256_file

    if sha256_file(init) != init_sha256:
        raise ExportError(f"{init}: not P*'s bytes")
    head = subprocess.run(["git", "-C", str(pin_root), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=False).stdout
    if not head.strip().startswith(pin):
        raise ExportError(f"{pin_root} is at {head.strip()[:12]}, not the play pin {pin}")
    pk = json.loads((fold_dir / "pick.json").read_text())["picked"]
    sd = th.load(fold_dir / pk["path"], weights_only=True, map_location="cpu")
    if ST._state_digest(sd) != pk["policy_sha256"]:
        raise ExportError(f"{pk['path']}: its weights are not the picked checkpoint's")
    dst = fold_dir / "model"
    dst.mkdir(parents=True, exist_ok=True)
    name = f"x65_{_student_of(fold_dir)}"
    zp = dst / f"{name}.zip"
    write_zip(init, sd, zp)
    shutil.copy(run_config(init), dst / "model_config.json")
    side = json.loads(init.with_suffix(".json").read_text()) if init.with_suffix(".json").exists() else {}
    side["x65_student"] = {"student": _student_of(fold_dir), "picked": pk, "init_sha256": init_sha256,
                           "note": "P*'s checkpoint with the X65 ExIt-0 student's policy weights; every other "
                                   "field of this sidecar is P*'s"}
    zp.with_suffix(".json").write_text(json.dumps(side, indent=2))
    fx = fixture()
    tr = trainer_logits(init, sd, fx)
    pn = pin_logits(zp, pin_root, dst / "pin_logits.npz")
    bad = verify(tr, pn, fx["sha256"])
    doc = {"schema": SCHEMA, "student": _student_of(fold_dir), "zip": str(zp), "zip_sha256": sha256_file(zp),
           "picked": pk, "init_sha256": init_sha256, "fixture_sha256": fx["sha256"], "pin": pin,
           "pin_commit": pn.get("commit"), "pin_loader": pn.get("loader"), "bit_identical": not bad,
           "problems": bad, "smoke": None}
    (dst / "export.json").write_text(json.dumps(doc, indent=1))
    if bad:
        raise ExportError("VOID: " + "; ".join(bad))
    return doc


def _student_of(fold_dir: Path) -> str:
    return str(json.loads((fold_dir / "pick.json").read_text())["student"])


def smoke(fold_dir: Path, pin_root: Path, teamset: str, py: str = sys.executable) -> dict[str, Any]:
    """ONE game of the exported student against a local Kakuna peer, at the pin (module docstring)."""
    from main.x35 import kakuna as K
    from main.x65.bank import sha256_file
    from utils.paths import repo_path

    ex = json.loads((fold_dir / "model" / "export.json").read_text())
    if not ex.get("bit_identical"):
        raise ExportError("no bit-identical export to smoke")
    zp = Path(ex["zip"])
    if sha256_file(zp) != ex["zip_sha256"]:
        raise ExportError(f"{zp}: sha256 differs from the export record")
    out = fold_dir / "model" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    u = K.Unit(name=f"smoke__{teamset}__{SMOKE_SEED}__g00", label=ex["student"], arm=ex["student"], seed=0,
               teamset=teamset, unit_seed=SMOKE_SEED, model=str(zp), model_sha256=ex["zip_sha256"], game=0, games=1)
    port, held = K.claim_port()
    try:
        argv = K.unit_argv(u, out, port, py=py, short=False)
        cap = [str(repo_path("scripts", "ops", "mem_cap.sh")), "12", "timeout", "2700"] + argv
        env = dict(os.environ, PYTHONPATH=str(pin_root / "src"), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(cap, cwd=str(pin_root), env=env, capture_output=True, text=True, check=False)
    finally:
        held.close()
    finished = r.returncode == 0 and (out / "games.jsonl").exists() and \
        sum(1 for x in (out / "games.jsonl").read_text().splitlines() if x.strip()) == 1
    ex["smoke"] = {"rc": r.returncode, "finished": finished, "unit_seed": SMOKE_SEED, "teamset": teamset,
                   "stderr_tail": r.stderr[-1500:]}
    (fold_dir / "model" / "export.json").write_text(json.dumps(ex, indent=1))
    if not finished:
        raise ExportError(f"the smoke game did not finish (rc {r.returncode})")
    return ex


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x65.export", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--init", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--teamset", required=True)
    for p in (w, s):
        p.add_argument("--fold-dir", required=True)
        p.add_argument("--pin-root", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            doc = write(Path(a.fold_dir), Path(a.init), Path(a.pin_root))
        else:
            doc = smoke(Path(a.fold_dir), Path(a.pin_root), a.teamset)
        print(json.dumps({k: doc[k] for k in ("student", "zip_sha256", "bit_identical", "smoke")}), flush=True)
        return 0
    except (ExportError, ST.StudentError) as exc:
        print(f"[x65 export] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
