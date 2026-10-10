"""Static-token screen look 3: verify the P_st play checkout WITHOUT running git inside it.

``play_look2.sh`` checked the pin with ``git rev-parse HEAD`` and ``git status --porcelain --untracked-files=no -- src
data`` run IN the pin checkout. A worktree-isolated agent's harness refuses git aimed at another checkout, so look 3
proves the same two facts by content: (1) the pin checkout's ``HEAD`` (read from its gitdir's ``HEAD`` file) is the
detached P_st commit, and (2) every TRACKED file under ``src/`` and ``data/`` at P_st (``git ls-tree -r`` in THIS
repository, whose object store is shared) exists in the pin checkout with the identical git blob hash
(sha1 of ``blob <len>\\0<bytes>``), and no tracked path is missing. Untracked files are ignored, as in look 2.

    python pin_check.py [--pin-dir D]      exit 0 = CLEAN at P_st, 5 = REFUSED (the reason printed)
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

PIN = "6c6d2e0942e2111703a7e6d79bfadb31c8e51f01"
PIN_DIR = Path("/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09")
HERE = Path(__file__).resolve().parent


def blob_sha(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-dir", default=str(PIN_DIR))
    a = ap.parse_args()
    pd = Path(a.pin_dir)
    gitfile = (pd / ".git").read_text().strip()
    gitdir = Path(gitfile.split("gitdir:", 1)[1].strip())
    head = (gitdir / "HEAD").read_text().strip()
    print(f"pin checkout {pd}: HEAD {head}")
    if head != PIN:
        print(f"REFUSED: HEAD is not the detached P_st {PIN}")
        return 5
    tree = subprocess.run(["git", "ls-tree", "-r", "--full-tree", PIN, "--", "src", "data"], cwd=HERE,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    bad, n = [], 0
    for ln in tree:
        meta, path = ln.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            continue          # a submodule entry (commit) has no file to compare
        n += 1
        f = pd / path
        if mode == "120000":
            ok = f.is_symlink() and hashlib.sha1(b"blob %d\0" % len(str(f.readlink()).encode())
                                                 + str(f.readlink()).encode()).hexdigest() == sha
        else:
            ok = f.is_file() and not f.is_symlink() and blob_sha(f) == sha
        if not ok:
            bad.append(path)
    print(f"{n} tracked files under src/ + data/ at P_st compared; {len(bad)} differ or are missing")
    if n == 0:
        print("REFUSED: git ls-tree listed no tracked file (a vacuous comparison)")
        return 5
    if bad:
        print("REFUSED: the pin checkout's src/ or data/ is modified:", bad[:20])
        return 5
    print("CLEAN: the pin checkout is P_st 6c6d2e09 with src/ and data/ unmodified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
