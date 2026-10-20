"""The Rust binaries read their data where they SIT, never where they were built
(``gen3_dex_runtime_data_dir_v1``, ``src/rust_sim/src/dex/data_dir.rs``).

The 2026-09-09 incident: the dex path was baked at compile time, so main's ``sim_bridge`` /
``search_driver``, built from a worktree, panicked once that worktree was deleted. Three smokes over
the real binary (this checkout's build, resolved — built if needed — exactly as production does):

- ``--version`` names THIS checkout's ``data/pokemon``, found from the binary's own location;
- a COPY of the binary in another repo-shaped tree reads THAT tree's data (fails on a revert to the
  compile-time path), and ``$POKESIM_DATA_DIR`` overrides both;
- no binary this checkout has built embeds ANOTHER worktree's path (the grep the row asked for).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.bridge import sim_bridge_bin
from utils.contention import scale_timeout
from utils.paths import repo_path, repo_root, src_path

pytestmark = pytest.mark.integration


def _version(binary: str | Path, env_extra: "dict[str, str] | None" = None) -> "dict[str, str]":
    env = {k: v for k, v in os.environ.items() if k != "POKESIM_DATA_DIR"}
    env.update(env_extra or {})
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env,
                         timeout=scale_timeout(30), check=True).stdout
    head, *fields = out.strip().split("\t")
    assert head.startswith("sim_bridge pokesim "), out
    return dict(f.split("=", 1) for f in fields)


@pytest.fixture(scope="module")
def sim_bridge() -> Path:
    return Path(sim_bridge_bin.resolve_sim_bridge_bin())


def test_version_names_this_checkouts_data_found_from_the_binary(sim_bridge):
    v = _version(sim_bridge)
    assert Path(v["data_dir"]).resolve() == repo_path("data", "pokemon").resolve()
    assert v["source"] == f"module:{sim_bridge}", v


def test_a_relocated_binary_reads_the_data_beside_it_not_its_build_tree(sim_bridge, tmp_path):
    """THE INCIDENT, inverted: the binary moves (a build from elsewhere landing in this tree);
    the data it reads must move with it."""
    fake = tmp_path / "other_repo"
    (fake / "data").mkdir(parents=True)
    (fake / "data" / "pokemon").symlink_to(repo_path("data", "pokemon").resolve())
    dest = fake / "src" / "rust_sim" / "target" / "release" / "sim_bridge"
    dest.parent.mkdir(parents=True)
    shutil.copy2(sim_bridge, dest)
    v = _version(dest)
    assert v["data_dir"] == str(fake / "data" / "pokemon"), v
    assert v["source"] == f"module:{dest}", v
    # The run-time override wins over the location.
    v = _version(dest, {"POKESIM_DATA_DIR": str(tmp_path / "elsewhere")})
    assert (v["data_dir"], v["source"]) == (str(tmp_path / "elsewhere"), "env:POKESIM_DATA_DIR"), v


def test_a_relocated_binary_LOADS_its_dex_from_the_data_beside_it(sim_bridge, tmp_path):
    """The load path itself, not the ``--version`` report: ``sim_bridge`` loads the dex at startup,
    so a relocated copy whose neighbouring data dir holds only the marker file must die naming THAT
    dir's missing ``gen3_moves.json``. On a compile-time path it would load this checkout's full
    data and exit 0 on an empty stdin."""
    fake = tmp_path / "other_repo"
    (fake / "data" / "pokemon").mkdir(parents=True)
    (fake / "data" / "pokemon" / "gen3_species.json").write_text("{}")
    dest = fake / "src" / "rust_sim" / "target" / "release" / "sim_bridge"
    dest.parent.mkdir(parents=True)
    shutil.copy2(sim_bridge, dest)
    env = {k: v for k, v in os.environ.items() if k != "POKESIM_DATA_DIR"}
    r = subprocess.run([str(dest)], input="", capture_output=True, text=True, env=env,
                       timeout=scale_timeout(30), check=False)
    assert r.returncode != 0, f"loaded a dex that is not beside it: {r.stderr[-500:]}"
    assert str(fake / "data" / "pokemon" / "gen3_moves.json") in r.stderr, r.stderr[-1000:]


_WORKTREE_MARKERS =(rb"/gen3ai-wt[^\x00/\s]*", rb"/\.claude/worktrees/[A-Za-z0-9._-]+")


def test_no_built_binary_embeds_another_worktrees_path(sim_bridge):
    """The grep smoke: every worktree path a binary of this checkout embeds is THIS checkout's
    (none at all in the main checkout). A hit names a binary built in a worktree that now sits in
    this tree — the class that panicked on the worktree's deletion."""
    here = str(repo_root().resolve()).encode()
    bins = [p for d in (src_path("rust_sim", "target"), src_path("rust_env", "target"))
            for prof in ("release", "selfcheck", "debug")
            for p in (d / prof).glob("*")
            if p.is_file() and os.access(p, os.X_OK) and p.suffix in ("", ".so")]
    bins.append(sim_bridge)
    foreign = {}
    for b in bins:
        data = b.read_bytes()
        for pat in _WORKTREE_MARKERS:
            for m in set(re.findall(pat, data)):
                if (m + b"/") not in (here + b"/"):
                    foreign.setdefault(str(b), set()).add(m.decode(errors="replace"))
    assert not foreign, f"binaries built in ANOTHER worktree sit in this checkout: {foreign}"
