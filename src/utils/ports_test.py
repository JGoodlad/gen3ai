"""``utils.ports`` / ``utils.filelock`` / ``utils.hashing``: the tools' converged helpers (2026-10-26) behave EXACTLY as
the copies they replaced. Each ``_old_*`` below is the replaced code VERBATIM (kakuna's ``choose_port`` /
``claim_port``, the anchors server's ``_free``, the team-strength lane's ``_port_free``, the four sha256 copies); a
recording fake socket proves the new call makes the same syscalls in the same order with the same result, over
every busy / claimed pattern of a small range. Deterministic: the fake decides which bind fails, no real port is
probed except in the one real-socket check, which binds its own listener."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import itertools
import socket
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import pytest

from utils import ports as P
from utils.filelock import try_exclusive
from utils.hashing import CHUNK, sha256_file
from utils.paths import src_path


# ------------------------------------------------------------------------------------- the replaced code, verbatim
class _KakunaError(RuntimeError):
    pass


def _old_choose_port(lo: int, hi: int) -> int:
    for p in range(lo, hi + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
            except OSError:
                continue
            return p
    raise _KakunaError(f"no free port in {lo}-{hi}")


def _old_claim_port(lo: int, hi: int, lock_dir: Path) -> Tuple[int, Any]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    for p in range(lo, hi + 1):
        fh = open(lock_dir / f"{p}.lock", "w")  # noqa: SIM115
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
            except OSError:
                fh.close()
                continue
        return p, fh
    raise _KakunaError(f"no free, unclaimed port in {lo}-{hi}")


def _old_anchors_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _old_team_strength_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _old_sha(path: Any) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- the recording socket
class _Trace:
    def __init__(self, busy: Set[int]) -> None:
        self.busy, self.calls = busy, []  # type: Set[int], List[Tuple[Any, ...]]


def _fake_socket_cls(trace: _Trace) -> Any:
    class FakeSocket:
        def __init__(self, family: int, kind: int) -> None:
            trace.calls.append(("socket", family, kind))

        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *exc: Any) -> None:
            trace.calls.append(("close",))

        def setsockopt(self, *a: Any) -> None:
            trace.calls.append(("setsockopt", *a))

        def bind(self, addr: Tuple[str, int]) -> None:
            trace.calls.append(("bind", *addr))
            if addr[1] in trace.busy:
                raise OSError(98, "Address already in use")

    return FakeSocket


def _outcome(fn: Any, *a: Any, **k: Any) -> Any:
    try:
        r = fn(*a, **k)
    except _KakunaError as exc:
        return ("raise", str(exc))
    if isinstance(r, tuple):                    # a claim: report the port, release the held file
        r[1].close()
        return ("claim", r[0])
    return ("ok", r)


LO, HI = 9590, 9593
PATTERNS = [set(c) for n in range(HI - LO + 2) for c in itertools.combinations(range(LO, HI + 1), n)]


def _run(monkeypatch: pytest.MonkeyPatch, busy: Set[int], fn: Any, *a: Any, **k: Any) -> Tuple[Any, List[Any]]:
    tr = _Trace(busy)
    monkeypatch.setattr(socket, "socket", _fake_socket_cls(tr))
    return _outcome(fn, *a, **k), tr.calls


@pytest.mark.parametrize("busy", PATTERNS, ids=lambda b: "busy" + "-".join(map(str, sorted(b))))
def test_kakuna_choose_port_makes_the_same_syscalls_as_its_old_copy(monkeypatch, busy):
    from main.x35 import kakuna as KA

    def new(lo: int, hi: int) -> int:
        try:
            return KA.choose_port(lo, hi)
        except KA.KakunaError as exc:
            raise _KakunaError(str(exc)) from None

    assert _run(monkeypatch, busy, new, LO, HI) == _run(monkeypatch, busy, _old_choose_port, LO, HI)


@pytest.mark.parametrize("busy", PATTERNS, ids=lambda b: "busy" + "-".join(map(str, sorted(b))))
@pytest.mark.parametrize("held", [(), (LO,), (LO, LO + 2), tuple(range(LO, HI + 1))], ids=lambda h: f"held{len(h)}")
def test_kakuna_claim_port_makes_the_same_syscalls_and_claims_as_its_old_copy(monkeypatch, tmp_path, busy, held):
    """Busy binds AND ports another claimant holds (a real flock on the real lock file, from another open file)."""
    from main.x35 import kakuna as KA

    def new(lo: int, hi: int, lock_dir: Path) -> Tuple[int, Any]:
        try:
            return KA.claim_port(lo, hi, lock_dir=lock_dir)
        except KA.KakunaError as exc:
            raise _KakunaError(str(exc)) from None

    got = {}
    for name, fn in (("old", _old_claim_port), ("new", new)):
        d = tmp_path / name
        d.mkdir()
        holders = [try_exclusive(d / f"{p}.lock") for p in held]
        assert all(h is not None for h in holders)
        try:
            got[name] = _run(monkeypatch, busy, fn, LO, HI, d)
        finally:
            for h in holders:
                h.close()  # type: ignore[union-attr]
        got[name] += (sorted(p.name for p in d.iterdir()),)      # the same lock files are left behind
    assert got["new"] == got["old"]


@pytest.mark.parametrize("busy", [set(), {LO}])
@pytest.mark.parametrize("old", [_old_anchors_free, _old_team_strength_port_free])
def test_the_anchors_and_team_strength_probes_make_the_same_syscalls_as_their_old_copies(monkeypatch, busy, old):
    from main.team_strength import cli as TS

    for new in (P.bindable, TS.bindable):
        assert _run(monkeypatch, busy, new, LO) == _run(monkeypatch, busy, old, LO)


def test_anchors_pick_port_skips_the_reserved_ports_and_picks_the_first_bindable(monkeypatch, tmp_path):
    from main.anchors import server as S

    (res, calls) = _run(monkeypatch, {7999}, S.pick_port, [7999, 8002], tmp_path)
    S.release_all_ports()
    assert res == ("ok", 8002) and [c[2] for c in calls if c[0] == "bind"] == [7999, 8002]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["7999.lock", "8002.lock"], "a reserved port got a lock file"


def test_a_real_listener_reads_busy_to_every_probe_and_free_once_closed():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert not P.bindable(port) and not _old_anchors_free(port)
    finally:
        srv.close()
    assert P.bindable(port) and _old_anchors_free(port)


# ---------------------------------------------------------------------------------------------- reserved ports
def test_every_in_repo_reserved_ports_dict_is_the_one_in_utils_ports():
    from main import play
    from main.anchors import server
    from utils.bridge import ws_frontend

    assert P.RESERVED_PORTS == {8000: "the shared DEV server", 8001: (
        "the reserved TRAINING server (the trainer itself is in-process and connects to none)")}
    assert server.RESERVED_PORTS is P.RESERVED_PORTS and ws_frontend.RESERVED_PORTS is P.RESERVED_PORTS
    assert play.RESERVED_PORTS is P.RESERVED_PORTS


def test_the_metamon_peers_literal_copy_equals_the_canonical_dict():
    """The Metamon peer never imports this tree (another interpreter), so it keeps a literal; it may not drift."""
    tree = ast.parse((src_path() / "main/anchors/peer_scripts/metamon_side.py").read_text())
    vals = [ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "RESERVED_PORTS" for t in n.targets)]
    assert vals == [P.RESERVED_PORTS]


# --------------------------------------------------------------------------------------------------- filelock
def test_try_exclusive_refuses_a_second_open_file_even_in_one_process_and_frees_on_close(tmp_path):
    p = tmp_path / "x.lock"
    a = try_exclusive(p)
    assert a is not None and try_exclusive(p) is None
    a.close()
    b = try_exclusive(p)
    assert b is not None
    b.close()


@pytest.mark.parametrize("mode,kept", [("w", ""), ("a+", "holder\n")])
def test_try_exclusive_keeps_the_callers_open_mode_even_when_refused(tmp_path, mode, kept):
    """``"w"`` truncates on open, as each replaced copy did (kakuna, the reads lane lock); ``"a+"`` keeps the holder
    record the refused supervisor names (``reads.supervisors``)."""
    p = tmp_path / "x.lock"
    p.write_text("holder\n")
    with open(p, "r+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert try_exclusive(p, mode) is None
        assert p.read_text() == kept


# ----------------------------------------------------------------------------------------------------- hashing
@pytest.mark.parametrize("n", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 7])
def test_sha256_file_equals_hashlib_and_the_replaced_copies(tmp_path, n):
    p = tmp_path / "f.bin"
    data = bytes((i * 131 + 7) % 256 for i in range(n))
    p.write_bytes(data)
    assert sha256_file(p) == sha256_file(str(p)) == _old_sha(p) == hashlib.sha256(data).hexdigest()


def test_the_converged_callers_use_the_one_sha256_file():
    from main.reads import models
    from main.x35 import gc, kakuna

    assert kakuna.sha256_file is sha256_file and models.sha256_file is sha256_file and gc.sha256_file is sha256_file
    assert not hasattr(kakuna, "_sha") and not hasattr(gc, "_file_sha")


def test_claim_returns_none_when_every_port_is_taken(tmp_path, monkeypatch):
    tr = _Trace(set(range(LO, HI + 1)))
    monkeypatch.setattr(socket, "socket", _fake_socket_cls(tr))
    got: Optional[Tuple[int, Any]] = P.claim(LO, HI, tmp_path)
    assert got is None and P.first_bindable(LO, HI) is None


# ------------------------------------------------------------------------------ claims for the tests' own servers
def test_claim_skips_the_ports_it_is_told_to_before_touching_their_lock_files(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "bindable", lambda *a, **k: True)
    first, second = (P.claim(7999, 8002, tmp_path, skip=P.RESERVED_PORTS) for _ in range(2))
    assert first is not None and second is not None and (first[0], second[0]) == (7999, 8002)
    assert P.claim(7999, 8002, tmp_path, skip=P.RESERVED_PORTS) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["7999.lock", "8002.lock"], "a reserved port got a lock file"
    first[1].close()
    second[1].close()


def test_claimed_holds_a_port_for_the_block_and_gives_it_back_after(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "bindable", lambda *a, **k: True)       # nothing has bound the port: only the claim differs
    with P.claimed(9650, 9651, tmp_path) as a:
        with P.claimed(9650, 9651, tmp_path) as b:
            assert {a, b} == {9650, 9651}
            with pytest.raises(RuntimeError, match="no free, unclaimed port"):
                with P.claimed(9650, 9651, tmp_path):
                    pass
        with P.claimed(9650, 9651, tmp_path) as again:
            assert again == b, "a released claim is the next picker's"
    with P.claimed(9650, 9651, tmp_path) as first:
        assert first == 9650


def test_claimed_never_hands_out_a_reserved_port(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "bindable", lambda *a, **k: True)
    with P.claimed(8000, 8002, tmp_path) as port:
        assert port == 8002 and not {"8000.lock", "8001.lock"} & {p.name for p in tmp_path.iterdir()}


def test_the_test_port_range_is_clear_of_the_lanes_and_the_anchors_default_range():
    from main.x35 import kakuna

    lo, hi = P.TEST_PORT_RANGE
    assert lo > kakuna.PORT_RANGE[1] and lo > 9599 and not set(range(lo, hi + 1)) & set(P.RESERVED_PORTS)


def test_the_claimed_port_fixture_claims_in_the_shared_lock_dir_and_each_call_is_distinct(claimed_port):
    a, b = claimed_port(), claimed_port()
    assert a != b and all(P.TEST_PORT_RANGE[0] <= p <= P.TEST_PORT_RANGE[1] for p in (a, b))
    for p in (a, b):
        assert try_exclusive(P.PORT_LOCK_DIR / f"{p}.lock") is None, f"{p} is not claimed in the shared lock dir"
    assert claimed_port() not in (a, b)
