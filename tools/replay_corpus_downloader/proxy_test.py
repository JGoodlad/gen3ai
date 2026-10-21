"""The downloader's proxy: every request goes through the owner's SOCKS5 proxy, and it fails CLOSED.

No test here reaches the internet: the "proxy" is a fake SOCKS5 server on loopback that records the CONNECT it is
asked for and refuses it, so a request that tried to bypass it would be the only way to see a different outcome.
"""
import socket
import struct
import threading

import pytest

from tools.replay_corpus_downloader import sync as S


class FakeSocks5:
    """A loopback SOCKS5 server (no auth) that records each CONNECT target and answers 'connection refused'."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.targets = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with c:
                try:
                    n = c.recv(2)[1]
                    c.recv(n)
                    c.sendall(b"\x05\x00")                          # no auth
                    head = c.recv(4)
                    if head[3] == 3:                                # a host NAME: socks5h resolves at the proxy
                        host = c.recv(c.recv(1)[0]).decode()
                    else:
                        host = socket.inet_ntoa(c.recv(4))
                    port = struct.unpack(">H", c.recv(2))[0]
                    self.targets.append((host, port))
                    c.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)    # REP 5: connection refused
                except (OSError, IndexError):
                    pass

    def close(self):
        self.srv.close()


@pytest.fixture
def fake_proxy():
    p = FakeSocks5()
    yield p
    p.close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_every_request_goes_to_the_proxy_and_a_refusal_never_falls_back(fake_proxy, monkeypatch):
    # an environment that would send a plain requests.Session DIRECT: trust_env must be off
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("HTTPS_PROXY", "")
    fetch = S.make_http_fetch(sleep_s=0.0, retries=1, timeout=5.0, proxy=f"socks5h://127.0.0.1:{fake_proxy.port}",
                              backoff_s=0.0)
    with pytest.raises(S.FetchError, match="gave up after 2 attempts"):
        fetch(f"{S.BASE}/search.json?format=gen3ou")
    assert fake_proxy.targets == [("replay.pokemonshowdown.com", 443)] * 2, \
        "both attempts went to the proxy, the host name unresolved (socks5h): no direct connection, no local DNS"


def test_check_proxy_fails_closed_on_an_unreachable_proxy():
    with pytest.raises(S.ProxyError, match="unreachable"):
        S.check_proxy(f"socks5h://127.0.0.1:{_free_port()}", timeout=2.0)


def test_check_proxy_passes_a_listening_proxy(fake_proxy):
    S.check_proxy(f"socks5h://127.0.0.1:{fake_proxy.port}")


@pytest.mark.parametrize("bad", ["http://127.0.0.1:1080", "socks5h://127.0.0.1", "127.0.0.1:1080", "socks4://h:1"])
def test_only_a_socks5_proxy_url_is_accepted(bad):
    with pytest.raises(S.ProxyError):
        S.proxy_endpoint(bad)


def test_a_credential_in_the_proxy_url_is_never_printed(capsys):
    assert S.redact_proxy("socks5h://user:s3cret@proxy.example:1080") == "socks5h://***@proxy.example:1080"
    assert S.redact_proxy("socks5h://127.0.0.1:1080") == "socks5h://127.0.0.1:1080"
    rc = S.main(["--out", "x", "--proxy", f"socks5h://user:s3cret@127.0.0.1:{_free_port()}"])
    err = capsys.readouterr().err
    assert rc == S.EXIT_PROXY_DOWN and "s3cret" not in err and "no request made" in err


def test_main_exits_3_on_a_dead_proxy_before_any_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "make_http_fetch", lambda *a, **k: pytest.fail("no fetcher may be built"))
    rc = S.main(["--out", str(tmp_path), "--proxy", f"socks5h://127.0.0.1:{_free_port()}"])
    assert rc == S.EXIT_PROXY_DOWN and not any(tmp_path.iterdir())


def test_a_second_downloader_is_refused_by_the_lock(tmp_path, monkeypatch, fake_proxy):
    lock = tmp_path / "dl.lock"
    monkeypatch.setattr(S, "lock_path", lambda: lock)
    monkeypatch.setattr(S, "make_http_fetch", lambda *a, **k: pytest.fail("no fetcher call while locked"))
    monkeypatch.setattr(S, "checked_fetch", lambda ap, a: (lambda url: pytest.fail("no request while locked")))
    with S.single_instance(lock):
        rc = S.main(["--out", str(tmp_path / "c"), "--proxy", f"socks5h://127.0.0.1:{fake_proxy.port}"])
    assert rc == S.EXIT_LOCKED
    with S.single_instance(lock):       # released on exit: the lock never goes stale
        pass
