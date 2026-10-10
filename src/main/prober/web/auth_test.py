"""The shared-password gate.

What this is protecting is CPU, not data — every read view is anonymous by design. So the tests
are about the properties that keep a low-entropy shared password from being worse than useless:
constant-time comparison, a signed cookie rather than the password itself, an expiry, throttling,
and a default that fails CLOSED when no password is configured.
"""

from __future__ import annotations

import time

import pytest

from main.prober.web.auth import (
    Auth, CookieKeyError, ENV_FILE_VAR, ENV_VAR, KEY_FILE_VAR, default_key_path, load_or_create_key,
    load_password,
)


# -- where the secret comes from ----------------------------------------------------------

def test_password_comes_from_the_environment():
    assert load_password(env={ENV_VAR: "test-only-password"}) == "test-only-password"
    assert load_password(env={ENV_VAR: "  test-only-password  "}) == "test-only-password"


def test_password_can_come_from_a_file(tmp_path):
    """The form a systemd unit should use — a mode-0600 path keeps the secret out of both argv
    and the unit text."""
    secret = tmp_path / "pw"
    secret.write_text("test-only-password\n")
    assert load_password(env={ENV_FILE_VAR: str(secret)}) == "test-only-password"


def test_the_file_wins_over_the_inline_variable(tmp_path):
    secret = tmp_path / "pw"
    secret.write_text("from-file")
    assert load_password(env={ENV_FILE_VAR: str(secret), ENV_VAR: "inline"}) == "from-file"


def test_an_unreadable_password_file_is_loud(tmp_path):
    """Silently falling back to "no password" would turn a typo'd path into a disabled gate."""
    with pytest.raises(RuntimeError):
        load_password(env={ENV_FILE_VAR: str(tmp_path / "missing")})


def test_no_password_configured_reads_as_none():
    assert load_password(env={}) is None
    assert load_password(env={ENV_VAR: "   "}) is None


# -- the gate -----------------------------------------------------------------------------

def test_unconfigured_fails_closed():
    """No password set => the expensive probes are OFF, not open. An operator who forgets to set
    the secret must not accidentally publish a CPU-burn button."""
    auth = Auth(None)
    assert auth.configured is False
    assert auth.unlocked(None) is False
    assert auth.check("anything", "ip") is False
    assert auth.unlocked(auth.issue()) is False


def test_open_access_bypasses_everything():
    """`--open` is the laptop escape hatch, and it must not need a password to exist."""
    auth = Auth(None, open_access=True)
    assert auth.required is False
    assert auth.unlocked(None) is True


def test_the_right_password_unlocks_and_the_wrong_one_does_not():
    auth = Auth("test-only-password")
    assert auth.check("test-only-password", "ip") is True
    assert auth.check("pidgey", "ip") is False
    assert auth.check("Test-Only-Password", "ip") is False   # exact match, no case folding
    assert auth.check("test-only-password ", "ip") is False   # no trimming of the ATTEMPT


def test_the_cookie_is_a_signature_not_the_password():
    """A cookie is client-visible. Putting the shared secret in it would hand the password to
    anyone who reads one out of a browser or a proxy log."""
    auth = Auth("test-only-password")
    token = auth.issue()
    assert "test-only-password" not in token
    assert auth.valid(token) is True
    assert auth.unlocked(token) is True


@pytest.mark.parametrize("forged", [
    "", "garbage", "9999999999", "9999999999.", ".sig", "9999999999.notasignature",
    "abc.def", "-1.x",
])
def test_a_forged_cookie_is_rejected(forged):
    assert Auth("test-only-password").valid(forged) is False


def test_a_cookie_signed_by_another_process_is_rejected():
    """With no key file the signing key is minted per process, so a token from a different
    instance was never valid here."""
    other = Auth("test-only-password").issue()
    assert Auth("test-only-password").valid(other) is False


def test_an_expired_cookie_is_rejected(monkeypatch):
    auth = Auth("test-only-password")
    token = auth.issue()
    assert auth.valid(token)
    monkeypatch.setattr(time, "time", lambda: time.struct_time and 10 ** 12)  # far future
    assert auth.valid(token) is False


def test_the_expiry_is_signed_so_it_cannot_be_extended():
    """Tampering with the plaintext half must invalidate the token — otherwise the expiry is a
    suggestion the client can edit."""
    auth = Auth("test-only-password")
    expiry, _, sig = auth.issue().partition(".")
    assert auth.valid(f"{int(expiry) + 10 ** 6}.{sig}") is False


# -- throttling ---------------------------------------------------------------------------

def test_repeated_failures_throttle_that_client():
    """A password handed out in Discord is guessable by construction, so the brute-force path is
    the one that needs closing."""
    auth = Auth("test-only-password")
    assert auth.throttled("1.2.3.4") == 0.0
    for _ in range(8):
        auth.check("wrong", "1.2.3.4")
    assert auth.throttled("1.2.3.4") > 0.0


def test_throttling_is_per_client():
    auth = Auth("test-only-password")
    for _ in range(8):
        auth.check("wrong", "1.2.3.4")
    assert auth.throttled("5.6.7.8") == 0.0


def test_a_success_clears_the_failure_count():
    auth = Auth("test-only-password")
    for _ in range(3):
        auth.check("wrong", "1.2.3.4")
    assert auth.check("test-only-password", "1.2.3.4") is True
    for _ in range(7):
        auth.check("wrong", "1.2.3.4")
    assert auth.throttled("1.2.3.4") == 0.0, "the counter should have restarted after the success"


# -- regressions from the 2026-08-09 adversarial review -------------------------------------
# Each of these FAILS if its fix is reverted. The throttle one is the important case: the
# per-client limit was completely defeated by rotating the client identity.

def test_a_rotating_client_identity_cannot_brute_force_the_password():
    """THE finding. With only a per-client cooldown, an attacker who presents a fresh identity
    each request gets unlimited guesses — measured at 500/500 accepted before the fix. The global
    cap must bite regardless of how many identities are used."""
    auth = Auth("test-only-password")
    accepted = 0
    for i in range(500):
        if auth.throttled(f"10.0.0.{i}") > 0:
            break
        auth.check("wrong", f"10.0.0.{i}")
        accepted += 1
    assert accepted < 500, "a rotating identity still got unlimited guesses"
    assert accepted <= 80, f"the global cap let {accepted} guesses through"


def test_the_global_cap_is_not_reset_by_a_successful_login():
    """Otherwise one legitimate user logging in hands the guesser a fresh budget."""
    auth = Auth("test-only-password")
    for i in range(70):
        auth.check("wrong", f"10.1.0.{i}")
    assert auth.throttled("10.1.0.999") > 0
    auth.check("test-only-password", "friendly")          # a real login
    assert auth.throttled("10.1.0.999") > 0, "a success cleared the global window"


def test_the_global_window_expires(monkeypatch):
    """It is a rate limit, not a permanent lockout — legitimate users must recover."""
    import main.prober.web.auth as A
    auth = Auth("test-only-password")
    for i in range(70):
        auth.check("wrong", f"10.2.0.{i}")
    assert auth.throttled("someone") > 0
    now = time.time()
    monkeypatch.setattr(A.time, "time", lambda: now + A._GLOBAL_WINDOW_SECONDS + 1)
    assert auth.throttled("someone") == 0.0


def test_the_failure_map_is_bounded():
    """It is written by anonymous requests; unbounded it is an attacker-controlled memory leak
    (measured: 3000 spoofed identities -> 3000 permanent entries)."""
    import main.prober.web.auth as A
    auth = Auth("test-only-password")
    for i in range(A._MAX_TRACKED_CLIENTS * 2):
        auth.check("wrong", f"10.3.{i // 256}.{i % 256}")
    assert len(auth._failures) <= A._MAX_TRACKED_CLIENTS


def test_a_single_client_is_still_throttled_quickly():
    """The global cap must not have replaced the per-client one — a single guesser should be cut
    off after a handful of tries, long before the global budget."""
    auth = Auth("test-only-password")
    for _ in range(8):
        auth.check("wrong", "1.2.3.4")
    assert auth.throttled("1.2.3.4") > 0
    assert auth.throttled("9.9.9.9") == 0.0, "one guesser must not lock out everyone else"


# -- the signing key outlives a restart ---------------------------------------------------

def test_a_cookie_survives_a_restart_with_the_same_key_file_and_password(tmp_path):
    """The owner's complaint (2026-10-09): every restart logged him out. A NEW Auth (the simulated
    restart) over the same key file and password must accept the old cookie."""
    keyfile = str(tmp_path / "key")
    token = Auth("test-only-password", key_file=keyfile).issue()
    restarted = Auth("test-only-password", key_file=keyfile)
    assert restarted.key_persisted is True
    assert restarted.valid(token) is True
    assert restarted.unlocked(token) is True


def test_changing_the_password_invalidates_existing_cookies(tmp_path):
    keyfile = str(tmp_path / "key")
    token = Auth("test-only-password", key_file=keyfile).issue()
    assert Auth("another-test-only-password", key_file=keyfile).valid(token) is False


def test_a_different_key_file_does_not_validate(tmp_path):
    token = Auth("test-only-password", key_file=str(tmp_path / "a")).issue()
    assert Auth("test-only-password", key_file=str(tmp_path / "b")).valid(token) is False


def test_the_key_file_is_created_0600_and_32_bytes(tmp_path):
    path = tmp_path / "sub" / "key"
    key = load_or_create_key(str(path))
    assert len(key) == 32 and path.read_bytes() == key
    assert (path.stat().st_mode & 0o777) == 0o600
    assert load_or_create_key(str(path)) == key           # stable across calls
    assert [p.name for p in path.parent.iterdir()] == ["key"]   # no temp litter


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o666])
def test_a_key_file_looser_than_0600_is_refused(tmp_path, mode):
    path = tmp_path / "key"
    path.write_bytes(b"k" * 32)
    path.chmod(mode)
    with pytest.raises(CookieKeyError):
        load_or_create_key(str(path))
    # Auth never signs with it: it falls back to a per-process key, and says so.
    auth = Auth("test-only-password", key_file=str(path))
    assert auth.key_persisted is False
    assert Auth("test-only-password", key_file=str(path)).valid(auth.issue()) is False
    assert path.read_bytes() == b"k" * 32                  # not overwritten


def test_a_key_file_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o600)
    monkeypatch.setattr("main.prober.web.auth.os.geteuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(CookieKeyError):
        load_or_create_key(str(path))


def test_a_symlinked_key_file_is_refused(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(b"k" * 32)
    real.chmod(0o600)
    link = tmp_path / "key"
    link.symlink_to(real)
    with pytest.raises(CookieKeyError):
        load_or_create_key(str(link))


def test_a_missing_key_file_is_regenerated_and_old_cookies_die(tmp_path):
    keyfile = tmp_path / "key"
    token = Auth("test-only-password", key_file=str(keyfile)).issue()
    keyfile.unlink()
    fresh = Auth("test-only-password", key_file=str(keyfile))
    assert keyfile.exists() and fresh.key_persisted is True
    assert fresh.valid(token) is False
    assert Auth("test-only-password", key_file=str(keyfile)).valid(fresh.issue()) is True


@pytest.mark.parametrize("junk", [b"", b"short", b"x" * 33])
def test_a_corrupt_key_file_is_regenerated_safely(tmp_path, junk):
    keyfile = tmp_path / "key"
    keyfile.write_bytes(junk)
    keyfile.chmod(0o600)
    first = Auth("test-only-password", key_file=str(keyfile))
    assert first.key_persisted is True
    assert len(keyfile.read_bytes()) == 32 and (keyfile.stat().st_mode & 0o777) == 0o600
    assert Auth("test-only-password", key_file=str(keyfile)).valid(first.issue()) is True


def test_the_expiry_still_applies_to_a_persisted_key(tmp_path, monkeypatch):
    keyfile = str(tmp_path / "key")
    auth = Auth("test-only-password", key_file=keyfile)
    token = auth.issue()
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 15 * 24 * 3600)
    assert Auth("test-only-password", key_file=keyfile).valid(token) is False


def test_the_key_path_honours_the_env_override_and_the_state_default(tmp_path):
    assert default_key_path(env={KEY_FILE_VAR: "/x/y"}) == "/x/y"
    assert default_key_path(env={}).endswith("/.local/state/gen3ai/prober_cookie_key")
