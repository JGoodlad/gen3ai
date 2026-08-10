"""A shared password that unlocks the expensive endpoints. Reading needs nothing.

THE SHAPE THE OWNER ASKED FOR (2026-08-09): the model is open source and its outcomes are meant to
be public, so every read view is anonymous. What needs gating is not the DATA, it is the WORK —
`falsify_scan` and `calibration` spawn Node re-rolls for minutes each, beside a live training run,
and a public endpoint that starts them is a free CPU-burn button. One shared password, handed out
in Discord, unlocks those. No usernames, no accounts, no email: helping out should not cost anyone
personal information.

So this is deliberately NOT a user system. There is one secret, it grants one capability, and the
only thing a session proves is "this visitor knows the password".

WHAT IT IS AND IS NOT GOOD FOR. A shared password distributed over chat is low-entropy and will
leak eventually; that is understood and acceptable, because the thing behind it is "may spend some
CPU", not "may read private data". It is a speed bump against drive-by and crawlers, not a
security boundary. The measures here are the ones that keep that speed bump honest:

  * **The secret never appears in argv.** It comes from `$GEN3AI_PROBER_PASSWORD` or a file — a
    command line is world-readable in `ps` on a shared box.
  * **Constant-time comparison** (`hmac.compare_digest`), so the check cannot be timed open.
  * **A signed cookie, not the password in a cookie.** HMAC-SHA256 over an expiry, with a signing
    key minted at startup — so a restart logs everyone out, and a stolen cookie expires.
  * **HttpOnly + SameSite=Lax.** Lax is what makes cross-site POSTs fail, which is the CSRF story
    for the job endpoints; there is no separate token to manage.
  * **Failure throttling per client**, because a shared password is guessable by definition.

With no password configured the app is read-only for everyone, including locally — `--open` is the
explicit opt-out that keeps a laptop session frictionless.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import OrderedDict

COOKIE = "prober_session"
_TTL_SECONDS = 14 * 24 * 3600          # a fortnight; a restart invalidates sooner anyway
_MAX_FAILURES = 8                      # per client, before the cooldown
_COOLDOWN_SECONDS = 300.0

# A GLOBAL cap, on top of the per-client one. The per-client throttle is only as good as the
# client identity, and identity behind a proxy is a header — an adversarial review of this file
# defeated the per-client limit completely by rotating `CF-Connecting-IP` (500/500 guesses, no
# cooldown). `app._client` now only trusts that header from a trusted peer, but a keying scheme
# is the kind of thing that gets quietly weakened again, so the guarantee "you cannot brute-force
# this password at request rate" should not depend on keying being right. This cap holds even if
# every request presents a brand-new identity.
_GLOBAL_MAX_FAILURES = 60
_GLOBAL_WINDOW_SECONDS = 300.0
# Bound the failure map: it is written by anonymous requests, so an unbounded dict is a slow
# memory leak an attacker controls (measured: 3000 spoofed identities -> 3000 permanent entries).
_MAX_TRACKED_CLIENTS = 2048

ENV_VAR = "GEN3AI_PROBER_PASSWORD"
ENV_FILE_VAR = "GEN3AI_PROBER_PASSWORD_FILE"


def load_password(*, env=None) -> "str | None":
    """The shared secret from the environment, or a file it names. None when unset.

    A file is the better form for a systemd unit — `EnvironmentFile=` or a mode-0600 secret path
    keeps it out of both argv and the unit text.
    """
    env = os.environ if env is None else env
    path = env.get(ENV_FILE_VAR)
    if path:
        try:
            with open(path) as fh:
                value = fh.read().strip()
        except OSError as exc:
            raise RuntimeError(f"{ENV_FILE_VAR}={path!r} could not be read: {exc}") from exc
        return value or None
    return (env.get(ENV_VAR) or "").strip() or None


class Auth:
    """The gate. `unlocked(request)` is the only question the app asks it."""

    def __init__(self, password: "str | None", *, open_access: bool = False) -> None:
        self._password = password
        self.open_access = bool(open_access)
        self._key = secrets.token_bytes(32)      # per-process: a restart ends every session
        # OrderedDict so the oldest entry is cheap to evict — see _MAX_TRACKED_CLIENTS.
        self._failures: "OrderedDict[str, list]" = OrderedDict()  # client -> [count, first_ts]
        self._global_failures: "list[float]" = []  # timestamps, pruned to the window
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        """Is there a password at all? When there is not, and `--open` was not passed, the
        expensive endpoints are simply off — which is the safe default for a hosted instance whose
        operator forgot to set the secret."""
        return self._password is not None

    @property
    def required(self) -> bool:
        """Does the visitor need to do anything to run a job?"""
        return not self.open_access

    # -- login ---------------------------------------------------------------------------

    def throttled(self, client: str) -> float:
        """Seconds remaining before this client may try again; 0.0 when it may try now.

        Two limits, whichever bites harder: this client's cooldown, and the global one that holds
        even when every request claims a different identity.
        """
        now = time.time()
        with self._lock:
            self._prune(now)
            wait = 0.0
            entry = self._failures.get(client)
            if entry and entry[0] >= _MAX_FAILURES:
                elapsed = now - entry[1]
                if elapsed >= _COOLDOWN_SECONDS:
                    del self._failures[client]
                else:
                    wait = _COOLDOWN_SECONDS - elapsed
            if len(self._global_failures) >= _GLOBAL_MAX_FAILURES:
                oldest = self._global_failures[0]
                wait = max(wait, _GLOBAL_WINDOW_SECONDS - (now - oldest))
            return max(wait, 0.0)

    def check(self, attempt: str, client: str) -> bool:
        """Verify a password attempt, recording the failure if it is wrong."""
        if not self.configured:
            return False
        ok = hmac.compare_digest(attempt.encode("utf-8"), self._password.encode("utf-8"))
        now = time.time()
        with self._lock:
            self._prune(now)
            if ok:
                self._failures.pop(client, None)
                # A success does NOT clear the global window. Otherwise one legitimate login
                # would reset the budget a guesser is burning through.
            else:
                entry = self._failures.get(client)
                if entry is None:
                    entry = [0, now]
                    self._failures[client] = entry
                entry[0] += 1
                self._failures.move_to_end(client)          # most-recently-seen last
                self._global_failures.append(now)
                while len(self._failures) > _MAX_TRACKED_CLIENTS:
                    self._failures.popitem(last=False)      # evict the least-recently-seen
        return ok

    def _prune(self, now: float) -> None:
        """Drop global failures that have aged out of the window. Caller holds the lock."""
        cutoff = now - _GLOBAL_WINDOW_SECONDS
        stamps = self._global_failures
        i = 0
        while i < len(stamps) and stamps[i] < cutoff:
            i += 1
        if i:
            del stamps[:i]

    # -- the cookie ----------------------------------------------------------------------

    def issue(self) -> str:
        expiry = int(time.time()) + _TTL_SECONDS
        return f"{expiry}.{self._sign(expiry)}"

    def valid(self, token: "str | None") -> bool:
        if not token or "." not in token:
            return False
        raw_expiry, _, sig = token.partition(".")
        try:
            expiry = int(raw_expiry)
        except ValueError:
            return False
        if expiry < time.time():
            return False
        return hmac.compare_digest(sig, self._sign(expiry))

    def _sign(self, expiry: int) -> str:
        mac = hmac.new(self._key, str(expiry).encode("ascii"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")

    # -- the one question the app asks ---------------------------------------------------

    def unlocked(self, token: "str | None") -> bool:
        """May this request run an expensive job?"""
        if self.open_access:
            return True
        return self.configured and self.valid(token)
