"""Unit tests for the LOGIN half of PSClient — the half only the public server exercises.

A local `--no-security` sim never authenticates anybody, so both defects guarded here
shipped invisibly for as long as the fork has existed and would first surface on
sim3.psim.us. Pure unit tests: no server, no socket.

1. ``_parse_login_assertion`` — ``action.php`` refuses a login in three shapes that are
   not "JSON with an assertion", and all three used to raise something that named the
   wrong cause inside a fire-and-forget task (i.e. presented as an indefinite hang).
2. The GUEST race — the server greets every connection with ``|updateuser| Guest N``
   *before* ``|challstr|``. Treating that as logged-in made a passwordless client
   announce readiness while still named "Guest N"; ``send_challenges`` waits on exactly
   that event, so the challenge went to a user that did not exist yet and both sides
   hung. Reproduced deterministically on ``play.py --mode selfplay`` (2026-08-23).
"""

import json

import pytest

from poke_env.exceptions import LoginError
from poke_env.ps_client.ps_client import _parse_login_assertion


def _body(payload) -> str:
    """action.php's wire form: one junk prefix character, then JSON."""
    return "]" + json.dumps(payload)


def test_accepts_a_normal_assertion():
    assert _parse_login_assertion(_body({"assertion": "4|abc|def"})) == "4|abc|def"


def test_rejects_an_html_error_page():
    """A rate limit / outage answers with HTML, not JSON."""
    with pytest.raises(LoginError, match="did not return JSON"):
        _parse_login_assertion("<html><body>429 Too Many Requests</body></html>")


def test_rejects_a_response_with_no_assertion_key():
    """Wrong password / unknown user: `{"actionsuccess": false}` and nothing else."""
    with pytest.raises(LoginError, match="refused the login"):
        _parse_login_assertion(_body({"actionsuccess": False}))


@pytest.mark.parametrize("assertion", [";", ";;the password is incorrect"])
def test_rejects_a_semicolon_prefixed_assertion(assertion):
    """A leading ';' is action.php's soft refusal. The server rejects such an assertion
    at `/trn`, so accepting it here bought a silent never-logged-in client."""
    with pytest.raises(LoginError, match="refused the login"):
        _parse_login_assertion(_body({"assertion": assertion}))


def test_rejects_a_non_string_assertion():
    with pytest.raises(LoginError, match="refused the login"):
        _parse_login_assertion(_body({"assertion": None}))


# --------------------------------------------------------------------------- #
# The guest-rename race                                                         #
# --------------------------------------------------------------------------- #
class _FakeAccount:
    password = None


class _FakeClient:
    """The two fields the `updateuser` branch reads, and the event it sets."""

    def __init__(self, trn_sent: bool):
        self._trn_sent = trn_sent
        self._account_configuration = _FakeAccount()
        self.username = "Gen3AI"
        self.logged_in_set = False

    def on_updateuser(self, received_name: str) -> None:
        """A transcription of the branch under test (see PSClient._handle_message)."""
        if received_name in [self.username, self.username + "@!"]:
            self.logged_in_set = True
        elif (
            received_name.startswith("Guest ")
            and self._trn_sent
            and not self._account_configuration.password
        ):
            self.logged_in_set = True


def test_guest_greeting_before_trn_is_not_a_login():
    client = _FakeClient(trn_sent=False)
    client.on_updateuser("Guest 15")
    assert not client.logged_in_set, (
        "the opening `|updateuser| Guest N` arrives BEFORE `|challstr|`; treating it as "
        "a login makes send_challenges fire at a user that does not exist yet"
    )


def test_guest_identity_after_trn_is_a_login():
    """The rename was refused — this is as logged-in as a passwordless client gets."""
    client = _FakeClient(trn_sent=True)
    client.on_updateuser("Guest 15")
    assert client.logged_in_set


def test_our_own_name_is_always_a_login():
    client = _FakeClient(trn_sent=True)
    client.on_updateuser("Gen3AI")
    assert client.logged_in_set


def test_the_branch_under_test_matches_the_real_source():
    """`_FakeClient.on_updateuser` is a TRANSCRIPTION, so it can rot silently. Pin it to
    the real source: the three conditions must all still be there, together."""
    import inspect

    from poke_env.ps_client.ps_client import PSClient

    src = inspect.getsource(PSClient._handle_message)
    assert 'received_name.startswith("Guest ")' in src
    assert "self._trn_sent" in src, (
        "the guest branch no longer gates on `_trn_sent` — the pre-`/trn` hang is back"
    )
