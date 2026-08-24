"""This module contains exceptions."""


class ShowdownException(Exception):
    """
    This exception is raised when a non-managed message
    is received from the server.
    """

    pass


class LoginError(ShowdownException):
    """Raised when the authentication endpoint refuses a login.

    Distinct from a transport failure: the socket is fine, the CREDENTIALS (or the
    rate limit) are not. Without it a refusal surfaced as a ``KeyError: 'assertion'``
    or a ``json.JSONDecodeError`` on an HTML error page — neither of which names the
    cause, and both of which reach the caller through a swallowed ``listen()`` as an
    indefinite hang on ``logged_in.wait()``.
    """

    pass
