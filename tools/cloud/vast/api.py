"""A minimal vast.ai REST client (stdlib urllib, Bearer auth). No vastai CLI, no third-party package.

THE KEY NEVER LEAVES THIS PROCESS: it is read from ``~/.config/gen3ai/vastai.key`` (mode 600, the owner's file;
``$GEN3AI_VAST_KEY_FILE`` overrides the path) into a header, and nothing here prints, logs or returns it. An HTTP
error is re-raised as :class:`VastError` carrying the method, path, status and the server's body only.

The endpoints and payload shapes follow the official CLI (``vastai`` 1.8.3, ``vastai/api/instances.py`` and
``keys.py``, read 2026-10-12): offers ``POST /bundles/``; create ``PUT /asks/<offer>/`` -> ``{"success", "new_contract"}``;
list ``GET /api/v1/instances/`` (paged); one ``GET /instances/<id>/?owner=me``; destroy ``DELETE /instances/<id>/``; account
keys ``GET|POST /ssh/``; an instance key ``POST /instances/<id>/ssh/``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

HOST = "https://console.vast.ai"
BASE = HOST + "/api/v0"
DEFAULT_KEY_FILE = "~/.config/gen3ai/vastai.key"


class VastError(RuntimeError):
    """A failed API call: method, path, HTTP status and the server's reply (never the key)."""


def key_file() -> str:
    return os.path.expanduser(os.environ.get("GEN3AI_VAST_KEY_FILE") or DEFAULT_KEY_FILE)


def _key() -> str:
    path = key_file()
    if not os.path.isfile(path):
        raise VastError(f"no vast.ai API key at {path} (the owner saves it there, mode 600)")
    if os.stat(path).st_mode & 0o077:
        raise VastError(f"{path} is readable by group/other — chmod 600 it first")
    with open(path, encoding="utf-8") as fh:
        k = fh.read().strip()
    if not k:
        raise VastError(f"{path} is empty")
    return k


def call(method: str, path: str, body: Optional[Dict[str, Any]] = None,
         query: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Any:
    url = (HOST if path.startswith("/api/") else BASE) + path
    if query:
        url += "?" + urllib.parse.urlencode({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                             for k, v in query.items()})
    req = urllib.request.Request(
        url, method=method, data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        raise VastError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise VastError(f"{method} {path} -> {e.reason!r}") from None
    return json.loads(raw) if raw.strip() else {}


# ---------------------------------------------------------------------------------------------- account
def account() -> Dict[str, Any]:
    """The account's money fields only (credit, balance, can_pay); never the whole user row."""
    u = call("GET", "/users/current/")
    return {k: u.get(k) for k in ("credit", "balance", "can_pay", "has_billing")}


def credit() -> float:
    return float(account().get("credit") or 0.0)


def ssh_keys() -> List[Dict[str, Any]]:
    r = call("GET", "/ssh/")
    return r if isinstance(r, list) else (r.get("keys") or r.get("ssh_keys") or [])


def add_account_ssh_key(pub: str) -> Any:
    return call("POST", "/ssh/", {"ssh_key": pub})


def attach_ssh_key(instance_id: int, pub: str) -> Any:
    return call("POST", f"/instances/{int(instance_id)}/ssh/", {"ssh_key": pub})


# ---------------------------------------------------------------------------------------------- offers
def search_offers(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = call("POST", "/bundles/", query)
    return list(r.get("offers", r if isinstance(r, list) else []))


# ---------------------------------------------------------------------------------------------- instances
def create_instance(offer_id: int, *, image: str, disk_gb: float, label: str, onstart: str = "",
                    env: Optional[Dict[str, str]] = None) -> int:
    """Rent ``offer_id`` (on-demand) with an SSH runtype; returns the new instance id."""
    body = {"client_id": "me", "image": image, "env": env or {}, "price": None, "disk": disk_gb,
            "label": label, "extra": None, "onstart": onstart or None, "image_login": None,
            "python_utf8": False, "lang_utf8": False, "use_jupyter_lab": False, "jupyter_dir": None,
            "force": False, "cancel_unavail": True, "template_hash_id": None, "user": None,
            "runtype": "ssh_direc ssh_proxy"}
    r = call("PUT", f"/asks/{int(offer_id)}/", body)
    if not r.get("success") or not r.get("new_contract"):
        raise VastError(f"create on offer {offer_id} refused: {r}")
    return int(r["new_contract"])


def instances() -> List[Dict[str, Any]]:
    """EVERY instance on the account, in any state (v1, paged; the v0 list answers 410 since 2026)."""
    rows: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}], "limit": 25}
    for _ in range(100):
        r = call("GET", "/api/v1/instances/", query=params)
        rows.extend(r.get("instances") or [])
        if not r.get("next_token"):
            return rows
        params["after_token"] = r["next_token"]
    raise VastError("GET /api/v1/instances/: more than 100 pages")


def instance(instance_id: int) -> Optional[Dict[str, Any]]:
    try:
        r = call("GET", f"/instances/{int(instance_id)}/", query={"owner": "me"})
    except VastError as e:
        if "HTTP 404" in str(e):
            return None
        raise
    inst = r.get("instances") if isinstance(r, dict) else None
    if isinstance(inst, list):
        return inst[0] if inst else None
    return inst or None


def destroy_instance(instance_id: int) -> Any:
    return call("DELETE", f"/instances/{int(instance_id)}/", {})
