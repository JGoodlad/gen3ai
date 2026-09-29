"""The Rust env's protocol helpers: the startup declaration and the typed error decoding (M5 Lane 0)."""
import json

import pytest

from utils.rust_env import protocol as P


def test_spec_json_names_every_key_and_nothing_else():
    s = json.loads(P.spec_json(n=2, threads=1, teams=["a"], names=("x", "y"), decision_tense=False,
                               switch_freeze=True, turn_limit=None, refusal_budget=3, bank_dir=None))
    assert tuple(s) == P.SPEC_KEYS
    assert s["turn_limit"] is None and s["switch_freeze"] is True


def test_every_spec_key_reaches_the_rust_contract():
    from utils.rust_env import columns as C

    assert "pub const SPEC_KEYS: [&str; %d] = [%s];" % (
        len(P.SPEC_KEYS), ", ".join(f'"{k}"' for k in P.SPEC_KEYS)) in C.render()


@pytest.mark.parametrize("status,cls", [(1, P.CoreFault), (2, P.CallerError), (3, P.CorePanic),
                                        (4, P.LifecycleViolation), (5, P.RefusalBudgetExceeded)])
def test_a_dispatch_error_decodes_to_its_typed_class(status, cls):
    text = json.dumps({"status": status, "env": 3, "kind": "fault", "class": None, "message": "m",
                       "script": "START {}\n"})
    e = P.error_from_json(text)
    assert type(e) is cls and e.env == 3 and e.script == "START {}\n" and e.kind == "fault"
