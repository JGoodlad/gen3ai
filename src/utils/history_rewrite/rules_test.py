"""The filter-repo rules: literal vs boundary regex vs scoped, and the invariant that no rule can re-match a replacement."""
from __future__ import annotations

import re

import pytest

from utils.history_rewrite import names as N
from utils.history_rewrite.rules import build_rules
from utils.history_rewrite.synthetic_repo_testkit import make_repo

SCOPE = {"designs/research_state/measurements/human_replay_quality_2026-10-25/results/results.json"}


def _ident(idx, tid, *forms, tier="flagged"):
    return N.Identity(idx, tid, tuple(forms), tier)


def _audit(per: dict[int, N.IdentityHits]) -> N.Audit:
    return N.Audit(per_identity=per)


def _hits(spellings=(), contaminated=(), paths=(), pre=0):
    h = N.IdentityHits()
    for s in spellings:
        h.spellings[s] += 1
        h.blobs.add(s.decode())
    for s in contaminated:
        h.contaminated[s] += 1
    h.paths |= set(paths)
    h.pre_existing_blobs = pre
    return h


def test_a_clean_identity_gets_literal_rules_for_every_spelling_longest_first():
    ids = [_ident(1, "zorblaxpilot", "zorblaxpilot", "ZorblaxPilot")]
    r = build_rules(_audit({1: _hits([b"ZORBLAXPILOT", b"zorblaxpilot"], paths=SCOPE)}), ids, SCOPE)
    assert [sp for sp, _ in r.literals] == sorted({b"ZORBLAXPILOT", b"zorblaxpilot", b"ZorblaxPilot"}, key=lambda b: (-len(b), b))
    assert not r.regexes and not r.scoped and not r.waived
    text = r.render(ids).decode()
    assert text.count("==>redacted-account-01") == 3 and text.startswith("literal:")


def test_a_spelling_inside_a_longer_token_becomes_a_boundary_regex_not_a_literal():
    ids = [_ident(1, "redf", "redf", "red f")]
    r = build_rules(_audit({1: _hits([b"red f"], contaminated=[b"red f"], paths=SCOPE)}), ids, SCOPE)
    assert [sp for sp, _ in r.regexes] == [b"red f"] and b"red f" not in [sp for sp, _ in r.literals]
    rule = r.render(ids).decode().splitlines()[-1]
    assert rule.startswith("regex:") and "(?<![A-Za-z0-9])" in rule
    rx = re.compile(rule[len("regex:"):rule.index("==>")].encode())
    assert rx.sub(b"X", b"say red f now, but red fixed stays") == b"say X now, but red fixed stays"


def test_an_identity_standing_alone_outside_the_account_data_files_is_scoped_and_waived():
    ids = [_ident(1, "zorblaxpilot", "zorblaxpilot"), _ident(2, "quibblefnord", "quibblefnord")]
    per = {1: _hits([b"zorblaxpilot"], paths=SCOPE), 2: _hits([b"quibblefnord"], paths=SCOPE | {"src/fixture.log"})}
    r = build_rules(_audit(per), ids, SCOPE)
    assert [i for _, i in r.literals] == [1] and [i for _, i in r.scoped] == [2]
    assert r.waived == {2: ("src/fixture.log",)}
    assert b"quibblefnord" not in r.render(ids)      # a scoped spelling never reaches the GLOBAL file
    body = r.render_callback(ids)
    assert "quibblefnord" in body and "redacted-account-02" in body and "src/fixture.log" not in body


def test_a_hit_blob_that_predates_the_introducing_commit_also_scopes_the_identity():
    ids = [_ident(1, "zorblaxpilot", "zorblaxpilot")]
    r = build_rules(_audit({1: _hits([b"zorblaxpilot"], paths=SCOPE, pre=1)}), ids, SCOPE)
    assert r.scoped and not r.literals and 1 in r.waived


def test_the_callback_body_compiles_and_scopes_by_path():
    ids = [_ident(2, "quibblefnord", "quibblefnord")]
    r = build_rules(_audit({2: _hits([b"quibblefnord"], paths=SCOPE | {"x.log"})}), ids, SCOPE)
    scope_path = sorted(SCOPE)[0].encode()

    class Value:
        def __init__(self): self.inserted = []
        def get_contents_by_identifier(self, b): return b"a quibblefnord b quibblefnordx"
        def is_binary(self, c): return False
        def apply_replace_text(self, c): return c
        def insert_file_with_contents(self, c): self.inserted.append(c); return b"NEWBLOB"

    body = "\n  ".join(r.render_callback(ids).splitlines())
    ns: dict = {"re": re}
    exec("def cb(filename, mode, blob_id, value):\n  " + body, ns)
    v = Value()
    assert ns["cb"](scope_path, b"100644", b"OLD", v) == (scope_path, b"100644", b"NEWBLOB")
    assert v.inserted == [b"a redacted-account-02 b quibblefnordx"]          # boundary-checked
    v2 = Value()
    assert ns["cb"](b"x.log", b"100644", b"OLD", v2) == (b"x.log", b"100644", b"OLD") and not v2.inserted  # out of scope
    assert ns["cb"](scope_path, b"120000", b"L", Value()) == (scope_path, b"120000", b"L")                   # symlinks untouched


def test_a_form_that_could_match_inside_the_replacement_token_is_refused():
    with pytest.raises(ValueError, match="replacement token"):
        build_rules(_audit({1: _hits([b"redacted"])}), [_ident(1, "redacted", "redacted")], SCOPE)


def test_two_identities_may_not_share_a_spelling():
    ids = [_ident(1, "zorblax", "Zorblax"), _ident(2, "zorblax2", "zorblax")]
    with pytest.raises(ValueError, match="share one spelling"):
        build_rules(_audit({}), ids, SCOPE)


def test_a_spelling_the_rules_file_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="cannot carry"):
        build_rules(_audit({}), [_ident(1, "ab", "a==>b long")], SCOPE)


def test_rules_from_a_real_audit_cover_every_derived_form(tmp_path):
    r = make_repo(tmp_path)
    ids = N.derive_identities(r["path"], r["intro"])
    a = N.audit_history(r["path"], ids, before=r["intro"])
    rules = build_rules(a, ids, set())
    spelled = {sp.lower() for sp, _ in [*rules.literals, *rules.regexes, *rules.scoped]}
    assert {f.lower().encode() for i in ids for f in i.forms} <= spelled
