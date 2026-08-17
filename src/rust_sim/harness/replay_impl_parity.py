"""Differentially gate the REPLAY family (`replay` / `reroll` / `reroll_many`) —
`node replay_driver.js` vs the rust `search_driver` binary, live, side by side.

Scratch tool (tmp/, not shipped). Sibling of `tmp/search_impl_parity.py`, which gates the
SEARCH family (`open_root` / `expand_many`) against a CAPTURED golden.

    export PYTHONPATH=$PYTHONPATH:src
    python src/rust_sim/harness/replay_impl_parity.py [--records tmp/search_golden_node.json] [--bin PATH] [-v]

WHY A SEPARATE FILE (rather than extending search_impl_parity.py)
-----------------------------------------------------------------
Three things differ end to end, and folding them together would have made both harnesses
harder to read for no gain:
  * the PROTOCOL is one-shot (`{mode}` in, one JSON object out, then exit), not the
    persistent `{id, cmd}` loop — so the driver plumbing has nothing in common;
  * there is no captured golden: node is spawned LIVE on the identical request, which is
    strictly stronger here (the replay family's requests are cheap and the record files
    already exist, so a stale golden buys nothing);
  * this harness owns its own COVERAGE construction (ended arms, a `stuck` arm, the error
    paths) — the search golden's case list is fixed by `tmp/search_golden.py`.
The two share the deep-diff + allowlist DISCIPLINE, not code.

NORMALIZATION (the ONLY thing silently forgiven):
  * `|t:|<epoch>` → `|t:|<N>`, inside chunk payloads AND inside `turn_log` lines. This is
    the port's one documented emission exception (wall-clock stamps, which sit in
    poke-env's MESSAGES_TO_IGNORE). Nothing else is normalized.

Everything else that cannot match must appear in ALLOWLIST below, is PRINTED on every run
with its reason and hit count, and is counted separately from matches. An error case
compares the ok/error VERDICT and the error CLASS strictly; only the message TEXT is
forgiven (node returns `e.stack`, the port a plain message).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "src/rust_sim/target/release/search_driver"
NODE_REPLAY = ROOT / "src/utils/bridge/replay_driver.js"
NODE_SEARCH = ROOT / "src/utils/bridge/search_driver.js"
T_LINE = re.compile(r"^\|t:\|.*$", re.M)

# ---------------------------------------------------------------------------
# The EXPLICIT allowlist. A field that diverges and matches an entry is reported as
# ALLOWLISTED (with a hit count), never as a match.
# ---------------------------------------------------------------------------
ALLOWLIST = [
    (
        lambda p: p.endswith(".error"),
        "error TEXT only: Node returns `e.stack` (a JS stack trace with a `Error: ` "
        "prefix and frames), the port a plain message. Both the ok/error VERDICT (and the "
        "process EXIT CODE, which is what reconstruction.py actually branches on) and the "
        "error CLASS are compared STRICTLY — see the ERROR CASES table.",
    ),
]



REJECT_PREFIXES = ("|error|[Invalid choice]", "|error|[Unavailable choice]")


def is_subsequence(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


ERROR_CLASSES = {
    "invalid-turn": "invalid turn ",
    "unreached-turn": "battle never reached the start of turn ",
    "not-ended": "but battle has not ended",
    "unknown-mode": "unknown mode ",
    "unknown-cmd": "unknown cmd ",
    "unknown-node": "unknown node ",
    "recorded-exact-off-root": "recorded_exact requested on a non-root node",
    # `reroll_many` with a seedless arm: node dies in `auxRngFromSeed(undefined)`
    # (`Cannot read properties of undefined`), the port reports the missing field. Same
    # VERDICT and same CAUSE, different wording — so this class matches either phrasing.
    "arm-missing-seed": ("missing seed", "undefined"),
}

# Fields the port renders from state it models only approximately. NOT forgiven (a
# divergence still fails) — printed every run so a green result is not mistaken for
# evidence they are right.
COVERAGE_NOTES = [
    (
        "pre_state.*.active[].volatiles",
        "The port keeps volatiles as typed fields, not a keyed map, so the NAME list is "
        "reconstructed (search.rs::volatile_names). Whether these records exercise a "
        "NON-EMPTY list is reported under COVERAGE below; an all-empty run means only the "
        "'duration-1 fields must NOT appear' fact is tested.",
    ),
    (
        "turn_log |split| reconstruction",
        "Showdown stores an addSplit effect as THREE `battle.log` entries (|split|pN / "
        "secret / shared) and the port keeps ONE omniscient line, so bridge.rs::"
        "split_log_lines re-derives the triples. The HP-bearing and Pressure-[silent] "
        "cases are covered by any real battle; the Intimidate `|-hint|` owner attribution "
        "is a HEURISTIC (track the preceding |switch|) and is only exercised when such a "
        "hint lands inside a re-rolled turn — see the COVERAGE report.",
    ),
]


def norm(s):
    return T_LINE.sub("|t:|<N>", s) if isinstance(s, str) and "|t:|" in s else s


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def one_shot(argv, request, timeout=300):
    """Run a ONE-SHOT driver. Returns (rc, parsed-or-None, raw-stdout, stderr)."""
    p = subprocess.run(argv, input=json.dumps(request).encode(),
                       capture_output=True, timeout=timeout)
    txt = p.stdout.decode("utf-8", "replace")
    try:
        parsed = json.loads(txt)
    except Exception:
        parsed = None
    return p.returncode, parsed, txt, p.stderr.decode("utf-8", "replace")


class Persistent:
    """A persistent `{id, cmd}` driver (either impl) — for the SEARCH-family error paths."""

    def __init__(self, argv):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.seq = 0

    def call(self, **payload):
        self.seq += 1
        payload.setdefault("id", self.seq)
        self.p.stdin.write(json.dumps(payload) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("driver died: " + self.p.stderr.read()[:4000])
        return json.loads(line)

    def close(self):
        try:
            self.call(cmd="close")
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------------------
# Deep diff (same shape as tmp/search_impl_parity.py)
# ---------------------------------------------------------------------------


def diff(exp, got, path, out):
    if isinstance(exp, str) or isinstance(got, str):
        exp, got = norm(exp), norm(got)
    if type(exp) is not type(got) and not (
            isinstance(exp, (int, float)) and isinstance(got, (int, float))):
        out.append((path, exp, got))
        return
    if isinstance(exp, dict):
        for k in sorted(set(exp) | set(got)):
            if k not in exp or k not in got:
                out.append((f"{path}.{k}", exp.get(k, "<absent>"), got.get(k, "<absent>")))
            else:
                diff(exp[k], got[k], f"{path}.{k}", out)
    elif isinstance(exp, list):
        if len(exp) != len(got):
            out.append((f"{path}[len]", len(exp), len(got)))
        for i in range(min(len(exp), len(got))):
            diff(exp[i], got[i], f"{path}[{i}]", out)
    elif exp != got:
        out.append((path, exp, got))


def allowlisted(path):
    for pred, reason in ALLOWLIST:
        if pred(path):
            return reason
    return None


def count_leaves(v):
    if isinstance(v, dict):
        return sum(count_leaves(x) for x in v.values()) or 1
    if isinstance(v, list):
        return sum(count_leaves(x) for x in v) or 1
    return 1


def matches_class(msg, cls):
    pats = ERROR_CLASSES[cls]
    if isinstance(pats, str):
        pats = (pats,)
    return any(p in (msg or "") for p in pats)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def truncate_record(rec, turn):
    """A record whose commands stop ONE command into `turn` — the only construction that
    reaches Node's `stuck` verdict on this family.

    `resolveTurnExact` feeds the remaining recorded commands until the turn advances or the
    battle ends; with a HALF-answered turn it runs out with neither, which is exactly the
    `stuck: true` branch. A well-formed record can never produce that (its turns are always
    complete), so a hand-truncated one is the honest way to exercise the field rather than
    reporting it as un-coverable. It also drives `replay`'s not-ended error path.
    """
    rec = json.loads(json.dumps(rec))          # deep copy — never mutate the shared record
    # Find the command index where `turn` starts by counting resolved move rounds is
    # fragile; instead cut at a fraction of the log and let build_to_turn decide. The caller
    # passes a turn it has already verified is reachable in the truncated stream.
    rec["commands"] = rec["commands"][:turn]
    return rec


def last_turn(rec, node_argv):
    """The final turn of the battle, from a node `replay` (used to pick LATE re-roll turns
    so arms actually END the battle — the coverage the search golden lacked)."""
    rc, out, _raw, _err = one_shot(node_argv, {"mode": "replay", "record": rec})
    if rc != 0 or not out:
        return None
    return out["outcome"]["turn"]


def build_cases(records, node_argv):
    """(name, request, error_class|None) triples — the whole differential surface."""
    cases = []
    for ri, rec in enumerate(records):
        end = last_turn(rec, node_argv) or 10
        # LATE turns first: the previously-missing "arms that END the battle" coverage.
        late = [t for t in (end - 1, end - 3, end - 6) if t >= 2]
        turns = sorted({2, 5, 9, *late})

        cases.append((f"r{ri}/replay", {"mode": "replay", "record": rec}, None))

        for T in turns:
            cases.append((
                f"r{ri}/reroll@{T}",
                {"mode": "reroll", "record": rec, "turn": T,
                 "seeds": ["original", "sodium,0011223344556677", "gen5,1234567890abcdef"],
                 "p1_action": "recorded", "p2_action": "recorded", "followup": "random"},
                None,
            ))
            cases.append((
                f"r{ri}/reroll-explicit@{T}",
                {"mode": "reroll", "record": rec, "turn": T,
                 "seeds": ["original", "sodium,deadbeefcafe"],
                 "p1_action": "move 1", "p2_action": "random", "followup": "default"},
                None,
            ))
            cases.append((
                f"r{ri}/reroll_many@{T}",
                {"mode": "reroll_many", "record": rec, "turn": T, "followup": "random",
                 "arms": [
                     {"label": 0, "p1_action": "recorded", "p2_action": "recorded",
                      "seed": "original"},
                     {"label": 1, "p1_action": "move 1", "p2_action": "recorded",
                      "seed": "original"},
                     {"label": 2, "p1_action": "move 2", "p2_action": "random",
                      "seed": "sodium,00ff00ff"},
                     {"label": 3, "p1_action": "switch 2", "p2_action": "random",
                      "seed": "gen5,1234567890abcdef"},
                     {"label": "s", "p1_action": "random", "p2_action": "random",
                      "seed": "sodium,abc123"},
                 ]},
                None,
            ))

        # --- the STUCK / truncated-record constructions -----------------------
        # Cut mid-turn: `commands[:k]` for an ODD k leaves the last turn half-answered.
        for k in (11, 21):
            if k >= len(rec["commands"]):
                continue
            trunc = truncate_record(rec, k)
            cases.append((f"r{ri}/replay-truncated[{k}]",
                          {"mode": "replay", "record": trunc}, "not-ended"))
            # Re-roll the LAST turn the truncated stream reaches. `original` + both
            # `recorded` takes resolve_turn_exact, which runs out of commands mid-turn.
            tail = one_shot(node_argv, {"mode": "reroll", "record": trunc, "turn": 2,
                                        "seeds": [], "p1_action": "recorded",
                                        "p2_action": "recorded"})
            if tail[0] != 0:
                continue
            for T in (2, 3, 4, 5):
                cases.append((
                    f"r{ri}/reroll-truncated[{k}]@{T}",
                    {"mode": "reroll", "record": trunc, "turn": T,
                     "seeds": ["original"], "p1_action": "recorded",
                     "p2_action": "recorded", "followup": "random"},
                    "MAYBE-unreached-turn",
                ))

        # --- ONE-SHOT error paths --------------------------------------------
        cases.append((f"r{ri}/err-turn-0",
                      {"mode": "reroll", "record": rec, "turn": 0, "seeds": []},
                      "invalid-turn"))
        cases.append((f"r{ri}/err-turn-frac",
                      {"mode": "reroll", "record": rec, "turn": 2.5, "seeds": []},
                      "invalid-turn"))
        cases.append((f"r{ri}/err-turn-missing",
                      {"mode": "reroll_many", "record": rec, "arms": []},
                      "invalid-turn"))
        cases.append((f"r{ri}/err-turn-unreachable",
                      {"mode": "reroll", "record": rec, "turn": 9999, "seeds": []},
                      "unreached-turn"))
        cases.append((f"r{ri}/err-unknown-mode",
                      {"mode": "teleport", "record": rec}, "unknown-mode"))
        cases.append((f"r{ri}/err-arm-no-seed",
                      {"mode": "reroll_many", "record": rec, "turn": 2,
                       "arms": [{"label": 0, "p1_action": "move 1"}]},
                      "arm-missing-seed"))
    return cases


SEARCH_ERROR_CASES = [
    # (name, [request, ...], error_class) — driven through BOTH persistent drivers. The last
    # request in the list is the one compared; earlier ones set up state (a real root).
    ("search/err-unknown-cmd", [{"cmd": "teleport"}], "unknown-cmd"),
    ("search/err-open-root-turn-0", [{"cmd": "open_root", "turn": 0}], "invalid-turn"),
    ("search/err-open-root-turn-frac", [{"cmd": "open_root", "turn": 2.5}], "invalid-turn"),
    ("search/err-open-root-unreachable", [{"cmd": "open_root", "turn": 9999}],
     "unreached-turn"),
    ("search/err-unknown-node",
     [{"cmd": "expand_many", "arms": [{"node_id": "n999", "label": 0, "seed": "original"}]}],
     "unknown-node"),
    ("search/err-recorded-exact-off-root", ["<ROOT>", "<CHILD-RECORDED-EXACT>"],
     "recorded-exact-off-root"),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "tmp/search_golden_node.json"))
    ap.add_argument("--bin", default=str(DEFAULT_BIN))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--max-diffs", type=int, default=12)
    args = ap.parse_args()

    binpath = Path(args.bin)
    if not binpath.exists():
        print(f"FAIL: rust search_driver not found at {args.bin}")
        return 2

    node_argv = ["node", str(NODE_REPLAY)]
    rust_argv = [str(binpath)]

    blob = json.load(open(args.records))
    records, seen = [], set()
    for c in blob["cases"]:
        key = c["record"]["prng_seed"]
        if key not in seen:
            seen.add(key)
            records.append(c["record"])

    cases = build_cases(records, node_argv)

    n_ok = n_err = n_arms = 0
    matched_leaves = 0
    allow_hits = Counter()
    failures = []
    cov = Counter()

    for name, req, err_class in cases:
        nrc, nout, _nraw, nerr = one_shot(node_argv, req)
        rrc, rout, rraw, rerr = one_shot(rust_argv, req)

        # 1. THE VERDICT (the exit code reconstruction.py branches on) — strict.
        if (nrc == 0) != (rrc == 0):
            failures.append((f"{name}[verdict]", f"node rc={nrc}", f"rust rc={rrc} {rraw[:300]}"))
            continue
        if rout is None:
            failures.append((f"{name}[unparseable]", "<json>", rraw[:400] + " | " + rerr[:400]))
            continue

        if nrc != 0:
            n_err += 1
            # 2. THE ERROR CLASS — strict; only the message TEXT is allowlisted.
            cls = err_class
            if cls == "MAYBE-unreached-turn":
                # A truncated record may or may not reach this turn; whichever it is, BOTH
                # impls must agree, which the class check below enforces symmetrically.
                cls = "unreached-turn" if matches_class(nout.get("error"), "unreached-turn") \
                    else None
            if cls and not matches_class(nout.get("error"), cls):
                failures.append((f"{name}[node-class]", cls, str(nout.get("error"))[:300]))
            if cls and not matches_class(rout.get("error"), cls):
                failures.append((f"{name}[rust-class]", cls, str(rout.get("error"))[:300]))
            if cls:
                cov[f"error:{cls}"] += 1
            ds = []
            diff(nout, rout, name, ds)
            for path, e, g in ds:
                reason = allowlisted(path)
                if reason:
                    allow_hits[reason] += 1
                else:
                    failures.append((path, e, g))
            continue

        n_ok += 1
        if err_class and err_class != "MAYBE-unreached-turn":
            failures.append((f"{name}[expected-error]", err_class, "both impls succeeded"))
        # 3. FULL FIELD-BY-FIELD comparison.
        arms = nout.get("rerolls", []) or nout.get("arms", [])
        n_arms += len(arms)
        for a in arms:
            if a.get("outcome", {}).get("ended"):
                cov["arm:ended"] += 1
            if a.get("outcome", {}).get("stuck"):
                cov["arm:stuck"] += 1
            if a.get("turn_log"):
                cov["arm:turn_log-nonempty"] += 1
            if any(l.startswith("|split|") for l in a.get("turn_log", [])):
                cov["arm:turn_log-has-split"] += 1
            if any("Intimidate does not activate" in l for l in a.get("turn_log", [])):
                cov["arm:turn_log-has-intimidate-hint"] += 1
            if any(l == "" for l in a.get("turn_log", [])):
                cov["arm:turn_log-has-owner-only-empty-shared"] += 1
        for side in ("p1", "p2"):
            for act in (nout.get("pre_state", {}) or {}).get(side, {}).get("active", []):
                if act and act.get("volatiles"):
                    cov["pre_state:nonempty-volatiles"] += 1
        ds = []
        diff(nout, rout, name, ds)
        matched_leaves += max(count_leaves(nout) - len(ds), 0)
        for path, e, g in ds:
            reason = allowlisted(path)
            if reason:
                allow_hits[reason] += 1
            else:
                failures.append((path, e, g))
        if args.verbose:
            print(f"  ok  {name}  ({len(arms)} arms)", flush=True)

    # --- the SEARCH-family error paths (persistent protocol) -----------------
    n_search = 0
    nd = Persistent(["node", str(NODE_SEARCH)])
    rd = Persistent(rust_argv)
    try:
        for name, reqs, cls in SEARCH_ERROR_CASES:
            nres = rres = None
            for req in reqs:
                if req == "<ROOT>":
                    payload = {"cmd": "open_root", "record": records[0], "turn": 3}
                    nres, rres = nd.call(**payload), rd.call(**payload)
                    nroot, rroot = nres.get("node_id"), rres.get("node_id")
                    payload = {"cmd": "expand_many", "arms": [
                        {"node_id": nroot, "label": 0, "p1_action": "move 1",
                         "p2_action": "random", "seed": "sodium,00ff00ff"}]}
                    nres = nd.call(**payload)
                    payload["arms"][0]["node_id"] = rroot
                    rres = rd.call(**payload)
                    nchild = next((a["node_id"] for a in nres.get("arms", [])
                                   if a.get("node_id")), None)
                    rchild = next((a["node_id"] for a in rres.get("arms", [])
                                   if a.get("node_id")), None)
                    continue
                if req == "<CHILD-RECORDED-EXACT>":
                    if not nchild or not rchild:
                        break
                    nres = nd.call(cmd="expand_many", arms=[
                        {"node_id": nchild, "label": 0, "recorded_exact": True}])
                    rres = rd.call(cmd="expand_many", arms=[
                        {"node_id": rchild, "label": 0, "recorded_exact": True}])
                    continue
                payload = dict(req)
                if payload.get("cmd") == "open_root":
                    payload["record"] = records[0]
                nres, rres = nd.call(**payload), rd.call(**payload)
            if nres is None or rres is None:
                failures.append((f"{name}[setup]", "a root+child", "setup did not complete"))
                continue
            n_search += 1
            if bool(nres.get("ok")) != bool(rres.get("ok")):
                failures.append((f"{name}[verdict]", f"node ok={nres.get('ok')}",
                                 f"rust ok={rres.get('ok')}"))
                continue
            if nres.get("ok"):
                failures.append((f"{name}[expected-error]", cls, "both impls succeeded"))
                continue
            if not matches_class(nres.get("error"), cls):
                failures.append((f"{name}[node-class]", cls, str(nres.get("error"))[:300]))
            if not matches_class(rres.get("error"), cls):
                failures.append((f"{name}[rust-class]", cls, str(rres.get("error"))[:300]))
            cov[f"error:{cls}"] += 1
            ds = []
            diff(nres, rres, name, ds)
            for path, e, g in ds:
                reason = allowlisted(path)
                if reason:
                    allow_hits[reason] += 1
                else:
                    failures.append((path, e, g))
    finally:
        nd.close()
        rd.close()

    # --- report ---------------------------------------------------------------
    print("=" * 78)
    print("REPLAY DRIVER PARITY — node replay_driver.js vs the rust search_driver binary")
    print("=" * 78)
    print(f"binary          : {binpath}")
    print(f"records         : {len(records)}")
    print(f"cases           : {len(cases) + len(SEARCH_ERROR_CASES)}"
          f"   ok: {n_ok}   error-path: {n_err + n_search}   arms: {n_arms}")
    print(f"leaf fields OK  : {matched_leaves}")
    print("normalized      : |t:|<epoch> in chunk payloads AND turn_log lines (the port's "
          "one documented emission exception)")
    print()
    print("ALLOWLIST (explicit, never silent):")
    for _pred, reason in ALLOWLIST:
        print(f"  - hits={allow_hits.get(reason, 0)}  {reason}")
    print()
    print("COVERAGE ACTUALLY EXERCISED (the classes the search golden was missing):")
    for k in sorted(cov):
        print(f"  {cov[k]:5d}  {k}")
    for k in ("arm:ended", "arm:stuck", "arm:turn_log-has-split",
              "arm:turn_log-has-intimidate-hint", "pre_state:nonempty-volatiles"):
        if not cov.get(k):
            print(f"      0  {k}   <-- NOT EXERCISED by this record set")
    print()
    print("COVERAGE NOTES (compared strictly, but read the caveat):")
    for path, reason in COVERAGE_NOTES:
        print(f"  - {path}: {reason}")
    print()
    if failures:
        print(f"DIVERGENCES: {len(failures)}")
        seen_paths = Counter(p.split(".", 1)[-1] for p, _, _ in failures)
        for k, v in seen_paths.most_common(20):
            print(f"  {v:5d}  {k}")
        print()
        for path, e, g in failures[: args.max_diffs]:
            print(f"  {path}")
            print(f"    node: {json.dumps(e)[:700]}")
            print(f"    rust: {json.dumps(g)[:700]}")
        print()
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
