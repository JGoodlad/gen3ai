"""Replay the NODE search-driver golden against the RUST `search_driver` and diff.

Scratch tool (tmp/, not shipped). The golden is `tmp/search_golden_node.json`, captured
from `src/utils/bridge/search_driver.js` by `tmp/search_golden.py`; this drives the Rust
binary over the IDENTICAL request sequence and compares every field of every response.

    export PYTHONPATH=$PYTHONPATH:src
    python src/rust_sim/harness/search_impl_parity.py [--golden PATH] [--bin PATH] [-v]

Deliberately talks raw stdin/stdout JSON — NOT through `SearchSession` — so it stays
valid while the Python seam is being reworked by another agent.

NORMALIZATION (the ONLY thing silently forgiven):
  * `|t:|<epoch>` lines inside chunk payloads → `|t:|<N>`. This is the port's one
    documented emission exception (`src/rust_sim/src/protocol.rs`: wall-clock stamps,
    which poke-env ignores). Nothing else is normalized.

Everything else that cannot match must appear in ALLOWLIST below, is PRINTED on every
run with its reason, and is counted separately from matches.
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
T_LINE = re.compile(r"^\|t:\|.*$", re.M)

# ---------------------------------------------------------------------------
# The EXPLICIT allowlist. Each entry: (path predicate, reason). A field that
# diverges and matches an entry is reported as ALLOWLISTED, never as a match.
# ---------------------------------------------------------------------------
ALLOWLIST = [
    (
        lambda p: p.endswith(".error"),
        "error TEXT only: Node returns `e.stack` (a JS stack trace), the port a plain "
        "message. The ok/ok:false VERDICT is still compared strictly.",
    ),
]


# Fields the port renders from state it models only approximately. These are NOT
# forgiven (a divergence still fails); they are printed every run so a green result is
# not mistaken for evidence they are right.
COVERAGE_NOTES = [
    (
        "pre_state.*.active[].volatiles",
        "The port keeps volatiles as typed fields, not a keyed map, so the NAME list is "
        "reconstructed (search.rs::volatile_names). This golden covers exactly ONE fact "
        "about it — that the port's duration-1 single-turn fields (focuspunch/pursuit/"
        "protect/flinch/beatup/counter/endure/snatch) must NOT be reported at a move "
        "boundary, since the port clears them at the next turn's top where Showdown "
        "removes them at the residual (including focuspunch diverged 2 of 12 pre_states). "
        "Every OTHER name (choicelock, perishsong, twoturnmove, ...) is UNVERIFIED: all 12 "
        "pre_states here end up with an EMPTY list.",
    ),
    (
        "pre_state.pseudoWeather",
        "The port has no pseudo-weather map; the list is derived from the format's clause "
        "flag (BattleState::sleep_clause). gen-3 has no pseudo-weather MOVE, so the clause "
        "rules are the only real entries.",
    ),
]


def norm_chunk(s):
    return T_LINE.sub("|t:|<N>", s) if isinstance(s, str) and "|t:|" in s else s


class RustDriver:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def call(self, payload):
        self.p.stdin.write(json.dumps(payload) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("rust driver died: " + self.p.stderr.read()[:4000])
        return json.loads(line), line

    def close(self):
        try:
            self.call({"id": -1, "cmd": "close"})
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ---------------------------------------------------------------------------
# Deep diff
# ---------------------------------------------------------------------------

def diff(exp, got, path, out):
    """Append (path, expected, got) for every leaf that differs."""
    if isinstance(exp, str) or isinstance(got, str):
        exp, got = norm_chunk(exp), norm_chunk(got)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "tmp/search_golden_node.json"))
    ap.add_argument("--bin", default=str(DEFAULT_BIN))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--max-diffs", type=int, default=12)
    args = ap.parse_args()

    binpath = Path(args.bin)
    if not binpath.exists():
        alt = Path("/tmp/pokesim_target_searchdrv/release/search_driver")
        if alt.exists():
            binpath = alt
        else:
            print(f"FAIL: rust search_driver not found at {args.bin}")
            return 2

    cases = json.load(open(args.golden))["cases"]

    # The golden was captured with ONE node driver per BATTLE (several turns each), and
    # node ids are monotonic for a driver's lifetime — so the rust process must be
    # grouped the same way or every `node_id` would legitimately differ.
    groups, order = {}, []
    for c in cases:
        key = c["record"]["prng_seed"] + "|" + str(len(c["record"]["commands"]))
        if key not in order:
            order.append(key)
            groups[key] = []
        groups[key].append(c)

    n_cases = n_calls = n_arms = 0
    matched_leaves = 0
    allow_hits = Counter()
    failures = []

    for key in order:
        drv = RustDriver(binpath)
        try:
            for c in groups[key]:
                n_cases += 1
                calls = [("open_root", {"cmd": "open_root", "record": c["record"],
                                        "turn": c["turn"]}, c["open_root"])]
                exp_arms = c["expand_many"].get("arms", [])
                arms_req = []
                for a in exp_arms:
                    # Reconstruct the request the golden was produced from. The golden
                    # records the RESPONSE, so the arm SPEC is rebuilt from tmp/
                    # search_golden.py's `arms_for` shape, keyed by the echoed label.
                    arms_req.append(arm_spec(c, a))
                calls.append(("expand_many",
                              {"cmd": "expand_many", "arms": arms_req}, c["expand_many"]))
                d2 = c.get("expand_depth2")
                if d2:
                    child = next((a["node_id"] for a in exp_arms if a.get("node_id")), None)
                    calls.append(("expand_many(d2)",
                                  {"cmd": "expand_many", "arms": [
                                      {"node_id": child, "label": 100,
                                       "p1_action": "move 1", "p2_action": "random",
                                       "seed": "sodium,abc123"}]}, d2))
                for name, payload, expected in calls:
                    n_calls += 1
                    payload = dict(payload, id=expected.get("id"))
                    got, _raw = drv.call(payload)
                    n_arms += len(expected.get("arms", []))
                    ds = []
                    diff(expected, got, f"{name}@turn{c['turn']}", ds)
                    matched_leaves += max(count_leaves(expected) - len(ds), 0)
                    for path, e, g in ds:
                        reason = allowlisted(path)
                        if reason:
                            allow_hits[reason] += 1
                        else:
                            failures.append((path, e, g))
        finally:
            drv.close()

    print("=" * 78)
    print("SEARCH DRIVER PARITY — node golden vs rust binary")
    print("=" * 78)
    print(f"binary          : {binpath}")
    print(f"cases           : {n_cases}   requests: {n_calls}   arms: {n_arms}")
    print(f"leaf fields OK  : {matched_leaves}")
    print(f"normalized      : |t:|<epoch> inside chunk payloads (the port's one "
          f"documented emission exception)")
    print()
    print("ALLOWLIST (explicit, never silent):")
    for _pred, reason in ALLOWLIST:
        print(f"  - hits={allow_hits.get(reason, 0)}  {reason}")
    print()
    print("COVERAGE NOTES (matched here, but the golden does not exercise them):")
    for path, reason in COVERAGE_NOTES:
        print(f"  - {path}: {reason}")
    print()
    if failures:
        print(f"DIVERGENCES: {len(failures)}")
        seen = Counter(p.split("@")[0] + "…" + p.split(".", 1)[-1] for p, _, _ in failures)
        for k, v in seen.most_common(20):
            print(f"  {v:5d}  {k}")
        print()
        for path, e, g in failures[: args.max_diffs]:
            print(f"  {path}")
            print(f"    node: {json.dumps(e)[:600]}")
            print(f"    rust: {json.dumps(g)[:600]}")
        print()
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


def arm_spec(case, arm_response):
    """Rebuild the arm REQUEST for a recorded arm RESPONSE (tmp/search_golden.py::arms_for)."""
    root = case["open_root"]
    node_id = root["node_id"]
    rec_p2 = root["recorded_choices"].get("p2")
    label = arm_response["label"]
    if label == 0:
        return {"node_id": node_id, "label": 0, "recorded_exact": True}
    if 1 <= label <= 4:
        mv = ["move 1", "move 2", "move 3", "switch 2"][label - 1]
        return {"node_id": node_id, "label": label, "p1_action": mv,
                "p2_action": rec_p2 or "random", "seed": "original"}
    if label in (10, 11):
        seed = ["sodium,0011223344556677", "gen5,1234567890abcdef"][label - 10]
        return {"node_id": node_id, "label": label, "p1_action": "move 1",
                "p2_action": "random", "seed": seed}
    if label == 20:
        return {"node_id": node_id, "label": 20, "p1_action": "random",
                "p2_action": "random", "seed": "sodium,deadbeefcafe", "followup": "random"}
    if label == 21:
        return {"node_id": node_id, "label": 21, "p1_action": "move 1",
                "p2_action": "random", "seed": "sodium,00ff00ff", "followup": "default"}
    raise SystemExit(f"unknown golden arm label {label!r}")


if __name__ == "__main__":
    sys.exit(main())
