"""No `set -e` script under `scripts/` puts a COMPOUND GROUP on the left of `||` / `&&`.

Bash ignores `set -e` inside any command that is the left side of `||` / `&&` (and inside `if` / `!`
conditions), so `( ruff; mypy; pytest; echo OK ) || handler` runs the body to its end and the group's status
is the LAST command's. `scripts/land.sh`'s gates were vacuous that way from creation until 21257d79
(2026-10-04). Audit 2026-10-06 of every shell script under `scripts/`: only `bootstrap.sh`, `land.sh` and the
three `workstation/*tunnel*|get_proxy_ip` scripts use `set -e`; none has the shape today (the ops scripts run
with `set -u` and check each status explicitly). This gate keeps it that way: a `( ... ) ||`, `{ ...; } ||`
or `( ... ) &&` group in an errexit script fails with the line, so the author writes `cmd || fail` per command.
"""
import re
from pathlib import Path

from utils.paths import repo_path

_ERREXIT = re.compile(r"^\s*set\s+-[A-Za-z]*e[A-Za-z]*\b", re.M)
# a line that CLOSES a group and chains: `) ||`, `} &&` (but not `$( … )` / `$(( … ))` / `${…}` expansions)
_CLOSE_CHAIN = re.compile(r"^\s*[)}]\s*(\|\||&&)")
# a one-line group chained: `( a; b ) || c`, `{ a; b; } && c`
_ONE_LINE = re.compile(r"^\s*(\([^()$]*\)|\{[^{}$]*;\s*\})\s*(\|\||&&)")


def errexit_group_violations(text: str) -> list:
    """`(lineno, line)` for each compound group chained with `||` / `&&` in a script that uses `set -e`."""
    if not _ERREXIT.search(text):
        return []
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if _CLOSE_CHAIN.match(line) or _ONE_LINE.match(line):
            out.append((n, line.strip()))
    return out


def test_the_detector_flags_the_land_sh_shape_and_ignores_safe_code():
    bad_multi = "set -Eeuo pipefail\n(\n  ruff check\n  pytest\n  echo OK\n) || { echo FAILED; exit 1; }\n"
    bad_one = "set -e\n( ruff check; pytest ) || exit 1\n"
    bad_brace = "set -euo pipefail\n{ ruff check; pytest; } && echo ok\n"
    assert errexit_group_violations(bad_multi) == [(6, ") || { echo FAILED; exit 1; }")]
    assert [n for n, _ in errexit_group_violations(bad_one)] == [2]
    assert [n for n, _ in errexit_group_violations(bad_brace)] == [2]
    safe = ('set -Eeuo pipefail\nX="$(cd "$d" && pwd)"\nruff check || fail ruff\n'
            'pytest -q || { echo F; exit 1; }\nn=$(( a + b )) && echo "$n"\n# ( a; b ) || c\n')
    assert errexit_group_violations(safe) == []
    # a script WITHOUT errexit (the ops scripts run `set -u`) is not this bug class
    assert errexit_group_violations("set -u\n( a; b ) || c\n") == []


def test_no_errexit_script_under_scripts_chains_a_compound_group():
    bad = {}
    for p in sorted(Path(repo_path("scripts")).rglob("*.sh")):
        v = errexit_group_violations(p.read_text())
        if v:
            bad[str(p.relative_to(repo_path()))] = v
    assert not bad, (
        "a `( … ) ||` / `{ …; } ||` group in a `set -e` script never stops at a failing command inside it "
        f"(bash ignores errexit on the left of ||/&&): check each command explicitly. {bad}")
