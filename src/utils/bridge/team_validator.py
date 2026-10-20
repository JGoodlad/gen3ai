import copy
import functools
import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Tuple

#: The per-process verdict memo (``Teambuilder.__init__`` re-validated the SAME pool in every construction, ~0.5 s of
#: Node start-up each: ``designs/ops/TECH_DEBT_BACKLOG.md``). Keyed by (validator version, format, team text), so a
#: Showdown bump (the submodule commit) or a rebuilt ``dist`` cannot serve a stale verdict. ONLY verdicts Node
#: actually returned are stored: an infrastructure failure (no script, Node error, no output) is never cached.
#: Values are stored and handed out DEEP-COPIED, so a caller that edits a result cannot poison the memo.
_MEMO: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
_MEMO_LOCK = threading.Lock()  # ws_frontend validates from ``asyncio.to_thread``


@functools.lru_cache(maxsize=1)
def validator_version() -> str:
    """What the verdicts depend on: the pinned Showdown commit, the built validator module and the script itself."""
    here = os.path.dirname(os.path.abspath(__file__))
    ps = os.path.abspath(os.path.join(here, '..', '..', '..', 'deps', 'pokemon-showdown'))
    try:
        commit = subprocess.run(['git', '-C', ps, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ''
    parts = [commit or 'no-commit']
    for path in (os.path.join(ps, 'dist', 'sim', 'team-validator.js'), os.path.join(here, 'validate_team.js')):
        try:
            st = os.stat(path)
            parts.append(f"{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append('absent')
    return '|'.join(parts)


def clear_memo() -> None:
    """Forget every memoised verdict (a test, or a caller that changed ``deps/`` mid-process)."""
    with _MEMO_LOCK:
        _MEMO.clear()


def validate_team_locally(format_id: str, team_text: str) -> Dict[str, Any]:
    """
    Validates a Pokémon team using the local Node.js Showdown environment.
    """
    results = validate_teams_locally(format_id, [team_text])
    return results[0]

def validate_teams_locally(format_id: str, teams: List[str]) -> List[Dict[str, Any]]:
    """
    Validates multiple Pokémon teams in a single batch using the local Node.js Showdown environment.

    Memoised per process by (validator version, format, team text): a team already judged is answered from the memo
    and only the MISSES go to one Node process (none when all hit). The semantics are the Node script's own.

    Args:
        format_id: The tier/format (e.g., 'gen3ou', 'gen9ou')
        teams: List of team texts in Showdown/PokePaste export format

    Returns:
        A list of dictionaries, each with 'valid' (bool) and 'errors' (List[str])
    """
    version = validator_version()
    keys = [(version, format_id, t) for t in teams]
    results: List[Any] = [None] * len(teams)
    with _MEMO_LOCK:
        for i, k in enumerate(keys):
            if k in _MEMO:
                results[i] = copy.deepcopy(_MEMO[k])
    # De-duplicate the misses (a pool may hold one text twice): one verdict per distinct text.
    misses = list(dict.fromkeys(teams[i] for i, r in enumerate(results) if r is None))
    if not misses:
        return results
    fresh, cacheable = _run_node(format_id, misses)
    by_text = dict(zip(misses, fresh))
    if cacheable:
        with _MEMO_LOCK:
            for text, verdict in by_text.items():
                _MEMO[(version, format_id, text)] = copy.deepcopy(verdict)
    for i, r in enumerate(results):
        if r is None:
            results[i] = copy.deepcopy(by_text[teams[i]])
    return results


def _run_node(format_id: str, teams: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    """One Node process over ``teams``: (results, whether they are real verdicts that may be memoised)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    node_script_path = os.path.join(current_dir, 'validate_team.js')

    if not os.path.exists(node_script_path):
        return [{"valid": False, "errors": [f"Cannot find the Node bridge script at: {node_script_path}"]} for _ in teams], False

    payload = {
        "format": format_id,
        "team": teams
    }

    try:
        process = subprocess.Popen(
            ['node', node_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=json.dumps(payload))

        if stderr:
            if "Error" in stderr or "cannot find module" in stderr.lower():
                return [{"valid": False, "errors": [f"Node execution error: {stderr}"]} for _ in teams], False

        if not stdout:
            return [{"valid": False, "errors": ["No output from Node script"]} for _ in teams], False

        out = json.loads(stdout)
        # A real answer is one verdict dict per team. Anything else (the script's own whole-request error is a
        # single dict) becomes one invalid result per team, never memoised.
        if not (isinstance(out, list) and len(out) == len(teams) and all(
                isinstance(v, dict) and 'valid' in v for v in out)):
            return [{"valid": False, "errors": [f"Unexpected Node output: {stdout.strip()[:500]}"]} for _ in teams], False
        return out, True
    except Exception as e:
        return [{"valid": False, "errors": [f"Subprocess failure: {str(e)}"]} for _ in teams], False


if __name__ == "__main__":
    # Quick test
    test_format = "gen3ou"
    test_team = """
Tyranitar @ Leftovers  
Ability: Sand Stream  
EVs: 252 HP / 4 Atk / 252 SpA  
Quiet Nature  
- Rock Slide  
- Crunch  
- Pursuit  
- Fire Blast  
    """
    
    result = validate_team_locally(test_format, test_team)
    print(f"Result: {result}")
