"""A grep guard: ``agents.gen3_data.hidden_power`` is THE one Python Hidden Power table and formula (owner
2026-10-19: "only 1 impl for the HP determinization"), and ``src/rust_sim/src/state.rs`` the engine's one Rust copy
of ``getHiddenPower`` (pinned by ``hidden_power_test.py`` and ``tests/hidden_power_fill_test.rs``). A second
type -> IVs table, IVs -> type / power formula, or formula-order type list anywhere else under ``src/`` FAILS here:
import the module instead. (JavaScript harnesses are dev oracles outside this guard; ``ou_random_teams.js`` keeps its
own solver because its output is the frozen P72 panel's provenance.)"""
import re

from utils.paths import src_root

HOMES = {"agents/gen3_data/hidden_power.py", "rust_sim/src/state.rs"}
SELF = "agents/gen3_data/hidden_power_single_impl_test.py"
PATTERNS = {
    "the type formula (x * 15 / 63)": re.compile(r"\*\s*15\s*//?\s*63"),
    "the power formula (x * 40 / 63)": re.compile(r"\*\s*40\s*//?\s*63"),
    "the formula-order type list": re.compile(r"""["']fighting["']\s*,\s*["']flying["']\s*,\s*["']poison["']""", re.I),
    "an HP IV table literal": re.compile(r"^\s*(?:pub\s+)?(?:const\s+|static\s+)?\w*HP_?IVS\w*\s*(?::[^=\n]*)?=\s*[\{\[]",
                                         re.I | re.M),
    "a type -> 30-IV row": re.compile(
        r"""["'](?:fighting|flying|poison|ground|rock|bug|ghost|steel|fire|water|grass|electric|psychic|ice|dragon|dark)"""
        r"""["']\s*:\s*[\{\(\[][^\n]*\b30\b""", re.I),
}


def _hits():
    root = src_root()
    out = []
    for path in sorted(list(root.rglob("*.py")) + list(root.rglob("*.rs"))):
        rel = path.relative_to(root).as_posix()
        if rel in HOMES or rel == SELF or "/target/" in f"/{rel}":
            continue
        text = path.read_text(errors="replace")
        for what, rx in PATTERNS.items():
            for m in rx.finditer(text):
                out.append(f"src/{rel}:{text.count(chr(10), 0, m.start()) + 1}: {what}")
    return out


def test_no_second_hidden_power_table_or_formula_in_src():
    hits = _hits()
    assert not hits, "a second Hidden Power impl (import agents.gen3_data.hidden_power):\n" + "\n".join(hits)


def test_the_guard_sees_a_planted_copy(tmp_path, monkeypatch):
    # The guard is not vacuous: each pattern fires on the shape the deleted copies had.
    planted = {
        "the type formula (x * 15 / 63)": "return HP_TYPES[v * 15 // 63]",
        "the power formula (x * 40 / 63)": "bp = (x * 40 / 63) + 30",
        "the formula-order type list": 'HP_TYPES = ("fighting", "flying", "poison", "ground")',
        "an HP IV table literal": "GEN3_HP_IVS = {\n",
        "a type -> 30-IV row": '    "Ghost": {"hp": 31, "atk": 31, "def": 30},',
    }
    for what, line in planted.items():
        assert PATTERNS[what].search(line), what
