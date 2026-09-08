"""Does `designs/ARCHITECTURE.md`'s PROSE still agree with the production mirror?

Sits beside `ruff_gate_test.py`, `file_size_gate_test.py` and `claude_md_freshness_gate_test.py`
at the `src/` root because, like those, its subject is a whole-tree artifact rather than any one
package.

**Why it exists.** `ARCHITECTURE.md` carries the always-current obligation and is read as fact —
it is the first thing the root `CLAUDE.md` sends an agent to before it reasons about the model. On
2026-09-07 it said belief heads run `label_only` while its OWN generated flag table, the
production mirror and the live arm all said `shaping` (hand-fixed in `22e757db`). Nothing caught
that, because the two halves of the document are gated differently: §6's flag table is GENERATED
from `designs/production_config.json` and pinned by `arch_tables_test.py`, while every sentence
around it is hand-written and pinned by nothing. A reader cannot tell the halves apart, so the
hand-written half inherits the generated half's authority without inheriting its gate. That is the
gap this closes: **a MODE-flag value stated in prose is compared against the mirror, key by key.**

**The extraction is DECLARED, never inferred.** `_CLAIMS` below is an explicit table of
(config key → the regex that finds the doc's statement of its value). Inferring "a backticked
identifier followed by a bolded token is a value claim" would be a parser for English, and it
would fail in the one direction that matters: it would go QUIET on a renamed key and report
nothing. A declared row instead FAILS when its pattern stops matching, so a rename is loud.

**Four checks.**

1. **Every declared claim's value equals the mirror's.** The mirror is read through
   `agents.training.baselines.production_config()` — the registry accessor — never a hardcoded
   path, because a second opinion about which file is "production" is the defect the registry
   exists to remove.
2. **Every declared key exists in the mirror**, and its pattern matches the doc at least once.
   Together these are the rename tripwire: a key renamed in the config fails (1st half), a
   sentence rewritten past its pattern fails (2nd half).
3. **The required MODE flags are all covered.** `_REQUIRED_COVERAGE` pins the set the backlog row
   names, so a row cannot be quietly deleted to make a failure go away.
4. **Every key §6's generated table marks `INERT` is called INERT wherever the prose names it.**
   `INERT` means "recorded, but does nothing given another setting" — exactly the value a reader
   will otherwise act on. The mirror decides which keys those are (the generated table is derived
   from it); the prose only has to agree.

**What is NOT checked, deliberately.** The generated blocks themselves — `arch_tables_test.py`
owns those, and re-deriving them here would be a second generator to keep in sync. And every claim
that has no referent in the mirror: a measured dependence, a design rationale, a dimension. The
subject is the config keys the mirror actually holds.

**Cost: milliseconds** — one markdown read, one JSON read, ~40 regex searches. No cost marker; it
runs in the fast inner loop. Opt out explicitly:

    GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import os
import pathlib
import re
from typing import Any, Dict, List, NamedTuple, Tuple

import pytest

from agents.training.baselines import production_config
from utils.paths import repo_path

_SKIP = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_MODE_FLAG_DOC_GATE") == "1",
    reason="GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1",
)

_DOC = "designs/ARCHITECTURE.md"


# --------------------------------------------------------------------------------------------
# The DECLARED table
# --------------------------------------------------------------------------------------------

class Claim(NamedTuple):
    """One sentence in `ARCHITECTURE.md` that states a config key's production value.

    `pattern` is searched against the doc's HAND-WRITTEN text (the generated blocks are stripped
    first) and must capture the stated value in a group named `value`. Every match is checked, not
    just the first — the same key is stated in several places and they have drifted apart from
    each other before.
    """
    key: str
    pattern: str
    where: str          # a human pointer to the sentence, for the failure message


# NOTE ON WRITING A ROW: anchor on the words around the value, not on the value itself. A pattern
# that hardcodes the expected value can only ever match when the doc is already right, which makes
# the check vacuous in exactly the way `designs/learning/vacuous_tests_and_guards.md` catalogues.
_CLAIMS: Tuple[Claim, ...] = (
    # ---- the four MODE strings ------------------------------------------------------------
    Claim("critic",
          r"\|\s*\*\*`(?P<value>[a-z]+)`\*\*\s*\(\*\*this config\*\*\)",
          "§3.4 the --critic table, the row marked (**this config**)"),
    Claim("belief_grad_mode",
          r"Belief heads run under `belief_grad_mode`\s*\*\*`(?P<value>\w+)`\*\*",
          "§3.4 'Belief heads run under ...'"),
    Claim("belief_grad_mode",
          r"production mirror `belief_grad_mode:\s*\"(?P<value>\w+)\"`",
          "§3.4 the parenthetical citing the mirror"),
    Claim("opp_intent_grad_mode",
          r"\(`opp_intent_grad_mode:\s*\"(?P<value>\w+)\"`\)",
          "§3.4 'The opponent-INTENT head is the exception'"),
    Claim("hp_belief_mode",
          r"`hp_belief_mode`\s*=\s*`\"(?P<value>\w+)\"`",
          "§3.2 the HPTypeBelief step"),
    Claim("hp_belief_mode",
          r"`hp_belief_mode\s*==\s*(?P<value>\w+)`",
          "§7 the hp_type_label row's emit condition"),
    Claim("win_prob_mode",
          r"`win_head`\s*\|\s*`win_prob_mode`\s*\*\*`(?P<value>\w+)`\*\*",
          "§3.4 the side-readout table"),
    Claim("q_winprob_mode",
          r"`q_winprob_mode`\s*`\"(?P<value>\w+)\"`\s*/\s*OFF",
          "§3.4 'the LATENT readout' closing sentence"),
    Claim("move_belief_mode",
          r"`move_belief_mode`\s*=\s*`\"(?P<value>\w+)\"`",
          "§3.2 the MoveBelief step"),

    # ---- the critic family: what the mode implies and refuses ------------------------------
    Claim("use_popart",
          r"\(`use_popart`\s*(?P<value>true|false),\s*refused here\)",
          "§3.4 the --critic table's PopArt column"),
    Claim("value_dist_mode",
          r"`value_dist_head` is not built here\*\*\s*\(`value_dist_mode`\s*`(?P<value>\w+)`",
          "§3.4 'value_dist_head is not built here'"),
    Claim("value_dist_bins",
          r"`value_dist_head` is not built here\*\*[^)]*`value_dist_bins`\s*(?P<value>\d+)\)",
          "§3.4 'value_dist_head is not built here'"),
    Claim("value_dist_coef",
          r"`value_dist_coef`\s*stays recorded at\s*(?P<value>[\d.]+)",
          "§3.4 'value_dist_coef stays recorded at ...'"),
    Claim("vf_coef",
          r"BCE against the terminal WIN INDICATOR\*\*, at `vf_coef`\s*\*\*(?P<value>[\d.]+)\*\*",
          "§3.4 the --critic table's winprob row"),
    Claim("win_prob_coef",
          r"`win_prob_mode`\s*\*\*`\w+`\*\*,\s*`win_prob_coef`\s*\*\*(?P<value>[\d.]+)\*\*",
          "§3.4 the side-readout table"),
    Claim("victory_value",
          r"`\+victory_value`\s*\(\*\*(?P<value>[\d.]+)\*\*\)\s*on a win",
          "§3.4 the --critic table's reward-stream column"),

    # ---- §6.3 the reward block (resume-immutable) ------------------------------------------
    Claim("hand_shaping",
          r"`hand_shaping`\s*\*\*(?P<value>true|false)\*\*",
          "§6.3 'The production reward is ONE TERMINAL TERM'"),
    Claim("terminal_indicator",
          r"`terminal_indicator`\s*\*\*(?P<value>true|false)\*\*",
          "§6.3 same sentence"),
    Claim("victory_value",
          r"`victory_value`\s*\*\*(?P<value>[\d.]+)\*\*",
          "§6.3 same sentence"),
    Claim("draw_penalty",
          r"`draw_penalty`\s*\*\*(?P<value>-?[\d.]+)\*\*",
          "§6.3 same sentence"),
    Claim("no_progress_tax_armed",
          r"`no_progress_tax_armed`\s*\*\*(?P<value>true|false)\*\*",
          "§6.3 same sentence"),
    Claim("all_shaping_pbrs",
          r"`all_shaping_pbrs`\s*\*\*(?P<value>true|false)\*\*,\s*`pbrs_material`",
          "§6.3 the three INERT reward fields"),
    Claim("pbrs_material",
          r"`pbrs_material`\s*\*\*(?P<value>true|false)\*\*",
          "§6.3 the three INERT reward fields"),
    Claim("pbrs_belief",
          r"`pbrs_belief`\s*\*\*(?P<value>true|false)\*\*",
          "§6.3 the three INERT reward fields"),
    # The recorded-and-inert DEFAULTS list. These are the values a resume must re-pass, so a
    # wrong one here is a FATAL nobody can debug from the document.
    Claim("no_progress_penalty",
          r"`no_progress_penalty`\s*(?P<value>[\d.]+)\s*·",
          "§6.3 the recorded defaults list"),
    Claim("mat_alive_weight",
          r"`mat_alive_weight`\s*(?P<value>[\d.]+)", "§6.3 the recorded defaults list"),
    Claim("bias_additivity",
          r"`bias_additivity`\s*(?P<value>[\d.]+)", "§6.3 the recorded defaults list"),
    Claim("self_ko_hp_penalty",
          r"`self_ko_hp_penalty`\s*(?P<value>[\d.]+)", "§6.3 the recorded defaults list"),
    Claim("switch_bias_weight",
          r"`switch_bias_weight`\s*(?P<value>[\d.]+)", "§6.3 the recorded defaults list"),
    Claim("bias_redesign",
          r"`bias_redesign`\s*(?P<value>true|false)\s*·", "§6.3 the recorded defaults list"),
    Claim("drop_redundant_bias",
          r"`drop_redundant_bias`\s*(?P<value>true|false)", "§6.3 the recorded defaults list"),
    Claim("drop_switch_bias",
          r"`drop_switch_bias`\s*(?P<value>true|false)", "§6.3 the recorded defaults list"),
    Claim("stall_pbrs",
          r"`stall_pbrs`\s*(?P<value>true|false)\.", "§6.3 the recorded defaults list"),

    # ---- head on/off coefficients (§7's emit-condition table states each one) ---------------
    Claim("opp_belief_aux_coef",
          r"\(`opp_belief_aux_coef`\s*(?P<value>[\d.]+)\)", "§7 the belief_species row"),
    Claim("move_belief_coef",
          r"\(`move_belief_coef`\s*(?P<value>[\d.]+)\)", "§7 the known_moves row"),
    Claim("item_belief_coef",
          r"\(`item_belief_coef`\s*(?P<value>[\d.]+)\)", "§7 the item_label row"),
    Claim("spread_belief_nature",
          r"\(`spread_belief_nature`\s*(?P<value>true|false)\)", "§7 the belief_nature row"),

    # ---- structural toggles the prose states ON / OFF / ABSENT ------------------------------
    Claim("value_entity_pool",
          r"\*\*(?P<value>ON|OFF) in production: `value_entity_pool`\*\*", "§3.3 critic routes"),
    Claim("value_entity_pool_full",
          r"\*\*`value_entity_pool_full`\*\*\s*\(v82,\s*\*\*(?P<value>ON|OFF)\*\*\)",
          "§3.3 critic routes"),
    Claim("intent_threshold",
          r"\*\*(?P<value>ON|OFF) in production: `intent_threshold`\*\*", "§3.2 pointer cells"),
    Claim("pair_outcome_cell",
          r"\*\*LIVE \((?P<value>ON|OFF) in the gen-17 base\): `pair_outcome_cell`\*\*",
          "§3.2 pointer cells"),
    Claim("pair_outcome_switch",
          r"\*\*LIVE \((?P<value>ON|OFF) in the gen-17 base\): `pair_outcome_switch`\*\*",
          "§3.2 pointer cells"),
    Claim("switch_branch_cell",
          r"\*\*LIVE \((?P<value>ON|OFF) in the gen-17 base\): `switch_branch_cell`\*\*",
          "§3.2 pointer cells"),
    Claim("conditional_threat_cell",
          r"\*\*LIVE \((?P<value>ON|OFF) in the gen-17 base\): `conditional_threat_cell`\*\*",
          "§3.2 pointer cells"),
    Claim("pair_value_route",
          r"\*\*Available but (?P<value>ON|OFF): `pair_value_route`\*\*", "§3.3 critic routes"),

    # ---- the op block's gates (§4) ----------------------------------------------------------
    Claim("op_drop_renders",
          r"`op_drop_renders`\s*=\s*\*\*(?P<value>true|false)\*\*",
          "§4 the op output-block table"),
    Claim("damage_matrices_outgoing",
          r"`damage_matrices_outgoing`\s*=\s*(?P<value>true|false)\b",
          "§4 the op output-block table, sub-block 5"),
    Claim("damage_matrices_incoming",
          r"`damage_matrices_incoming`\s*=\s*(?P<value>true|false)\b",
          "§4 the op output-block table, sub-block 6"),
    Claim("damage_topk_k",
          r"K\s*=\s*`damage_topk_k`\s*=\s*(?P<value>\d+)", "§4 the op output-block table"),

    # ---- the seat / family counts --------------------------------------------------------
    Claim("entity_topk_seats",
          r"K\s*=\s*`entity_topk_seats`\s*=\s*(?P<value>\d+)", "§2.3 the E4 seat row"),
    Claim("edge_bias_families",
          r"`edge_bias_families\s*=\s*\"(?P<value>[a-z0-9,]+)\"`", "§5 the opening sentence"),

    # ---- identity ---------------------------------------------------------------------------
    Claim("config_version",
          r"`config_version`\s*\*\*(?P<value>\d+)\*\*", "§0 the Production run row"),
    Claim("arch_signature",
          r"`arch_signature`\s*\*\*`(?P<value>[a-z0-9_]+)`\*\*", "§0 the Production run row"),
    Claim("total_dim",
          r"One flat `float32` vector of \*\*(?P<value>\d+)\*\* dims", "§1 the observation"),
)

# The MODE flags the backlog row names. A row above may be rewritten, but one of these keys may
# not simply stop being covered — that is how a gate becomes decorative.
_REQUIRED_COVERAGE = frozenset({
    "belief_grad_mode", "opp_intent_grad_mode", "critic", "hp_belief_mode",
    "hand_shaping", "terminal_indicator", "win_prob_mode", "move_belief_mode",
    "value_dist_mode", "q_winprob_mode",
    # each head's on/off coefficient
    "win_prob_coef", "vf_coef", "opp_belief_aux_coef", "move_belief_coef",
    "item_belief_coef", "value_dist_coef",
})


# --------------------------------------------------------------------------------------------
# Reading the document
# --------------------------------------------------------------------------------------------

_GENERATED_BLOCK = re.compile(
    r"<!--\s*BEGIN GENERATED:\s*(?P<name>[\w-]+)\s*-->(?P<body>.*?)<!--\s*END GENERATED:\s*\1\s*-->",
    re.DOTALL)


def doc_text() -> str:
    return pathlib.Path(repo_path("designs", "ARCHITECTURE.md")).read_text()


def prose_of(text: str) -> str:
    """The HAND-WRITTEN half: the document with every generated block blanked out.

    Blanked rather than deleted, so a reported offset still lines up with the file.
    """
    def _blank(m: "re.Match[str]") -> str:
        return "\n" * m.group(0).count("\n")
    return _GENERATED_BLOCK.sub(_blank, text)


def generated_block(text: str, name: str) -> str:
    for m in _GENERATED_BLOCK.finditer(text):
        if m.group("name") == name:
            return m.group("body")
    return ""


_TRUEY = {"true", "on", "live", "active", "yes"}
_FALSEY = {"false", "off", "no", "absent"}


def parse_value(token: str) -> Any:
    """A doc token → the Python value a `model_config.json` field would hold."""
    t = token.strip().strip("`*\"' ").replace("−", "-")   # U+2212 MINUS SIGN
    low = t.lower()
    if low in _TRUEY:
        return True
    if low in _FALSEY:
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def values_agree(doc_value: Any, cfg_value: Any) -> bool:
    if isinstance(doc_value, bool) or isinstance(cfg_value, bool):
        return doc_value is cfg_value
    if isinstance(doc_value, (int, float)) and isinstance(cfg_value, (int, float)):
        return float(doc_value) == float(cfg_value)
    return doc_value == cfg_value


class Violation(NamedTuple):
    key: str
    where: str
    doc_value: Any
    cfg_value: Any
    detail: str

    def line(self) -> str:
        return (f"  {self.key:26s} doc says {self.doc_value!r:14s} "
                f"mirror says {self.cfg_value!r:14s} — {self.where}"
                + (f"\n      {self.detail}" if self.detail else ""))


def check_claims(text: str, cfg: Dict[str, Any],
                 claims: Tuple[Claim, ...] = _CLAIMS) -> List[Violation]:
    """The whole comparison, over an arbitrary document + mirror.

    Factored out so the planted-contradiction test can run it on a MUTATED COPY of the real
    document rather than on a fixture that could drift away from the real one, and without ever
    writing to the tree.
    """
    prose = prose_of(text)
    out: List[Violation] = []
    for c in claims:
        if c.key not in cfg:
            out.append(Violation(c.key, c.where, None, None,
                                 "this key is not in the production mirror at all — renamed?"))
            continue
        matches = list(re.finditer(c.pattern, prose))
        if not matches:
            out.append(Violation(c.key, c.where, None, cfg[c.key],
                                 "the declared pattern matched NOTHING — the sentence was "
                                 "rewritten or removed; re-anchor the row or delete it"))
            continue
        for m in matches:
            got = parse_value(m.group("value"))
            if not values_agree(got, cfg[c.key]):
                out.append(Violation(c.key, c.where, got, cfg[c.key], ""))
    return out


def paragraphs(text: str) -> List[Tuple[int, str]]:
    """(1-based start line, block) for every blank-line-separated block."""
    out, start, buf = [], 1, []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not buf:
                start = lineno
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, "\n".join(buf)))
    return out


def inert_keys(text: str) -> List[str]:
    """The keys §6's GENERATED flag table marks `INERT` — i.e. what the mirror says is inert."""
    out = []
    for line in generated_block(text, "flag-table").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 4 and cells[3].startswith("INERT"):
            out.append(cells[1].strip("`"))
    return out


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------

@_SKIP
def test_architecture_prose_agrees_with_the_production_mirror():
    violations = check_claims(doc_text(), production_config())
    assert not violations, (
        "`designs/ARCHITECTURE.md`'s hand-written prose disagrees with the production mirror "
        "(`agents.training.baselines.production_config()`). The doc is loaded and read as FACT; "
        "a wrong mode value there is acted on.\n\n"
        + "\n".join(v.line() for v in violations)
        + "\n\nFix the DOC (state the new truth; never narrate the change inline). If the mirror "
          "is what moved, refresh it top-down per `designs/production_config.README.md` and "
          "regenerate §6 with `python -m agents.model.arch_tables`."
    )


@_SKIP
def test_every_required_mode_flag_is_covered():
    covered = {c.key for c in _CLAIMS}
    missing = sorted(_REQUIRED_COVERAGE - covered)
    assert not missing, (
        "These MODE flags have no row in `_CLAIMS`, so the gate says nothing about them:\n  "
        + "\n  ".join(missing)
        + "\n\nThe backlog row that ordered this gate names them. Add a row anchored on the "
          "sentence that states the value — do not shrink the required set."
    )


@_SKIP
def test_the_mirror_declares_inert_keys_and_the_prose_agrees():
    """A key the mirror marks INERT must be CALLED inert wherever the prose names it.

    `INERT` is the status a reader most needs and is least able to derive: the field is recorded
    `true`, and it does nothing. §6.3's three PBRS fields are the live example — `all_shaping_pbrs`
    reads `true` in `model_config.json` and emits no term.
    """
    text = doc_text()
    keys = inert_keys(text)
    assert keys, (
        "§6's generated flag table lists no INERT rows. Either the table's format changed (this "
        "parser reads `| `key` | value | STATUS |`) or the generator stopped emitting the status "
        "— both make this check silently vacuous. Regenerate with "
        "`python -m agents.model.arch_tables`."
    )

    # PARAGRAPH granularity, not line: this prose hard-wraps, so "`value_dist_coef`\nstays
    # recorded at 1.0 and §6 marks it `INERT`" is one claim across two lines and a line-wise check
    # would report it as a violation. A paragraph is the smallest unit that is always a whole
    # claim.
    bad = []
    for key in keys:
        for start, para in paragraphs(prose_of(text)):
            if re.search(r"`" + re.escape(key) + r"`", para) and "INERT" not in para:
                bad.append(f"  {key:24s} ARCHITECTURE.md:{start}  {para.strip()[:150]}")
    assert not bad, (
        "The mirror marks these keys INERT, but the prose names them without saying so — a "
        "reader takes the recorded value for the effective one:\n\n"
        + "\n".join(bad)
        + "\n\nSay INERT on the line, or move the mention into a sentence that does."
    )


@_SKIP
def test_a_planted_contradiction_FAILS_the_gate():
    """The gate's own reachability check — it must fail on a doc that lies.

    Runs against a MUTATED COPY of the real document held in memory: a fixture would drift away
    from the file the gate actually reads, and editing the real file would leave the tree dirty on
    a failure. Each planted flip is applied to the sentence the corresponding `_CLAIMS` row
    anchors on, so this proves the ROW fires, not merely that the comparison works.
    """
    text = doc_text()
    cfg = production_config()
    assert not check_claims(text, cfg), "the real document must be clean before planting"

    plants: Tuple[Tuple[str, str, str], ...] = (
        # (key, original substring, the lie)
        ("belief_grad_mode",
         "Belief heads run under `belief_grad_mode` **`shaping`**",
         "Belief heads run under `belief_grad_mode` **`label_only`**"),
        ("hand_shaping", "`hand_shaping` **false**", "`hand_shaping` **true**"),
        ("critic", "| **`winprob`** (**this config**)", "| **`shaped`** (**this config**)"),
        ("vf_coef", "at `vf_coef` **0.5**", "at `vf_coef` **1.0**"),
    )
    for key, original, lie in plants:
        assert original in text, (
            f"the planted-contradiction test cannot find its anchor for {key!r}:\n  {original}\n"
            "The sentence moved — re-anchor this plant AND the matching `_CLAIMS` row, or the "
            "proof that the gate can fail is itself vacuous."
        )
        found = check_claims(text.replace(original, lie), cfg)
        assert any(v.key == key and v.doc_value is not None for v in found), (
            f"planting a contradiction on {key!r} did NOT fail the gate — its `_CLAIMS` row "
            f"cannot fire, which makes it decorative.\n  planted: {lie}\n  found: {found}"
        )

    # And the rename tripwire: a key that vanishes from the mirror is reported, not skipped.
    stripped = {k: v for k, v in cfg.items() if k != "critic"}
    renamed = check_claims(text, stripped)
    assert any(v.key == "critic" for v in renamed), (
        "a config key missing from the mirror was silently skipped — the rename tripwire is dead"
    )
