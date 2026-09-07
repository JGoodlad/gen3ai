"""Staleness + drift gates for the GENERATED sections of designs/ARCHITECTURE.md.

Three claims, each of which has historically failed silently as prose:
  1. The committed marker sections equal what HEAD generates (`--check` green) — the table cannot
     describe yesterday's architecture.
  2. Each head table's total equals the LIVE `in_features` — the concat parts and their order
     cannot drift from `ProjectionAssembler.forward` / `forward_internal`.
  3. `designs/production_config.json` matches the CONSTRUCTION the BASELINE REGISTRY declares
     for its `production` entry (`designs/baselines.json`, `gen3_baselines_registry_v1`) — the
     committed mirror cannot quietly stop describing what it claims to describe. This replaced a
     NEWEST-RUN heuristic on 2026-09-06; the test's own docstring records why that heuristic
     failed in both directions. (Skipped where `models/` does not exist — a fresh clone / CI has
     no run archive. `models/` lives only in the MAIN checkout, so the archive is resolved with
     `utils.paths.main_models_dir()` and a worktree reaches across.)
"""
import json
import os

import pytest

from agents.model.model_version import ARCH_SIGNATURE

from agents.model import arch_tables
from utils.paths import main_models_dir, models_skip_reason


@pytest.fixture(scope="module")
def fe_and_cfg():
    """One extractor build shared by every test here — construction is the expensive part."""
    return arch_tables.build_extractor()


def _arch_md_text() -> str:
    with open(arch_tables._ARCH_MD) as fh:
        return fh.read()


def test_marker_pairs_exist_exactly_once():
    text = _arch_md_text()
    for name in arch_tables.SECTIONS:
        begin, end = arch_tables._begin(name), arch_tables._end(name)
        assert text.count(begin) == 1, f"{begin!r} must appear exactly once"
        assert text.count(end) == 1, f"{end!r} must appear exactly once"
        # extract_section validates ordering (BEGIN before END) by construction.
        arch_tables.extract_section(text, name)


def test_committed_sections_match_generated(fe_and_cfg):
    """The `--check` gate: committed ARCHITECTURE.md == what HEAD generates, section by section.

    Regenerated inline from the shared fixture rather than via `main(["--check"])` so a failure
    names the drifted section instead of just returning 1 (and so the extractor is built once
    for the whole module, not once per test).
    """
    fe, cfg = fe_and_cfg
    text = _arch_md_text()
    sections = {
        "modules": arch_tables.modules_section(fe),
        "head-inputs": arch_tables.head_inputs_section(fe),
        "flag-table": arch_tables.flag_table_section(fe, cfg),
    }
    stale = [name for name in arch_tables.SECTIONS
             if arch_tables.extract_section(text, name) != sections[name].strip("\n")]
    assert not stale, (
        f"generated section(s) {stale} in designs/ARCHITECTURE.md are STALE — regenerate with "
        "`python -m agents.model.arch_tables` in the same commit as the architecture change")


def test_head_totals_equal_live_in_features(fe_and_cfg):
    fe, _ = fe_and_cfg
    pi, vf = arch_tables.head_input_parts(fe)
    assert sum(d for _, d, _ in pi) == fe.projection.in_features
    assert sum(d for _, d, _ in vf) == fe.value_projection.in_features


def test_production_config_matches_the_registry():
    """DRIFT GATE: `designs/production_config.json` matches the CONSTRUCTION the BASELINE REGISTRY
    declares for its `production` entry (`designs/baselines.json`, `gen3_baselines_registry_v1`).

    🚨 **This REPLACES `test_production_config_matches_newest_run`, which answered a question
    nobody asked.** That test compared the mirror against whichever run directory in `models/` had
    the newest mtime — a heuristic that cannot tell a production generation from a two-hour
    ablation arm, and that on 2026-09-06 would have been pointed at a MIS-LAUNCHED arm: an argv
    copied out of a design-doc block, carrying the critic block and the hyperparameters and
    nothing else, so 31 architecture fields had silently reverted to their OFF defaults. It
    therefore failed in both directions — green when the mirror described a fiction, red whenever
    any research arm trained last.

    Which run is production is a JUDGEMENT, so it is DECLARED (a name, a checkpoint, a ledger
    entry) rather than inferred, and the mirror is checked against that declaration. Since
    2026-09-06 the declaration is a CONSTRUCTION rather than a copy — gen-17's surface migrated
    v97 → v109 with a 13-key critic override block — and `baselines.compare_production` checks all
    three of its parts (surface equality, the declared migration's key delta, every override
    present at its declared value and actually differing). The full statement of what it checks
    and why lives there; `src/main/baselines_test.py` owns the unit coverage of each clause.

    Kept HERE as well as there because this file is the one a reader opens when asking "what
    guards the mirror?" — and because a mirror that stops describing a real config makes
    ARCHITECTURE.md, the delivery graph and `extractor_compiles_test`'s "production arch" all
    describe a fiction with a straight face.
    """
    from agents.training import baselines
    models_dir = main_models_dir()
    if models_dir is None:
        pytest.skip(models_skip_reason())
    b = baselines.get("production")
    run_cfg_path = os.path.join(str(models_dir), b.run, "model_config.json")
    if not os.path.exists(run_cfg_path):
        # Names the escape hatch, like every other archive skip in this tree: a contributor
        # reading it must not go hunting for a broken test (utils/paths_test.py pins this).
        pytest.skip(f"the `production` baseline's run {b.run} is not on this box "
                    f"({run_cfg_path}) — {models_skip_reason()}")
    with open(run_cfg_path) as fh:
        run_cfg = json.load(fh)
    problems = baselines.compare_production(run_cfg, arch_tables.load_config(), b)
    assert not problems, (
        "designs/production_config.json no longer matches the construction "
        f"designs/baselines.json declares for `production` ({b.run}):\n\n"
        + "\n\n".join(problems))


def test_the_registry_and_the_live_code_agree_on_the_arch_signature():
    """THE SIGNATURE-BUMP WINDOW, restated for the registry.

    Two live requirements pull the mirror in opposite directions: `extractor_compiles_test` needs
    it to match the LIVE code (it compiles the "production arch" from it) and the drift gate needs
    it to describe a real run. Both hold in the steady state and CANNOT both hold between an
    `ARCH_SIGNATURE` bump and the next launch, because for that window every existing run is by
    construction at the previous architecture. Relaxing either side would be wrong — dropping the
    live-code match makes the compile gate compile a fiction, dropping the run match lets the
    mirror rot silently — so the window is DETECTED and REPORTED rather than papered over.
    """
    from agents.training import baselines
    b = baselines.get("production")
    if b.arch_signature != ARCH_SIGNATURE:
        pytest.skip(
            f"the `production` baseline ({b.run}) records arch_signature {b.arch_signature!r} "
            f"and the live code is {ARCH_SIGNATURE!r} — a SIGNATURE-BUMP WINDOW. "
            "designs/production_config.json tracks the live code until a run exists at the new "
            "signature; re-point the baseline with `python -m main.baselines set production "
            "<run>/<ckpt> --reason \"<ledger entry title>\"` when one does. This skip ENDS then.")
    assert arch_tables.load_config().get("arch_signature") == ARCH_SIGNATURE
