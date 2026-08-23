"""Staleness + drift gates for the GENERATED sections of designs/ARCHITECTURE.md.

Three claims, each of which has historically failed silently as prose:
  1. The committed marker sections equal what HEAD generates (`--check` green) — the table cannot
     describe yesterday's architecture.
  2. Each head table's total equals the LIVE `in_features` — the concat parts and their order
     cannot drift from `ProjectionAssembler.forward` / `forward_internal`.
  3. `designs/production_config.json` still reflects the newest production run's
     `model_config.json` on every field BOTH carry — the committed mirror cannot quietly stop
     describing the run everything derives from. (Skipped where `models/` does not exist —
     a fresh clone / CI has no run archive. `models/` lives only in the MAIN checkout, so the
     archive is resolved with `utils.paths.main_models_dir()` and a worktree reaches across.)
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


def _newest_run_config():
    """The newest run dir (by mtime) under models/ that carries a model_config.json."""
    models_dir = main_models_dir()
    if models_dir is None:
        return None
    runs = []
    for d in os.listdir(models_dir):
        run_dir = os.path.join(models_dir, d)
        cfg_path = os.path.join(run_dir, "model_config.json")
        if os.path.exists(cfg_path):
            runs.append((os.path.getmtime(run_dir), cfg_path))
    if not runs:
        return None
    return max(runs)[1]


def test_production_config_matches_newest_run():
    """DRIFT GATE: designs/production_config.json is the newest production run's config carried
    forward to HEAD's schema (see the ARCHITECTURE.md header). Fields only ONE side has are the
    schema delta and are fine; a differing value on a field BOTH have means the mirror no longer
    reflects the production run, and everything generated from it describes a fictional config.
    """
    run_cfg_path = _newest_run_config()
    if run_cfg_path is None:
        pytest.skip(f"no run carries a model_config.json — {models_skip_reason()}")
    with open(run_cfg_path) as fh:
        run_cfg = json.load(fh)
    prod_cfg = arch_tables.load_config()

    # THE SIGNATURE-BUMP WINDOW. Two live requirements pull this file in opposite directions:
    # `extractor_compiles_test` needs it to match the LIVE code (it compiles the "production
    # arch" from it), and this test needs it to mirror the newest RUN. Both hold in the steady
    # state and CANNOT both hold between an ARCH_SIGNATURE bump and the next launch — for that
    # window every existing run is, by construction, at the previous architecture.
    #
    # Resolving it by relaxing either side would be wrong: dropping the live-code match makes the
    # compile gate compile a fiction, and dropping the run match lets the mirror rot silently.
    # So the window is DETECTED instead, from the run's own recorded signature, and the mirror
    # follows the live code while it lasts. `gen3_frame_deletion_v1` (2026-08-17) is the first
    # bump to hit this; before it, the two requirements had never been in conflict.
    if run_cfg.get("arch_signature") != ARCH_SIGNATURE:
        pytest.skip(
            f"newest run {run_cfg_path} is at arch_signature "
            f"{run_cfg.get('arch_signature')!r}, live code is {ARCH_SIGNATURE!r} — no run exists "
            f"at the current architecture yet, so production_config.json tracks the live code "
            f"until the next generation launches. This skip ENDS the moment one does.")
    shared = (set(run_cfg) & set(prod_cfg)) - {"config_version", "arch_signature"}
    diffs = {k: {"run": run_cfg[k], "production_config": prod_cfg[k]}
             for k in sorted(shared) if run_cfg[k] != prod_cfg[k]}
    assert not diffs, (
        f"designs/production_config.json no longer reflects the newest run "
        f"({run_cfg_path}) on shared fields:\n{json.dumps(diffs, indent=2)}")
