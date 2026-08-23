"""Unit tests for the capacity-battery CLI (`main.capacity`) — no checkpoint, no model.

Everything model-shaped is covered by `agents/model/capacity_probes_test.py`; what is left here
is the part that decides WHICH checkpoint gets measured, whether the artifact is strict JSON, and
whether the table renders — the three ways this command can be wrong without failing.

The checkpoint-resolution tests are the ones that earn their keep: a run root can carry several
ARMS (`legA_final_model.zip`, `legB_final_model.zip`, `final_model.zip`) and a battery that
silently picked the wrong one would emit a correct-looking artifact about a model nobody asked
about — the same shape as the audit-sampler bug that drew every state from one step dir while
labelling itself a pool average.
"""
import json
import os

import pytest

from main.capacity import VALIDITY, build_parser, render, resolve_checkpoint


# --------------------------------------------------------------------------- resolution

def test_an_explicit_zip_is_taken_verbatim(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("final_model.zip", "legA_final_model.zip", "legB_final_model.zip"):
        (run / name).write_bytes(b"")
    ckpt, run_dir = resolve_checkpoint(str(run / "legB_final_model.zip"))
    assert os.path.basename(ckpt) == "legB_final_model.zip"
    assert run_dir == str(run)


def test_a_missing_zip_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "nope.zip"))


def test_a_run_dir_follows_latest_txt(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint_99_steps.zip").write_bytes(b"")
    (run / "final_model.zip").write_bytes(b"")
    (run / "latest.txt").write_text("checkpoints/checkpoint_99_steps.zip\n")
    ckpt, run_dir = resolve_checkpoint(str(run))
    assert ckpt.endswith("checkpoint_99_steps.zip")
    assert run_dir == str(run)


def test_a_stale_latest_txt_falls_through_to_final_model(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.txt").write_text("checkpoints/gone.zip\n")
    (run / "final_model.zip").write_bytes(b"")
    ckpt, _ = resolve_checkpoint(str(run))
    assert ckpt.endswith("final_model.zip")


def test_a_run_dir_with_only_checkpoints_takes_the_newest(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    old, new = run / "checkpoints" / "a.zip", run / "checkpoints" / "b.zip"
    old.write_bytes(b"")
    new.write_bytes(b"")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    ckpt, _ = resolve_checkpoint(str(run))
    assert ckpt.endswith("b.zip")


def test_an_empty_run_dir_raises_with_the_fix_in_the_message(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="Name the .zip explicitly"):
        resolve_checkpoint(str(run))


def test_a_nonexistent_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "absent"))


# --------------------------------------------------------------------------- parser

def test_parser_defaults():
    args = build_parser().parse_args(["models/x"])
    assert args.target == "models/x"
    assert args.max_states == 3000 and args.n_targets == 8 and args.folds == 5
    assert args.seed == 0 and args.device == "cpu" and args.out is None


def test_every_help_string_renders():
    """The `checkargs` lesson: one unescaped `%` in a help string makes `--help` raise, and
    nothing else in the tree ever renders them."""
    build_parser().format_help()


# --------------------------------------------------------------------------- report

def _synthetic_report():
    taps = ["role_tokens", "team_tokens", "value_pooled", "pi_features", "vf_features"]
    rank = {t: {"pr": 3.5, "srank99": 40.0, "effrank": 9.0, "n90": 5.0, "n95": 7.0,
                "dim": 128, "n_rows": 1200, "pr_frac": 0.027} for t in taps}
    return {
        "battery_version": 1,
        "meta": {"run_dir": "/models/run_x", "run_name": "run_x", "checkpoint": "/m/final.zip",
                 "arch_signature": "sig_v1", "num_timesteps": 1234567, "n_states": 100,
                 "seed": 0, "sampling": {"n_files_read": 4, "per_step": {"step_1": 100},
                                         "per_opponent": {"random": 100}}},
        "rank": {"trained": rank, "fresh": rank},
        "trainability": {"n_targets": 8, "folds": 5, "seed": 0, "target_family": "f",
                         "taps": {t: {"r2_trained": 0.4, "r2_fresh": 0.5, "nmse_trained": 0.6,
                                      "nmse_fresh": 0.5, "capacity_ratio": None,
                                      "per_target_r2_trained": [0.4] * 8} for t in taps}},
        "decodability": {"skipped": {"opp_alive": "degenerate target: zero variance"},
                         "facts": {"our_active_hp": {
                             "task": "regression", "note": "n", "target_std": 0.3,
                             "taps": {t: {"trained": 0.8, "fresh": None, "l2": 10.0}
                                      for t in taps}}}},
        "params": {"n_params_total": 42, "phases": {"projection": {
            "n_params": 42, "param_share": 1.0, "l2_norm": 1.0, "rms": 0.1, "zero_frac": 0.0}}},
        "validity": VALIDITY,
    }


def test_render_covers_every_section_and_the_tripwire_line():
    text = render(_synthetic_report())
    for needle in ("CAPACITY BATTERY v1", "run_x", "1,234,567",
                   "(a) REPRESENTATION EFFECTIVE RANK", "(b) TRAINABILITY",
                   "(c) PROBE DECODABILITY", "(d) PARAMETER CENSUS",
                   "TRIPWIRE, NOT VERDICT"):
        assert needle in text, needle


def test_render_shows_a_skipped_fact_with_its_reason():
    text = render(_synthetic_report())
    assert "opp_alive" in text and "degenerate target" in text


def test_render_tolerates_missing_numbers():
    """`None` (a non-finite value the sanitizer nulled) must print as a dash, not crash the run
    that just spent minutes producing it."""
    text = render(_synthetic_report())
    assert "-" in text


def test_report_is_strict_json_round_trippable(tmp_path):
    from agents.model.capacity_probes import jsonable

    clean = jsonable(_synthetic_report())
    path = tmp_path / "capacity_battery.json"
    with open(path, "w") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)
    assert json.loads(path.read_text()) == clean


def test_validity_carries_all_three_clauses_for_every_metric():
    """The non-negotiable discipline: every metric ships what movement MEANS, what PAIRED
    behavioural evidence would confirm it, and the explicit no-verdict line."""
    assert set(VALIDITY) == {"rank", "trainability", "decodability", "params"}
    for name, note in VALIDITY.items():
        assert set(note) == {"movement", "confirm", "no_verdict"}, name
        assert "no kill/build decision" in note["no_verdict"].lower(), name
        assert len(note["movement"]) > 40 and len(note["confirm"]) > 40, name
