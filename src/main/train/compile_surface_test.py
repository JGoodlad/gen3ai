"""The shared compile cache keys a run on the values it COMPILES AT (`main.train.compile_surface`).

The defect (B1 quit probe, 2026-11-12): the key read the ARGV's `--batch-size`, which is inert on a `--model`
launch (SB3 restores the checkpoint's), so a fresh B = 2048 start and its own restart — the same graph — got
two keys (`1bdf57369a43` / `70105b399543`, stamps differing in `fields.run.compile_args.batch_size` alone)
and the first restart compiled cold.
"""
from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest

from agents.model import compile_cache as CC
from agents.model import compile_cache_shared as SH
from main.train import compile_surface as CS
from main.train import lifecycle
from main.train.parser import build_parser


def _zip(path, **data):
    """A checkpoint as far as the key reads one: SB3's plain-JSON `data` member."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps(data))
    return str(path)


class _Resolved(SimpleNamespace):                    # `resolve_config` fills every toggle before the key is read
    def __getattr__(self, name):
        return 0.0 if name.endswith("_coef") else None


def _args(*argv):
    ns = vars(build_parser().parse_args(list(argv)))
    return _Resolved(**{k: (0.0 if v is None and k.endswith("_coef") else v) for k, v in ns.items()})


def _declared(monkeypatch, args, run_dir):
    """The `run` surface `_declare_compile_cache` hands `prepare_run_cache` (None = this run shares nothing)."""
    seen = []
    monkeypatch.setattr(CC, "prepare_run_cache", lambda _dir, **kw: seen.append(kw["run"]))
    lifecycle._declare_compile_cache(args, str(run_dir))
    assert len(seen) == 1
    return seen[0]


def test_the_effective_keys_are_exactly_the_keyed_flags_that_are_inert_on_a_resume():
    """The CLASS, not one field: a flag that is in the key AND restored from the checkpoint must be resolved."""
    from main.train.recipe_surface import INERT_ON_RESUME
    assert set(SH.EFFECTIVE_COMPILE_ARG_KEYS) == set(SH.COMPILE_ARG_KEYS) & set(INERT_ON_RESUME)


def test_a_fresh_start_and_its_resume_share_ONE_key_when_only_argv_defaults_differ(tmp_path, monkeypatch):
    default_b = build_parser().get_default("batch_size")
    assert default_b != 2048                         # else the resume's argv would agree by accident
    ckpt = _zip(tmp_path / "checkpoint_1000192_steps.zip", batch_size=2048, n_steps=384)
    fresh = _declared(monkeypatch, _args("--batch-size", "2048"), tmp_path / "a")
    resumed_args = _args("--model", ckpt)
    assert resumed_args.batch_size == default_b      # the argv a launcher restart really carries
    resume = _declared(monkeypatch, resumed_args, tmp_path / "b")
    assert fresh["compile_args"]["batch_size"] == resume["compile_args"]["batch_size"] == 2048
    assert fresh == resume and SH.key_of(fresh) == SH.key_of(resume)


def test_a_different_EFFECTIVE_batch_size_is_a_different_key(tmp_path, monkeypatch):
    """Never a merge: two checkpoints at different micro-batches under the SAME argv get two keys, and a typed
    (inert) `--batch-size` does not move a resume's key off its checkpoint's."""
    a = _declared(monkeypatch, _args("--model", _zip(tmp_path / "a.zip", batch_size=2048)), tmp_path / "ra")
    b = _declared(monkeypatch, _args("--model", _zip(tmp_path / "b.zip", batch_size=1024)), tmp_path / "rb")
    assert a["compile_args"]["batch_size"] == 2048 and b["compile_args"]["batch_size"] == 1024
    assert SH.key_of(a) != SH.key_of(b)
    typed = _declared(monkeypatch, _args("--model", str(tmp_path / "a.zip"), "--batch-size", "512"), tmp_path / "rt")
    assert typed == a
    fresh = _declared(monkeypatch, _args("--batch-size", "1024"), tmp_path / "rf")
    assert SH.key_of(fresh) == SH.key_of(b) != SH.key_of(a)


@pytest.mark.parametrize("data", [{}, {"batch_size": 0}, {"batch_size": "2048"}, {"batch_size": True}, None])
def test_an_unreadable_checkpoint_shares_NOTHING(tmp_path, monkeypatch, capsys, data):
    """A value the key cannot read is never guessed: the run falls back to its own cache (`run=None`)."""
    path = tmp_path / "c.zip"
    if data is None:
        path.write_bytes(b"not a zip")
    else:
        _zip(path, **data)
    assert _declared(monkeypatch, _args("--model", str(path)), tmp_path / "r") is None
    err = capsys.readouterr().err
    assert "will not share a compile cache" in err and ("ValueError" in err or "BadZipFile" in err), err
    _zip(path, batch_size=2048)                      # the same launch with a readable checkpoint DOES share
    assert _declared(monkeypatch, _args("--model", str(path)), tmp_path / "r")["compile_args"]["batch_size"] == 2048
    assert _declared(monkeypatch, _args("--model", str(tmp_path / "absent.zip")), tmp_path / "r2") is None


def test_run_fields_REFUSES_a_model_launch_without_the_effective_values(tmp_path):
    """The guard behind the fix: no caller can key a `--model` launch on its argv again."""
    args = _args("--model", _zip(tmp_path / "c.zip", batch_size=2048))
    with pytest.raises(ValueError, match="inert"):
        SH.run_fields(args)
    with pytest.raises(ValueError, match="not an effective compile key"):
        SH.run_fields(args, {"batch_size": 2048, "device": "cpu"})
    assert SH.run_fields(args, CS.effective_compile_args(args))["compile_args"]["batch_size"] == 2048
    assert SH.run_fields(_args())["compile_args"]["batch_size"] == build_parser().get_default("batch_size")


def test_a_capped_fresh_batch_keys_at_the_cap_its_checkpoint_will_hold(tmp_path, monkeypatch):
    """`model_build` caps a fresh micro-batch at the rollout capacity AFTER the key is declared; the key applies the
    same rule (`fresh_batch_size`), so that run's checkpoint — which holds the cap — resumes onto the same key."""
    assert CS.fresh_batch_size(batch_size=4096, n_steps=8, n_envs=4) == 32
    assert CS.fresh_batch_size(batch_size=2048, n_steps=384, n_envs=256) == 2048
    fresh = _declared(monkeypatch, _args("--batch-size", "4096", "--n-steps", "8", "--n-envs", "4"), tmp_path / "f")
    assert fresh["compile_args"]["batch_size"] == 32
    debug = _declared(monkeypatch, _args("--batch-size", "4096", "--n-steps", "8", "--n-envs", "4", "--debug"),
                      tmp_path / "d")
    assert debug["compile_args"]["batch_size"] == 8          # --debug runs ONE env
    resume = _declared(monkeypatch, _args("--model", _zip(tmp_path / "c.zip", batch_size=32), "--n-steps", "8",
                                          "--n-envs", "4"), tmp_path / "r")
    assert SH.key_of(resume) == SH.key_of(fresh)
    import inspect
    from main.train import model_build
    assert "fresh_batch_size(" in inspect.getsource(model_build.build_and_train)


def test_the_goldens_lookup_is_the_loaders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "goldens" / "g1").mkdir(parents=True)
    _zip(tmp_path / "models" / "goldens" / "g1" / "final_model.zip", batch_size=768)
    assert CS.resolve_model_arg("g1") == "models/goldens/g1"
    assert CS.restored_compile_args("g1") == {"batch_size": 768}
    assert CS.resolve_model_arg("nowhere") == "nowhere" and CS.resolve_model_arg("x.zip") == "x.zip"


# ── the `--compile-trainer` AUTO default judges the EFFECTIVE update shape, not the argv's ─────────────────────────

def _auto(args):
    from main.train.config import _auto_compile_trainer
    return _auto_compile_trainer(args)


def _cuda_resume(ckpt, **argv):
    """A `--model` resume on an explicit cuda device (the auto default's device half is then ON without a card);
    `n_envs` 16, no rollout target, so an update is n_steps * 16 rows. The argv's `n_steps` / `batch_size` are the
    parser defaults unless typed: INERT on a resume."""
    a = _args("--model", ckpt, "--device", "cuda", "--n-envs", "16",
              *[x for k, v in argv.items() for x in (f"--{k.replace('_', '-')}", str(v))])
    a.rollout_target_samples = 0
    return a


def test_auto_compile_on_a_resume_judges_the_checkpoints_shape_ON_where_the_argvs_would_refuse(tmp_path):
    """The run trained compiled at n_steps 128 x 16 envs = 2048 rows, batch 1024 (divides). The argv typed a
    batch of 1536 (inert on a resume): judged on the argv, the auto default turned the compile OFF."""
    ckpt = _zip(tmp_path / "checkpoint_1000_steps.zip", batch_size=1024, n_steps=128)
    args = _cuda_resume(ckpt, n_steps=128, batch_size=1536)
    assert (128 * 16) % 1536 != 0                    # the argv's shape is refused ...
    assert _auto(args) == (True, None)               # ... the learner's is not


def test_auto_compile_on_a_resume_judges_the_checkpoints_shape_OFF_where_the_argvs_would_pass(tmp_path):
    """The reverse: the checkpoint's batch 1536 does not divide its 2048-row update, the argv's 1024 would."""
    ckpt = _zip(tmp_path / "checkpoint_1000_steps.zip", batch_size=1536, n_steps=128)
    enabled, why = _auto(_cuda_resume(ckpt, n_steps=128, batch_size=1024))
    assert enabled is False and "1536" in why, why


def test_auto_compile_reads_the_checkpoints_n_steps_too(tmp_path):
    ckpt = _zip(tmp_path / "checkpoint_1000_steps.zip", batch_size=1024, n_steps=64)     # 64 x 16 = 1024 rows
    assert _auto(_cuda_resume(ckpt, n_steps=96, batch_size=1024)) == (True, None)      # argv: 1536 rows, refused


def test_control_a_fresh_launch_still_judges_its_argv():
    assert _auto(_cuda_resume_fresh(n_steps=128, batch_size=1024)) == (True, None)
    assert _auto(_cuda_resume_fresh(n_steps=128, batch_size=1536))[0] is False


def _cuda_resume_fresh(**argv):
    a = _args("--device", "cuda", "--n-envs", "16",
              *[x for k, v in argv.items() for x in (f"--{k.replace('_', '-')}", str(v))])
    a.rollout_target_samples = 0
    return a


def test_an_unreadable_checkpoint_falls_back_to_the_argv_and_says_so(tmp_path, capsys):
    bad = tmp_path / "checkpoint_1000_steps.zip"
    bad.write_text("not a zip")
    assert _auto(_cuda_resume(str(bad), n_steps=128, batch_size=1024)) == (True, None)
    assert "could not read --model's n_steps / batch_size" in capsys.readouterr().out
