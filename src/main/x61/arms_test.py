"""X61's arms are the toy recipe from step 0, one lever apart, at one pin (``main.x61.arms``)."""
import shlex

import pytest

from main.x61 import arms as A

#: ``models/rb_toy_ent_s4002/metadata.json:original_command`` (the T-ENT toy behind T*), verbatim but for the checkout
#: prefix of its entry point.
SOURCE_COMMAND = (
    "<main>/src/main/launcher/__main__.py --restart-interval-hours 6 --pin-commit "
    "3899277ad0d2e622d9768ba13dd51ec74cd28fc3 --arch production --allow-nonproduction-arch --unified-moves off "
    "--no-damage-op --no-damage-outgoing --damage-matrices off --damage-topk 0 --move-belief-mode off "
    "--no-move-prior-fusion --no-move-latent --opp-belief-cls-k 0 --no-item-belief --no-spread-belief "
    "--no-spread-belief-nature --no-t0-species-prior --no-species-prior-fusion --entity-topk-seats 0 "
    "--no-entity-tail-seats --edge-bias-families off --no-history-events --no-value-entity-pool "
    "--no-value-entity-pool-full --move-resolution off --speed-physics off --op-reduction max --mon-hazard-cost off "
    "--trunk-layers 2 --switch-hazard-cost off --eot-residual off --move-resolution-facts off --status-facts off "
    "--ko-ramp ramp --g-ledger coarse --effective-stats off --move-set-closure off --no-intent-move-cell "
    "--no-intent-threshold --no-intent-conditional --no-pair-outcome-cell --no-pair-outcome-switch "
    "--no-switch-branch-cell --no-conditional-threat-cell --no-op-drop-renders --no-op-believed-lean "
    "--opp-belief-aux-coef 0 --opp-intent-coef 0 --move-belief-coef 0 --move-belief-latent-coef 0 "
    "--spread-belief-coef 0 --item-belief-coef 0 --hp-type-belief-coef 0 --steps 100000000 --seed 4002 "
    "--checkpoint-every-steps 1000000 --snapshot-ladder-games 0 --device cuda --run-name rb_toy_ent_s4002 "
    "--supply-starve-cycles self_play_pool=0")
PIN = "0" * 40
#: The flags (with their value count) the 2026-10-15 legacy deletions removed (``designs/deleted_flags.md`` U1 / U3):
#: each the OFF value of a damage-operator family, vacuous under ``--no-damage-op``.
DELETED_INERT = {
    "--move-resolution": 1, "--speed-physics": 1, "--op-reduction": 1, "--switch-hazard-cost": 1,
    "--eot-residual": 1, "--move-resolution-facts": 1, "--status-facts": 1, "--ko-ramp": 1, "--g-ledger": 1,
    "--effective-stats": 1, "--no-intent-move-cell": 0, "--no-intent-threshold": 0, "--no-intent-conditional": 0,
    "--no-pair-outcome-cell": 0, "--no-pair-outcome-switch": 0, "--no-switch-branch-cell": 0,
    "--no-conditional-threat-cell": 0}
_DROP = {**DELETED_INERT, "--steps": 1, "--seed": 1, "--run-name": 1, "--restart-interval-hours": 1, "--pin-commit": 1}


def toy_recipe_from(original_command):
    """The recipe from a toy run's command: the entry point, launcher flags, run tokens and deleted flags dropped."""
    toks = shlex.split(original_command)[1:]
    out, i = [], 0
    while i < len(toks):
        if toks[i] in _DROP:
            i += 1 + _DROP[toks[i]]
            continue
        out.append(toks[i])
        i += 1
    return tuple(out)


def test_the_recipe_is_the_source_toy_command_minus_the_deleted_inert_flags_and_the_run_tokens():
    assert toy_recipe_from(SOURCE_COMMAND) == A.TOY_RECIPE
    toks = shlex.split(SOURCE_COMMAND)
    assert sum(1 for t in toks if t in DELETED_INERT) == len(DELETED_INERT) == 17
    assert "--no-damage-op" in A.TOY_RECIPE, "every deleted flag is inert only because the damage op is off"


def test_the_source_command_is_the_committed_runs_when_models_is_present():
    from utils.paths import main_models_dir

    models = main_models_dir()
    meta = models / A.TOY_SOURCE_RUN / "metadata.json" if models else None
    if meta is None or not meta.exists():
        pytest.skip("models/ is only in the main checkout")
    import json

    assert json.loads(meta.read_text())["original_command"].split("--restart-interval-hours")[1] == \
        SOURCE_COMMAND.split("--restart-interval-hours")[1]


def test_the_two_arms_are_one_lever_apart_at_the_same_pin_and_seed():
    only_n1, only_c0 = A.lever_diff(6101, PIN)
    assert only_n1 == list(A.N1_LEVER) and only_c0 == []
    c0 = A.build_argv("toy_c0f", 6101, PIN)
    assert c0[:4] == ["--restart-interval-hours", "6", "--pin-commit", PIN]
    assert c0[c0.index("--steps") + 1] == "100000000" and c0[c0.index("--seed") + 1] == "6101"
    assert c0[c0.index("--run-name") + 1] == "rb_x61t_c0f_s6101"
    assert A.build_argv("toy_n1", 6102, PIN)[-4:] == ["--team-mix", "n1_proc_v1@ALL:0.7", "--team-mix-sides", "both"]
    assert "--team-mix" not in c0 and "--model" not in c0, "a FRESH run, the pool only"
    assert c0[c0.index("--behaviour-check") + 1] == "warn", "C3: toy arms run K9(b) as warn"


def test_an_unknown_arm_or_seed_or_a_missing_pin_is_refused(monkeypatch):
    with pytest.raises(A.ArmError):
        A.build_argv("toy_n4", 6101, PIN)
    with pytest.raises(A.ArmError):
        A.build_argv("toy_n1", 4002, PIN)
    monkeypatch.setattr(A, "PIN", None)
    with pytest.raises(A.ArmError, match="pin"):
        A.build_argv("toy_n1", 6101)


def test_the_lever_names_a_registered_corpus_and_the_parser_accepts_its_share():
    from agents.training.matchup_spec import parse_team_mix
    from utils.team_corpora import CORPORA

    assert A.CORPUS in CORPORA
    corpus, tier, share = parse_team_mix(A.N1_LEVER[1])
    assert corpus == A.CORPUS and tier == A.CORPUS_TIER and abs(share - 0.7) < 1e-12


def test_the_corpus_check_loads_this_trees_corpus():
    from utils.paths import repo_root

    ident = A.check_corpus_at(str(repo_root()))
    assert ident["corpus"] == A.CORPUS and ident["n"] >= 10_000
