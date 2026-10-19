"""``--team-mix`` in the MatchupSpec, the Rust collector's team table / stager and the eval ledger's team-set id
(``gen3_team_mix_v1``; designs/training/team_curriculum.md "The team mix").

Pure unit tests: a tiny corpus in ``tmp_path`` (``team_mix_testkit``), the Node validator mocked to all-valid (the
team strings are real gen3ou exports, so parse / pack run for real). Each test names the revert it fails on.
"""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from agents.training.matchup_spec import (
    MatchupSpec, TeamSource, parse_team_mix, validate_exploiter_trainee_is_sample)
from agents.training.team_mix_testkit import TEST_CORPUS, clear_corpus_cache, install_corpus
from utils.teambuilder_test import TEAM_A, TEAM_B

#: The spec hashes of the two flagless matchups, recorded BEFORE `--team-mix` existed (origin/main 8989c137).
#: `fb823b7177` is X26's own record (`models/rb_x26_s2601/metadata.json` `cli_args._matchup_spec_hash`).
NO_FLAG_HASH = "7cdbc69447"
NO_FLAG_SELF_PLAY_HASH = "fb823b7177"


@pytest.fixture(autouse=True)
def _fresh_corpus_cache():
    clear_corpus_cache()
    yield
    clear_corpus_cache()


@pytest.fixture
def all_valid():
    with mock.patch("utils.bridge.team_validator.validate_teams_locally",
                    side_effect=lambda fmt, ts: [{"valid": True} for _ in ts]):
        yield


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    return install_corpus(monkeypatch, tmp_path, [TEAM_B, TEAM_A], tiers={"A": 1, "AB": 2})


def _args(**kw):
    base = dict(trainee_team=None, trainee_teams=None, exploiter=None, self_play=False,
                bot_weights=None, exploiter_keep_bots=False, exploiter_bot_fraction=0.5,
                stable_opponent_temp=1.0)
    base.update(kw)
    return SimpleNamespace(**base)


# ── byte-identical when absent ─────────────────────────────────────────────────────────────────────────────────

def test_no_flag_spec_hash_is_pinned_to_its_pre_mix_value():
    """The flagless matchups hash exactly as they did before the flag existed (X26's record included). Fails on
    revert to writing a ``mix`` key (even ``None``) into every source's dict."""
    assert MatchupSpec.from_args(_args()).spec_hash() == NO_FLAG_HASH
    assert MatchupSpec.from_args(_args(self_play=True)).spec_hash() == NO_FLAG_SELF_PLAY_HASH
    assert MatchupSpec.from_args(_args(team_mix=None, team_mix_sides=None)).spec_hash() == NO_FLAG_HASH
    d = MatchupSpec.from_args(_args()).to_dict()
    assert all("mix" not in d[k] for k in ("trainee_teams", "opponent_teams", "eval_trainee_teams"))


# ── the flag's syntax ──────────────────────────────────────────────────────────────────────────────────────────

def test_parse_team_mix_accepts_the_documented_forms(corpus):
    assert parse_team_mix(f"{TEST_CORPUS}:0.5") == (TEST_CORPUS, "", 0.5)
    assert parse_team_mix(f"{TEST_CORPUS}@AB:0.25") == (TEST_CORPUS, "AB", 0.25)
    assert parse_team_mix(f"{TEST_CORPUS}@A:1") == (TEST_CORPUS, "A", 1.0)
    assert parse_team_mix("w_forum_v1@ABC:0.5") == ("w_forum_v1", "ABC", 0.5)   # the registered real corpus


@pytest.mark.parametrize("bad, why", [
    ("nope_v9:0.5", "unknown team corpus"),
    (f"{TEST_CORPUS}:0", r"in \(0, 1\]"),
    (f"{TEST_CORPUS}:1.5", r"in \(0, 1\]"),
    (f"{TEST_CORPUS}:nan", r"in \(0, 1\]"),
    (f"{TEST_CORPUS}:-0.1", r"in \(0, 1\]"),
    (f"{TEST_CORPUS}:half", "not a number"),
    (f"{TEST_CORPUS}", "expected <corpus>"),
    (f"{TEST_CORPUS}@:0.5", "expected <corpus>"),
    (f"{TEST_CORPUS}@A@B:0.5", "expected <corpus>"),
])
def test_parse_team_mix_refuses_bad_values(corpus, bad, why):
    with pytest.raises(ValueError, match=why):
        parse_team_mix(bad)


# ── the spec: sides, eval, provenance ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sides, on_trainee, on_opponent", [
    (None, True, True), ("both", True, True), ("trainee", True, False), ("opponent", False, True)])
def test_sides_and_eval_stays_on_the_pool(corpus, sides, on_trainee, on_opponent):
    """The declared side(s) carry the mix; EVAL never does, so a mix arm and its control evaluate on the same
    teams. Fails on revert to letting ``eval_trainee_teams`` default to the (mixed) trainee source."""
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5", team_mix_sides=sides))
    assert (spec.trainee_teams.mix is not None) == on_trainee
    assert (spec.opponent_teams.mix is not None) == on_opponent
    assert spec.eval_trainee_teams.mix is None
    assert spec.eval_trainee_teams == TeamSource(kind="default_biased", bias_prob=0.1)
    d = spec.to_dict()
    assert "mix" not in d["eval_trainee_teams"]
    # the head-to-head reads `eval_trainee_teams` to accept a player: still the default biased pool builder
    assert d["eval_trainee_teams"] == MatchupSpec.from_args(_args()).to_dict()["eval_trainee_teams"]


def test_provenance_records_the_corpus_identity_and_changes_the_hash(corpus):
    """The record carries the corpus ``ident()`` + the share, and a different share / tier is a different
    measurement regime (hash). Fails on revert to omitting the mix from ``to_dict``."""
    from utils.team_corpora import load_corpus

    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5"))
    c = load_corpus(TEST_CORPUS, "AB")
    rec = spec.to_dict()["trainee_teams"]["mix"]
    ident = c.ident()
    for k in ("tier", "n", "tier_sha256", "data_sha256", "manifest_sha256"):
        assert rec[k] == ident[k], k
    assert rec["corpus"] == TEST_CORPUS and rec["share"] == 0.5 and rec["schema"] == "gen3_team_mix_v1"
    assert spec.to_dict()["opponent_teams"]["mix"] == rec
    hashes = {spec.spec_hash(), NO_FLAG_HASH,
              MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.25")).spec_hash(),
              MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@A:0.5")).spec_hash(),
              MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5", team_mix_sides="opponent")).spec_hash()}
    assert len(hashes) == 5


def test_no_tier_means_the_manifests_default_tier(corpus):
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}:0.5"))
    assert spec.trainee_teams.mix.tier == "AB" and spec.trainee_teams.mix.n == 2


def test_summary_echo_names_the_corpus(corpus):
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@A:0.5"))
    text = "\n".join(spec.summary_lines())
    assert f"corpus {TEST_CORPUS}@A (1 teams" in text
    assert "eval: trainee on full pool + 10% sample-team bias" in text


def test_an_unverifiable_corpus_raises_corpus_error(corpus):
    """A missing tier or a data file that no longer matches its manifest raises ``CorpusError`` (the trainer's
    startup turns it into FATAL_CONFIG)."""
    from utils.team_corpora import CorpusError

    with pytest.raises(CorpusError, match="tier 'Z'"):
        MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@Z:0.5"))
    (corpus / "teams.jsonl.gz").write_bytes(b"tampered")
    clear_corpus_cache()
    with pytest.raises(CorpusError, match="sha256"):
        MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@A:0.5"))


def test_build_hands_the_corpus_to_both_builders(corpus, all_valid):
    """``TeamSource.build`` passes the mix to the builder of every declared side. Fails on revert to dropping the
    ``mix_teams`` kwargs in ``build``."""
    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5"))
    pool, sample = [TEAM_A], [TEAM_A]
    tt, ot = spec.trainee_teams.build(pool, sample), spec.opponent_teams.build(pool, sample)
    assert len(tt.mix_packed_teams) == 2 and tt.mix_prob == 0.5
    assert ot.mix_packed_teams == tt.mix_packed_teams and ot.mix_prob == 0.5
    assert tt.bias_prob == 0.1 and tt.bias_packed_teams       # the trainee's usual draw is intact
    no_mix = spec.eval_trainee_teams.build(pool, sample)
    assert no_mix.mix_packed_teams == () and "mix_packed_teams" not in vars(no_mix)


def test_exploiter_trainee_may_not_draw_from_the_corpus(corpus):
    """F-DV-3: an exploiter pilots vetted sample teams only — a trainee-side mix is refused, an opponent-side one is
    not. Fails on revert of the mix branch in ``validate_exploiter_trainee_is_sample`` (an unpinned exploiter
    trainee is otherwise out of its scope and passes quietly)."""
    for sides in (None, "both", "trainee"):
        spec = MatchupSpec.from_args(_args(exploiter="models/x", team_mix=f"{TEST_CORPUS}:0.5", team_mix_sides=sides))
        with pytest.raises(ValueError, match="exploiter's TRAINEE may not draw"):
            validate_exploiter_trainee_is_sample(spec, [TEAM_A])
    spec = MatchupSpec.from_args(_args(exploiter="models/x", team_mix=f"{TEST_CORPUS}:0.5",
                                       team_mix_sides="opponent"))
    validate_exploiter_trainee_is_sample(spec, [TEAM_A])           # no raise


# ── the Rust collector's team table and stager; the eval ledger's team-set id ──────────────────────────────────

def test_team_table_declares_the_corpus_and_the_stager_draws_it(corpus, all_valid):
    """``TeamTable.add_builder`` declares the mix teams (else ``TeamTableMiss`` at the first corpus episode), and a
    ``TeamStager`` over mixed builders stages 200 episodes whose corpus count equals the seeded reference stream
    exactly (rule 8) and lies in [70, 130] of 200 at share 0.5. Fails on revert of ``add_builder``'s mix line."""
    import random

    from agents.training import keyed_draw as KD
    from agents.training.rust_rollout.teams import STREAM_TEAM_RNG, TeamStager, TeamTable

    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@A:0.5"))   # corpus = TEAM_B; pool = TEAM_A
    tt = spec.trainee_teams.build([TEAM_A], [TEAM_A])
    ot = spec.opponent_teams.build([TEAM_A], [TEAM_A])
    table = TeamTable()
    n_envs, run_seed, per_env = 4, 12345, 50
    st = TeamStager(table, n_envs, trainee_builder=tt, opponent_builder=ot, run_seed=run_seed)
    assert len(table) == 2 and tt.mix_packed_teams[0] in table.teams
    cols = {"ep_team": np.zeros((n_envs, 2), np.int64), "ep_seed": np.zeros((n_envs, 4), np.int64)}
    corpus_idx = table.index(tt.mix_packed_teams[0], "test")
    n_corpus = [0, 0]
    for k in range(per_env):
        st.stage(cols, list(range(n_envs)), [0] * n_envs, [k] * n_envs)
        n_corpus[0] += int((cols["ep_team"][:, 0] == corpus_idx).sum())
        n_corpus[1] += int((cols["ep_team"][:, 1] == corpus_idx).sum())

    def ref(env, which):
        seed = int(KD.draw_keys(run_seed, STREAM_TEAM_RNG, env, which, 0)) & 0x7FFFFFFFFFFFFFFF
        rng, hits = random.Random(seed), 0
        for _ in range(per_env):
            hit = rng.random() < 0.5
            if not hit and which == 0:
                rng.random()                                  # the trainee's 10 % bias branch
            rng.choice([None])                                # one choice over a one-team list
            hits += hit
        return hits

    assert n_corpus[0] == sum(ref(e, 0) for e in range(n_envs))
    assert n_corpus[1] == sum(ref(e, 1) for e in range(n_envs))
    assert all(70 <= n <= 130 for n in n_corpus), n_corpus


def test_team_table_without_a_mix_is_unchanged(all_valid):
    from agents.training.rust_rollout.teams import TeamTable

    spec = MatchupSpec.from_args(_args())
    table = TeamTable()
    table.add_builder(spec.trainee_teams.build([TEAM_A, TEAM_B], [TEAM_B]))
    assert len(table) == 2


def test_builder_ident_carries_the_mix_only_when_declared(corpus, all_valid):
    """The eval ledger's team-set id keys on ``builder_ident``: unchanged keys without a mix (every existing id is
    stable), the corpus + share with one. Fails on revert of either half of ``builder_ident``'s mix branch."""
    from agents.training.cycle_ledger import builder_ident

    spec = MatchupSpec.from_args(_args(team_mix=f"{TEST_CORPUS}@AB:0.5"))
    plain = builder_ident(spec.eval_trainee_teams.build([TEAM_A], [TEAM_A]))
    assert set(plain) == {"teams", "bias_teams", "bias_prob"}
    mixed = builder_ident(spec.trainee_teams.build([TEAM_A], [TEAM_A]))
    assert len(mixed["mix_teams"]) == 2 and mixed["mix_prob"] == 0.5
