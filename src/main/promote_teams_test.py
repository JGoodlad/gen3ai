"""Gates for ``main.promote_teams`` — the seed-recorded random team promotion.

Everything here runs on a SYNTHETIC team tree except the last section, which pins the tool against
the real pool and the real exclusion artifact. No node, no battles, no models.
"""
from __future__ import annotations

import json
import os

import pytest

from agents.training.team_archetypes import team_sha
from main import promote_teams as pt


# ── a synthetic team tree ───────────────────────────────────────────────────────────────────────

def _team(i: int) -> str:
    return f"Tyranitar @ Leftovers\nAbility: Sand Stream\n- Rock Slide {i}\n"


def _tree(tmp_path, n_other: int = 30, n_sample: int = 3):
    """`<tmp>/data/teams/{sample,others/authorA}/` with manifests, mirroring the real layout."""
    root = tmp_path / "root"
    teams = root / "data" / "teams"
    (teams / "sample").mkdir(parents=True)
    (teams / "others" / "authorA").mkdir(parents=True)

    sample_meta = []
    for i in range(n_sample):
        txt = _team(1000 + i)
        name = f"s{i}"
        (teams / "sample" / f"{name}.txt").write_text(txt)
        sample_meta.append({"id": name, "name": name, "file": f"teams/sample/{name}.txt"})
    (teams / "sample" / "teams.json").write_text(json.dumps(sample_meta, indent=2))

    other_meta = []
    for i in range(n_other):
        txt = _team(i)
        name = f"o{i:03d}"
        (teams / "others" / "authorA" / f"{name}.txt").write_text(txt)
        other_meta.append({"id": name, "name": name, "format": "gen3ou", "valid": True,
                           "errors": [], "file": f"teams/others/authorA/{name}.txt",
                           "source": f"https://example/{name}"})
    (teams / "others" / "authorA" / "teams.json").write_text(json.dumps(other_meta, indent=2))
    return str(root)


@pytest.fixture()
def tree(tmp_path):
    return _tree(tmp_path)


def _fake_arch(pool):
    return {sha: {"archetype": "balance", "tags": []} for sha in pool}


# ── the draw ────────────────────────────────────────────────────────────────────────────────────

def test_pool_load_matches_the_loader(tree):
    """load_pool must see exactly what TeamLoader sees — it is a mirror, and mirrors drift."""
    pool = pt.load_pool(tree)
    pt._cross_check_pool(tree, pool)          # raises on drift
    assert len(pool) == 33
    assert sum(1 for t in pool.values() if t.category == "sample") == 3


def test_same_seed_gives_the_same_draw(tree):
    pool = pt.load_pool(tree)
    a = pt.draw_teams(pool, set(), 10, 7, None)
    b = pt.draw_teams(pool, set(), 10, 7, None)
    assert a.accepted == b.accepted
    assert pt.draw_teams(pool, set(), 10, 8, None).accepted != a.accepted


def test_the_draw_does_not_touch_the_global_rng(tree):
    """A per-call ``random.Random(seed)``, never the module-level stream any import can perturb."""
    import random
    random.seed(0)
    before = [random.random() for _ in range(3)]
    random.seed(0)
    pt.draw_teams(pt.load_pool(tree), set(), 5, 1234, None)
    assert [random.random() for _ in range(3)] == before


def test_the_draw_is_a_function_of_the_seed_not_of_dict_order(tree):
    """Eligible shas are sorted before the shuffle, so a different insertion order draws the same."""
    pool = pt.load_pool(tree)
    shuffled = dict(reversed(list(pool.items())))
    assert pt.draw_teams(shuffled, set(), 10, 99, None).accepted == \
           pt.draw_teams(pool, set(), 10, 99, None).accepted


def test_an_excluded_team_can_never_be_drawn(tree):
    pool = pt.load_pool(tree)
    excluded = set(sorted(pool)[:20])
    for seed in range(25):
        got = pt.draw_teams(pool, excluded, 10, seed, None).accepted
        assert not (set(got) & excluded), f"seed {seed} drew an excluded team"
        assert len(set(got)) == 10


def test_drawing_more_than_is_eligible_raises(tree):
    pool = pt.load_pool(tree)
    with pytest.raises(ValueError, match="only 3 are eligible"):
        pt.draw_teams(pool, set(sorted(pool)[3:]), 10, 1, None)


# ── replacement ─────────────────────────────────────────────────────────────────────────────────

def test_an_invalid_team_is_replaced_by_the_next_candidate_not_dropped(tree):
    pool = pt.load_pool(tree)
    clean = pt.draw_teams(pool, set(), 8, 5, None).accepted        # the unfiltered seeded order
    bad = {clean[2], clean[5]}
    sha_of = {t.text: s for s, t in pool.items()}

    def validator(texts):
        return [{"valid": sha_of[t] not in bad, "errors": [] if sha_of[t] not in bad else ["illegal"]}
                for t in texts]

    res = pt.draw_teams(pool, set(), 8, 5, validator)
    assert len(res.accepted) == 8                                   # replaced, never short
    assert not (set(res.accepted) & bad)
    assert [r["rejected_sha"] for r in res.replacements] == [clean[2], clean[5]]
    assert res.replacements[0]["errors"] == ["illegal"]
    # the survivors keep their seeded order, and the replacements come from further down the SAME
    # shuffle — so the draw is still a function of the seed alone.
    assert res.accepted[:8] == [s for s in res.accepted if s not in bad]
    assert res.considered > 8


def test_a_broken_validator_aborts_instead_of_condemning_the_pool(monkeypatch):
    """The positive control is the whole point: a dead node bridge must not read as 693 bad teams."""
    import utils.bridge.team_validator as tv
    monkeypatch.setattr(tv, "validate_teams_locally",
                        lambda f, ts: [{"valid": False, "errors": ["Node execution error: boom"]}
                                       for _ in ts])
    with pytest.raises(RuntimeError, match="BROKEN, not the teams"):
        pt.make_validator("gen3ou", "control")(["a", "b"])

    monkeypatch.setattr(tv, "validate_teams_locally",
                        lambda f, ts: [{"valid": False, "errors": ["Illegal"]} for _ in ts])
    with pytest.raises(RuntimeError, match="positive control"):
        pt.make_validator("gen3ou", "control")(["a", "b"])


def test_a_good_batch_strips_the_control_from_the_verdicts(monkeypatch):
    import utils.bridge.team_validator as tv
    monkeypatch.setattr(tv, "validate_teams_locally",
                        lambda f, ts: [{"valid": True, "errors": []} for _ in ts])
    assert len(pt.make_validator("gen3ou", "control")(["a", "b"])) == 2


# ── promotion ───────────────────────────────────────────────────────────────────────────────────

def _promote(tree, shas, seed=1):
    pool = pt.load_pool(tree)
    actions = pt.plan_promotion(tree, pool, shas)
    pt.apply_promotion(tree, pool, actions, _fake_arch(pool), seed, "now")
    return pool, actions


def test_promotion_moves_a_team_it_does_not_duplicate_it(tree):
    """The draw-weight guard: a promoted team left in BOTH manifests is drawn twice as often."""
    pool = pt.load_pool(tree)
    picks = pt.draw_teams(pool, set(), 4, 3, None).accepted
    _promote(tree, picks)
    counts = pt.check_invariants(tree, expect_sample=3 + 4, expect_total=33)   # raises on a dupe
    assert counts == {"total": 33, "sample": 7, "other": 26}
    after = pt.load_pool(tree)
    for sha in picks:
        assert after[sha].category == "sample"
        assert after[sha].rel_path == f"data/teams/sample/{sha}.txt"


def test_promotion_is_idempotent(tree):
    pool = pt.load_pool(tree)
    picks = pt.draw_teams(pool, set(), 3, 11, None).accepted
    _promote(tree, picks)
    first = json.loads((open(os.path.join(tree, "data/teams/sample/teams.json"))).read())

    pool2 = pt.load_pool(tree)
    actions = pt.plan_promotion(tree, pool2, picks)
    assert {a.kind for a in actions} == {"already_curated"}      # already IN the sample manifest
    pt.apply_promotion(tree, pool2, actions, _fake_arch(pool2), 11, "now")
    assert json.loads(open(os.path.join(tree, "data/teams/sample/teams.json")).read()) == first
    pt.check_invariants(tree, expect_sample=6, expect_total=33)


def test_a_dest_holding_a_different_team_is_refused(tree):
    pool = pt.load_pool(tree)
    sha = pt.draw_teams(pool, set(), 1, 2, None).accepted[0]
    dest = os.path.join(tree, "data", "teams", "sample", f"{sha}.txt")
    open(dest, "w").write("Blissey @ Leftovers\n- Softboiled\n")
    with pytest.raises(RuntimeError, match="REFUSING to overwrite"):
        pt.plan_promotion(tree, pool, [sha])


def test_a_dest_already_holding_the_same_team_is_a_noop(tree):
    pool = pt.load_pool(tree)
    sha = pt.draw_teams(pool, set(), 1, 2, None).accepted[0]
    dest = os.path.join(tree, "data", "teams", "sample", f"{sha}.txt")
    open(dest, "w").write(pool[sha].text)
    assert [a.kind for a in pt.plan_promotion(tree, pool, [sha])] == ["already_promoted"]


def test_the_promoted_manifest_entry_carries_its_provenance(tree):
    pool = pt.load_pool(tree)
    sha = pt.draw_teams(pool, set(), 1, 4, None).accepted[0]
    src = pool[sha].rel_path
    _promote(tree, [sha], seed=4242)
    entry = next(e for e in json.loads(open(os.path.join(tree, "data/teams/sample/teams.json")).read())
                 if e["id"] == sha)
    assert entry["file"] == f"teams/sample/{sha}.txt"
    assert entry["promoted"] == {"from": src, "seed": 4242, "at": "now",
                                 "by": "python -m main.promote_teams"}
    assert os.path.exists(os.path.join(tree, src)), "the source .txt must survive — only the entry moves"


# ── the manifest ────────────────────────────────────────────────────────────────────────────────

def test_manifest_round_trips_and_renders(tree, tmp_path):
    pool = pt.load_pool(tree)
    excl = pt.Exclusions(union=sorted(pool)[:2], counts={"demo": 2}, path="x.json",
                         raw={"categories": {"demo": {"shas": sorted(pool)[:2], "reason": "r —"}}})
    res = pt.draw_teams(pool, excl.as_set(), 5, 17, None)
    actions = pt.plan_promotion(tree, pool, res.accepted)
    man = pt.build_manifest(res, pool, _fake_arch(pool), excl, actions, "now", "gen3ou", True)

    p = tmp_path / "m.json"
    p.write_text(json.dumps(man, indent=2))
    back = json.loads(p.read_text())
    assert back == man
    assert back["seed"] == 17 and back["n_requested"] == 5
    assert [r["sha"] for r in back["draw"]] == res.accepted
    assert back["exclusions"]["by_category"] == {"demo": 2}
    assert sum(back["archetype_composition"].values()) == 5

    md = pt.render_manifest_md(man)
    assert "seed `17`" in md and "UNIFORM RANDOM" in md
    for sha in res.accepted:
        assert sha in md
    assert "strip-normalized" in md


def test_the_manifest_records_the_seed_even_when_it_was_minted(tree, capsys, monkeypatch):
    """A seed the user did not type still lands in the manifest — that is what makes it auditable."""
    pool = pt.load_pool(tree)
    res = pt.draw_teams(pool, set(), 3, 918273, None)
    man = pt.build_manifest(res, pool, _fake_arch(pool), pt.Exclusions([], {}, "x", {"categories": {}}),
                            pt.plan_promotion(tree, pool, res.accepted), "now", "gen3ou", False)
    assert man["seed"] == 918273
    assert man["_meta"]["validated"] is False


# ── the hash convention ─────────────────────────────────────────────────────────────────────────

def test_the_key_convention_is_the_STRIPPED_sha1_prefix():
    """Pinned because an UNSTRIPPED ``sha1(text)`` is a recorded derived-key defect in this tree
    (``coverage_sample.py``; the tell was every row carrying ``"class": "?"``). Any change that makes
    this tool hash raw text must fail here, not silently produce keys nothing else can join on."""
    import hashlib
    body = "Tyranitar @ Leftovers\nAbility: Sand Stream\n"
    padded = f"\n  {body}  \n\n"
    assert team_sha(padded) == team_sha(body)
    assert team_sha(body) == hashlib.sha1(body.strip().encode()).hexdigest()[:10]
    assert team_sha(padded) != hashlib.sha1(padded.encode()).hexdigest()[:10]
    assert len(team_sha(body)) == 10


def test_the_tool_keys_teams_with_team_sha_and_nothing_else(tmp_path):
    """The whole module must route through ``team_archetypes.team_sha`` — one implementation."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(pt.__file__)), "promote_teams.py")).read()
    assert "from agents.training.team_archetypes import load_team_archetypes, team_sha" in src
    assert "hashlib" not in src, "promote_teams must not hash teams itself — call team_sha"

    root = _tree(tmp_path)
    pool = pt.load_pool(root)
    for sha, t in pool.items():
        assert sha == team_sha(t.text) == team_sha(t.text + "\n\n  ")


# ── against the real tree ───────────────────────────────────────────────────────────────────────

def test_the_committed_exclusion_artifact_is_coherent_with_the_real_pool():
    from utils.paths import repo_path
    excl = pt.load_exclusions(str(repo_path("designs", "ai_v12", "promotion_exclusions.json")))
    pool = pt.load_pool(str(repo_path()))
    assert len(pool) == 719, "the pool moved — re-verify the slate and the exclusions before drawing"
    assert set(excl.union) <= set(pool), "an exclusion names a team that is not in the pool"
    assert len(excl.union) == 26
    assert excl.counts == {"taught_F5": 9, "taught_F6": 12, "rev4_pending": 24,
                           "held_out_instruments": 2}
    assert len(pool) - len(excl.union) == 693


def test_the_committed_exclusions_agree_with_recorded_run_provenance():
    """THE DRIFT GATE. The artifact was first built from FROZEN ARGV FILES in a session-scoped job
    directory, before the runs existed; the launched runs dealt different teams. On 2026-08-31 its
    ``rev4_pending`` block was stale on all three arms — 4 teams it named were never pinned, 4 it
    missed were — and the union SIZE stayed 26 throughout, so the coherence test above passed the
    whole time. The only authority is each run's own ``metadata.json``.

    Repair: ``python -m main.promote_teams --regenerate-exclusions``.
    """
    from utils.paths import main_models_dir, repo_path
    md = main_models_dir()
    if not md:
        pytest.skip("no models/ archive on this box — recorded provenance cannot be read")
    excl = pt.load_exclusions(str(repo_path("designs", "ai_v12", "promotion_exclusions.json")))
    prov = pt.recorded_provenance(excl, str(md), str(repo_path()))
    assert prov, "the artifact names no runs at all — the derivation seam is broken, not clean"
    assert any(a.present for a in prov), (
        "every run named by the exclusion artifact is missing from models/ — this test would "
        "pass vacuously; it is asserting nothing")
    drift = pt.exclusion_drift(excl, prov)
    assert not drift, "\n".join(
        [f"{len(drift)} exclusion arm(s) disagree with recorded run provenance — repair with "
         "`python -m main.promote_teams --regenerate-exclusions`:"]
        + [f"  {d['category']}/{d['arm']} vs {d['run']}/metadata.json: "
           f"named but NEVER PINNED {d['in_artifact_never_pinned']}; "
           f"PINNED but missing {d['pinned_but_missing']}" for d in drift])


def test_the_real_draw_never_returns_a_taught_team():
    from utils.paths import repo_path
    excl = pt.load_exclusions(str(repo_path("designs", "ai_v12", "promotion_exclusions.json")))
    pool = pt.load_pool(str(repo_path()))
    for seed in (0, 1, 20260830, 2 ** 31 - 1):
        got = pt.draw_teams(pool, excl.as_set(), 40, seed, None).accepted
        assert len(set(got)) == 40
        assert not (set(got) & excl.as_set())


def test_the_demo_dry_run_manifest_is_the_committed_seed():
    """The committed demo is a real draw: re-running its seed must reproduce it exactly."""
    from utils.paths import repo_path
    demo = repo_path("designs", "ai_v12", "promotion_dry_run_demo.json")
    if not os.path.exists(demo):
        pytest.skip("no committed demo draw")
    man = json.loads(open(demo).read())
    excl = pt.load_exclusions(str(repo_path("designs", "ai_v12", "promotion_exclusions.json")))
    pool = pt.load_pool(str(repo_path()))
    res = pt.draw_teams(pool, excl.as_set(), man["n_requested"], man["seed"], None)
    assert res.accepted == [r["sha"] for r in man["draw"]]


def test_the_cli_parser_accepts_the_documented_surface():
    p = pt.build_parser()
    a = p.parse_args(["--dry-run", "--seed", "5", "--n", "12"])
    assert (a.dry_run, a.seed, a.n) == (True, 5, 12)
    assert p.parse_args([]).n == pt.DEFAULT_N and p.parse_args([]).seed is None
    assert p.parse_args(["--draw-only"]).draw_only
    assert p.parse_args(["--verify-exclusions"]).verify_exclusions
    assert p.parse_args(["--regenerate-exclusions"]).regenerate_exclusions
    assert not p.parse_args([]).regenerate_exclusions


def test_dry_run_and_draw_only_are_mutually_exclusive():
    assert pt.main(["--dry-run", "--draw-only"]) == 2
