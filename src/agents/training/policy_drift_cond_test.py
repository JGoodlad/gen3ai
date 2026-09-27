"""Unit tests for the POLICY DRIFT meter's CONDITIONAL class rates — no model, no bridge.

Each fails on revert: eligibility and USEFUL eligibility counted on hand-built states (every rule),
the obs facts read through the layout (pinned against the encoders themselves), the Wilson and the
paired-bootstrap intervals, the backfill equalling a fresh compute, and the verdict naming only the
classes whose interval excludes 0."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.training import policy_drift as pd
from agents.training import policy_drift_cond as pc
from agents.training.policy_drift_test import _Probs, _fake_run

A = 11


def _num(mid):
    from agents.gen3_data import moves
    return moves.move_data(mid).num


def _facts(n, **over):
    f = {"opp_spikes": np.zeros(n, np.int64), "hp": np.full(n, 0.5), "asleep": np.zeros(n, bool),
         "stockpile": np.zeros(n, bool), "wish_pending": np.zeros(n, bool),
         "boosts": {s: np.zeros(n, np.int64) for s in pc.BOOST_STATS}}
    for k, v in over.items():
        if k in pc.BOOST_STATS:
            f["boosts"][k] = np.asarray(v, np.int64)
        else:
            f[k] = np.asarray(v)
    return f


# ── eligibility + useful eligibility on hand-built states ─────────────────────────────────────
# One state per rule. Each state has switch 0 + the move in slot 0 (+ earthquake in slot 1) legal.
_CASES = [
    # (move, facts overrides for this state, useful?)
    ("spikes", {"opp_spikes": 3}, False),        # 3 layers: onSideRestart fails
    ("spikes", {"opp_spikes": 2}, True),
    ("recover", {"hp": 1.0}, False),             # heal() fails at full HP
    ("recover", {"hp": 0.99}, True),
    ("rest", {"hp": 0.5, "asleep": True}, False),  # Rest.onTry: already asleep
    ("rest", {"hp": 0.5}, True),
    ("wish", {"hp": 1.0}, True),                 # Wish at full HP still heals the slot next turn
    ("wish", {"hp": 0.5, "wish_pending": True}, False),   # addSlotCondition: already pending
    ("swallow", {"hp": 0.5}, False),             # no Stockpile
    ("swallow", {"hp": 0.5, "stockpile": True}, True),
    ("swordsdance", {"atk": 6}, False),          # getCappedBoost: +6 cannot rise
    ("swordsdance", {"atk": 5}, True),
    ("dragondance", {"atk": 6}, True),           # Speed can still rise
    ("dragondance", {"atk": 6, "spe": 6}, False),
    ("calmmind", {"spa": 6, "spd": 6}, False),
    ("bellydrum", {"hp": 0.5}, False),           # bellydrum.onHit: hp <= maxhp/2
    ("bellydrum", {"hp": 0.8}, True),
    ("bellydrum", {"hp": 0.8, "atk": 6}, False),
    ("curse", {"atk": 6}, True),                 # non-Ghost Curse: Def can still rise
    ("minimize", {"evasion": 6}, False),
]


def _case_probe():
    n = len(_CASES)
    mask = np.zeros((n, A), np.float32)
    mask[:, [0, 6, 7]] = 1
    nums = np.zeros((n, 4), np.int64)
    over = {}
    for i, (mid, f, _) in enumerate(_CASES):
        nums[i, :2] = (_num(mid), _num("earthquake"))
        for k, v in f.items():
            over.setdefault(k, [None] * n)[i] = v
    base = _facts(n)
    for k, vals in over.items():
        arr = (base["boosts"][k] if k in pc.BOOST_STATS else base[k]).copy()
        for i, v in enumerate(vals):
            if v is not None:
                arr[i] = v
        if k in pc.BOOST_STATS:
            base["boosts"][k] = arr
        else:
            base[k] = arr
    return mask, nums, base


def test_every_useful_rule_on_hand_built_states():
    mask, nums, facts = _case_probe()
    _, useful = pc.action_tables(mask, nums, pd.move_class_by_num(), pc.move_ids_by_num(), facts)
    got = [bool(useful[i, 6]) for i in range(len(_CASES))]
    assert got == [u for _, _, u in _CASES]
    assert useful[:, 0].all() and useful[:, 7].all()      # switch / attack: useful whenever legal


def test_eligible_and_useful_counts_in_the_block():
    mask, nums, facts = _case_probe()
    cls_act, useful = pc.action_tables(mask, nums, pd.move_class_by_num(), pc.move_ids_by_num(), facts)
    p = np.zeros_like(mask)
    p[:, 6] = 0.6                                         # greedy = the gated move, everywhere
    p[:, 0] = 0.3
    p[:, 7] = 0.1
    c = pc.cond_block(p, {"prev": None}, mask, cls_act, useful)["classes"]
    n_by = {k: sum(1 for m, _, _ in _CASES if pd.move_class_by_num()[_num(m)] == k)
            for k in pc.USEFUL_CLASSES}
    u_by = {k: sum(1 for m, _, u in _CASES if u and pd.move_class_by_num()[_num(m)] == k)
            for k in pc.USEFUL_CLASSES}
    assert n_by == {"hazard": 2, "recovery": 8, "setup": 10}
    for k in pc.USEFUL_CLASSES:
        assert (c[k]["elig"]["n"], c[k]["elig"]["k"]) == (n_by[k], n_by[k])
        assert (c[k]["useful"]["n"], c[k]["useful"]["k"]) == (u_by[k], u_by[k])
        assert c[k]["elig"]["rate"] == 1.0 and np.isclose(c[k]["elig"]["mass"], 0.6)
    # switch and attack legal in every state, never greedy; a class never legal has n = 0
    assert (c["switch"]["elig"]["n"], c["switch"]["elig"]["k"]) == (len(_CASES), 0)
    assert np.isclose(c["attack"]["elig"]["mass"], 0.1) and c["attack"]["useful"] is None
    assert c["phazing"]["elig"]["n"] == 0 and c["phazing"]["elig"]["rate"] is None


def test_a_forced_state_is_outside_the_universe():
    mask = np.zeros((2, A), np.float32)
    mask[0, 6] = 1                                          # only Spikes legal: forced
    mask[1, [0, 6]] = 1
    nums = np.array([[_num("spikes"), 0, 0, 0]] * 2)
    cls_act, _ = pc.action_tables(mask, nums, pd.move_class_by_num())
    p = mask / mask.sum(-1, keepdims=True)
    blk = pc.cond_block(p, {}, mask, cls_act, None)
    assert blk["n_universe"] == 1 and blk["classes"]["hazard"]["elig"]["n"] == 1
    assert blk["useful_available"] is False and blk["classes"]["hazard"]["useful"] is None


def test_every_setup_move_in_the_dex_has_a_stat_map():
    t = pd.move_class_by_num()
    ids = pc.move_ids_by_num()
    missing = [ids[n] for n, c in t.items() if c == "setup" and not pc.setup_stats(ids[n])]
    assert missing == []


# ── the obs facts through the layout, pinned against the encoders ────────────────────────────
def test_state_facts_read_the_layout_the_encoders_write():
    from agents.observation import constants as C
    from agents.observation.active_context import ActiveContextEncoder
    from agents.observation.global_env import GlobalEnvEncoder
    from agents.observation.pokemon import _STATUS_STR_IDX
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    lay = Gen3ObservationEncoder(load_mappings()).get_features_extractor_kwargs()["layout"]
    obs = np.zeros((1, lay["total_dim"]), np.float32)
    ot = lay["parts"]["our_team"]
    w = ot["reshape"][1]
    slot = ot["start"] + 2 * w                                   # our active is slot 2
    obs[0, slot + C.POKEMON_ACTIVE_OFFSET] = 1
    obs[0, slot + lay["pokemon"]["hp"]["offset"]] = 0.37
    obs[0, slot + lay["pokemon"]["condition"]["offset"] + _STATUS_STR_IDX["slp"]] = 1
    obs[0, ot["start"] + lay["pokemon"]["hp"]["offset"]] = 0.9  # a bench mon: must not be read
    live = SimpleNamespace(boosts={"atk": 6, "spe": -2, "evasion": 1}, volatiles=["stockpile2"])
    cx = lay["parts"]["context"]
    obs[0, cx["start"]: cx["start"] + cx["reshape"][1]] = ActiveContextEncoder().encode(live)
    g = lay["parts"]["global"]
    hz = g["start"] + lay["global_layout"]["hazards"]["offset"]
    obs[0, hz + 1] = 2 / C.MAX_SPIKES
    r = lay["parts"]["reactive"]["start"] + lay["reactive_layout"]["wish_floating_our"]["offset"]
    obs[0, r] = 0.5
    f = pc.state_facts(obs, lay)
    assert np.isclose(f["hp"][0], 0.37) and f["asleep"][0] and f["stockpile"][0] and f["wish_pending"][0]
    assert {s: int(f["boosts"][s][0]) for s in pc.BOOST_STATS} == {
        "atk": 6, "def": 0, "spa": 0, "spd": 0, "spe": -2, "accuracy": 0, "evasion": 1}
    # the Spikes count agrees with the global encoder's own decoder
    assert int(f["opp_spikes"][0]) == GlobalEnvEncoder().describe_vector(obs[0, g["start"]: g["end"]])["opp_spikes"] == 2


# ── intervals ─────────────────────────────────────────────────────────────────────────────────
def test_wilson_interval():
    lo, hi = pc.wilson(5, 10)
    assert np.isclose(lo, 0.2366, atol=1e-4) and np.isclose(hi, 0.7634, atol=1e-4)
    lo, hi = pc.wilson(0, 12)
    assert np.isclose(lo, 0.0, atol=1e-12) and np.isclose(hi, 0.2425, atol=1e-4)
    assert pc.wilson(0, 0) is None


def test_paired_delta_interval():
    same = np.array([1, 0, 1, 0, 1] * 10)
    assert pc.paired_delta_ci(same, same) == (0.0, 0.0)
    assert pc.paired_delta_ci(np.ones(30), np.zeros(30)) == (1.0, 1.0)
    # one flip in twelve states: the point delta is +8pp but the interval reaches 0
    x = np.zeros(12); x[0] = 1
    lo, hi = pc.paired_delta_ci(x, np.zeros(12))
    assert lo == 0.0 and hi > 0 and not pc.interval_clear((lo, hi))
    # 30 of 100 states moved up, 5 down: clearly positive
    x, y = np.zeros(100), np.zeros(100)
    x[:30] = 1; y[30:35] = 1
    lo, hi = pc.paired_delta_ci(x, y)
    assert 0.1 < lo < 0.25 < hi < 0.4
    assert pc.paired_delta_ci(x, y) == (lo, hi)                   # seeded: reproducible


def test_conditional_rate_ref_delta_and_interval_in_the_block():
    n = 40
    mask = np.zeros((n, A), np.float32)
    mask[:, [0, 6, 7]] = 1
    nums = np.tile([_num("spikes"), _num("earthquake"), 0, 0], (n, 1))
    cls_act, _ = pc.action_tables(mask, nums, pd.move_class_by_num())
    cur = np.zeros((n, A), np.float32); cur[:, 7] = .6; cur[:, 6] = .3; cur[:, 0] = .1
    cur[:20, 6] = .7; cur[:20, 7] = .2                             # spikes greedy in 20/40
    ref = np.zeros((n, A), np.float32); ref[:, 7] = .6; ref[:, 6] = .3; ref[:, 0] = .1
    h = pc.cond_block(cur, {"back": (7, ref)}, mask, cls_act, None)["classes"]["hazard"]["elig"]
    assert (h["n"], h["k"], h["rate"]) == (40, 20, 0.5)
    assert h["ci"] == pc.wilson(20, 40)
    b = h["refs"]["back"]
    assert b["step"] == 7 and b["rate"] == 0.0 and b["delta"] == 0.5 and pc.interval_clear(b["ci"])
    assert np.isclose(h["mass"], 0.5) and np.isclose(b["mass_delta"], 0.2)


# ── backfill == fresh compute ─────────────────────────────────────────────────────────────────
def test_backfill_equals_a_fresh_compute_and_never_rewrites_rows(tmp_path):
    n = 60
    rng = np.random.default_rng(3)
    mask = np.zeros((n, A), np.float32)
    mask[:, [0, 1, 6, 7, 8]] = 1
    nums = np.tile([_num("spikes"), _num("recover"), _num("swordsdance"), 0], (n, 1))
    facts = _facts(n, opp_spikes=rng.integers(0, 4, n), hp=rng.choice([0.4, 1.0], n),
                   atk=rng.choice([0, 6], n))
    obs = np.zeros((n, 5), np.float32)
    run = _fake_run(tmp_path, [4_000_000, 6_000_000, 16_000_000])
    out = tmp_path / "out"
    kw = dict(table=pd.move_class_by_num(), ids=pc.move_ids_by_num(), facts=facts, log=lambda s: None)
    fresh = pd.process_pending(run, out, obs, mask, nums, _Probs(mask), **kw)
    assert all(r["cond"]["useful_available"] for r in fresh)
    # simulate the pre-change rows: the same rows WITHOUT the block
    old = [{k: v for k, v in json.loads(line).items() if k != "cond"}
           for line in (out / "rows.jsonl").read_text().splitlines()]
    (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in old))
    before = (out / "rows.jsonl").read_bytes()
    kw.pop("log")
    done = pd.backfill_cond(out, mask, nums, log=lambda s: None, **kw)
    assert done == [4_000_000, 6_000_000, 16_000_000]
    assert (out / "rows.jsonl").read_bytes() == before              # originals untouched
    merged = pd.load_rows_merged(out)
    for f, m in zip(fresh, merged):
        assert m["cond_backfilled"] and m["cond"] == json.loads(json.dumps(f["cond"]))
        assert pd.verdict(m) == f["verdict"]
    # idempotent: a second backfill finds nothing to do
    assert pd.backfill_cond(out, mask, nums, log=lambda s: None, **kw) == []


# ── the verdict names only interval-clear classes ─────────────────────────────────────────────
def _grain(delta, ci, n=100, prev=None):
    refs = {"back": {"step": 1, "delta": delta, "ci": ci}}
    if prev is not None:
        refs["prev"] = prev
    return {"n": n, "k": 0, "rate": 0.5, "ci": [0.4, 0.6], "mass": 0.5, "refs": refs}


def _row(classes):
    flat = {"n": 10, "flips": 0, "rate": 0.0,
            "buckets": {lab: {"n": 3, "flips": 0, "rate": 0.0} for lab in pd.bucket_labels()}}
    ref = {"step": 1, "kl_mean": .1, "kl_median": .1, "flip": flat,
           "share_delta": {c: (0.3 if c == "attack" else 0.0) for c in pd.CLASSES}}
    cond = {"classes": {c: {"elig": _grain(0.0, [-0.01, 0.01]), "useful": None} for c in pc.COND_CLASSES}}
    cond["classes"].update(classes)
    return {"status": "ok", "cycling": [], "refs": {"prev": ref, "back": ref, "anchor": ref}, "cond": cond}


def test_verdict_names_only_classes_whose_interval_excludes_zero():
    row = _row({
        "hazard": {"elig": _grain(0.5, [0.3, 0.7], n=60), "useful": _grain(0.4, [0.2, 0.6], n=50)},
        "setup": {"elig": _grain(0.3, [-0.05, 0.6]), "useful": None},        # straddles 0
        "switch": {"elig": _grain(-0.2, [-0.3, -0.1]), "useful": None},
        "status": {"elig": _grain(0.02, [0.01, 0.03]), "useful": None},      # clear but < SHIFT_ABS
    })
    v = pd.verdict(row)
    assert v == "shifting (hazard|useful +40pp (n=50), switch|legal -20pp (n=100))"
    # the old overall-share rule would have named attack (+30pp share); the conditional one must not
    assert "attack" not in v and "setup" not in v and "status" not in v


def test_verdict_refining_when_no_interval_clears():
    row = _row({"setup": {"elig": _grain(0.3, [-0.05, 0.6]), "useful": None}})
    assert pd.verdict(row) == "refining"


def test_render_shows_the_conditional_table(tmp_path):
    from main.policy_drift import render
    n = 20
    mask = np.zeros((n, A), np.float32); mask[:, [0, 6, 7]] = 1
    nums = np.tile([_num("spikes"), _num("earthquake"), 0, 0], (n, 1))
    run = _fake_run(tmp_path, [4_000_000, 6_000_000])
    rows = pd.process_pending(run, tmp_path / "o", np.zeros((n, 3), np.float32), mask, nums,
                              _Probs(mask), facts=_facts(n), ids=pc.move_ids_by_num(),
                              log=lambda s: None)
    txt = render(rows)
    assert "CONDITIONAL greedy rate" in txt and "hazard*" in txt and "secondary" in txt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
