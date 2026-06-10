"""Tests for the agent-facing ProbeSession + query CLI (no real checkpoint)."""

import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from agents.action.constants import MOVE_START
from main.prober.model import ObsOffsets
from main.prober.session import ProbeSession

_OFF = ObsOffsets(mm_off=10, om_off=20, tm_off=164, active_block_dim=5,
                  turn_history_offset=200, turn_history_dim=10)
_OBS_LEN = 256


class _FakeModel:
    offsets = _OFF

    def action_dist(self, obs, mask):
        w = np.ones(len(mask), dtype=np.float64)
        for s in range(4):
            w[MOVE_START + s] = 1.0 + float(obs[_OFF.mm_off + s]) * 10.0
        w = w * mask.astype(np.float64)
        return w / w.sum(), np.arange(len(mask), dtype=np.float64)

    def logit_grad(self, obs, mask, idx):
        return np.arange(len(obs), dtype=np.float64)

    def value(self, obs, mask):
        return 1.23


def _build_run(tmp_path):
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_2000000" / "staller"
    os.makedirs(bd, exist_ok=True)
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({"thunderbolt": {"prob": "92.1%", "valid": True},
                    "earthquake": {"prob": "2.8%", "valid": True},
                    "move2": {"prob": "0.0%", "valid": False},
                    "move3": {"prob": "0.0%", "valid": False},
                    "struggle": {"prob": "0.0%", "valid": False}})
    invs = [
        {"i": 1, "turn": 1, "phase": "move_selection", "chosen": "earthquake",
         "our": {"species": "zapdos"}, "opp": {"species": "jynx"}, "actions": actions,
         "outcome": {"reward": {"total": -1.4, "base": "hp=-1.2"}, "events": []}},
        {"i": 2, "turn": 2, "phase": "move_selection", "chosen": "switch:m0",
         "our": {"species": "zapdos"}, "opp": {"species": "jynx"}, "actions": actions,
         "outcome": {"reward": {"total": 0.5}, "events": ["opp:jynx:fainted"]}},
    ]
    summary = {"meta": {"step": 2000000, "result": "WIN", "turns": 4, "invocations": 2},
               "invocations": invs}
    with open(bd / "win_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((2, _OBS_LEN), dtype=np.float32)
    obs[:, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.25, 0.0, 0.125]
    np.savez(bd / "win_001_states.npz", obs=obs,
             has_state=np.array([1, 1], dtype=np.int8),
             values=np.array([2.0, 5.0], dtype=np.float32))
    # manifest (no retained snapshot → resolution stays 'nearest'); identity for run_summary
    with open(bd.parent / "eval_manifest.json", "w") as f:
        json.dump({"step": 2000000, "git_hash": "abc123",
                   "arch_signature": "gen3_x", "snapshot": None}, f)
    (run / "checkpoint_3200000_steps.zip").write_text("")  # gives the ladder a path
    with open(run / "metadata.json", "w") as f:
        json.dump({"gamma": 0.95}, f)
    return str(run), str(bd / "win_001_summary.json")


def test_battles_filter(tmp_path):
    run, _ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert len(sess.battles()) == 1
    assert sess.battles(outcome="loss") == []
    assert sess.battles(opponent="staller")[0]["step"] == 2000000


def test_battle_overview_is_model_free(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)  # no model_loader → must not load a model
    ov = sess.battle_overview(summ)
    assert ov["meta"]["result"] == "WIN"
    assert ov["model_resolution"]["tier"] == "nearest"
    r0, r1 = ov["invocations"]
    assert r0["chosen"] == "earthquake" and r0["value"] == 2.0 and r0["reward_total"] == -1.4
    assert "switch" in r1["flags"] and "faint" in r1["flags"]


def test_find_model_free(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert sess.find(summ, "switch") == [1]
    assert sess.find(summ, "faint") == [1]


def test_analyze_loads_resolved_model(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    d = sess.analyze(summ, 0)
    assert d["chosen"] == "earthquake"
    assert d["model_resolution"]["tier"] == "nearest"
    assert d["value"]["recorded"] == 2.0 and d["value"]["rerun"] == 1.23
    assert d["value"]["next_recorded"] == 5.0 and abs(d["value"]["delta"] - 3.0) < 1e-9
    # JSON-serializable
    json.dumps(d)


def test_find_disagree_uses_model(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    # inv0 chose earthquake but the fake favors thunderbolt → disagree
    assert 0 in sess.find(summ, "disagree")


def test_run_summary(tmp_path):
    run, _ = _build_run(tmp_path)
    s = ProbeSession(run).run_summary()
    assert s["n_steps"] == 1 and s["gamma"] == 0.95
    assert s["totals"] == {"win": 1, "loss": 0, "battles": 1}
    step = s["steps"][0]
    assert step["step"] == 2000000
    assert step["identity"]["git_hash"] == "abc123"
    assert step["identity"]["snapshot_available"] is False
    assert step["opponents"][0] == {"name": "staller", "win": 1, "loss": 0, "battles": 1}
    assert s["checkpoints"][0]["step"] == 3200000


def test_overview_delta_td_and_notable(tmp_path):
    run, summ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview(summ)
    r0, r1 = ov["invocations"]
    assert r0["delta_v"] == 3.0                       # 5.0 - 2.0
    assert abs(r0["td_residual"] - (-1.4 + 0.95 * 5.0 - 2.0)) < 1e-9   # γ from metadata
    assert r1["delta_v"] is None and r1["td_residual"] is None         # no next decision
    nt = ov["notable"]
    assert nt["faints"] == [1] and nt["switches"] == [1] and nt["uncertain_count"] == 2
    assert nt["biggest_value_drops"] == [{"inv": 0, "delta_v": 3.0}]


def test_overview_has_active_board(tmp_path):
    run, summ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview(summ)
    assert ov["invocations"][0]["our_active"].startswith("zapdos")
    assert ov["invocations"][0]["opp_active"].startswith("jynx")


def test_analyze_carries_board(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    d = sess.analyze(summ, 0)
    assert d["board"]["ours"]["active_species"] == "zapdos"


def test_find_value_criteria_and_limit(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert sess.find(summ, "value_drop") == [0]       # only inv0 has a next V
    assert sess.find(summ, "low_value") == [0, 1]     # V=2.0 then 5.0
    assert sess.find(summ, "high_value") == [1, 0]
    assert sess.find(summ, "low_value", limit=1) == [0]


def test_short_id_resolves(tmp_path):
    run, _ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview("step_2000000/staller/win_001")
    assert ov["meta"]["step"] == 2000000


def _write_battle(run, opponent, name, invs, values):
    """Minimal trace pair (summary + npz with `values`) for scan tests (model-free)."""
    bd = run / "eval_traces" / "step_2000000" / opponent
    os.makedirs(bd, exist_ok=True)
    outcome = name.split("_")[0]
    summary = {"meta": {"step": 2000000, "result": outcome.upper(), "turns": len(invs),
                        "invocations": len(invs)}, "invocations": invs}
    with open(bd / f"{name}_summary.json", "w") as f:
        json.dump(summary, f)
    np.savez(bd / f"{name}_states.npz",
             obs=np.zeros((len(invs), _OBS_LEN), dtype=np.float32),
             has_state=np.ones(len(invs), dtype=np.int8),
             values=np.array(values, dtype=np.float32))
    if not os.path.exists(bd.parent / "eval_manifest.json"):
        with open(bd.parent / "eval_manifest.json", "w") as f:
            json.dump({"step": 2000000, "git_hash": "abc", "arch_signature": "x",
                       "snapshot": None}, f)
    with open(run / "metadata.json", "w") as f:
        json.dump({"gamma": 0.95}, f)


def _inv(turn, chosen, reward_total, *, faint=False, species=("zapdos", "jynx")):
    return {"i": turn, "turn": turn, "phase": "move_selection", "chosen": chosen,
            "our": {"species": species[0]}, "opp": {"species": species[1]},
            "actions": {chosen: {"prob": "80.0%", "valid": True}},
            "outcome": {"reward": {"total": reward_total},
                        "events": (["opp:jynx:fainted"] if faint else [])}}


def _build_scan_run(tmp_path):
    """Two losses (distinct worst drops/TDs) + one win, for cross-battle scan."""
    run = tmp_path / "run"
    # loss_006: values 10→2→6  ⇒ worst ΔV -8 @inv0; r0=-1 ⇒ td0 = -1+0.95*2-10 = -9.10
    _write_battle(run, "aggressive_v2", "loss_006",
                  [_inv(1, "earthquake", -1.0), _inv(2, "switch:m0", 0.5, faint=True)],
                  [10.0, 2.0, 6.0])
    # loss_007: values 5→1 ⇒ worst ΔV -4 @inv0; r0=-10 ⇒ td0 = -10+0.95*1-5 = -14.05 (worst TD)
    _write_battle(run, "aggressive_v2", "loss_007",
                  [_inv(1, "fireblast", -10.0, faint=True)], [5.0, 1.0])
    # a win — must be excluded by outcome="loss"
    _write_battle(run, "heuristic", "win_001",
                  [_inv(1, "thunderbolt", 0.5), _inv(2, "thunderbolt", 1.0)], [3.0, 8.0])
    (run / "checkpoint_3200000_steps.zip").write_text("")
    return str(run)


def test_scan_ranks_worst_turning_point_per_battle(tmp_path):
    run = _build_scan_run(tmp_path)
    rows = ProbeSession(run).scan(outcome="loss")
    assert [r["short_id"].split("/")[-1] for r in rows] == ["loss_006", "loss_007"]  # -8 before -4
    w = rows[0]["worst"]
    assert w["inv"] == 0 and abs(w["delta_v"] - (-8.0)) < 1e-9
    assert abs(w["td_residual"] - (-1.0 + 0.95 * 2.0 - 10.0)) < 1e-9
    assert w["chosen"] == "earthquake" and w["our_active"].startswith("zapdos")
    assert rows[0]["opponent"] == "aggressive_v2" and rows[0]["turns"] == 2


def test_scan_outcome_and_opponent_filters(tmp_path):
    run = _build_scan_run(tmp_path)
    sess = ProbeSession(run)
    assert len(sess.scan(outcome="loss")) == 2          # the win is excluded
    assert len(sess.scan()) == 3                          # unfiltered: all three battles
    assert [r["short_id"].split("/")[-1] for r in sess.scan(outcome="win")] == ["win_001"]
    assert all(r["opponent"] == "aggressive_v2"
               for r in sess.scan(opponent="aggressive_v2"))


def test_scan_metric_td_reorders(tmp_path):
    run = _build_scan_run(tmp_path)
    # by ΔV: loss_006(-8) first; by TD: loss_007(-14.05) first
    by_dv = ProbeSession(run).scan(outcome="loss", metric="value_drop")
    by_td = ProbeSession(run).scan(outcome="loss", metric="td_residual")
    assert by_dv[0]["short_id"].endswith("loss_006")
    assert by_td[0]["short_id"].endswith("loss_007")
    assert abs(by_td[0]["worst"]["td_residual"] - (-10.0 + 0.95 * 1.0 - 5.0)) < 1e-9


def test_scan_limit_and_bad_metric(tmp_path):
    run = _build_scan_run(tmp_path)
    assert len(ProbeSession(run).scan(outcome="loss", limit=1)) == 1
    with pytest.raises(ValueError):
        ProbeSession(run).scan(metric="nonsense")


def test_cli_scan_emits_json(tmp_path):
    import sys
    import main.prober.query as q
    run = _build_scan_run(tmp_path)
    argv = sys.argv
    sys.argv = ["query", "scan", run, "--outcome", "loss", "--metric", "td_residual", "--limit", "1"]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            q.main()
    finally:
        sys.argv = argv
    parsed = json.loads(buf.getvalue())
    assert len(parsed) == 1 and parsed[0]["short_id"].endswith("loss_007")


def test_cli_error_envelope(tmp_path):
    import sys
    import main.prober.query as q
    argv = sys.argv
    sys.argv = ["query", "analyze", str(tmp_path / "nope_summary.json"), "0"]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as ei:
                q.main()
        assert ei.value.code == 1
        assert "error" in json.loads(buf.getvalue())
    finally:
        sys.argv = argv


def test_cli_overview_emits_json(tmp_path):
    import main.prober.query as q
    import sys
    run, summ = _build_run(tmp_path)
    argv = sys.argv
    sys.argv = ["query", "overview", summ]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            q.main()
    finally:
        sys.argv = argv
    parsed = json.loads(buf.getvalue())
    assert parsed["meta"]["step"] == 2000000 and len(parsed["invocations"]) == 2


# --- triage (loss attribution orchestration) -------------------------------

def _loss_inv(turn, chosen, *, hp, hp_delta):
    """One decision carrying our active HP + the resolved HP delta (so faint / v_at categorize).
    No belief block in the test obs → active_pko is None, so only the model-free categories fire."""
    return {"i": turn, "turn": turn, "phase": "move_selection", "chosen": chosen,
            "our": {"species": "zapdos", "hp": f"{hp * 100:.0f}%"},
            "opp": {"species": "jynx"},
            "actions": {chosen: {"prob": "80.0%", "valid": True}},
            "outcome": {"reward": {"total": -1.0},
                        "our": {"hp_delta": f"{hp_delta * 100:+.0f}%"}, "events": []}}


def _build_triage_run(tmp_path):
    """Four losses spanning the model-free buckets across a bot + a non-bot opponent, plus an
    eval_results.jsonl so only the bot opponents are rating-weighted."""
    run = tmp_path / "run"
    # heuristic (a bot): a critic_blindspot (V>0 then craters) + a greedy_setup
    _write_battle(run, "heuristic", "loss_001",
                  [_loss_inv(1, "thunderbolt", hp=0.8, hp_delta=-0.1)], [5.0, -10.0])
    _write_battle(run, "heuristic", "loss_002",
                  [_loss_inv(1, "dragondance", hp=0.9, hp_delta=-0.1)], [5.0, -10.0])
    # aggressive (a bot): an attrition_death (mon died, no belief block → not surprise/ignored)
    _write_battle(run, "aggressive", "loss_003",
                  [_loss_inv(1, "earthquake", hp=0.5, hp_delta=-0.5)], [5.0, -10.0])
    # sentinel_0 (NOT a bot — absent from eval_results): a positional_grind (V<0, no death)
    _write_battle(run, "sentinel_0", "loss_004",
                  [_loss_inv(1, "protect", hp=0.8, hp_delta=-0.1)], [-5.0, -12.0])
    with open(run / "eval_results.jsonl", "w") as f:
        f.write(json.dumps({"step": 2000000, "bots": {"heuristic": 0.8, "aggressive": 0.7}}) + "\n")
    (run / "checkpoint_3200000_steps.zip").write_text("")
    return str(run)


def test_triage_categorizes_and_ranks_by_recoverable(tmp_path):
    run = _build_triage_run(tmp_path)
    out = ProbeSession(run).triage()
    assert out["step"] == 2000000 and out["n_losses_analyzed"] == 4
    assert out["n_bot_opponents"] == 2                          # heuristic + aggressive only
    cats = {c["category"]: c for c in out["categories"]}
    assert set(cats) == {"critic_blindspot", "greedy_setup", "attrition_death", "positional_grind"}
    # attrition_death (1 of 1 aggressive losses, loss_rate 0.30) outranks the heuristic buckets
    # (1 of 2 losses, loss_rate 0.20): 0.30·1·/2 = 15% vs 0.20·0.5/2 = 5%.
    assert out["categories"][0]["category"] == "attrition_death"
    assert abs(cats["attrition_death"]["est_recoverable_winrate_pct"] - 15.0) < 1e-6
    assert abs(cats["critic_blindspot"]["est_recoverable_winrate_pct"] - 5.0) < 1e-6
    # sentinel loss is counted in the share but NOT rating-weighted (no bot win-rate for it)
    assert cats["positional_grind"]["est_recoverable_winrate_pct"] == 0.0
    assert cats["positional_grind"]["by_opponent"] == {"sentinel_0": 1}


def test_triage_opponent_filter_and_levers(tmp_path):
    run = _build_triage_run(tmp_path)
    out = ProbeSession(run).triage(opponent="heuristic")
    cats = {c["category"] for c in out["categories"]}
    assert cats == {"critic_blindspot", "greedy_setup"} and out["n_losses_analyzed"] == 2
    # every category carries a concrete lever + examples (the actionable output)
    for c in out["categories"]:
        assert c["lever"] and c["examples"]


def test_triage_no_eval_results_falls_back_to_volume(tmp_path):
    run = _build_triage_run(tmp_path)
    os.remove(os.path.join(run, "eval_results.jsonl"))           # no bot win-rates available
    out = ProbeSession(run).triage()
    assert out["n_bot_opponents"] == 0
    assert all(c["est_recoverable_winrate_pct"] == 0.0 for c in out["categories"])
    # ranked by raw loss volume; the fallback is announced in the metric + a caveat
    counts = [c["n"] for c in out["categories"]]
    assert counts == sorted(counts, reverse=True)
    assert "raw loss volume" in out["ranking_metric"].lower() or "raw" in out["ranking_metric"].lower()
    assert any("bot win-rates" in cav.lower() for cav in out["caveats"])


def test_triage_empty_run_is_graceful(tmp_path):
    run = tmp_path / "empty"
    (run / "eval_traces").mkdir(parents=True)
    out = ProbeSession(str(run)).triage()
    assert out["n_losses_analyzed"] == 0 and out["categories"] == []


def test_cli_triage_emits_json(tmp_path):
    import sys
    import main.prober.query as q
    run = _build_triage_run(tmp_path)
    argv = sys.argv
    sys.argv = ["query", "triage", run]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            q.main()
    finally:
        sys.argv = argv
    parsed = json.loads(buf.getvalue())
    assert parsed["categories"][0]["category"] == "attrition_death"
    assert parsed["n_bot_opponents"] == 2


# --- representation probe (orchestration end-to-end, fake model) ------------

# Real Gen3 base speeds (the is_faster label): zapdos 100, blissey 55 (gap 45 = easy),
# tyranitar 61, swampert 60 (gap 1 = contested). Cycle 4 matchups so labels + both groups appear.
_PROBE_MATCHUPS = [("zapdos", "blissey", 1), ("blissey", "zapdos", 0),
                   ("tyranitar", "swampert", 1), ("swampert", "tyranitar", 0)]
_PROBE_OBS_LEN = 64


class _FakeFeatureModel:
    """Returns vf features whose dim 0 IS the label signal baked into obs[0] — so a probe on vf
    must recover the label (and a probe on noise must not). Mirrors ProbeModel.features' contract."""
    def features(self, obs, mask):
        v = np.asarray(obs, dtype=np.float64)[:16].copy()
        return {"vf": v, "pi": v}


def _fake_choice():
    from main.prober.discovery import ModelChoice
    return ModelChoice("ckpt.zip", "recent", "fake", None, 2000000)


_FAKE_CHOICE = _fake_choice()


def _build_probe_run(tmp_path, n_per_battle=18, n_battles=3):
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    for bi in range(n_battles):
        invs, obs = [], []
        for k in range(n_per_battle):
            ours, opp, label = _PROBE_MATCHUPS[(bi * n_per_battle + k) % len(_PROBE_MATCHUPS)]
            invs.append({
                "i": k, "turn": k + 1, "phase": "move_selection", "chosen": "tackle",
                "our": {"species": ours, "hp": "80%"}, "opp": {"species": opp},
                "actions": {"tackle": {"prob": "80.0%", "valid": True}},
                "outcome": {"reward": {"total": -1.0}, "our": {"hp_delta": "-20%"}, "events": []}})
            row = rng.standard_normal(_PROBE_OBS_LEN) * 0.3
            row[0] = (3.0 if label == 1 else -3.0) + 0.2 * rng.standard_normal()   # label baked in
            obs.append(row)
        bd = run / "eval_traces" / "step_2000000" / f"opp{bi}"
        os.makedirs(bd, exist_ok=True)
        with open(bd / f"loss_{bi:03d}_summary.json", "w") as f:
            json.dump({"meta": {"step": 2000000, "result": "LOSS", "turns": n_per_battle,
                                "invocations": n_per_battle},
                       "teams": {"ours": [], "opponent": []}, "invocations": invs}, f)
        np.savez(bd / f"loss_{bi:03d}_states.npz",
                 obs=np.array(obs, dtype=np.float32),
                 logits=np.zeros((n_per_battle, 11), dtype=np.float32),
                 values=np.zeros(n_per_battle, dtype=np.float32),
                 has_state=np.ones(n_per_battle, dtype=np.int8))
        if not os.path.exists(bd.parent / "eval_manifest.json"):
            with open(bd.parent / "eval_manifest.json", "w") as f:
                json.dump({"step": 2000000, "git_hash": "abc", "arch_signature": "x",
                           "snapshot": None}, f)
    with open(run / "metadata.json", "w") as f:
        json.dump({"gamma": 0.99}, f)
    (run / "checkpoint_2000000_steps.zip").write_text("")   # for path resolution; loader is faked
    return str(run)


def test_probe_recovers_label_from_representation(tmp_path):
    run = _build_probe_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda p: _FakeFeatureModel())
    out = sess.probe("is_faster", max_decisions=200)
    assert out["target"] == "is_faster" and out["task"] == "classification"
    assert out["n_decisions"] >= 40
    rep = out["representation_probe"]["overall"]
    assert rep["accuracy"] > 0.9 and rep["auc"] > 0.95     # the baked-in signal is recovered
    # both groups appear (easy = zapdos/blissey, contested = ttar/swampert)
    assert set(out["representation_probe"]["by_group"]) == {"easy", "contested"}
    # synthetic 64-dim obs → the live-offset belief decode finds nothing → no provided baseline
    assert out["provided_feature_baseline"] is None


def test_probe_noise_features_do_not_recover(tmp_path):
    run = _build_probe_run(tmp_path)

    class _NoiseModel:
        def features(self, obs, mask):
            r = np.random.default_rng(int(abs(obs[1] * 1e6)) % (2**32)).standard_normal(16)
            return {"vf": r, "pi": r}                       # features independent of the label

    out = ProbeSession(run, model_loader=lambda p: _NoiseModel()).probe("is_faster", max_decisions=200)
    assert abs(out["representation_probe"]["overall"]["lift"]) < 0.15   # no signal → ~baseline


def test_probe_unknown_target_and_too_few(tmp_path):
    run = _build_probe_run(tmp_path, n_per_battle=4, n_battles=1)
    sess = ProbeSession(run, model_loader=lambda p: _FakeFeatureModel())
    with pytest.raises(ValueError):
        sess.probe("not_a_target")
    out = sess.probe("is_faster")                            # only 4 decisions → graceful error
    assert "error" in out and out["n_decisions"] < 30


class _FakeGradModel:
    """Fake exposing the grad/dist surface history_saliency needs. Policy grad concentrates on the
    first two history slots; value grad is flat across history — so the per-slot breakdown is checkable."""
    offsets = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5,
                         turn_history_offset=40, turn_history_dim=20, turn_delta_dim=4)  # 5 slots × 4

    def action_dist(self, obs, mask):
        n = len(mask)
        return np.ones(n) / n, np.zeros(n)

    def logit_grad(self, obs, mask, idx):
        g = np.zeros(_PROBE_OBS_LEN)
        g[40:44] = 1.0          # history slot 0 high
        g[44:48] = 0.5          # slot 1 medium; slots 2-4 zero
        return g

    def value_grad(self, obs, mask):
        g = np.zeros(_PROBE_OBS_LEN)
        g[40:60] = 0.2          # flat across all 5 history slots
        return g


def test_history_saliency_per_slot_breakdown(tmp_path):
    run = _build_probe_run(tmp_path)
    out = ProbeSession(run, model_loader=lambda p: _FakeGradModel()).history_saliency(max_decisions=50)
    assert out["n_history_turns"] == 5 and out["n_decisions"] >= 40
    sl = out["slots"]
    # policy: slot0 > slot1 > slots2-4 (which are exactly 0)
    assert sl[0]["policy_saliency_norm"] > sl[1]["policy_saliency_norm"] > 0
    assert sl[2]["policy_saliency_norm"] == 0.0 and sl[4]["policy_saliency_norm"] == 0.0
    # value: flat across slots
    assert sl[0]["value_saliency_norm"] == sl[4]["value_saliency_norm"] > 0


def test_cli_probe_emits_json(tmp_path):
    import sys
    import main.prober.query as q
    import main.prober.session as sess_mod
    run = _build_probe_run(tmp_path)
    # Patch ProbeModel.load (the CLI builds its own session) to the fake feature model.
    orig = sess_mod.ProbeSession._model_for
    sess_mod.ProbeSession._model_for = lambda self, b: (_FakeFeatureModel(), _FAKE_CHOICE)
    argv = sys.argv
    sys.argv = ["query", "probe", run, "is_faster", "--max-decisions", "200"]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            q.main()
    finally:
        sys.argv = argv
        sess_mod.ProbeSession._model_for = orig
    parsed = json.loads(buf.getvalue())
    assert parsed["target"] == "is_faster"
    assert parsed["representation_probe"]["overall"]["accuracy"] > 0.9
