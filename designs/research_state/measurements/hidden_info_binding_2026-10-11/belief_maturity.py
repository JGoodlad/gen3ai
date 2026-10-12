"""BELIEF MATURITY meter (DESCRIPTIVE, no outcome read): how accurate is the network's OWN hidden-information
belief at a checkpoint, on the value-discrimination pilot's 600 registered root states of its run, and how far
above the Smogon prior it fuses with?

One CPU forward per root, the checkpoint loaded at ITS OWN pin (``run_at_pin.sh``; PYTHONPATH = the pin checkout).
The root rows are re-encoded from the bank's input log by the pin's ``core_events`` (the replay the pilot used,
`main.policy_spectrum.replay`), and the truth is the bank battle's packed team of the side OPPOSITE the decision's.

What the searched side has SEEN (its own stream, up to the root decision's tracker entry ``after``):
- a species: a ``switch`` / ``drag`` line naming an opponent mon;
- an item: any ``-item`` / ``-enditem`` line on that mon, or any line on that mon carrying ``[from] item:``
  (an APPROXIMATION of "revealed": a Trick / Knock Off reveal counts; an item named only on the OTHER mon's line,
  e.g. ``[of]`` — none in gen 3 that we rely on — does not).
- moves: read from the network's own obs (the roster's revealed move ids), which is what ``move_w`` excludes.

Three meters, per checkpoint:
- MOVES, per SEEN opponent mon (obs slot not a hypothesis) with r < 4 revealed moves: the X5 roster's fixed-mass
  ``move_w`` (Σ over the unrevealed candidates = 4 − r): ``exp_recall`` = Σ_{true unrevealed m} π_m / (4 − r);
  ``topk_recall`` = |top-(4 − r) by π ∩ true unrevealed| / (4 − r). The SMOGON baseline: the species' Smogon move
  usage (``gen3_move_priors.json``) renormalised over the SAME candidate set, same two numbers. A true
  ``hiddenpower<type>`` counts as that typed move channel; when Hidden Power is already revealed (typeless) its typed
  channels carry a TYPE distribution, so they are dropped from the candidates and the true HP is not "unrevealed".
  A mon whose count of true unrevealed moves differs from 4 − r (a < 4-move set, an obs / truth disagreement) is
  EXCLUDED and counted (``moves_excluded_count_mismatch``).
- ITEMS, per SEEN opponent mon whose item the searched side has not seen revealed: P(true item) under the network's
  item posterior (softmax of ``ext.last_item_logits``, the item-num axis) and top-1 accuracy; the SMOGON baseline is
  the head's own fused prior buffer ``item_belief_head.item_prior[species num]`` (the cold-start posterior).
  ``nothing`` (no item) is item num 0, a legitimate class.
- SPECIES, per root with a never-seen slot: Σ_{true hidden species} π_s / k with π = ``ext.last_hypothesis.species.pi``
  (Σ π = k, the unrevealed count) — `belief_accuracy_read.py`'s ``belief_expected_hit_share``; plus the top-k recall.

Every meter is a mean over its units; 95 % intervals are a CLUSTER bootstrap over BATTLES (2,000 draws,
``numpy.random.default_rng(20261011)``); the 15M − 10M difference is PAIRED (same units, same battle draw).

    run_at_pin.sh <end2|prod> 1 8 2h belief_maturity.py --steps 1M,3M,... [--limit N]
    → belief_maturity.json (this folder; per-cp blocks merged in)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
sys.path.insert(0, str(PILOT))
import common as C  # noqa: E402  (read-only import of the pilot's helpers)
import move_belief as MB  # noqa: E402

TEAM = 6
B_DRAWS = 2000
SEED = 20261011
OUT = HERE / "belief_maturity.json"

#: step label → checkpoint file, per run (the periodic checkpoints; 15M = final_model.zip)
CKPTS = {
    "end2": {"1M": "checkpoints/checkpoint_1000192_steps.zip", "3M": "checkpoints/checkpoint_3000064_steps.zip",
             "5M": "checkpoints/checkpoint_5000095_steps.zip", "7M": "checkpoints/checkpoint_7000147_steps.zip",
             "10M": "checkpoints/checkpoint_10000009_steps.zip", "12M": "checkpoints/checkpoint_12000157_steps.zip",
             "15M": "final_model.zip"},
    "prod": {"5M": "checkpoints/checkpoint_5000029_steps.zip", "10M": "checkpoints/checkpoint_10000002_steps.zip",
             "15M": "final_model.zip"},
}


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def parse(e: str):
    f = e.split("|")
    return f + [""] * (12 - len(f))


def seen_hp(tm: str, seen) -> bool:
    return tm in seen or (tm.startswith("hiddenpower") and "hiddenpower" in seen)


# ── the searched side's view up to the root ────────────────────────────────────────────────────
def stream_view(res: dict, viewer: int, after: int):
    """({species id: nick set}, {species id: item revealed?}) from the viewer's chunks 0..after."""
    opp = f"p{2 - viewer}a: "
    nick, seen, item_seen = {}, set(), set()
    for side, lines in res["chunks"][: after + 1]:
        if side != viewer:
            continue
        for line in lines:
            p = line.split("|")
            if len(p) < 3 or not p[2].startswith(opp):
                continue
            if p[1] in ("switch", "drag") and len(p) > 3:
                sid = toid(p[3].split(",")[0])
                nick[p[2]] = sid
                seen.add(sid)
            elif p[2] in nick and (p[1] in ("-item", "-enditem") or any(x.startswith("[from] item:") for x in p[3:])):
                item_seen.add(nick[p[2]])
    return seen, item_seen


def build_roots(bank, decs):
    """Per decision: the row, mask, the true opponent sets and the searched side's stream view."""
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.replay import _viewer_decisions

    bids = sorted({d["battle"] for d in decs})
    res_by = {}
    for i in range(0, len(bids), 16):
        part = bids[i:i + 16]
        for b, r in zip(part, run_core([bank.battles[bank.battle_index[x]].recorded() for x in part],
                                       trackers=True, obs=True)):
            if not r["ok"]:
                raise RuntimeError(f"{b}: replay failed: {r.get('error')}")
            res_by[b] = r
    roots = []
    for d in decs:
        res = res_by[d["battle"]]
        v = 0 if d["side"] == "p1" else 1
        ds = _viewer_decisions(res, v)
        ent = [e for e in res["trackers"][v] if "obs" in e]
        dec = ds[d["n"]]
        if dec.choice is None:
            raise RuntimeError(f"{d['id']}: the root decision was never answered")
        seen, item_seen = stream_view(res, v, int(ent[d["n"]]["after"]))
        b = bank.battles[bank.battle_index[d["battle"]]]
        opp = b.p2 if d["side"] == "p1" else b.p1
        true = {}
        for e in opp["team"].split("]"):
            f = parse(e)
            # packed: nick | species | ITEM | ability | moves | nature | evs | ...
            true[toid(f[1] or f[0])] = {"item": toid(f[2]) or "nothing", "moves": [toid(m) for m in f[4].split(",") if m]}
        roots.append({"id": d["id"], "battle": d["battle"], "row": np.array(dec.row, dtype=np.float32),
                      "mask": np.asarray(dec.mask, dtype=np.uint8), "seen": seen, "item_seen": item_seen,
                      "true": true})
    return roots


# ── one checkpoint ─────────────────────────────────────────────────────────────────────────────
def measure(model, roots, P_moves, sp_num, num2id, item_num, n2m, threads):
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals

    ext = model.policy.features_extractor
    if getattr(ext, "item_belief_head", None) is None:
        raise RuntimeError("this checkpoint has no item belief head")
    item_prior = ext.item_belief_head.item_prior.float().cpu().numpy()
    units = defaultdict(list)                 # meter → [(battle, value)]
    counts = defaultdict(int)
    with inference_globals(threads), MB.capture_roster() as box:
        for i in range(0, len(roots), 64):
            part = roots[i:i + 64]
            box.clear()
            logits_and_values(model, np.stack([r["row"] for r in part]), np.stack([r["mask"] for r in part]), batch=64)
            slots = MB.slots_from_box(box, n2m)
            hs = ext.last_hypothesis
            pi = hs.species.pi.float().cpu().numpy()
            kk = hs.species.k.cpu().numpy()
            it_post = ext.last_item_logits.float().softmax(-1).cpu().numpy()          # [B,6,I]
            obs_sp = box["ctx"].species_ids[:, TEAM:].long().cpu().numpy()
            for j, r in enumerate(part):
                bt = r["battle"]
                true_by_num = {}
                for sid, t in r["true"].items():
                    true_by_num.setdefault(sp_num.get(sid, 0), []).append((sid, t))
                # SPECIES
                hidden = set(r["true"]) - r["seen"]
                k = float(kk[j])
                if hidden:
                    hn = {sp_num.get(s, 0) for s in hidden}
                    if 0 in hn:
                        counts["species_true_without_num"] += 1
                    if round(k) != len(hidden):
                        counts["species_k_mismatch"] += 1
                    if k > 0:
                        units["species_exp_hit"].append((bt, sum(float(pi[j, n]) for n in hn if n) / k))
                        top = set(int(n) for n in np.argsort(-pi[j])[: int(round(k))])
                        units["species_topk_recall"].append((bt, len(top & hn) / len(hn)))
                # per SEEN slot (obs species known, not a hypothesis)
                for s, sl in enumerate(slots[j]):
                    if sl["hyp"] or sl["num"] <= 0:
                        continue
                    cands = true_by_num.get(sl["num"], [])
                    if len(cands) != 1:
                        counts["seen_slot_no_unique_true_mon"] += 1
                        continue
                    sid, t = cands[0]
                    if sid not in r["seen"]:
                        counts["seen_slot_not_in_stream"] += 1
                    # MOVES
                    rev = set(sl["revealed"])
                    nrev = len(rev)
                    if nrev < 4:
                        hp_known = "hiddenpower" in rev
                        cand = {m: v for m, v in sl["w"].items() if not (hp_known and m.startswith("hiddenpower"))}
                        tm_un = [m for m in t["moves"] if not seen_hp(m, rev)]
                        kslot = 4 - nrev
                        if len(tm_un) != kslot:
                            counts["moves_excluded_count_mismatch"] += 1
                        elif cand:
                            counts["moves_units"] += 1
                            counts["moves_true_not_candidate"] += sum(m not in cand for m in tm_un)
                            netw = cand
                            pri = P_moves.get(sid) or P_moves.get(num2id.get(sl["num"], ""), {})
                            smw = {m: float(pri.get(m, 0.0)) for m in cand}
                            ssum = sum(smw.values())
                            for tag, w, tot in (("net", netw, float(kslot)), ("smogon", smw, ssum)):
                                if tot <= 0:
                                    counts[f"moves_{tag}_zero_mass"] += 1
                                    continue
                                units[f"moves_exp_recall_{tag}"].append(
                                    (bt, sum(w.get(m, 0.0) for m in tm_un) / tot))
                                top = sorted(w, key=lambda m: (-w[m], m))[:kslot]
                                units[f"moves_topk_recall_{tag}"].append((bt, len(set(top) & set(tm_un)) / kslot))
                            units["moves_net_mass_sum_err"].append((bt, abs(sum(cand.values()) - kslot)))
                    # ITEMS
                    if sid in r["item_seen"]:
                        counts["items_revealed_skipped"] += 1
                        continue
                    inum = item_num(t["item"])
                    if inum is None:
                        counts["items_true_unknown_num"] += 1
                        continue
                    if int(obs_sp[j, s]) != sl["num"]:
                        counts["items_slot_species_disagree"] += 1
                        continue
                    post = it_post[j, s]
                    prior = item_prior[sl["num"]] / item_prior[sl["num"]].sum()
                    units["items_p_true_net"].append((bt, float(post[inum])))
                    units["items_top1_net"].append((bt, float(int(np.argmax(post)) == inum)))
                    units["items_p_true_smogon"].append((bt, float(prior[inum])))
                    units["items_top1_smogon"].append((bt, float(int(np.argmax(prior)) == inum)))
    return units, dict(counts)


# ── statistics ────────────────────────────────────────────────────────────────────────────────
def boot_draws(battles, n=B_DRAWS):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(battles), size=(n, len(battles)))
    return idx


def cluster_mean_ci(pairs, battles, draws):
    """mean over units; 95 % percentile CI of a battle-cluster bootstrap with the SHARED draw matrix."""
    bi = {b: i for i, b in enumerate(battles)}
    s = np.zeros(len(battles)); c = np.zeros(len(battles))
    for b, v in pairs:
        s[bi[b]] += v; c[bi[b]] += 1
    if c.sum() == 0:
        return None
    ss, cc = s[draws].sum(1), c[draws].sum(1)
    bs = ss / np.where(cc > 0, cc, np.nan)
    return {"mean": round(float(s.sum() / c.sum()), 4), "ci95": [round(float(np.nanpercentile(bs, 2.5)), 4),
                                                                   round(float(np.nanpercentile(bs, 97.5)), 4)],
            "n_units": int(c.sum()), "n_battles": int((c > 0).sum())}, (s, c)


def paired_diff_ci(sc_a, sc_b, draws):
    """(B − A) of two meters over the SAME battles (the same unit set per battle), the shared draws."""
    (sa, ca), (sb, cb) = sc_a, sc_b
    da = sa[draws].sum(1) / ca[draws].sum(1)
    db = sb[draws].sum(1) / cb[draws].sum(1)
    d = db - da
    return {"diff": round(float(sb.sum() / cb.sum() - sa.sum() / ca.sum()), 4),
            "ci95": [round(float(np.nanpercentile(d, 2.5)), 4), round(float(np.nanpercentile(d, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=None, help="comma list of step labels (default: all for the cp)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from agents.gen3_data import items as gitems
    from agents.inference.belief_decode import _num_to_species_id
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import load_checkpoint

    t0 = time.time()
    data = Path(os.environ["VDP_PIN"]) / "data" / "pokemon"
    P_moves = json.loads((data / "gen3_move_priors.json").read_text())
    sp_num = MB.species_nums(data)
    n2m = MB.move_ids(data)
    num2id = _num_to_species_id()

    def item_num(iid):
        if iid == "nothing":
            return 0
        it = gitems.get(iid)
        return int(it.num) if it is not None else None

    bank = load_bank(C.ARCHIVE / c / "bank")
    decs = sorted(bank.decisions, key=lambda x: x["id"])[: a.limit]
    roots = build_roots(bank, decs)
    battles = sorted({r["battle"] for r in roots})
    draws = boot_draws(battles)
    print(f"[belief_maturity] {c}: {len(roots)} roots / {len(battles)} battles replayed in {time.time() - t0:.0f}s",
          flush=True)
    steps = a.steps.split(",") if a.steps else list(CKPTS[c])
    run_dir = C.MODELS / C.CPS[c]["run"]
    per_step, sc = {}, {}
    for st in steps:
        zp = run_dir / CKPTS[c][st]
        t1 = time.time()
        model = load_checkpoint(zp, device="cpu")
        units, counts = measure(model, roots, P_moves, sp_num, num2id, item_num, n2m, C.threads())
        del model
        blk = {"ckpt": str(zp), "counts": counts}
        sc[st] = {}
        for k, pairs in sorted(units.items()):
            r = cluster_mean_ci(pairs, battles, draws)
            if r is None:
                continue
            blk[k], sc[st][k] = r
        per_step[st] = blk
        print(f"[belief_maturity] {c} {st}: " + json.dumps({k: blk[k]["mean"] for k in blk if isinstance(blk[k], dict)
                                                             and "mean" in blk[k]}) + f" ({time.time() - t1:.0f}s)",
              flush=True)
    diffs = {}
    for lo, hi in (("10M", "15M"), ("5M", "10M"), ("5M", "15M")):
        if lo in sc and hi in sc:
            diffs[f"{hi}-{lo}"] = {k: paired_diff_ci(sc[lo][k], sc[hi][k], draws)
                                   for k in sc[hi] if k in sc[lo] and not k.endswith("mass_sum_err")}
    # net − smogon at each step (paired, same units)
    vs_prior = {}
    for st in sc:
        vs_prior[st] = {}
        for base in ("moves_exp_recall", "moves_topk_recall", "items_p_true", "items_top1"):
            kn, ks = (f"{base}_net", f"{base}_smogon")
            if kn in sc[st] and ks in sc[st]:
                vs_prior[st][base] = paired_diff_ci(sc[st][ks], sc[st][kn], draws)
    block = {"cp": c, "run": C.CPS[c]["run"], "pin": os.environ.get("VDP_PIN_SHA"), "roots": len(roots),
             "battles": len(battles), "bootstrap": {"draws": B_DRAWS, "rng": f"numpy default_rng({SEED})",
                                                    "cluster": "battle"},
             "steps": per_step, "diffs": diffs, "net_minus_smogon": vs_prior,
             "item_revealed_rule": "-item / -enditem on the mon, or any line on the mon with [from] item:",
             "wall_s": round(time.time() - t0, 1)}
    out = Path(a.out)
    cur = json.loads(out.read_text()) if out.exists() else {}
    cur[c] = block
    out.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
    print(f"[belief_maturity] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
