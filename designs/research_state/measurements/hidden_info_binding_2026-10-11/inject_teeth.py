"""THE INJECTION'S TEETH (brief section 1) on registered ROOT states of one checkpoint. Counts only: no KL, no pick.

    [HIB_CKPT=end2_s2002] run_at_pin.sh <end2|prod> 2 8 2h inject_teeth.py [--need 64] [--max 200]

(a) TRUE p = 0 WITH targets: all 12 outputs (11 raw logits + V) bit-identical to the unhooked forward, on >= --need
    states that have >= 1 targeted slot; also: every hook call count is exactly 1 per forward, and the extractor
    instance is clean (no hook attribute) and reproduces the unhooked bytes after the context exits.
(b) p = 1, every family: on every targeted slot the PUBLISHED posterior (`last_move_belief_logits`) has
    sigma >= 0.999 on each target and <= 1e-5 on every other non-pinned channel; every pinned channel (revealed, 237,
    a revealed HP's typed channels) and every untargeted slot keeps its exact bytes.
(c) p = 1, TRUE (and TRUEHYP): the X5 op roster's fixed-mass `move_w` puts >= 0.99 of each targeted slot's
    unpinned mass on its targets (a seen slot's true unrevealed moves; a matched hypothesis slot's true moves).
(d) TRUE p = 1 changes the 12 outputs (legal logits + V): the count of states with any change, and with a change
    > 1e-3 (this one count; nothing else about the outputs).
Also: PLACEBO targets are disjoint from the true and pinned moves, hold <= 1 Hidden Power, number 4 - r unless the
pool runs short (counted), and are reproducible (a second PLACEBO@1.00 forward: same targets, same bytes); SELFMAP's
targets are the top-(4 - r) of the base posterior (no unchosen candidate above the lowest chosen, HP rule aside).
-> inject_teeth_<cp>.json here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hib_common as H  # noqa: E402

H.pin_paths()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=64)
    ap.add_argument("--max", type=int, default=200)
    a = ap.parse_args()
    cp = H.hib_cp()
    H.check_pin(cp)

    import inject as I
    import move_belief as MB  # the pilot's roster capture
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals

    st = json.loads(H.states_file(cp).read_text())
    ids = sorted(st["states"])[: a.max]
    bank = load_bank(H.bank_dir(cp))
    R = H.roots(bank, ids)
    mapper = I.TeamMapper(H.data_dir())
    res = {"cp": cp, "pin": H.CPS[cp]["pin_cp"], "states_checked": 0, "states_with_targets": 0,
           "a_bit_identical": 0, "a_fail": [], "calls_ok": True, "clean_after": None,
           "b": {}, "c": {}, "d": {"changed_any": 0, "changed_gt_1e-3": 0, "of": 0},
           "placebo": {"slots": 0, "untargeted_by_true": 0, "overlap_true_or_pinned": 0, "gt1_hp": 0, "short": 0,
                       "wrong_count": 0,
                       "repro_targets_equal": 0, "repro_bytes_equal": 0, "of": 0},
           "selfmap": {"slots": 0, "not_topk": 0, "short": 0},
           "truehyp": {"hyp_slots": 0, "hyp_matched": 0, "never_seen": 0, "never_seen_covered": 0},
           "seen_unmatched": 0, "true_unrev_lt_k": 0, "true_not_cand": 0}
    for fam in ("TRUE", "PLACEBO", "SELFMAP", "TRUEHYP"):
        res["b"][fam] = {"slots": 0, "target_min_sigma": 1.0, "other_max_sigma": 0.0, "fail_slots": 0,
                         "pinned_or_untargeted_changed": 0}
    for fam in ("TRUE", "TRUEHYP"):
        res["c"][fam] = {"slots": 0, "min_ratio": 1.0, "fail_slots": 0}
    t0 = time.monotonic()
    with inference_globals(2):
        model = H.load_model(cp)
        ext = model.policy.features_extractor
        res["move_set_closure"] = str(getattr(ext, "move_set_closure", "absent"))
        inj = I.Injector(ext)

        def fwd(row, mask):
            return logits_and_values(model, row[None], mask[None], batch=1)[0]

        for did in ids:
            if res["states_with_targets"] >= a.need:
                break
            r = R[did]
            row, mask = r["row"], r["mask"]
            legal = mask.astype(bool)
            true_sets = mapper.true_sets(r["opp_team"])
            base = fwd(row, mask)
            res["states_checked"] += 1
            with inj.installed():
                def arm(fam, p, record=True, roster=False):
                    inj.spec = I.Spec(fam, p, did, true_sets, record=record)
                    c0 = inj.calls
                    if roster:
                        with MB.capture_roster() as box:
                            out = fwd(row, mask)
                            ro = box["roster"]
                    else:
                        out, ro = fwd(row, mask), None
                    if inj.calls != c0 + 1:
                        res["calls_ok"] = False
                    pub = ext.last_move_belief_logits.detach().float().cpu().numpy()[0]
                    s = inj.spec.stats
                    inj.spec = None
                    return out, ro, pub, s

                out0, _, _, s0 = arm("TRUE", 0.0)
                has = bool(s0["slots"])
                res["seen_unmatched"] += s0["seen_unmatched"]
                if has:
                    res["states_with_targets"] += 1
                    res["true_unrev_lt_k"] += s0["true_unrev_lt_k"]
                    res["true_not_cand"] += s0["true_not_cand"]
                    if np.array_equal(out0.view(np.uint32), base.view(np.uint32)):
                        res["a_bit_identical"] += 1
                    else:
                        res["a_fail"].append(did)
                # (b) / (c) at p = 1
                outs = {}
                for fam in ("TRUE", "PLACEBO", "SELFMAP", "TRUEHYP"):
                    out1, ro, pub, s = arm(fam, 1.0, roster=fam in ("TRUE", "TRUEHYP"))
                    outs[fam] = (out1, s)
                    tgt, on, pin = s["tgt"][0], s["slot_on"][0], s["pinned"][0]
                    bl, nl = s["base_logits"][0], s["new_logits"][0] if "new_logits" in s else s["base_logits"][0]
                    if not np.array_equal(pub.view(np.uint32), nl.view(np.uint32)):
                        res["b"][fam]["pinned_or_untargeted_changed"] += 1000   # the publication is not the rewrite
                    sig = 1.0 / (1.0 + np.exp(-pub.astype(np.float64)))
                    B = res["b"][fam]
                    for sl in range(6):
                        keep = pin[sl] if on[sl] else np.ones_like(pin[sl])
                        if not np.array_equal(pub[sl][keep].view(np.uint32), bl[sl][keep].view(np.uint32)):
                            B["pinned_or_untargeted_changed"] += 1
                        if not on[sl]:
                            continue
                        B["slots"] += 1
                        tmin = float(sig[sl][tgt[sl]].min())
                        other = ~tgt[sl] & ~pin[sl]
                        omax = float(sig[sl][other].max()) if other.any() else 0.0
                        B["target_min_sigma"] = min(B["target_min_sigma"], tmin)
                        B["other_max_sigma"] = max(B["other_max_sigma"], omax)
                        if tmin < 0.999 or omax > 1e-5:
                            B["fail_slots"] += 1
                    if ro is not None:
                        w = ro.move_w.float().cpu().numpy()[0]
                        C = res["c"][fam]
                        for sl in range(6):
                            if not on[sl]:
                                continue
                            un = ~pin[sl]
                            tot = float(w[sl][un].sum())
                            ratio = float(w[sl][tgt[sl]].sum()) / tot if tot > 0 else 0.0
                            C["slots"] += 1
                            C["min_ratio"] = min(C["min_ratio"], ratio)
                            if ratio < 0.99:
                                C["fail_slots"] += 1
                    if fam == "TRUE" and has:
                        d = np.abs(out1[:11][legal].astype(np.float64) - base[:11][legal]).max()
                        d = max(d, abs(float(out1[11]) - float(base[11])))
                        res["d"]["of"] += 1
                        res["d"]["changed_any"] += int(d > 0)
                        res["d"]["changed_gt_1e-3"] += int(d > 1e-3)
                    if fam == "TRUEHYP":
                        for k in ("hyp_slots", "hyp_matched", "never_seen", "never_seen_covered"):
                            res["truehyp"][k] += s[k]
                # placebo / selfmap target validity (row 0 of each family's recorded stats)
                sP, sS, sT = outs["PLACEBO"][1], outs["SELFMAP"][1], outs["TRUE"][1]
                sig_b = 1.0 / (1.0 + np.exp(-sP["base_logits"][0].astype(np.float64)))
                for sl in range(6):
                    tP, tS, tT = sP["targets"][0][sl], sS["targets"][0][sl], sT["targets"][0][sl]
                    if (tP or tS) and not tT:
                        res["placebo"]["untargeted_by_true"] += 1
                    if not tT:
                        continue
                    k = sT["k_slot"][0][sl]
                    pin = set(np.flatnonzero(sP["pinned"][0][sl]).tolist())
                    true_all = set(sT["true_slot"][0][sl])
                    res["placebo"]["slots"] += 1
                    if set(tP) & (true_all | pin):
                        res["placebo"]["overlap_true_or_pinned"] += 1
                    if sum(1 for n in tP if I.is_hp(n)) > 1:
                        res["placebo"]["gt1_hp"] += 1
                    if len(tP) > k:
                        res["placebo"]["wrong_count"] += 1
                    elif len(tP) < k:
                        res["placebo"]["short"] += 1
                    res["selfmap"]["slots"] += 1
                    cand = set(np.flatnonzero(sS["cand"][0][sl]).tolist())
                    if (len(tS) > k or set(tS) & pin or sum(1 for n in tS if I.is_hp(n)) > 1
                            or any(n not in cand for n in tS)):
                        res["selfmap"]["not_topk"] += 1
                        continue
                    if len(tS) < k:
                        res["selfmap"]["short"] += 1
                    if tS:
                        lo = min(sig_b[sl][n] for n in tS)
                        chosen_hp = any(I.is_hp(n) for n in tS)
                        if any(n not in tS and sig_b[sl][n] > lo and not (I.is_hp(n) and chosen_hp) for n in cand):
                            res["selfmap"]["not_topk"] += 1
                # reproducibility of PLACEBO@1.00
                outP2, _, _, sP2 = arm("PLACEBO", 1.0)
                res["placebo"]["of"] += 1
                res["placebo"]["repro_targets_equal"] += int(sP2["targets"] == sP["targets"])
                res["placebo"]["repro_bytes_equal"] += int(np.array_equal(outP2.view(np.uint32),
                                                                         outs["PLACEBO"][0].view(np.uint32)))
            after = fwd(row, mask)
            clean = ("_typed_hp_posterior" not in ext.__dict__ and "_apply_move_belief" not in ext.__dict__
                     and np.array_equal(after.view(np.uint32), base.view(np.uint32)))
            res["clean_after"] = clean if res["clean_after"] is None else (res["clean_after"] and clean)
    res["wall_s"] = round(time.monotonic() - t0, 1)
    ok = (res["states_with_targets"] >= a.need and res["a_bit_identical"] == res["states_with_targets"]
          and res["calls_ok"] and res["clean_after"]
          and all(v["fail_slots"] == 0 and v["pinned_or_untargeted_changed"] == 0 for v in res["b"].values())
          and all(v["fail_slots"] == 0 for v in res["c"].values())
          and res["placebo"]["overlap_true_or_pinned"] == 0 and res["placebo"]["gt1_hp"] == 0
          and res["placebo"]["untargeted_by_true"] == 0 and res["placebo"]["wrong_count"] == 0
          and res["placebo"]["repro_targets_equal"] == res["placebo"]["of"]
          and res["placebo"]["repro_bytes_equal"] == res["placebo"]["of"]
          and res["selfmap"]["not_topk"] == 0 and res["seen_unmatched"] == 0)
    res["PASS"] = bool(ok)
    out = H.HERE / f"inject_teeth_{cp}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("cp", "PASS", "states_with_targets", "a_bit_identical", "d")}), flush=True)
    if not ok:
        sys.exit(4)


if __name__ == "__main__":
    main()
