"""THE MODEL-INPUT RUNNER (brief section 3): per registered state, the ROOT forward and the critic's ONE-PLY
successor forwards under every arm, raw outputs only, written under OPAQUE ARM CODES (``arm_key.json``; blinded).

    [HIB_CKPT=end2_s2002] run_at_pin.sh <end2|prod> 2 8 24h model_input.py [--shard i --of W] [--limit N]

Per state (the cp's registered 600; end2_s2002 runs on END2's states):
1. the root row (re-derived by replaying the bank battle at the pin) and the TRUE opponent team (the packed team of the
   side opposite the decision's);
2. the critic's one-ply successors on seeds 40..47 (`main.policy_spectrum.qhat.one_ply`, the TRUE world, the other
   side's open decision answered greedily by the UN-injected network through the pilot's `Fwd`, exactly as
   `run_playouts.one_state`), generated ONCE; their rows / masks saved (``succ/<sha>.npz``);
3. BASE = the forward with no hook installed, ONE batch: row 0 the root, rows 1.. the captured successors;
4. per arm code: the hook installed, its spec set, the same one-batch forward; the target recomputed inside every
   forward from that row's own ctx (`inject.py`), against the same true team.
TEETH (stop the run, exit 3, nothing written for that state; the failure goes to ``fail.jsonl``):
- end2 / prod: BASE's successor q (2V - 1; a terminal branch = its end value) reproduces Phase 1's ``crit`` within 1e-5
  for every (action, seed), with the same branch set and source letters; BASE's root logits reproduce
  ``states_<cp>.json``'s registered root logits within 1e-4 (they are stored rounded to 1e-6).
  end2_s2002 has NO Phase-1 teeth (its successors are its own; its states' registered logits are end2's).
- every cp: the TRUE@0.00 arm (the hook ON, p = 0, targets computed) is bit-identical to BASE at the root and at every
  successor.
A one_ply / replay refusal is a row with ``ok`` false (counted, never dropped), as in the pilot.

-> archive ``~/gen3ai_archive/hidden_info_binding/<cp>/raw/raw_<i>_of_<W>.jsonl`` (one fsync'd row per state),
``audit/audit_<i>_of_<W>.jsonl`` (each code's ROOT target move nums per slot; the scorer never reads it), ``succ/``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hib_common as H  # noqa: E402

H.pin_paths()
import common as PC  # noqa: E402  (the pilot's: cpu_seconds, write_row, read_rows)

ACT = 11
CRIT_LO, CRIT_HI = 40, 48
TOL_CRIT = 1e-5
TOL_ROOT = 1e-4


def f32(a) -> list:
    return [float(x) for x in np.asarray(a, dtype=np.float32).ravel()]


def phase1_crit(cp: str, want: set) -> dict:
    out = {}
    for p in sorted(H.phase1_rows_dir(cp).glob("shard_*.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                i = line.find('"id": "')
                if i < 0:
                    continue
                j = line.index('"', i + 7)
                if line[i + 7:j] not in want:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    out[r["id"]] = r["crit"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cp = H.hib_cp()
    H.check_pin(cp)
    import inject as I
    import run_playouts as RP  # the pilot's Fwd + greedy (Phase 1's exact one-ply answer)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import end_value, logits_and_values, one_ply
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    key_file = json.loads((H.HERE / "arm_key.json").read_text())
    if key_file["codes"] != json.loads(json.dumps(H.arm_codes())):
        raise SystemExit("arm_key.json differs from hib_common.arm_codes()")
    codes = sorted(key_file["codes"])
    null_code = next(c for c in codes if key_file["codes"][c]["family"] == "TRUE" and key_file["codes"][c]["p"] == 0.0)
    teeth_p1 = cp in ("end2", "prod")
    st = json.loads(H.states_file(cp).read_text())
    E = st["states"]
    ids = sorted(E)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard]
    if a.limit:
        ids = ids[: a.limit]
    od = H.out_dir(cp)
    for sub in ("raw", "audit", "succ"):
        (od / sub).mkdir(exist_ok=True)
    raw_path = od / "raw" / f"raw_{a.shard}_of_{a.of}.jsonl"
    aud_path = od / "audit" / f"audit_{a.shard}_of_{a.of}.jsonl"
    done = {r["id"] for q in sorted((od / "raw").glob("raw_*.jsonl")) for r in PC.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[model_input] {cp} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    crit1 = phase1_crit(cp, set(todo)) if teeth_p1 else {}
    if teeth_p1 and set(todo) - set(crit1):
        raise SystemExit(f"{len(set(todo) - set(crit1))} states have no Phase-1 crit row")
    bank = load_bank(H.bank_dir(cp))
    R = H.roots(bank, todo)
    mapper = I.TeamMapper(H.data_dir())
    lib = F.load(F.default_path("release"))
    core = SearchCore(lib=lib)
    t0 = time.monotonic()
    with inference_globals(int(os.environ.get("VDP_THREADS", "2"))):
        model = H.load_model(cp)
        ext = model.policy.features_extractor
        inj = I.Injector(ext)
        fwd_gen = RP.Fwd(model)

        def lav(rows, masks):
            return logits_and_values(model, rows, masks, batch=max(1, len(rows)))

        fr = open(raw_path, "a")
        fa = open(aud_path, "a")
        try:
            for i, did in enumerate(todo):
                r = R[did]
                d = r["d"]
                b = r["battle"]
                side = d["side"]
                row = {"id": did, "cp": cp, "side": side}
                w0, c0 = time.monotonic(), PC.cpu_seconds()
                at = cmd_index(b, side, d["n"])
                if b.commands[at][1] != d["played"]:
                    raise RuntimeError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
                seeds = turn_seeds(did, CRIT_HI)[CRIT_LO:CRIT_HI]
                true_sets = mapper.true_sets(r["opp_team"])

                def greedy_answer(rr, m, j):
                    return RP.greedy(fwd_gen(rr, m.astype(np.uint8)), m)

                try:
                    op = one_ply(core, to_log(b), at, side, seeds, greedy_answer)
                except SuccessorsError as exc:
                    row.update({"ok": False, "error": "one_ply: " + str(exc)[:500]})
                    PC.write_row(fr, row)
                    continue
                if op["tokens"] != d["tokens"]:
                    row.update({"ok": False, "error": "one_ply root tokens differ from the bank"})
                    PC.write_row(fr, row)
                    continue
                c_gen = PC.cpu_seconds() - c0
                brs = op["branches"]
                cap = [k for k, br in enumerate(brs) if br["row"] is not None]
                S = (np.stack([brs[k]["row"] for k in cap]).astype(np.float32) if cap
                     else np.zeros((0, core.obs_dim), np.float32))
                SM = (np.stack([np.asarray(brs[k]["mask"]) for k in cap]).astype(np.uint8) if cap
                      else np.zeros((0, ACT), np.uint8))
                branches = []
                for k, br in enumerate(brs):
                    if br["row"] is None:
                        q = end_value(br["end"], side)
                        branches.append([int(br["action"]), int(br["seed"]), {1.0: "+", -1.0: "-", 0.0: "0"}[q], q])
                    else:
                        branches.append([int(br["action"]), int(br["seed"]), "S" if br["switch_only"] else "V", None])
                tmp = od / "succ" / (H.safe_name(did) + ".tmp.npz")
                np.savez_compressed(tmp, rows=S, masks=SM, meta=np.array(json.dumps({"id": did, "branches": branches})))
                with open(tmp, "rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(tmp, od / "succ" / (H.safe_name(did) + ".npz"))
                # ONE batch per forward: row 0 = the root, rows 1.. = the captured successors (a row's output does
                # not depend on its batch peers up to float reassociation; BASE and every arm share the batch)
                X = np.concatenate([r["row"][None], S], 0)
                XM = np.concatenate([r["mask"][None], SM], 0)
                # BASE (no hook installed)
                c1 = PC.cpu_seconds()
                o = lav(X, XM)
                c_base = PC.cpu_seconds() - c1
                base_root, base_succ = o[0], o[1:, ACT]
                # TEETH vs Phase 1 / the registered root
                teeth = {}
                fail = None
                if teeth_p1:
                    reg = np.asarray(E[did]["logits"], dtype=np.float64)
                    dr = float(np.abs(base_root[:ACT].astype(np.float64) - reg).max())
                    teeth["root_max_abs"] = dr
                    if dr > TOL_ROOT:
                        fail = f"root logits differ from the registered ones by {dr}"
                    p1 = crit1[did]
                    mine = {}
                    for k, (act, s, src, q) in enumerate(branches):
                        mine.setdefault(str(act), {})[s] = (src, q)
                    for j, k in enumerate(cap):
                        act, s = branches[k][0], branches[k][1]
                        mine[str(act)][s] = (branches[k][2], 2.0 * float(base_succ[j]) - 1.0)
                    worst = 0.0
                    if sorted(mine, key=int) != sorted(p1, key=int):
                        fail = f"action sets differ: {sorted(mine)} vs {sorted(p1)}"
                    else:
                        for act, e in p1.items():
                            for s in range(CRIT_HI - CRIT_LO):
                                if s not in mine[act]:
                                    fail = f"action {act} seed {s}: no branch"
                                    continue
                                src, q = mine[act][s]
                                if src != e["src"][s]:
                                    fail = f"action {act} seed {s}: source {src} vs Phase 1 {e['src'][s]}"
                                dq = abs(q - float(e["q"][s]))
                                worst = max(worst, dq)
                                if dq > TOL_CRIT:
                                    fail = f"action {act} seed {s}: q {q:.7f} vs Phase 1 {e['q'][s]} (|d| {dq:.2e})"
                    teeth["crit_max_abs"] = worst
                arms, audit = {}, {}
                c_arms = 0.0
                true_root = None
                with inj.installed():
                    for code in codes:
                        info = key_file["codes"][code]
                        inj.spec = I.Spec(info["family"], float(info["p"]), did, true_sets)
                        c1 = PC.cpu_seconds()
                        o = lav(X, XM)
                        c_arms += PC.cpu_seconds() - c1
                        s = inj.spec.stats
                        inj.spec = None
                        if len(s["per_row"]) != len(X):
                            fail = f"the hook saw {len(s['per_row'])} rows, the batch has {len(X)}"
                        o_root, o_succ = o[0], o[1:, ACT]
                        if code == null_code:
                            if not (np.array_equal(o_root.view(np.uint32), base_root.view(np.uint32))
                                    and np.array_equal(o_succ.view(np.uint32), base_succ.view(np.uint32))):
                                fail = "the hook-on p = 0 arm is not bit-identical to BASE"
                            true_root = s["per_row"][0]
                        rest = s["per_row"][1:]
                        arms[code] = {"root": f32(o_root), "succ": f32(o_succ),
                                      "root_stats": s["per_row"][0],
                                      "succ_slots": int(sum(x["slots"] for x in rest)),
                                      "succ_moves": int(sum(x["moves"] for x in rest))}
                        audit[code] = s["targets"][0]
                if fail is not None:
                    with open(od / "fail.jsonl", "a") as ff:
                        PC.write_row(ff, {"id": did, "cp": cp, "teeth": teeth, "error": fail})
                    print(f"[model_input] TEETH FAILURE {did}: {fail}", flush=True)
                    sys.exit(3)
                row.update({
                    "ok": True, "mask": [int(x) for x in r["mask"]], "branches": branches,
                    "base": {"root": f32(base_root), "succ": f32(base_succ)}, "arms": arms,
                    "n_true_target_slots": int(true_root["slots"]), "n_true_target_moves": int(true_root["moves"]),
                    "teeth": teeth,
                    "cost": {"cpu_gen": round(c_gen, 4), "cpu_base": round(c_base, 4),
                             "cpu_arms": round(c_arms, 4), "n_arms": len(codes), "batch_rows": int(len(X)),
                             "succ_rows": int(len(S)), "branches": len(branches),
                             "cpu_total": round(PC.cpu_seconds() - c0, 4),
                             "wall_s": round(time.monotonic() - w0, 3)},
                })
                PC.write_row(fr, row)
                PC.write_row(fa, {"id": did, "cp": cp, "root_targets": audit})
                el = time.monotonic() - t0
                print(f"[model_input] {cp} {i + 1}/{len(todo)} {el / 60:.1f} min, eta "
                      f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min; succ {len(S)}, "
                      f"cpu {row['cost']['cpu_total']:.1f}s", flush=True)
        finally:
            fr.close()
            fa.close()
            core.close()


if __name__ == "__main__":
    main()
