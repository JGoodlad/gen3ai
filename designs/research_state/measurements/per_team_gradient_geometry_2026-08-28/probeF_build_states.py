"""PROBE F step 1 — build PER-TEAM state batches for the 9 R2 slice teams.

Source: the five R2 fork runs' own eval_traces.  Each fork was launched with
--trainee-teams <2 files>, so its traces contain decisions on exactly those two
teams and nothing else -> guaranteed volume on precisely the slice teams.

Writes /tmp/probeF/states_per_team.npz  (numpy only, no torch, no agents import)
"""
import glob
import json
import os
import re

import numpy as np

MODELS = "/home/goodlad/dev/gen3ai/models"
REPO = "/home/goodlad/dev/gen3ai"
OUT = "/tmp/probeF"
PER_TEAM = 1536          # 6 disjoint batches of 256
SEED = 0

# fork -> (run dir, its two --trainee-teams files)
FORKS = {
    "F5a": ("ai_v9_53_R2F5a_0826",
            ["eccfe630ec08de27", "023a2d47648b85e6"]),
    "F5b": ("ai_v9_54_R2F5b_0826",
            ["8e768980fc8f3b5f", "710d8d529538ff90"]),
    "F5c": ("ai_v9_55_R2F5c_0826",
            ["63eda9d8d491d6a4", "f5a4f4f0f5dc49ce"]),
    "F5d": ("ai_v9_56_R2F5d_0826",
            ["e541f7be8713393c", "9eb3abdc52876a63"]),
    "F5e": ("ai_v9_57_R2F5e_0826",
            ["e0d97b0ed592889d", "eccfe630ec08de27"]),
}
# eccfe630 is pinned by BOTH F5a and F5e; the mission fixes its teacher to F5a.
TEACHER_OVERRIDE = {"eccfe630ec08de27": "F5a"}


def species_key(sha):
    p = os.path.join(REPO, "data/teams/sample", sha + ".txt")
    s = set()
    for line in open(p):
        m = re.match(r"^([A-Za-z][A-Za-z0-9'.:\- ]*?)\s*(?:\([MF]\)\s*)?@", line)
        if m:
            s.add(m.group(1).strip().lower().replace(" ", "").replace("-", ""))
    assert len(s) == 6, (sha, s)
    return tuple(sorted(s))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    key2sha, sha2teacher = {}, {}
    for fk, (_run, shas) in FORKS.items():
        for sha in shas:
            key2sha[species_key(sha)] = sha
            sha2teacher.setdefault(sha, fk)
    for sha, fk in TEACHER_OVERRIDE.items():
        sha2teacher[sha] = fk
    shas = sorted(sha2teacher)
    print(f"{len(shas)} distinct slice teams")

    # collect per (team sha) -> list of (obs, logits, values, mask, actions, srcrun)
    pool = {s: [] for s in shas}
    for fk, (run, _shas) in FORKS.items():
        files = sorted(glob.glob(f"{MODELS}/{run}/eval_traces/*/*/*_states.npz"))
        got = {}
        for f in files:
            try:
                d = np.load(f)
                sm = json.load(open(f.replace("_states.npz", "_summary.json")))
            except Exception:
                continue
            tk = tuple(sorted(m["species"].lower().replace(" ", "").replace("-", "")
                              for m in sm["teams"]["ours"]))
            sha = key2sha.get(tk)
            if sha is None:
                got["UNKNOWN"] = got.get("UNKNOWN", 0) + 1
                continue
            keep = d["has_state"].astype(bool)
            if keep.sum() == 0:
                continue
            # PROXY ADVANTAGE (named, directional only): the within-battle TD
            # residual of the RECORDED critic.  Â_t = V(s_{t+1}) - V(s_t) on
            # interior decisions; Â_T = R - V(s_T) at the last, with
            # R = +1 win / -1 loss / 0 tie.  No intermediate reward is
            # recoverable offline, so this is a bootstrap-only residual.
            v_all = d["values"].astype(np.float64)
            res = str(sm["meta"].get("result", "")).upper()
            R = 1.0 if res == "WIN" else (-1.0 if res == "LOSS" else 0.0)
            adv_all = np.empty_like(v_all)
            if len(v_all) > 1:
                adv_all[:-1] = v_all[1:] - v_all[:-1]
            adv_all[-1] = R - v_all[-1]
            # SECOND proxy: the MC (return-to-go) advantage.  gamma = 1 and no
            # intermediate reward is recoverable offline, so R - V(s_t) with the
            # terminal outcome R is the whole return-to-go baseline residual.
            # Far less per-state noise than the 1-step TD residual above, at the
            # cost of being fully credit-blind within a battle.
            advmc_all = (R - v_all).astype(np.float32)
            pool[sha].append(dict(
                obs=d["obs"][keep], logits=d["logits"][keep],
                values=d["values"][keep], mask=d["action_mask"][keep],
                actions=d["actions"][keep], adv=adv_all[keep].astype(np.float32),
                advmc=advmc_all[keep],
                run=run, fileid=files.index(f)))
            got[sha] = got.get(sha, 0) + int(keep.sum())
        print(f"  {fk} ({run}): " + " ".join(f"{k[:8]}={v}" for k, v in sorted(got.items())))

    out = {}
    meta = {"per_team_target": PER_TEAM, "seed": SEED, "teams": {}}
    for sha in shas:
        chunks = pool[sha]
        obs = np.concatenate([c["obs"] for c in chunks])
        lg = np.concatenate([c["logits"] for c in chunks])
        vl = np.concatenate([c["values"] for c in chunks])
        mk = np.concatenate([c["mask"] for c in chunks])
        ac = np.concatenate([c["actions"] for c in chunks])
        ad = np.concatenate([c["adv"] for c in chunks])
        admc = np.concatenate([c["advmc"] for c in chunks])
        fid = np.concatenate([np.full(len(c["obs"]), c["fileid"], np.int32)
                              for c in chunks])
        srcrun = np.concatenate([np.full(len(c["obs"]),
                                         [FORKS[f][0] for f in FORKS].index(c["run"]),
                                         np.int8) for c in chunks])
        n = obs.shape[0]
        # shuffle at the BATTLE level first so disjoint batches are not the same
        # game; then subsample.  (fid is a per-file id = one battle.)
        order = rng.permutation(n)
        take = order[:PER_TEAM]
        out[f"{sha}__obs"] = obs[take]
        out[f"{sha}__rec_logits"] = lg[take]
        out[f"{sha}__rec_values"] = vl[take]
        out[f"{sha}__mask"] = mk[take]
        out[f"{sha}__actions"] = ac[take]
        out[f"{sha}__adv_proxy"] = ad[take]
        out[f"{sha}__adv_mc"] = admc[take]
        out[f"{sha}__fileid"] = fid[take]
        out[f"{sha}__srcrun"] = srcrun[take]
        meta["teams"][sha] = {
            "species": list(species_key(sha)),
            "teacher": sha2teacher[sha],
            "available": int(n),
            "taken": int(min(n, PER_TEAM)),
            "n_battles_available": int(len(np.unique(fid))),
            "n_battles_taken": int(len(np.unique(fid[take]))),
            "src_runs": sorted({FORKS[f][0] for f in FORKS
                                if sha in FORKS[f][1]}),
        }
        print(f"  {sha} teacher={sha2teacher[sha]:4s} avail={n:6d} "
              f"take={min(n, PER_TEAM)} battles={len(np.unique(fid[take]))}")

    np.savez_compressed(f"{OUT}/states_per_team.npz", **out)
    json.dump(meta, open(f"{OUT}/states_per_team_meta.json", "w"), indent=1)
    print("wrote", f"{OUT}/states_per_team.npz")


if __name__ == "__main__":
    main()
