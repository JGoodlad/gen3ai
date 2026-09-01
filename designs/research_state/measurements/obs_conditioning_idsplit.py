"""M8 / part 2 — split the observation into EMBEDDING-INDEX columns and SCALAR columns, and
re-read the participation ratio on each.

WHY THIS IS THE DECIDING MEASUREMENT. A covariance participation ratio treats every column as a
scalar magnitude. The gen3 observation is not all scalars: a fixed set of positions carry RAW DEX
NUMBERS that the extractor casts with `.long()` and feeds to `nn.Embedding` — species num (1..386),
move num (1..370), item num (1..600), type/ability/status/cant/faint-cause/item-transition ids.
Those columns never touch a Linear as a number (`slice_pokemon_categoricals` even ZEROES the raw
last-action column after extracting it — "a raw dex num must never reach a Linear; the manifest
rule"). Their variance is a property of the DEX NUMBERING, not of the state, so every direction
they contribute to the obs covariance is a direction the network does not consume as scale.

So the question "is the observation badly conditioned?" has to be asked of the SCALAR columns
alone — the ones that actually arrive at a weight matrix as magnitudes.

WHICH COLUMNS ARE IDs is read from each era's OWN declarations, never hardcoded:
  * per-mon: `Gen3ObservationEncoder.get_layout()['pokemon']` + the same field names
    `slice_pokemon_categoricals` reads (species_id / items.id / types.type{1,2} /
    abilities.id{1,2} / moves[i].{id,type}) + `POKEMON_LAST_ACTION_OFFSET`
  * board: `reactive_layout['active_req_moves']` — the first 2*per entries are move ids + type ids
    (`extractor_ctx` casts exactly those two runs with `.long()`)
  * event window: the `EventCol` members `EventSeats.forward` casts with `.long()`
  * turn frames: `turn_delta_encoder.TURN_DELTA_EMBEDDED_IDS`, the declared manifest

The dumper runs INSIDE each era's pinned worktree, so gen-12's and gen-13's manifests are their
own rather than today's projected backwards.

Run: nice -n 15 python designs/research_state/measurements/obs_conditioning_idsplit.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Reads models/ READ-ONLY. Writes /tmp/m8obs/obs_idsplit.json
"""
import json
import os
import subprocess
import sys

import numpy as np

OUT = "/tmp/m8obs/obs_idsplit.json"


DUMPER = r'''
import json, sys
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation import constants as K

L = Gen3ObservationEncoder(load_mappings()).get_layout()
D = int(L["total_dim"])
ids = {}          # abs col -> which table it routes to


def mark(col, kind):
    ids[int(col)] = kind


# ---- per-mon categoricals: the exact fields slice_pokemon_categoricals reads ----------------
pk = L["pokemon"]
for team in ("our_team", "opp_team"):
    base0 = L["parts"][team]["start"]
    n_slots, w = L["parts"][team]["reshape"]
    for s in range(int(n_slots)):
        b = int(base0) + s * int(w)
        mark(b + pk["species"]["offset"] + pk["species"]["layout"]["species_id"]["offset"],
             "species")
        mark(b + pk["items"]["offset"] + pk["items"]["layout"]["id"]["offset"], "item")
        mark(b + pk["types"]["offset"] + pk["types"]["layout"]["type1"]["offset"], "type")
        mark(b + pk["types"]["offset"] + pk["types"]["layout"]["type2"]["offset"], "type")
        mark(b + pk["abilities"]["offset"] + pk["abilities"]["layout"]["id1"]["offset"],
             "ability")
        mark(b + pk["abilities"]["offset"] + pk["abilities"]["layout"]["id2"]["offset"],
             "ability")
        mv = pk["moves"]
        toff = mv["layout"]["slot_layout"]["type"]["offset"]
        for sl in mv["layout"]["slots"]:
            mark(b + mv["offset"] + sl["offset"], "move")
            mark(b + mv["offset"] + sl["offset"] + toff, "type")
        lm = getattr(K, "POKEMON_LAST_ACTION_OFFSET", None)
        if lm is not None:
            mark(b + int(lm), "move")

# ---- board: active-request move ids + their type ids -----------------------------------------
rl = L["reactive_layout"]
if "active_req_moves" in rl:
    arm = rl["active_req_moves"]
    b = L["parts"]["reactive"]["start"] + int(arm["offset"])
    per = int(arm["per"])
    for j in range(per):
        mark(b + j, "move")
        mark(b + per + j, "type")

# ---- event window: exactly the columns EventSeats casts with .long() -------------------------
if "event_window_offset" in L:
    tok = int(L["event_token_dim"])
    C = getattr(K, "EVENT_COL", None) or getattr(K, "EventCol", None)
    if C is not None:
        names = [("TYPE", "event_kind"), ("ACTOR_SPECIES", "species"),
                 ("TARGET_SPECIES", "species"), ("MOVE", "move"), ("STATUS", "status"),
                 ("CANT", "cant"), ("FAINT_CAUSE", "faint"), ("ITEM_TRANSITION", "itemtr")]
        cols = [(int(getattr(C, n)), k) for n, k in names if hasattr(C, n)]
    else:
        # gen-13 predates `gen3_event_col_names_v1` — the columns were a documented block of
        # literals in its own `constants.py` (lines 214-226 of that revision). Those five
        # positions are the SAME five the named enum later froze, and the assert makes the
        # fallback unusable against any other token layout.
        assert tok == 19, f"unnamed event columns at token dim {tok}"
        cols = [(0, "event_kind"), (1, "species"), (3, "species"), (4, "move"), (15, "status")]
    for t in range(int(L["event_window_n"])):
        b = int(L["event_window_offset"]) + t * tok
        for c, k in cols:
            mark(b + c, k)

# ---- turn frames: the declared TURN_DELTA_EMBEDDED_IDS manifest -------------------------------
if "turn_history_offset" in L:
    from agents.observation.turn_delta_encoder import TURN_DELTA_EMBEDDED_IDS
    w = int(L["turn_delta_dim"])
    for t in range(int(L["n_history_turns"])):
        b = int(L["turn_history_offset"]) + t * w
        for pos, kind in TURN_DELTA_EMBEDDED_IDS:
            mark(b + int(pos), kind)

print(json.dumps({"total_dim": D, "id_cols": ids}))
'''


def dump_manifest(worktree):
    """Run the dumper inside `worktree` (None = the live tree) and return {col: kind}."""
    env = dict(os.environ)
    root = worktree or os.getcwd()
    env["PYTHONPATH"] = f"{root}/src"
    p = subprocess.run([sys.executable, "-c", DUMPER], cwd=root, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"manifest dump failed in {root}:\n{p.stderr[-2000:]}")
    d = json.loads(p.stdout)
    return int(d["total_dim"]), {int(k): v for k, v in d["id_cols"].items()}


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from obs_conditioning_probe import GENERATIONS as GENS
    from obs_conditioning_probe import blocks_of, column_names, live_mask, load_states
    from obs_conditioning_probe import boot_ci, pr_cov, zscore
    _LIVE = []

    def live_layout():
        if not _LIVE:
            from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
            _LIVE.append(Gen3ObservationEncoder(load_mappings()).get_layout())
        return _LIVE[0]

    os.makedirs("/tmp/m8obs", exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    WT = {f"{here}/obs_layout_gen12.json": "/tmp/m8_gen12",
          f"{here}/obs_layout_gen13.json": "/tmp/m8_gen13", "live": None}
    for wt in (v for v in WT.values() if v):
        if not os.path.isdir(wt):
            raise SystemExit(
                f"missing era worktree {wt}. This script EXECUTES each era's own manifest "
                "declarations, so a cached layout dump is not enough. Create them with:\n"
                "  git -C <main checkout> worktree add --detach /tmp/m8_gen12 ede5a887ea05\n"
                "  git -C <main checkout> worktree add --detach /tmp/m8_gen13 1fa47332deb8\n"
                "(each hash is that run's own metadata.json git_hash; remove them afterwards)")
    res = {"probe": "M8 part 2 — embedding-index vs scalar columns", "generations": {}}
    for label, run, step, lsrc in GENS:
        if lsrc is None:            # v8: no era layout reconstructed, deliberately out of scope
            continue
        D_man, idmap = dump_manifest(WT[lsrc])
        X, S, meta = load_states(run, step)
        D = X.shape[1]
        assert D == D_man, (label, D, D_man)
        is_id = np.zeros(D, bool)
        is_id[list(idmap)] = True
        m = live_mask(X)
        v = X.astype(np.float64).var(0)
        tot = float(v[m].sum())
        idm, scm = is_id & m, (~is_id) & m
        by_kind = {}
        for c, k in idmap.items():
            if m[c]:
                cell = by_kind.setdefault(k, {"n": 0, "var": 0.0})
                cell["n"] += 1
                cell["var"] += float(v[c])
        for k in by_kind:
            by_kind[k]["var_share"] = by_kind[k]["var"] / tot
        row = {"run": run, "trace": step, "obs_dim": D, **meta,
               "n_id_cols": int(is_id.sum()), "n_id_cols_live": int(idm.sum()),
               "n_scalar_cols_live": int(scm.sum()),
               "id_var_share": float(v[idm].sum() / tot),
               "scalar_var_share": float(v[scm].sum() / tot),
               "pr_full_raw": pr_cov(X[:, m]),
               "pr_ids_only_raw": pr_cov(X[:, idm]),
               "pr_scalars_only_raw": pr_cov(X[:, scm]),
               "pr_scalars_only_z": pr_cov(zscore(X[:, scm], np.ones(int(scm.sum()), bool))),
               "id_kinds": by_kind,
               # Cluster bootstrap over trace files. NOTE the estimator is biased DOWNWARD under
               # resampling-with-replacement (a duplicated battle is a perfectly correlated row
               # pair, which concentrates the covariance), so the point estimate can sit above the
               # interval. Read the WIDTH as the sampling scale, not the location as a bound.
               "pr_scalars_ci95": boot_ci(X[:, scm], S, np.random.default_rng(20260831))}
        row["pr_scalars_per_live_dim"] = row["pr_scalars_only_raw"] / max(1, int(scm.sum()))
        # top variance columns, each labelled ID-or-scalar — the claim "the loudest columns are
        # the ones the network never reads as numbers" is checkable row by row here.
        order = np.argsort(np.where(m, v, -1.0))[::-1][:20]
        row["top_variance_columns"] = [
            {"col": int(c), "std": float(np.sqrt(v[c])),
             "var_share": float(v[c] / tot),
             "kind": idmap.get(int(c), "SCALAR")} for c in order]
        row["top20_id_fraction"] = float(
            sum(1 for c in order if int(c) in idmap) / len(order))
        # ---- the conditioning question, asked of the SCALAR columns only -------------------
        # If the network's real scalar inputs span a huge dynamic range, that IS an optimization
        # liability regardless of what the ID columns do. If they do not, there is nothing here
        # for a normalizer to fix.
        L = live_layout() if lsrc == "live" else json.load(open(lsrc))
        names = column_names(L, blocks_of(L), D)
        sv = v[scm]
        sidx = np.flatnonzero(scm)
        o2 = np.argsort(sv)[::-1]
        row["scalar_scale"] = {
            "std_max": float(np.sqrt(sv[o2[0]])),
            "std_p99": float(np.sqrt(np.percentile(sv, 99))),
            "std_median": float(np.sqrt(np.median(sv))),
            "std_min": float(np.sqrt(sv[o2[-1]])),
            "max_over_median": float(np.sqrt(sv[o2[0]] / max(np.median(sv), 1e-300))),
            "top10": [{"col": int(sidx[i]), "name": names[sidx[i]],
                       "std": float(np.sqrt(sv[i]))} for i in o2[:10]]}
        res["generations"][label] = row
        print(f"{label:32s} D={D:5d} idcols={int(idm.sum()):4d} "
              f"idvar={row['id_var_share']:.4f} PRfull={row['pr_full_raw']:7.2f} "
              f"PRscalar={row['pr_scalars_only_raw']:7.2f} "
              f"PRscalar_z={row['pr_scalars_only_z']:7.2f}")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
