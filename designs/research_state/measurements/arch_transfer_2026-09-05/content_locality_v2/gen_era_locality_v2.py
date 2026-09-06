"""CONTENT LOCALITY v2, GEN ERA — the same measurement as content_locality/gen_era_locality.py
with THREE corrections. Everything else — teams, seeds, battles, pilot, opponent, bridge,
concurrency, the KL function — is copied verbatim, so the two artifacts are comparable line for
line.

CORRECTION 1 — CHECKPOINT RESOLUTION.
  v1 loaded each teacher from `{run}/final_model.zip`. A fold does NOT: `main/train/model_build.py`
  calls `agents.training.fixed_opponent_pool._resolve_zip_and_config(teacher_path, None)`, whose
  directory rung is `best_model/best_model.zip` -> `final_model.zip` -> `best_model.zip`, and every
  fold in this batch named a RUN DIRECTORY (verified from each fold's recorded
  `cli_args.distill_teacher`). That resolver is IMPORTED here, never re-implemented, and the
  resolved (zip, config) pair is recorded per teacher. All 19 teachers resolve to a file whose
  sha256 differs from what v1 scored (see resolved_teachers.json).

  The PARENTS are unchanged — the folds loaded them by explicit `--model <zip>`, so the resolver
  has nothing to do there.

CORRECTION 2 — TWO REFERENCES.
  The 8 R5F exploiters do not fork from the fold parent. They fork from
  `ai_v9_29_rev1_0823/final_model.zip` @25,067,760; `ai_v9_59_R2ACTION_0827` is their `--exploiter`
  TARGET and is itself a sibling fork of that same checkpoint (verified with
  `python -m main.lineage`). The 8 R5FUND teachers continue from their own R5F final, so rev-1's
  final is the true origin of both halves. Every statistic is therefore reported under

    REF-A  the FOLD PARENT   ai_v9_59_R2ACTION_0827/final_model.zip   (what the fold sees)
    REF-B  the TRUE ORIGIN   ai_v9_29_rev1_0823/final_model.zip       (what the exploiter did)

  Only the reference distribution changes; the states, the teams and the teacher networks are
  identical between the two columns.

  Each reference carries its OWN matched-noise floor. REF-A keeps v1's two adjacent R2ACTION
  checkpoints exactly. REF-B needs a floor of its own — two arbitrary near-identical policies
  measured against rev-1's final — so rev-1's two nearest retained checkpoints are added. That
  addition is declared: it is beyond "keep the same two pairs per era", which is honoured for REF-A.

CORRECTION 3 — THE CLUSTER BOOTSTRAP IS SIZED FROM ITS OWN ARRAY.
  v1 drew one index matrix per EXPECTED cluster count and reused it by name; the v8 arm's pooled-L
  then resampled a 23-cell array with indices drawn in [0, 22). See boot.py, which derives the
  matrix from len(vals) and ASSERTS the drawn range equals the cluster count. The gen arm's own
  sizes were all correct in v1, so this changes gen point estimates not at all and its CIs only by
  the different (correctly sized) draws.

Run: python gen_era_locality_v2.py <out.json> [battles_per_team=3]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import asyncio, json, random, sys, time
import numpy as np
import torch as th
th.set_num_threads(1)
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from agents.training.fixed_opponent_pool import _resolve_zip_and_config
from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boot import Boot   # noqa: E402  -- CORRECTION 3, the size-derived cluster bootstrap

MAIN = "/home/goodlad/dev/gen3ai"
MD = f"{MAIN}/models"
REV1 = f"{MD}/ai_v9_29_rev1_0823"
CFG = f"{REV1}/snapshots/model_config.json"
PARENT = f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"       # REF-A, and the state PILOT
ORIGIN = f"{REV1}/final_model.zip"                            # REF-B, the exploiters' fork origin
OPPONENT = f"{REV1}/snapshots/snapshot_000024000000.zip"

# --- MATCHED-NOISE FLOOR, REF-A: two nearby same-run checkpoints of the PARENT'S OWN run ------
#     (identical to v1 — the brief's "keep the same two adjacent-checkpoint pairs per era")
FLOOR_A = [("FLOORA_ckptA", f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_28067760_steps.zip"),
           ("FLOORA_ckptB", f"{MD}/ai_v9_59_R2ACTION_0827/checkpoints/checkpoint_27917760_steps.zip")]
# --- MATCHED-NOISE FLOOR, REF-B: rev-1's own two nearest retained checkpoints (-79k, -229k) ----
FLOOR_B = [("FLOORB_ckptA", f"{REV1}/checkpoints/checkpoint_24988992_steps.zip"),
           ("FLOORB_ckptB", f"{REV1}/checkpoints/checkpoint_24838992_steps.zip")]


def T(b):
    return f"{MAIN}/data/teams/sample/{b}.txt"


# indices 0..7 -- the reuse batch's UNTAUGHT 8, IN ITS ORDER. Verbatim from v1.
UNTAUGHT = ["61590463ee85d456", "9283210847f806ee", "ce35b7368c3d692e", "9909f2e98e981ccc",
            "9d5f845869e899ee", "f7ba5702fe856292", "90b94599967c6b77", "dbf81d8ecae51c39"]

# The 8 teacher PAIRS of the teacher-content 2x2. RUN DIRECTORIES now (v1 named .zip files) --
# the resolver takes the directory, exactly as the fold's --distill-teacher spec did.
PAIRS = [
    ("00", f"{MD}/ai_v9_92_R5F00_0831",  f"{MD}/ai_v9_120_R5FUND00_0901", ["8bdb5796b9", "436335607f"]),
    ("02", f"{MD}/ai_v9_94_R5F02_0831",  f"{MD}/ai_v9_122_R5FUND02_0901", ["6916e13879", "e702a104eb"]),
    ("04", f"{MD}/ai_v9_96_R5F04_0831",  f"{MD}/ai_v9_124_R5FUND04_0901", ["64a691c473", "e28069562f"]),
    ("06", f"{MD}/ai_v9_98_R5F06_0831",  f"{MD}/ai_v9_126_R5FUND06_0901", ["9e95fb59d7", "9d8391d864"]),
    ("08", f"{MD}/ai_v9_100_R5F08_0831", f"{MD}/ai_v9_128_R5FUND08_0901", ["750d056194", "a9831a5f5e"]),
    ("10", f"{MD}/ai_v9_102_R5F10_0831", f"{MD}/ai_v9_130_R5FUND10_0901", ["6ebe9ebc4d", "9b454d9ea7"]),
    ("12", f"{MD}/ai_v9_104_R5F12_0831", f"{MD}/ai_v9_132_R5FUND12_0901", ["009e3d0244", "b904dbe059"]),
    ("14", f"{MD}/ai_v9_106_R5F14_0831", f"{MD}/ai_v9_134_R5FUND14_0901", ["aea50f207e", "37d717a93a"]),
]
TAUGHT = [b for _, _, _, bs in PAIRS for b in bs]          # 16, indices 8..23
TEAMS = [(b, "untaught") for b in UNTAUGHT] + [(b, "taught") for b in TAUGHT]
IDX = {b: i for i, (b, _) in enumerate(TEAMS)}


def _strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    return m


class PinnedTeam(Gen3Teambuilder):
    def __init__(self, path): super().__init__([open(path).read()])
    def yield_team(self): return self.packed_teams[0]


class PairedPool(Gen3Teambuilder):
    def __init__(self, teams):
        super().__init__(teams); self._seq, self._i = [], 0
    def set_sequence(self, seq):
        self._seq, self._i = seq, 0; return self
    def yield_team(self):
        t = self.packed_teams[self._seq[self._i % len(self._seq)]]; self._i += 1; return t


class Capturing(RLPlayer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.captured = []
    def embed_battle(self, battle):
        d = super().embed_battle(battle)
        if d is not None and d.get("action_mask") is not None and int(d["action_mask"].sum()) > 0:
            self.captured.append({"observation": d["observation"].copy(),
                                  "action_mask": d["action_mask"].copy()})
        return d


def main(out_path, per_team=3):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p, cfg=CFG):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=cfg)
        return _strip_debugger(m)

    parent = load(PARENT)
    opp_model = load(OPPONENT)

    # CORRECTION 1: the training path's own resolver, IMPORTED. Recorded per teacher.
    resolved = {}
    for pid, unf, fund, _bs in PAIRS:
        for pref, run in (("UNF", unf), ("FUND", fund)):
            z, c, base = _resolve_zip_and_config(run, None)
            resolved[f"{pref}{pid}"] = {"run": os.path.relpath(run, MD),
                                        "zip": os.path.relpath(z, MD),
                                        "config": os.path.relpath(c, MD),
                                        "v1_scored": os.path.relpath(
                                            os.path.join(run, "final_model.zip"), MD)}
            print(f"  RESOLVE {pref}{pid}: {resolved[f'{pref}{pid}']['zip']}", flush=True)
    if any(r["zip"].endswith("final_model.zip") for r in resolved.values()):
        print("  !! NOTE: at least one teacher still resolves to final_model.zip", flush=True)

    loader = TeamLoader(); pool = PairedPool(loader.get_all_teams())
    n_pool = len(pool.packed_teams)

    t0 = time.time(); states = []; team_of = []
    for ti, (b, kind) in enumerate(TEAMS):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(per_team)]
        pilot = Capturing(model=parent, team=PinnedTeam(T(b)), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=AccountConfiguration(f"CL{ti}a", "pw"),
                          stochastic=True, start_listening=False, policy_seed=71000 + ti)
        opp = RLPlayer(model=opp_model, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=AccountConfiguration(f"CL{ti}b", "pw"),
                       stochastic=True, start_listening=False, policy_seed=72000 + ti)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, per_team, concurrency=1, impl="rust",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti:2d}] {kind:8s} {b} +{len(pilot.captured):4d} states "
              f"(total {len(states)}, {time.time()-t0:.0f}s)", flush=True)
    n_untaught_states = sum(1 for t in team_of if t < 8)
    per_team_counts = [int(sum(1 for t in team_of if t == i)) for i in range(len(TEAMS))]
    print(f"  UNTAUGHT STATE CROSS-CHECK: {n_untaught_states} states on indices 0..7 "
          f"(canonical offline_collateral_kl batch = 1100 at 3 battles/team)", flush=True)
    # The two published reproductions of this batch, asserted rather than hoped for.
    if per_team == 3:
        assert n_untaught_states == 1100, (
            f"n=3 must reproduce the canonical 1100-state untaught batch, got {n_untaught_states}")
        print("  CROSS-CHECK PASS: 1100 == canonical", flush=True)
    if per_team == 9:
        expect9 = [280, 399, 333, 458, 714, 592, 391, 301]      # teacher_distance's gen arm
        assert per_team_counts[:8] == expect9, (
            f"n=9 untaught per-team counts {per_team_counts[:8]} != published {expect9}")
        print(f"  CROSS-CHECK PASS: untaught per-team {per_team_counts[:8]} == published",
              flush=True)
    if len(states) < 500:
        raise SystemExit(f"FATAL: only {len(states)} states")

    obs = {"observation": th.as_tensor(np.array([s["observation"] for s in states]), dtype=th.float32),
           "action_mask": th.as_tensor(np.array([s["action_mask"] for s in states]), dtype=th.float32)}
    cl = np.array(team_of)

    def logits_of(model, chunk=256):
        sp = model.observation_space.spaces
        out = []
        with th.no_grad():
            for i in range(0, obs["observation"].shape[0], chunk):
                o = {k: v[i:i + chunk] for k, v in obs.items() if k in sp}
                out.append(model.policy.get_distribution(o).distribution.logits)
        return th.cat(out, 0)

    # CORRECTION 2: the two references.
    a_log = logits_of(parent)                       # REF-A, the fold parent
    origin = load(ORIGIN)
    b_log = logits_of(origin)                       # REF-B, the true fork origin
    del origin

    # A single anchor number the README quotes: how far apart the two references themselves are.
    ref_gap = float(masked_kl_rows(a_log, b_log, obs["action_mask"]).mean())
    ref_gap_rev = float(masked_kl_rows(b_log, a_log, obs["action_mask"]).mean())
    print(f"\n  REFERENCE GAP  KL(parent||origin) {ref_gap:.4f}   "
          f"KL(origin||parent) {ref_gap_rev:.4f}", flush=True)

    def per_team_kl(zip_path, tag, cfg=CFG):
        m = load(zip_path, cfg)
        q = logits_of(m)
        del m
        out = {}
        for ref, r_log in (("A", a_log), ("B", b_log)):
            fwd = masked_kl_rows(q, r_log, obs["action_mask"]).detach().cpu().numpy()
            rev = masked_kl_rows(r_log, q, obs["action_mask"]).detach().cpu().numpy()
            out[ref] = (np.array([fwd[cl == t].mean() for t in range(len(TEAMS))]),
                        np.array([rev[cl == t].mean() for t in range(len(TEAMS))]))
        print(f"  {tag:16s} refA untaught {out['A'][0][:8].mean():.4f} taught16 "
              f"{out['A'][0][8:].mean():.4f}  |  refB untaught {out['B'][0][:8].mean():.4f} "
              f"taught16 {out['B'][0][8:].mean():.4f}", flush=True)
        return out

    kl = {}
    for tag, path in FLOOR_A + FLOOR_B:
        kl[tag] = per_team_kl(path, tag)
    for pid, _u, _f, _bs in PAIRS:
        for pref in ("UNF", "FUND"):
            k = f"{pref}{pid}"
            kl[k] = per_team_kl(f"{MD}/{resolved[k]['zip']}", k, f"{MD}/{resolved[k]['config']}")

    # ACID: no two networks may produce an identical per-team KL vector.
    vecs = {k: v["A"][0] for k, v in kl.items()}
    dup = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
           if np.allclose(vecs[a], vecs[b], atol=1e-9)]
    if dup:
        print(f"  !! ACID: duplicate KL vectors {dup}", flush=True)

    res = {"_meta": {
        "corrections_vs_v1": [
            "CHECKPOINT: teachers resolved by agents.training.fixed_opponent_pool."
            "_resolve_zip_and_config(run_dir, None) -- the call main/train/model_build.py makes",
            "REFERENCE: every statistic reported under REF-A (fold parent) and REF-B (the "
            "exploiters' true fork origin, ai_v9_29_rev1_0823/final_model.zip @25,067,760)"],
        "statistic": "forward KL(teacher||reference) over legal actions; masked_kl_rows IMPORTED "
                     "from agents.training.instrumented_ppo.distill_anchor",
        "also_reported": "KL(reference||teacher)",
        "state_source": "PARENT pilots each of 24 teams vs rev-1's 24M snapshot, "
                        f"{per_team} battles/team, concurrency=1, seeded (VERBATIM from v1)",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pilot_policy": "71000+team_index",
                  "opponent_policy": "72000+team_index", "pool_sequence": "61000+team_index"},
        "teams": [{"i": i, "basename": b, "kind": k} for i, (b, k) in enumerate(TEAMS)],
        "n_states": len(states), "n_untaught_states": int(n_untaught_states),
        "states_per_team": per_team_counts,
        "ref_A_parent": PARENT, "ref_B_origin": ORIGIN, "opponent": OPPONENT,
        "ref_gap_kl_parent_given_origin": ref_gap, "ref_gap_kl_origin_given_parent": ref_gap_rev,
        "resolved_teachers": resolved,
        "acid_all_distinct": not dup, "acid_duplicates": [f"{a}|{b}" for a, b in dup],
        "wall_s_states": round(time.time() - t0, 1)}}
    for ref in ("A", "B"):
        res[f"per_team_kl_fwd_ref{ref}"] = {k: [float(x) for x in v[ref][0]] for k, v in kl.items()}
        res[f"per_team_kl_rev_ref{ref}"] = {k: [float(x) for x in v[ref][1]] for k, v in kl.items()}
    json.dump(res, open(out_path, "w"), indent=1)

    # ------------------------------------------------------------------ analysis --------------
    # CORRECTION 3: every CI resamples over the clusters of the array it is given. The matrix is
    # derived from len(vals) and asserted to span [0, n-1], so an array can never be under-sampled.
    boot = Boot()

    fl = {}
    for tag, _p in FLOOR_A + FLOOR_B:
        ref = "A" if tag.startswith("FLOORA") else "B"
        f = kl[tag][ref][0]
        fl[tag] = {"reference": ref, "untaught_mean": float(f[:8].mean()),
                   "untaught_ci95": list(boot.ci(f[:8])),
                   "taught16_mean": float(f[8:].mean()),
                   "taught16_ci95": list(boot.ci(f[8:])),
                   "L_floor": float(f[8:].mean() / f[:8].mean())}
        print(f"\n  FLOOR {tag} (ref {ref}): untaught {fl[tag]['untaught_mean']:.4f}  "
              f"taught {fl[tag]['taught16_mean']:.4f}  floor L {fl[tag]['L_floor']:.4f}",
              flush=True)
    res["floor"] = fl

    for ref in ("A", "B"):
        per_teacher = {}
        for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
            for pid, _u, _f, bs in PAIRS:
                f = kl[f"{pref}{pid}"][ref][0]
                own = np.array([f[IDX[b]] for b in bs])
                unt = f[:8]
                per_teacher[f"{pref}{pid}"] = {
                    "half": half, "taught_teams": bs,
                    "kl_taught": float(own.mean()), "kl_taught_per_team": [float(x) for x in own],
                    "kl_untaught": float(unt.mean()),
                    "kl_untaught_per_team": [float(x) for x in unt],
                    "L": float(own.mean() / unt.mean())}
        res[f"primary_A_per_teacher_ref{ref}"] = per_teacher

        groups = {}
        for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
            Ls = np.array([per_teacher[f"{pref}{p}"]["L"] for p, *_ in PAIRS])
            kt = np.array([per_teacher[f"{pref}{p}"]["kl_taught"] for p, *_ in PAIRS])
            ku = np.array([per_teacher[f"{pref}{p}"]["kl_untaught"] for p, *_ in PAIRS])
            groups[half] = {"L_mean": float(Ls.mean()), "L_ci95": list(boot.ci(Ls)),
                            "L_per_teacher": [float(x) for x in Ls],
                            "kl_taught_mean": float(kt.mean()),
                            "kl_taught_ci95": list(boot.ci(kt)),
                            "kl_untaught_mean": float(ku.mean()),
                            "kl_untaught_ci95": list(boot.ci(ku))}
        for name, key in (("L", "L"), ("kl_untaught", "kl_untaught"), ("kl_taught", "kl_taught")):
            d = np.array([per_teacher[f"FUND{p}"][key] - per_teacher[f"UNF{p}"][key]
                          for p, *_ in PAIRS])
            lo, hi = boot.ci(d)
            groups[f"FUNDED_minus_UNFUNDED_{name}"] = {
                "delta": float(d.mean()), "ci95": [lo, hi],
                "paired_on": "the 8 teacher PAIRS (same 2 teams each)",
                "separates_from_zero": not (lo <= 0 <= hi)}
        res[f"primary_A_groups_ref{ref}"] = groups

        B = {}
        for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
            own, sib = [], []
            for pid, _u, _f, bs in PAIRS:
                for b in bs:
                    i = IDX[b]
                    own.append(kl[f"{pref}{pid}"][ref][0][i])
                    sib.append(np.mean([kl[f"{pref}{q}"][ref][0][i] for q, *_ in PAIRS
                                        if q != pid]))
            own, sib = np.array(own), np.array(sib)
            d, r = own - sib, own / sib
            lo, hi = boot.ci(d); rlo, rhi = boot.ci(r)
            B[half] = {"kl_own_mean": float(own.mean()), "kl_siblings_mean": float(sib.mean()),
                       "delta_own_minus_siblings": float(d.mean()), "delta_ci95": [lo, hi],
                       "delta_separates_from_zero": not (lo <= 0 <= hi),
                       "R_ratio_mean": float(r.mean()), "R_ci95": [rlo, rhi],
                       "per_team_own": [float(x) for x in own],
                       "per_team_siblings": [float(x) for x in sib]}
        rf = np.array(B["funded"]["per_team_own"]) / np.array(B["funded"]["per_team_siblings"])
        ru = np.array(B["unfunded"]["per_team_own"]) / np.array(B["unfunded"]["per_team_siblings"])
        lo, hi = boot.ci(rf - ru)
        B["FUNDED_minus_UNFUNDED_R"] = {"delta": float((rf - ru).mean()), "ci95": [lo, hi],
                                        "separates_from_zero": not (lo <= 0 <= hi)}
        res[f"primary_B_sibling_control_ref{ref}"] = B

        print(f"\n  === REF-{ref} "
              f"({'fold parent R2ACTION' if ref == 'A' else 'true origin rev-1 final'}) ===",
              flush=True)
        for half in ("unfunded", "funded"):
            g = groups[half]; b = B[half]
            print(f"  {half:9s} KL_taught {g['kl_taught_mean']:.4f}  KL_untaught "
                  f"{g['kl_untaught_mean']:.4f}  L {g['L_mean']:.4f} "
                  f"[{g['L_ci95'][0]:.4f},{g['L_ci95'][1]:.4f}]   R {b['R_ratio_mean']:.4f} "
                  f"[{b['R_ci95'][0]:.4f},{b['R_ci95'][1]:.4f}]", flush=True)
        b = B["FUNDED_minus_UNFUNDED_R"]
        print(f"  FUNDED-UNFUNDED R {b['delta']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] "
              f"{'SEPARATES' if b['separates_from_zero'] else 'SPANS ZERO'}", flush=True)

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\n  wrote {out_path}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
