"""H7 Z-SWAP, v8 ERA — does the FiLM team-code z carry the exploiters' specialisation?

MUST BE RUN FROM AN ERA-PINNED CHECKOUT (`PYTHONPATH=<era>/src`). The v8 checkpoints are
config_version 45 and current code REFUSES them.

WHAT IS SWAPPED. At b13b30b2, `z` is OBS-DERIVED (a DeepSets code over OUR team's invariant facts,
`ZArchEncoder`, features_extractor.py L3450) and enters the network at EXACTLY ONE site: the
post-projection / pre-ReLU FiLM in `Gen3FeaturesExtractor.forward` (L4425), `h*(1+dg)+db` with
`[dg||db] = film_{pi,vf}(last_zarch)`. z NEVER touches the trunk. So substituting a different
[B,32] tensor for `last_zarch` changes exactly the FiLM modulation and nothing else. "P's z" means
the tensor P's OWN zarch_encoder produces on the same observation, captured from P's own forward.

The substitution is done by replacing `fe.zarch_encoder` with `_ZShim`, which returns a pinned
tensor when armed and delegates to the real encoder otherwise. TWO ACID gates run before any
measurement: (i) shim disarmed reproduces the unpatched logits bit-for-bit; (ii) shim armed with
the model's OWN captured z reproduces them bit-for-bit.

CONDITIONS (P = fold parent ai_v8_04, T = one of v8_14's three teachers):
  a   KL(T[z_T] || P[z_P])   baseline (== content_locality's statistic; ACID-checked against it)
  b   KL(T[z_P] || P[z_P])   teacher's weights, parent's z
  c1  KL(P[z_T] || P[z_P])   parent's weights, teacher's z -- P diverging from ITSELF (= z-sens of P)
  c2  KL(T[z_T] || P[z_T])   both on the teacher's z
  d0  KL(T[0]   || P[0])     both with z zeroed (NOT identity: the FiLM biases are free after init)
  dmu KL(T[zbar_T] || P[zbar_P])  both at their own state-mean z (kills team-to-team variation only)
  zsens_T  KL(T[z_P] || T[z_T])   how much swapping z moves the TEACHER at all

State batch: verbatim content_locality/v8_era_locality.py -- the FOLD PARENT pilots each team vs
the fixed reference ai_v8_03_zarch_control final, stochastic=False both sides, node bridge,
concurrency=1, sim seed [i+1,2,3,4], pool sequence random.Random(61000+i).

Pre-registration: PREREGISTRATION.md (frozen before any state was generated).

Run (from the era tree):
  PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 ERA_ROOT=/tmp/v8rep_era \
      GEN3AI_TIMEOUT_SCALE=12 nice -n 10 python era_zswap.py <out.json> [per_team=3] [min_states=400]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import asyncio, hashlib, itertools, json, random, sys, time
import numpy as np
import torch as th
th.set_num_threads(1)
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.inference.player import RLPlayer
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "content_locality"))
from era_kl import masked_kl_rows_era as masked_kl_rows   # noqa: E402

MAIN = "/home/goodlad/dev/gen3ai"
ERA_ROOT = os.environ.get("ERA_ROOT", os.getcwd())
MD = f"{MAIN}/models"
PAR_RUN = f"{MD}/ai_v8_04_distill_4teacher_0722"
PARENT = f"{PAR_RUN}/final_model_interrupted.zip"
PAR_CFG = f"{PAR_RUN}/model_config.json"
REF = f"{MD}/ai_v8_03_zarch_control_0718/final_model_interrupted.zip"
REF_CFG = f"{MD}/ai_v8_03_zarch_control_0718/model_config.json"

FLOORS = [("FLOOR_c277178", f"{PAR_RUN}/checkpoints/checkpoint_277178472_steps.zip", PAR_CFG),
          ("FLOOR_c275758", f"{PAR_RUN}/checkpoints/checkpoint_275758296_steps.zip", PAR_CFG)]

# WHICH FILE IS THE TEACHER? v8_14's recorded argv names each teacher as a RUN DIRECTORY
# (`.../ai_v8_09_pool10_exploiter_0723:*`), and `fixed_opponent_pool._resolve_zip_and_config`
# resolves a directory through `best_model/best_model.zip` -> `final_model.zip` ->
# `best_model.zip`. All three teacher runs HAVE best_model/best_model.zip, so THAT is the
# checkpoint the fold actually distilled -- `final_model_interrupted.zip` is not a rung at all.
# `$ZSWAP_TEACHER_FILE=final_interrupted` reproduces the earlier (wrong-file) pass as a secondary.
# The PARENT is unaffected: the fold names it as a direct `--model .../final_model_interrupted.zip`.
TEACHER_FILE = os.environ.get("ZSWAP_TEACHER_FILE", "best_model")
_RUNGS = {"best_model": ["best_model/best_model.zip", "final_model.zip", "best_model.zip"],
          "final_interrupted": ["final_model_interrupted.zip"]}


def resolve_teacher(run_dir):
    """The resolver's rung order, applied to a run dir. Returns (zip_path, which_rung)."""
    rungs = _RUNGS[TEACHER_FILE]
    for r in rungs:
        c = os.path.join(run_dir, r)
        if os.path.isfile(c):
            return c, r
    raise SystemExit(f"[GIGO] no teacher zip for {run_dir} (tried {rungs})")


_T_RUNS = [
    ("pool10", f"{MD}/ai_v8_09_pool10_exploiter_0723",
     {"sha": ["564b9be3ae", "d3c1cd0952", "7594a34f82", "47dc388b25", "4c9552cd01",
              "552d5857a3", "f5d46ca0fc", "3a83154c2a", "9b454d9ea7", "24db4aacdd"]}),
    ("semistall3", f"{MD}/ai_v8_06_semistall_3team_exploiter_0722",
     {"files": [f"{ERA_ROOT}/data/teams/sample/9d5f845869e899ee.txt",
                f"{ERA_ROOT}/data/teams/sample/f7ba5702fe856292.txt",
                f"{ERA_ROOT}/data/teams/sample/0972146213a667c9.txt"]}),
    ("defensive10", f"{MD}/ai_v8_13_defensive10_exploiter_0725",
     {"sha": ["9278913bce", "fc908f1bf4", "83aee9db7e", "044da80d78", "bef089d2cf",
              "3f95b25e9a", "8b6b8c8f52", "5c88ff9ca5", "3e9bdcee48", "65bfb2e8b4"]}),
]
TEACHER_RUNG = {}
TEACHERS = []
for _n, _rd, _spec in _T_RUNS:
    _zip, _rung = resolve_teacher(_rd)
    TEACHER_RUNG[_n] = _rung
    TEACHERS.append((_n, _zip, f"{_rd}/model_config.json", _spec))

UNTAUGHT_SHA = ["d0a4d2bcb8", "c90e782cad", "a6b630e6b4", "a577a735b7",
                "9292a21833", "eaa88395e7", "7c2cb5cec1", "89fcef3b53"]

# The z path -- the parameter groups the owner's hypothesis says an exploiter would use.
Z_PREFIXES = ("zarch_encoder.", "film_pi.", "film_vf.")

_ACCT = itertools.count(1)


def sha10(s):
    return hashlib.sha1(s.strip().encode()).hexdigest()[:10]


def _acct(tag):
    return AccountConfiguration(f"ZS{tag[:2]}{next(_ACCT):05d}", "pw")


def _strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    if hasattr(m, "policy"):
        m.policy.set_training_mode(False)
    return m


class _ZShim(th.nn.Module):
    """Wraps the real ZArchEncoder. Armed -> return the pinned z; disarmed -> the real code."""

    def __init__(self, real):
        super().__init__()
        self.real = real
        self.pinned = None          # a plain attribute, never a buffer (must not enter state_dict)

    def forward(self, ctx, embeddings):
        if self.pinned is None:
            return self.real(ctx, embeddings)
        return self.pinned

    def recon_logits(self, z):
        return self.real.recon_logits(z)


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
            self.captured.append({"observation": np.asarray(d["observation"]).copy(),
                                  "action_mask": np.asarray(d["action_mask"]).copy()})
        return d


# --------------------------------------------------------------------------- forward machinery --
CHUNK = 128


def _fe(model):
    return model.policy.features_extractor


def run_logits(model, obs, z=None, capture_z=False):
    """Logits [N,11] with the FiLM code forced to `z` ([N,32] or None = the model's own).

    `capture_z` additionally returns the z the model itself produced (only meaningful with z=None).
    """
    fe = _fe(model)
    shim = fe.zarch_encoder
    assert isinstance(shim, _ZShim), "zarch_encoder must be shimmed before use"
    sp = model.observation_space.spaces
    n = obs["observation"].shape[0]
    outs, zs = [], []
    with th.no_grad():
        for i in range(0, n, CHUNK):
            j = min(i + CHUNK, n)
            shim.pinned = None if z is None else z[i:j]
            o = {k: v[i:j] for k, v in obs.items() if k in sp}
            outs.append(model.policy.get_distribution(o).distribution.logits)
            if capture_z:
                zs.append(fe.last_zarch.detach().clone())
    shim.pinned = None
    L = th.cat(outs, 0)
    return (L, th.cat(zs, 0)) if capture_z else L


def acid_shim(model, obs, tag):
    """Gate the override mechanism: disarmed == unpatched, and armed-with-own-z == unpatched.

    INSTALLS the shim and leaves it installed -- callers must NOT wrap again (a second wrap is
    behaviourally harmless, since the outer shim short-circuits and the inner stays disarmed, but
    it renames every z-path state_dict key and would silently reshape the parameter split).
    """
    fe = _fe(model)
    real = fe.zarch_encoder
    assert not isinstance(real, _ZShim), f"{tag}: zarch_encoder is already shimmed"
    sp = model.observation_space.spaces
    sub = {k: v[:CHUNK] for k, v in obs.items() if k in sp}
    with th.no_grad():
        ref = model.policy.get_distribution(sub).distribution.logits.clone()
    fe.zarch_encoder = _ZShim(real)
    with th.no_grad():
        fe.zarch_encoder.pinned = None
        off = model.policy.get_distribution(sub).distribution.logits.clone()
        z_own = fe.last_zarch.detach().clone()
        fe.zarch_encoder.pinned = z_own
        on = model.policy.get_distribution(sub).distribution.logits.clone()
        fe.zarch_encoder.pinned = None
    d_off = float((off - ref).abs().max())
    d_on = float((on - ref).abs().max())
    print(f"  ACID shim {tag:16s} disarmed max|d| {d_off:.3e}   armed-with-own-z max|d| {d_on:.3e}",
          flush=True)
    if d_off != 0.0 or d_on != 0.0:
        raise SystemExit(f"[ACID] shim is not faithful for {tag}: {d_off} / {d_on}")
    return {"disarmed_max_abs_delta": d_off, "own_z_max_abs_delta": d_on}


def film_magnitude(model, obs, z, chunk=CHUNK):
    """How much does FiLM actually MODULATE the head features, relative to their own scale?

    A forward hook on `fe.projection` / `fe.value_projection` captures `pi_pre` / `vf_pre` BEFORE
    the FiLM line (FiLM rebinds a local, so the Linear's output is exactly the pre-FiLM tensor).
    The modulation is `h*(1+dg)+db - h = h*dg + db`; we report its L2 relative to ||h||.

    This is the number that decides whether a z-swap CAN move the policy at all.
    """
    fe = _fe(model)
    cap = {}

    def mk(key):
        def hook(_m, _i, out):
            cap[key] = out.detach()
        return hook

    h1 = fe.projection.register_forward_hook(mk("pi"))
    h2 = fe.value_projection.register_forward_hook(mk("vf"))
    sp = model.observation_space.spaces
    n = obs["observation"].shape[0]
    acc = {"pi": [0.0, 0.0, 0.0], "vf": [0.0, 0.0, 0.0]}   # sum||h||, sum||mod||, sum rel
    tot = 0
    try:
        with th.no_grad():
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                fe.zarch_encoder.pinned = None if z is None else z[i:j]
                o = {k: v[i:j] for k, v in obs.items() if k in sp}
                model.policy.get_distribution(o)
                zz = fe.last_zarch
                for key, gen in (("pi", fe.film_pi), ("vf", fe.film_vf)):
                    dg, db = gen(zz).chunk(2, dim=-1)
                    h = cap[key]
                    mod = h * dg + db
                    nh = h.norm(dim=-1)
                    nm = mod.norm(dim=-1)
                    acc[key][0] += float(nh.sum())
                    acc[key][1] += float(nm.sum())
                    acc[key][2] += float((nm / nh.clamp_min(1e-9)).sum())
                tot += j - i
        fe.zarch_encoder.pinned = None
    finally:
        h1.remove(); h2.remove()
    return {key: {"mean_norm_head_features": acc[key][0] / tot,
                  "mean_norm_film_modulation": acc[key][1] / tot,
                  "mean_relative_modulation": acc[key][2] / tot}
            for key in acc}


def param_split(sd_t, sd_p):
    """Fraction of ||theta_T - theta_P|| living in the z path vs the shared trunk/heads.

    Keys are deduped by storage pointer first: share_features_extractor=True makes SB3 alias the
    extractor under pi_/vf_ prefixes, which would triple-count every shared tensor.
    """
    seen, keys = set(), []
    for k, v in sd_p.items():
        if not th.is_floating_point(v):
            continue
        ptr = v.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr); keys.append(k)
    def _group(k):
        # FOUR groups. `recon_head` is separated because it is an AUX-LOSS-ONLY readout whose
        # output is never fed forward -- displacement there cannot change behaviour, so folding it
        # into the z path biases the z path's enrichment DOWNWARD.
        if "zarch_encoder." in k:
            return "recon_head" if "recon_head" in k else "z_encoder"
        if "film_pi." in k or "film_vf." in k:
            return "film_generators"
        return "shared_trunk_and_heads"

    g = {n: {"n": 0, "sq": 0.0, "l1": 0.0, "keys": []}
         for n in ("z_encoder", "recon_head", "film_generators", "shared_trunk_and_heads")}
    for k in keys:
        if k not in sd_t or sd_t[k].shape != sd_p[k].shape:
            continue
        d = (sd_t[k].double() - sd_p[k].double())
        e = g[_group(k)]
        e["n"] += d.numel(); e["sq"] += float((d * d).sum()); e["l1"] += float(d.abs().sum())
        if _group(k) != "shared_trunk_and_heads":
            e["keys"].append(k)
    tot_n = sum(e["n"] for e in g.values())
    tot_sq = sum(e["sq"] for e in g.values())
    tot_l1 = sum(e["l1"] for e in g.values())
    for e in g.values():
        e["param_frac"] = e["n"] / tot_n if tot_n else 0.0
        e["disp_frac_sq"] = e["sq"] / tot_sq if tot_sq else 0.0
        e["disp_frac_l1"] = e["l1"] / tot_l1 if tot_l1 else 0.0
        e["enrichment_sq"] = (e["disp_frac_sq"] / e["param_frac"]) if e["param_frac"] else 0.0
        e["keys"] = sorted(e["keys"])
    # The BEHAVIOURAL z path = everything z-related that can actually change an output
    # (encoder + generators; recon_head excluded).
    zb_n = g["z_encoder"]["n"] + g["film_generators"]["n"]
    zb_sq = g["z_encoder"]["sq"] + g["film_generators"]["sq"]
    zb_l1 = g["z_encoder"]["l1"] + g["film_generators"]["l1"]
    zall_n = zb_n + g["recon_head"]["n"]
    zall_sq = zb_sq + g["recon_head"]["sq"]
    return {"groups": g, "n_params_total": tot_n, "l2_norm_total": tot_sq ** 0.5,
            # behavioural z path (recon_head EXCLUDED) -- the headline
            "n_params_z": zb_n, "param_frac_z": zb_n / tot_n if tot_n else 0.0,
            "sq_z": zb_sq, "disp_frac_z_sq": zb_sq / tot_sq if tot_sq else 0.0,
            "l1_z": zb_l1, "disp_frac_z_l1": zb_l1 / tot_l1 if tot_l1 else 0.0,
            "enrichment_sq": ((zb_sq / tot_sq) / (zb_n / tot_n)) if tot_sq and zb_n else 0.0,
            # the same including the aux-only recon head, for continuity with the first pass
            "param_frac_z_incl_recon": zall_n / tot_n if tot_n else 0.0,
            "disp_frac_z_sq_incl_recon": zall_sq / tot_sq if tot_sq else 0.0,
            "enrichment_sq_incl_recon": ((zall_sq / tot_sq) / (zall_n / tot_n))
                                        if tot_sq and zall_n else 0.0}


# ----------------------------------------------------------------------------------- main -------
def main(out_path, per_team=3, min_states=400):
    maps = load_mappings(); cv = current_model_version(maps)

    def load(p, cfg):
        m, _ = load_foreign_opponent(p, current_version=cv, device="cpu", config_path=cfg)
        return _strip_debugger(m)

    pool_teams = TeamLoader().get_all_teams()
    by_sha = {sha10(t): t for t in pool_teams}

    taught_of = {}
    for name, _p, _c, spec in TEACHERS:
        shas = list(spec["sha"]) if "sha" in spec else [sha10(open(f).read()) for f in spec["files"]]
        missing = [s for s in shas if s not in by_sha]
        if missing:
            raise SystemExit(f"[GIGO] teacher {name}: {missing} not resolvable in the era pool")
        taught_of[name] = shas
    seen, taught_union = set(), []
    for n in taught_of:
        for s in taught_of[n]:
            if s not in seen:
                seen.add(s); taught_union.append(s)
    shared = sorted({s for s in seen if sum(s in taught_of[n] for n in taught_of) > 1})
    if shared:
        print(f"[zswap] NOTE: {len(shared)} team(s) taught by >1 teacher: {shared} "
              f"-- excluded from the sibling control", flush=True)
    bad = [s for s in UNTAUGHT_SHA if s in set(taught_union)]
    if bad:
        raise SystemExit(f"[GIGO] untaught set overlaps the taught union: {bad}")
    miss = [s for s in UNTAUGHT_SHA if s not in by_sha]
    if miss:
        raise SystemExit(f"[GIGO] untaught {miss} not in the era pool")

    TEAMS = [(s, "untaught") for s in UNTAUGHT_SHA] + [(s, "taught") for s in taught_union]
    IDX = {s: i for i, (s, _) in enumerate(TEAMS)}
    n_unt = len(UNTAUGHT_SHA)
    print(f"[zswap] TEACHER FILE MODE = {TEACHER_FILE!r}; per teacher: "
          f"{ {k: v for k, v in TEACHER_RUNG.items()} }", flush=True)
    print(f"[zswap] {len(TEAMS)} teams = {n_unt} untaught + {len(taught_union)} taught "
          f"({ {k: len(v) for k, v in taught_of.items()} }), pool {len(pool_teams)}", flush=True)

    parent = load(PARENT, PAR_CFG)
    ref = load(REF, REF_CFG)
    pool = PairedPool(pool_teams); n_pool = len(pool.packed_teams)

    # ------------------------------------------------------------------ states (content_locality)
    t0 = time.time(); states = []; team_of = []
    for ti, (s, kind) in enumerate(TEAMS):
        rng = random.Random(61000 + ti)
        seq = [rng.randrange(n_pool) for _ in range(per_team)]
        pilot = Capturing(model=parent, team=Gen3Teambuilder([by_sha[s]]), battle_format="gen3ou",
                          server_configuration=LocalhostServerConfiguration, mappings=maps,
                          account_configuration=_acct("pi"), stochastic=False, start_listening=False)
        opp = RLPlayer(model=ref, team=pool.set_sequence(seq), battle_format="gen3ou",
                       server_configuration=LocalhostServerConfiguration, mappings=maps,
                       account_configuration=_acct("op"), stochastic=False, start_listening=False)
        pilot.reset_battles(); opp.reset_battles()
        asyncio.run(run_local_battles(pilot, opp, per_team, concurrency=1, impl="node",
                                      seed=[ti + 1, 2, 3, 4]))
        states.extend(pilot.captured); team_of.extend([ti] * len(pilot.captured))
        print(f"  [{ti:2d}] {kind:8s} {s} +{len(pilot.captured):4d} states "
              f"(total {len(states)}, {time.time()-t0:.0f}s)", flush=True)
    if len(states) < min_states:
        raise SystemExit(f"FATAL: only {len(states)} states")
    del ref

    obs = {"observation": th.as_tensor(np.array([x["observation"] for x in states]), dtype=th.float32),
           "action_mask": th.as_tensor(np.array([x["action_mask"] for x in states]), dtype=th.float32)}
    mask = obs["action_mask"]
    cl = np.array(team_of)
    nT_teams = len(TEAMS)
    t_states = time.time() - t0

    def per_team_mean(v):
        return np.array([float(v[cl == t].mean()) for t in range(nT_teams)])

    def KL(a, b):
        return per_team_mean(masked_kl_rows(a, b, mask).detach().cpu().numpy())

    # ------------------------------------------------------------------------- parent conditions
    print(f"\n[zswap] {len(states)} states in {t_states:.0f}s; scoring", flush=True)
    acid = {"parent": acid_shim(parent, obs, "parent")}
    # acid_shim already INSTALLED the shim; wrapping again would rename every z-path key.
    tf = time.time()
    LP_zP, z_P = run_logits(parent, obs, z=None, capture_z=True)
    print(f"  parent forward {time.time()-tf:.0f}s   z_P {tuple(z_P.shape)} "
          f"|z| mean {float(z_P.norm(dim=-1).mean()):.3f}", flush=True)
    zbar_P = z_P.mean(0, keepdim=True).expand_as(z_P).contiguous()
    zero = th.zeros_like(z_P)
    LP_0 = run_logits(parent, obs, z=zero)
    LP_mu = run_logits(parent, obs, z=zbar_P)
    sd_p = {k: v for k, v in parent.policy.state_dict().items()}

    # z geometry: how much of z's variation is one shared direction, and how far apart the
    # models' codes are. Recorded because it decides whether a swap CAN do anything.
    def z_geom(z, zref=None):
        zc = z - z.mean(0, keepdim=True)
        # top singular direction's share of the centred energy
        try:
            sv = th.linalg.svdvals(zc.double())
            share = float((sv[0] ** 2) / (sv ** 2).sum())
        except Exception:
            share = float("nan")
        g = {"mean_norm": float(z.norm(dim=-1).mean()),
             "centred_rms": float(zc.pow(2).sum(-1).mean().sqrt()),
             "top_dir_energy_share": share}
        if zref is not None:
            d = z - zref
            g["rms_dist_to_ref"] = float(d.pow(2).sum(-1).mean().sqrt())
            g["rel_dist_to_ref"] = g["rms_dist_to_ref"] / max(g["mean_norm"], 1e-9)
        return g

    res_z = {"parent": z_geom(z_P)}
    fmag = {"parent": film_magnitude(parent, obs, None)}
    print(f"  FiLM modulation (parent): pi {fmag['parent']['pi']['mean_relative_modulation']*100:.2f}% "
          f"of ||h||, vf {fmag['parent']['vf']['mean_relative_modulation']*100:.2f}%", flush=True)

    kl = {}          # tag -> {condition: per-team vector}
    zg = {}
    psplit = {}

    def score_model(tag, path, cfg, is_floor=False):
        m = load(path, cfg)
        acid[tag] = acid_shim(m, obs, tag)
        # acid_shim already installed the shim -- do not wrap again (see its docstring).
        LT_zT, z_T = run_logits(m, obs, z=None, capture_z=True)
        LT_zP = run_logits(m, obs, z=z_P)
        LT_0 = run_logits(m, obs, z=zero)
        zbar_T = z_T.mean(0, keepdim=True).expand_as(z_T).contiguous()
        LT_mu = run_logits(m, obs, z=zbar_T)
        LP_zT = run_logits(parent, obs, z=z_T)          # parent's weights, teacher's z
        c = {"a": KL(LT_zT, LP_zP),
             "b": KL(LT_zP, LP_zP),
             "c1": KL(LP_zT, LP_zP),
             "c2": KL(LT_zT, LP_zT),
             "d0": KL(LT_0, LP_0),
             "dmu": KL(LT_mu, LP_mu),
             "zsens_T": KL(LT_zP, LT_zT),
             # How far does each z intervention move THIS network away from itself? These are the
             # mechanism numbers: if a network's own behaviour barely depends on z, z cannot be
             # carrying its specialisation, whatever the cross-model contrasts say.
             "zsens0_T": KL(LT_0, LT_zT),
             "zsensmu_T": KL(LT_mu, LT_zT),
             "rev_a": KL(LP_zP, LT_zT)}
        kl[tag] = c
        zg[tag] = z_geom(z_T, z_P)
        psplit[tag] = param_split(m.policy.state_dict(), sd_p)
        fmag[tag] = film_magnitude(m, obs, None)
        del m
        p = psplit[tag]
        print(f"  {tag:16s} a {c['a'][n_unt:].mean():.4f}  b {c['b'][n_unt:].mean():.4f}  "
              f"c1 {c['c1'][n_unt:].mean():.4f}  zsensT {c['zsens_T'][n_unt:].mean():.4f}  "
              f"| z rel-dist {zg[tag]['rel_dist_to_ref']:.4f}  "
              f"| dtheta z-share {p['disp_frac_z_sq']*100:.3f}% of {p['param_frac_z']*100:.3f}% params",
              flush=True)

    for tag, path, cfg in FLOORS:
        score_model(tag, path, cfg, is_floor=True)
    for name, path, cfg, _s in TEACHERS:
        score_model(name, path, cfg)

    vecs = {k: v["a"] for k, v in kl.items()}
    dup = [(a, b) for i, a in enumerate(vecs) for b in list(vecs)[i + 1:]
           if np.allclose(vecs[a], vecs[b], atol=1e-9)]
    if dup:
        print(f"  !! ACID: duplicate baseline KL vectors {dup}", flush=True)

    res = {"_meta": {
        "probe": "H7 z-swap",
        "era": "v8 (b13b30b289c5eaba136a930a4ab63451e209fbe5)",
        "prereg": "PREREGISTRATION.md (frozen before any state was generated)",
        "what_z_is": "OBS-DERIVED DeepSets code over OUR team's invariant facts (ZArchEncoder, "
                     "features_extractor.py L3450); consumed at EXACTLY ONE site -- the "
                     "post-projection/pre-ReLU head FiLM (L4425). z never touches the trunk.",
        "swap_mechanism": "_ZShim replaces fe.zarch_encoder and returns a pinned [B,32]; gated by "
                          "acid_shim (disarmed and armed-with-own-z both bit-identical to unpatched)",
        "statistic": "forward KL over legal actions, era_kl.masked_kl_rows_era (gated bit-identical "
                     "to the gen-era import by content_locality/kl_unit_test.py)",
        "conditions": {"a": "KL(T[z_T]||P[z_P])", "b": "KL(T[z_P]||P[z_P])",
                       "c1": "KL(P[z_T]||P[z_P])", "c2": "KL(T[z_T]||P[z_T])",
                       "d0": "KL(T[0]||P[0])", "dmu": "KL(T[zbar_T]||P[zbar_P])",
                       "zsens_T": "KL(T[z_P]||T[z_T])", "rev_a": "KL(P[z_P]||T[z_T])"},
        "state_source": f"PARENT pilots each of {nT_teams} teams vs ai_v8_03_zarch_control final, "
                        f"{per_team} battles/team, GREEDY both sides, concurrency=1, node bridge",
        "seeds": {"sim": "[team_index+1,2,3,4]", "pool_sequence": "61000+team_index"},
        "teams": [{"i": i, "sha10": s, "kind": k} for i, (s, k) in enumerate(TEAMS)],
        "taught_of": taught_of, "shared_teams": shared, "n_untaught": n_unt,
        "n_states": len(states),
        "states_per_team": [int((cl == t).sum()) for t in range(nT_teams)],
        "parent": PARENT, "reference_opponent": REF,
        "teacher_file_mode": TEACHER_FILE,
        "teacher_rung": TEACHER_RUNG,
        "teacher_zip": {n: p for n, p, _c, _s in TEACHERS},
        "teacher_resolution_note":
            "v8_14's argv names each teacher as a RUN DIR, which "
            "fixed_opponent_pool._resolve_zip_and_config resolves through "
            "best_model/best_model.zip first. mode=best_model reproduces the fold's own "
            "resolution; mode=final_interrupted reproduces content_locality's (wrong-file) pass.",
        "per_team": per_team,
        "acid_shim": acid, "acid_all_distinct": not dup,
        "acid_duplicates": [f"{a}|{b}" for a, b in dup],
        "wall_s_states": round(t_states, 1), "wall_s_total": round(time.time() - t0, 1)}}
    res["per_team_kl"] = {tag: {c: [float(x) for x in v] for c, v in d.items()}
                          for tag, d in kl.items()}
    # The parent's own z-dependence, the reference for every zsens*_T above.
    res["parent_self_kl"] = {
        "zsens0_P": [float(x) for x in KL(LP_0, LP_zP)],
        "zsensmu_P": [float(x) for x in KL(LP_mu, LP_zP)]}
    res["z_geometry"] = {"parent": res_z["parent"], **zg}
    res["param_split"] = psplit
    res["film_magnitude"] = fmag
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\n  wrote {out_path}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1],
         int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         int(sys.argv[3]) if len(sys.argv) > 3 else 400)
