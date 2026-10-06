"""The PURPOSE metrics (§7.4) and the ROLE reads (§4.2 R1–R4) from one checkpoint's per-decision columns.

Each metric is ONE value per run (its mean over the fixed bank rows of a stratum) — the unit §7.4's
across-seed inference takes (:mod:`main.belief_roles.infer`). The battle-clustered intervals beside
them are DESCRIPTIVE only (conditional on the run and the bank; Field & Welsh 2007), never the
inference.

**(1) Opponent-intent log loss — the ADOPTION GATE is the CONDITIONAL form (Amendment 3(b), §7.7(b)),
``intent_logloss_conditional``:** per row E_row = blob's named set (:mod:`.eset`; a blob run's OWN; a
fixed_mass run is scored on EVERY blob run of the look and its value is the MEAN over those sets), and on the rows whose realised event e ∈ E_row the loss is
``−log [P(e) / Σ_{e′ ∈ E_row} P(e′)]`` with the arm's probabilities RENORMALISED over E_row — same rows,
same support, both arms, so neither blob's selection on the outcome (its misses dropped) nor its
structural over-certainty (all mass on its named set, while fixed_mass reserves mass for OTHER) moves
the score. **Coverage is REPORTED** beside it (``intent_coverage``): the mass outside E_row against the
observed outside frequency (calibration-in-the-large + a reliability curve on fixed bins); for blob its
miss rate. Rule 8: a row whose denominator is within 1e-12 of 0 is excluded and counted.

**The as-built form stays REPORTED, descriptive only** (``intent_logloss``): mean ``−log P(realised
event)`` over the labelled rows the arm's OWN candidates cover; the rows they cannot cover are the MISS
column (``intent_miss_rate``), never floored into the score (§7.4). Split by kind (move / switch) and by
opponent class. A Struggle label (forced, no PP) is excluded and counted (in both forms).

**(2) Species presence** — per decision the team Brier ``Σ_{s∈V} (π_s − y_s)²`` (y = the true unseen
set) and the set log score ``Σ_V BCE(π_s, y_s)``; the Murphy reliability / resolution / uncertainty split
over the pooled per-species indicators on FIXED bins; class-wise mean calibration (Gupta & Ramdas 2022)
for the most frequent true species.

**(3) OTHER calibration (R4)** — OTHER = Σ_tail π over the species past the k-th in the ONE stable order
(for ``fixed_mass`` the arm's own OTHER, checked equal to the model's ``other_mass``; for ``blob`` and the
prior, the same construction on their presence — a DERIVED OTHER, so the three columns are comparable)
against the realised count of true unseen species in the tail, by r; reliability over OTHER-mass bins
and the Spiegelhalter-type Z = Σ(N − M) / √Σ π(1 − π).

**(4) Roles R1 / R2 / R3** (§4.2) — R1: Δ_ρ = mean(M_ρ − N_ρ) by r, its usage-weighted mean |Δ_ρ| the
summary; R2: Z_ρ = Σ(N − M) / √Σ Var (CONSERVATIVE under the fixed group mass, reported, never the sole
verdict); R3: on the substitute pairs, the rate at which BOTH have π ≥ 0.25 and the excess π₁ + π₂ − 1.

**Rule 8** (every exclusion counted in the output): a row whose selection the read depends on sits
within 1e-6 of a boundary (``fixed_mass``: ``near_tie_rows``; the derived OTHER: the k-th / (k+1)-th π
gap; ``blob``'s intent read: its E4 seat cut) is excluded from that read; R3 excludes a row where either
π lies in [0.245, 0.255]. A row whose labels do not count k true unseen species inside V is excluded from
(2)–(4) and counted (``label_mismatch``; the set BCE's own consistency rule).
"""
from __future__ import annotations

import functools
from typing import Dict, List, Optional

import numpy as np

from main.belief_roles.bank_rows import NO_EVENT, SWITCH_BASE, BankRows
from main.belief_roles.forward import Columns
from main.belief_roles.roles import RoleSet

#: Descriptive bootstrap: resamples and seed (battle clusters).
N_BOOT = 400
BOOT_SEED = 0
#: Fixed reliability bins on a presence π.
PI_BINS = (0.0, 0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 + 1e-12)
#: Fixed reliability bins on OTHER's mass (an expected count).
MASS_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 + 1e-12)
#: §4.2 R3's presence bar and its rule-8 exclusion band.
R3_BAR = 0.25
R3_BAND = (0.245, 0.255)
#: Fixed reliability bins on the mass OUTSIDE E_row (Amendment 3(b) coverage).
OUTSIDE_BINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 + 1e-12)
#: Class-wise calibration: how many of the most frequent true species to tabulate.
N_CLASSES = 20
CLASSES = ("bot", "pool_snapshot", "exploiter")


def _f(x) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = N_BOOT,
               seed: int = BOOT_SEED) -> Optional[List[float]]:
    """95 % percentile CI of the mean of ``values`` by a battle-cluster bootstrap (DESCRIPTIVE)."""
    if values.size == 0:
        return None
    u, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(u), size=(n_boot, len(u)))
    w = np.zeros((n_boot, len(u)))
    for b in range(n_boot):
        w[b] = np.bincount(draw[b], minlength=len(u))
    num, den = w @ sums, w @ cnts
    est = num / np.maximum(den, 1.0)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


# ------------------------------------------------------------------------------------------- truth
def unseen_labels(br: BankRows, cols: Columns):
    """``(y [N,S] bool, consistent [N] bool)`` — the true unseen set (true team minus the revealed
    nums) and whether it is exactly k species inside V (the set BCE's own rule)."""
    n, S = cols.cand.shape
    y = np.zeros((n, S), dtype=bool)
    for i in range(n):
        rev = set(int(x) for x in cols.revealed_nums[i] if x > 0)
        for s in br.true_species[i]:
            if int(s) not in rev and 0 < int(s) < S:
                y[i, int(s)] = True
    consistent = (~(y & ~cols.cand)).all(1) & (y.sum(1) == cols.k)
    return y, consistent


def role_counts(br: BankRows, roles: RoleSet) -> np.ndarray:
    """``[N,R]`` — N_ρ, the opponent team members whose TRUE moveset holds the role move."""
    nums = roles.nums
    out = np.zeros((br.n, len(nums)), dtype=np.float64)
    for i, team in enumerate(br.true_moves):
        for j, m in enumerate(nums):
            out[i, j] = sum(1 for mv in team if m in mv)
    return out


def stable_rank(pi: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """``[N,S]`` each species' rank in the ONE order (π descending, ties to the lower num,
    non-candidates last) — `hypothesis_set.stable_order` in numpy."""
    key = np.where(cand, -pi, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(pi.shape[1])[None, :].repeat(pi.shape[0], 0), axis=1)
    return rank


# ------------------------------------------------------------------------------------------ intent
@functools.lru_cache(maxsize=1)
def struggle_num() -> int:
    from agents import gen3_data

    md = gen3_data.moves.get("struggle")
    assert md is not None
    return int(md.num)


@functools.lru_cache(maxsize=None)
def _learnset_nums(species: int) -> Optional[frozenset]:
    """The species' legal move nums on the common event space (HP collapsed), None if unknown."""
    from agents import gen3_data
    from main.belief_roles.bank_rows import move_event_num

    sid = next((s for s in gen3_data.species.base_form_ids()
                if int(gen3_data.species.get(s).num) == species), None)  # type: ignore[union-attr]
    legal = gen3_data.learnset.get_legal_moves(sid) if sid else None
    if legal is None:
        return None
    out = set()
    for m in legal:
        try:
            out.add(move_event_num(m))
        except KeyError:
            continue
    return frozenset(out)


def miss_breakdown(br: BankRows, cols: Columns, miss: np.ndarray) -> dict:
    """What the MISSED events are (U4's F-X5-34 asks for the composition before a miss is read as a
    belief property): a move inside / outside the active species' learnset (outside = copied by
    Transform / Mimic / Metronome …, or a learnset-table gap), a switch-in."""
    out = {"move_in_learnset": 0, "move_outside_learnset": 0, "move_species_unknown": 0, "switch": 0}
    top: Dict[int, int] = {}
    for i in np.flatnonzero(miss):
        e = int(br.event[i])
        if e >= SWITCH_BASE:
            out["switch"] += 1
            continue
        legal = _learnset_nums(int(cols.opp_active_species[i]))
        if legal is None:
            out["move_species_unknown"] += 1
        elif e in legal:
            out["move_in_learnset"] += 1
        else:
            out["move_outside_learnset"] += 1
        top[e] = top.get(e, 0) + 1
    out["top_missed_moves"] = [[m, n] for m, n in sorted(top.items(), key=lambda x: (-x[1], x[0]))[:10]]
    return out


def intent_read(br: BankRows, cols: Columns, rows: np.ndarray) -> dict:
    """The intent read. A STRUGGLE label is excluded and counted: it is forced (no PP left), not a
    choice, and no candidate set of either arm can name it (training labels it a MOVE, so α masks it)."""
    struggle = rows & (br.event == struggle_num())
    lab = rows & (br.event != NO_EVENT) & ~struggle
    tie = lab & cols.read_tie_arm
    L = lab & ~cols.read_tie_arm
    cov = L & cols.intent_covered
    is_sw = br.event >= SWITCH_BASE
    nll = -cols.intent_logp

    def block(sel_rows) -> dict:
        n_l = int(sel_rows.sum())
        c = sel_rows & cols.intent_covered
        n_c = int(c.sum())
        v = nll[c]
        return {"n_labeled": n_l, "n_covered": n_c, "n_miss": n_l - n_c,
                "miss_rate": _f((n_l - n_c) / n_l) if n_l else None,
                "logloss": _f(v.mean()) if n_c else None,
                "logloss_ci95": cluster_ci(v, br.battle_index[c]) if n_c else None}

    out = {"all": block(L), "move": block(L & ~is_sw), "switch": block(L & is_sw),
           "by_opp_class": {k: block(L & (np.asarray(br.opp_class) == k)) for k in CLASSES},
           "excluded_rule8": int(tie.sum()), "excluded_struggle": int(struggle.sum()),
           "miss_breakdown": miss_breakdown(br, cols, L & ~cols.intent_covered)}
    assert np.isfinite(nll[cov]).all(), "a covered event has a non-finite log loss"
    return out


def intent_conditional(br: BankRows, cols: Columns, rows: np.ndarray) -> dict:
    """Metric (1) over every E_row the run is scored on: one :func:`intent_conditional_read` per
    reference (a blob run: its own set, one entry) and their MEAN — the run's value. A reference whose
    value is None makes the mean None (never a mean over the survivors)."""
    if not cols.esets:
        return {"available": False,
                "reason": "no E_row: a fixed_mass run is read on EVERY blob run of the look "
                          "(read --reference <label>=<blob>[,<blob>...])"}
    per = []
    for es in cols.esets:
        r = intent_conditional_read(br, cols, rows, es)
        r["reference_label"], r["reference_checkpoint_sha256"] = es.label, es.checkpoint_sha256
        per.append(r)

    def mean(get) -> Optional[float]:
        v = [get(r) for r in per]
        return None if any(x is None for x in v) else float(np.mean(v))

    return {"available": True, "n_references": len(per), "per_reference": per,
            "mean": {"logloss": mean(lambda r: r["all"]["logloss"]),
                     "set_miss_rate": mean(lambda r: r["all"]["set_miss_rate"]),
                     "outside_mass": mean(lambda r: r["coverage"].get("mean_outside_mass")),
                     "citl": mean(lambda r: r["coverage"].get("citl"))}}


def intent_conditional_read(br: BankRows, cols: Columns, rows: np.ndarray, es) -> dict:
    """Amendment 3(b)'s metric (1) and coverage on ONE E_row ``es`` (``eset.ESet``; module docstring).
    Rows: the labelled, non-Struggle
    rows; rule 8 excludes a row whose E_row is ill-determined (the reference blob's E4 seat cut is a
    near-tie) or whose arm read is (``read_tie_arm``), and a row whose denominator Σ_{E_row} P is within
    :data:`eset.DENOM_EPS` of 0. A realised event IN E_row to which the arm gives exactly zero mass has
    an infinite conditional loss: it is counted (``n_zero_event_mass``) and the metric is then None —
    never floored, never dropped in silence."""
    from main.belief_roles.eset import DENOM_EPS

    assert cols.eset_logp_event is not None
    struggle = rows & (br.event == struggle_num())
    lab = rows & (br.event != NO_EVENT) & ~struggle
    tie = lab & (es.tie | cols.read_tie_arm)
    L = lab & ~tie
    inset = L & es.e_in
    den0 = inset & (es.mass <= DENOM_EPS)
    U = inset & ~den0
    zero = U & ~np.isfinite(cols.eset_logp_event)
    is_sw = br.event >= SWITCH_BASE
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = -(cols.eset_logp_event - np.log(np.where(U, es.mass, 1.0)))

    def block(sel) -> dict:
        n_l = int((sel & L).sum())
        u = sel & U
        n_z = int((sel & zero).sum())
        v = cond[u]
        ok = n_z == 0 and v.size > 0
        return {"n_rows": n_l, "n_in_set": int((sel & inset).sum()), "n_scored": int(u.sum()),
                "n_zero_event_mass": n_z, "set_miss_rate": _f((n_l - int((sel & inset).sum())) / n_l)
                if n_l else None,
                "logloss": _f(v.mean()) if ok else None,
                "logloss_ci95": cluster_ci(v, br.battle_index[u]) if ok else None}

    allr = np.ones(br.n, dtype=bool)
    out = {"all": block(allr), "move": block(~is_sw), "switch": block(is_sw),
           "by_opp_class": {k: block(np.asarray(br.opp_class) == k) for k in CLASSES},
           "excluded_rule8_tie": int(tie.sum()), "excluded_rule8_denominator": int(den0.sum()),
           "excluded_struggle": int(struggle.sum())}
    # ---- coverage (REPORTED): the arm's mass outside E_row vs the observed outside frequency
    mass_out = np.clip(1.0 - es.mass[L], 0.0, 1.0)
    obs_out = (~es.e_in[L]).astype(np.float64)
    cov: dict = {"n": int(L.sum())}
    if L.any():
        edges = np.asarray(OUTSIDE_BINS)
        b = np.clip(np.searchsorted(edges, mass_out, side="right") - 1, 0, len(edges) - 2)
        bins = [{"lo": float(edges[i]), "n": int((b == i).sum()), "mass": float(mass_out[b == i].mean()),
                 "freq": float(obs_out[b == i].mean())} for i in range(len(edges) - 1) if (b == i).any()]
        d = mass_out - obs_out
        var = (mass_out * (1 - mass_out)).sum()
        cov.update({"mean_outside_mass": float(mass_out.mean()), "outside_freq": float(obs_out.mean()),
                    "citl": float(d.mean()), "citl_ci95": cluster_ci(d, br.battle_index[L]),
                    "z": float((obs_out - mass_out).sum() / np.sqrt(var)) if var > 0 else None,
                    "bins": bins})
    out["coverage"] = cov
    return out


# ---------------------------------------------------------------------------------------- presence
def presence_read(pi: np.ndarray, y: np.ndarray, cand: np.ndarray, k: np.ndarray,
                  clusters: np.ndarray, rows: np.ndarray) -> dict:
    P = rows & (k >= 1)
    if not P.any():
        return {"n": 0}
    p, t, c = pi[P], y[P].astype(np.float64), cand[P]
    sq = np.where(c, (p - t) ** 2, 0.0)
    brier = sq.sum(1)
    pc = np.clip(p, 1e-300, 1.0)
    qc = np.clip(1.0 - p, 1e-300, 1.0)
    bce = np.where(c, -(t * np.log(pc) + (1 - t) * np.log(qc)), 0.0).sum(1)
    kk = k[P].astype(np.float64)
    # Murphy split over the pooled per-species indicators, fixed bins
    pv, tv = p[c], t[c]
    edges = np.asarray(PI_BINS)
    b = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, len(edges) - 2)
    nb = np.bincount(b, minlength=len(edges) - 1).astype(np.float64)
    pbar = np.bincount(b, weights=pv, minlength=len(nb)) / np.maximum(nb, 1)
    obar = np.bincount(b, weights=tv, minlength=len(nb)) / np.maximum(nb, 1)
    o = tv.mean()
    N = float(len(pv))
    rel = float((nb * (pbar - obar) ** 2).sum() / N)
    res = float((nb * (obar - o) ** 2).sum() / N)
    # class-wise mean calibration, the most frequent true species
    freq = t.sum(0)
    top = [int(s) for s in np.argsort(-freq, kind="stable")[:N_CLASSES] if freq[s] > 0]
    classwise = []
    for s in top:
        m = c[:, s]
        classwise.append({"num": s, "n": int(m.sum()), "mean_pi": float(p[m, s].mean()),
                          "freq": float(t[m, s].mean())})
    return {"n": int(P.sum()), "brier": float(brier.mean()),
            "brier_ci95": cluster_ci(brier, clusters[P]),
            "brier_per_mon": float((brier / kk).mean()), "bce": float(bce.mean()),
            "murphy": {"reliability": rel, "resolution": res, "uncertainty": float(o * (1 - o)),
                       "n_indicators": int(N),
                       "bins": [{"lo": float(edges[i]), "n": int(nb[i]), "mean_pi": _f(pbar[i]),
                                 "freq": _f(obar[i])} for i in range(len(nb)) if nb[i] > 0]},
            "classwise": classwise}


def other_read(pi: np.ndarray, y: np.ndarray, cand: np.ndarray, k: np.ndarray, tie: np.ndarray,
               clusters: np.ndarray, rows: np.ndarray,
               model_mass: Optional[np.ndarray] = None) -> dict:
    base = rows & (k >= 1)
    O = base & ~tie
    if not O.any():
        return {"n": 0, "excluded_rule8": int((base & tie).sum())}
    rank = stable_rank(pi[O], cand[O])
    kk = k[O]
    tail = cand[O] & (rank >= kk[:, None])
    mass = np.where(tail, pi[O], 0.0).sum(1)
    real = (tail & y[O]).sum(1).astype(np.float64)
    var = np.where(tail, pi[O] * (1 - pi[O]), 0.0).sum(1)
    if model_mass is not None:
        gap = float(np.abs(mass - model_mass[O]).max())
        if gap > 1e-4:
            raise AssertionError(f"the read's OTHER mass disagrees with the model's own by {gap:.3g}")
    err = mass - real
    by_r = {}
    for r in range(1, 6):
        m = (6 - kk) == r
        if m.any():
            by_r[str(r)] = {"n": int(m.sum()), "mass": float(mass[m].mean()),
                            "realised": float(real[m].mean()), "err": float(err[m].mean())}
    edges = np.asarray(MASS_BINS)
    b = np.clip(np.searchsorted(edges, mass, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for i in range(len(edges) - 1):
        m = b == i
        if m.any():
            bins.append({"lo": float(edges[i]), "n": int(m.sum()), "mass": float(mass[m].mean()),
                         "realised": float(real[m].mean())})
    return {"n": int(O.sum()), "excluded_rule8": int((base & tie).sum()),
            "mean_mass": float(mass.mean()), "mean_realised": float(real.mean()),
            "mean_err": float(err.mean()), "mean_err_ci95": cluster_ci(err, clusters[O]),
            "z": float((real - mass).sum() / np.sqrt(var.sum())) if var.sum() > 0 else None,
            "share_of_k": float((mass / kk).mean()), "by_r": by_r, "bins": bins}


# -------------------------------------------------------------------------------------------- roles
def roles_read(M: np.ndarray, V: np.ndarray, Nt: np.ndarray, k: np.ndarray, roles: RoleSet,
               clusters: np.ndarray, rows: np.ndarray, tie: np.ndarray) -> dict:
    r = 6 - k
    base = rows & (r >= 1) & (r <= 5)
    use = base & ~tie
    w = np.asarray([x.carriers for x in roles.roles])
    per = []
    abs_d = []
    for j, ro in enumerate(roles.roles):
        d = M[use, j] - Nt[use, j]
        strata = {}
        for rr in range(1, 7):
            m = rows & ~tie & (r == rr)
            if m.any():
                strata[str(rr)] = {"n": int(m.sum()), "delta": float((M[m, j] - Nt[m, j]).mean())}
        vs = V[use, j].sum()
        per.append({"num": ro.num, "move": ro.move, "carriers": ro.carriers,
                    "mean_M": _f(M[use, j].mean()) if use.any() else None,
                    "mean_N": _f(Nt[use, j].mean()) if use.any() else None,
                    "delta": _f(d.mean()) if use.any() else None,
                    "delta_ci95": cluster_ci(d, clusters[use]) if use.any() else None,
                    "z": _f((Nt[use, j] - M[use, j]).sum() / np.sqrt(vs)) if vs > 0 else None,
                    "by_r": strata})
        abs_d.append(abs(d.mean()) if use.any() else np.nan)
    a = np.asarray(abs_d)
    ok = np.isfinite(a)
    summary = float((w[ok] * a[ok]).sum() / w[ok].sum()) if ok.any() else None
    return {"n": int(use.sum()), "excluded_rule8": int((base & tie).sum()),
            "weighted_abs_delta": summary, "roles": per}


def r3_read(pi: np.ndarray, y: np.ndarray, cand: np.ndarray, roles: RoleSet,
            rows: np.ndarray) -> dict:
    """R3 over (row, pair) instances (module docstring)."""
    both = n = excl = 0
    exc_sum = 0.0
    per = []
    for p in roles.pairs:
        a, b = p.s1, p.s2
        m = rows & cand[:, a] & cand[:, b] & (y[:, a] ^ y[:, b])
        pa, pb = pi[m, a], pi[m, b]
        band = ((pa >= R3_BAND[0]) & (pa <= R3_BAND[1])) | ((pb >= R3_BAND[0]) & (pb <= R3_BAND[1]))
        pa, pb = pa[~band], pb[~band]
        bt = (pa >= R3_BAR) & (pb >= R3_BAR)
        n_i = int(len(pa))
        per.append({"s1": p.name1, "s2": p.name2, "role": p.role, "n": n_i, "excluded_rule8": int(band.sum()),
                    "both_rate": _f(bt.mean()) if n_i else None,
                    "excess": _f((pa + pb - 1).mean()) if n_i else None})
        both += int(bt.sum())
        exc_sum += float((pa + pb - 1).sum())
        n += n_i
        excl += int(band.sum())
    return {"n_instances": n, "excluded_rule8": excl, "both_rate": _f(both / n) if n else None,
            "excess": _f(exc_sum / n) if n else None, "pairs": per}


# ----------------------------------------------------------------------------------------- the read
def read_all(br: BankRows, cols: Columns, roles: RoleSet) -> dict:
    """Every read on the on-pool (primary) and off-pool strata, both columns, plus ``per_run``."""
    y, consistent = unseen_labels(br, cols)
    Nt = role_counts(br, roles)
    out: Dict[str, dict] = {"label_mismatch": int(((cols.k >= 1) & ~consistent).sum())}
    fm = cols.arm == "fixed_mass"
    for stratum, sel in (("on_pool", br.on_pool), ("off_pool", ~br.on_pool)):
        good = sel & consistent
        blk: Dict[str, dict] = {"n_rows": int(sel.sum())}
        blk["intent"] = intent_read(br, cols, sel)
        blk["intent_conditional"] = intent_conditional(br, cols, sel)
        for col, pi, tie, rM, rV, rtie, mm in (
                ("arm", cols.pi_arm, cols.sel_tie_arm, cols.role_M_arm, cols.role_V_arm,
                 cols.read_tie_arm if fm else np.zeros(br.n, dtype=bool),
                 cols.other_mass_model if fm else None),
                ("prior", cols.pi_prior, cols.sel_tie_prior, cols.role_M_prior, cols.role_V_prior,
                 np.zeros(br.n, dtype=bool), None)):
            blk[col] = {
                "presence": presence_read(pi, y, cols.cand, cols.k, br.battle_index, good),
                "other": other_read(pi, y, cols.cand, cols.k, tie, br.battle_index, good, mm),
                "roles_r1_r2": roles_read(rM, rV, Nt, cols.k, roles, br.battle_index, sel, rtie),
                "roles_r3": r3_read(pi, y, cols.cand, roles, good & ~rtie),
            }
        out[stratum] = blk
    out["per_run"] = per_run(out["on_pool"])
    out["per_run_off_pool"] = per_run(out["off_pool"])
    return out


#: Amendment 3(b): purpose metric (1)'s ADOPTION-GATE form (the old ``intent_logloss`` is descriptive).
ADOPTION_GATE_METRIC = "intent_logloss_conditional"
#: The per-run scalars §7.4's across-seed inference reads, with the direction that is BETTER.
PER_RUN_DIRECTION = {
    ADOPTION_GATE_METRIC: "lower",
    "intent_logloss": "lower", "intent_miss_rate": "lower",
    "presence_brier": "lower", "presence_bce": "lower", "presence_resolution": "higher",
    "other_abs_err": "lower", "roles_r1_weighted_abs_delta": "lower", "roles_r3_both_rate": "lower",
}


def per_run(blk: dict) -> Dict[str, Optional[float]]:
    a = blk["arm"]
    oe = a["other"].get("mean_err")
    icm = (blk.get("intent_conditional") or {}).get("mean") or {}
    return {
        ADOPTION_GATE_METRIC: icm.get("logloss"),
        "intent_set_miss_rate": icm.get("set_miss_rate"),
        "intent_outside_mass": icm.get("outside_mass"),
        "intent_outside_citl": icm.get("citl"),
        "intent_logloss": blk["intent"]["all"]["logloss"],
        "intent_miss_rate": blk["intent"]["all"]["miss_rate"],
        "presence_brier": a["presence"].get("brier"),
        "presence_bce": a["presence"].get("bce"),
        "presence_resolution": (a["presence"].get("murphy") or {}).get("resolution"),
        "other_mean_err": oe,
        "other_abs_err": None if oe is None else abs(oe),
        "roles_r1_weighted_abs_delta": a["roles_r1_r2"]["weighted_abs_delta"],
        "roles_r3_both_rate": a["roles_r3"]["both_rate"],
    }
