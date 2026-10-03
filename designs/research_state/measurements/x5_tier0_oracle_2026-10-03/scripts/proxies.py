"""The three species beliefs of the X5 revision (`x5_revision_2026-10-03/scripts/m1m3.py`), re-built here so they
apply to REAL decision states. The construction and the per-r fits are copied from m1m3.py line for line and
refitted on m1m3's own synthetic rows (same seed), and `check_fits()` asserts the fitted values equal the ones m1m3
logged — so the beliefs here ARE the revision's beliefs.

- `prior`: the Smogon T0 teammate naive Bayes (cold start).
- `memo_nb`: a geometric blend of the prior with an IN-SAMPLE pool naive Bayes (719 archetype teams), blend fitted
  per number of reveals r to the banked head's per-slot NLL (L21301). MEASUREMENT proxy only; never a prior.
- `memo_exact`: P(s | revealed set) = share of pool teams containing the revealed set that also contain s, plus
  eps x prior, eps fitted per r the same way. Near-oracle on pool teams once the revealed set identifies the team.
"""
from __future__ import annotations

import json
import random

import torch

from agents import gen3_data
from agents.model.belief_tables import build_species_cooccur_prior
from agents.model.t0_species import species_team_prior_logits
from agents.observation.constants import MAX_SPECIES_ID
from utils.paths import repo_path

S = MAX_SPECIES_ID + 1
NEG = -1e30
HEAD = {1: 2.24, 2: 2.51, 3: 2.70, 4: 2.61, 5: 2.54}
# What m1m3.log printed (x5_revision_2026-10-03/out/m1m3.log lines 4 and 7).
M1M3_LAMS = {1: 1.0, 2: 1.0, 3: 0.252, 4: 0.066, 5: 0.0}
M1M3_EPS = {1: "1e-12", 2: "1e-12", 3: "31", 4: "183", 5: "1e+03"}


def _num(n):
    sp = gen3_data.species.get(n)
    return sp.num if sp else None


class Beliefs:
    def __init__(self):
        self.lm, self.ll = build_species_cooccur_prior(S)
        teams_raw = json.load(open(repo_path('data', 'teams', 'gen3_team_archetypes.json')))['teams']
        teams = []
        for _k, v in teams_raw.items():
            nums = [_num(s) for s in v['species']]
            if len(nums) == 6 and all(nums) and len(set(nums)) == 6:
                teams.append(nums)
        self.teams = teams
        self.team_sets = [set(t) for t in teams]
        C = torch.zeros(S, S, dtype=torch.float64); M = torch.zeros(S, dtype=torch.float64)
        for t in teams:
            for a in t:
                M[a] += 1
                for b in t:
                    if a != b:
                        C[a, b] += 1
        alpha = 0.5
        self.Pm = (M + alpha) / (M.sum() + alpha * S)
        cond = (C + alpha * self.Pm[None, :] * 5) / (M[:, None] * 5 + alpha * 5)
        self.pool_lift = torch.log(cond / self.Pm[None, :]).clamp(-6, 6)
        self._fit()

    # ---- the per-decision pieces ------------------------------------------------------------------
    def _parts(self, ids: torch.Tensor, hid: torch.Tensor):
        """ids [B,6] opp species nums (0 at hidden), hid [B,6] bool -> (valid, P_log, pool_log, ex, r)."""
        B = ids.shape[0]
        valid = torch.ones(B, S, dtype=torch.bool); valid[:, 0] = False
        onehot = torch.zeros(B, S, dtype=torch.float64)
        rev = ids * (~hid).long()
        valid.scatter_(1, rev, False); valid[:, 0] = False
        onehot.scatter_(1, rev, 1.0); onehot[:, 0] = 0.0
        prior_log = species_team_prior_logits(self.lm, self.ll, ids, hid).double()
        pool_log = torch.log(self.Pm)[None, :] + onehot @ self.pool_lift
        r = (~hid).sum(-1)
        ex = torch.zeros(B, S, dtype=torch.float64)
        for i in range(B):
            R_ = set(int(x) for x in rev[i].tolist() if x > 0)
            sup = [t for t in self.team_sets if R_ <= t]
            for t in sup:
                for s_ in t - R_:
                    ex[i, s_] += 1.0 / len(sup)
        return valid, prior_log, pool_log, ex, r

    @staticmethod
    def _norm(a, valid):
        a = torch.where(valid, a, torch.full_like(a, NEG))
        return torch.log_softmax(a, -1)

    def logbeliefs(self, ids: torch.Tensor, hid: torch.Tensor, pad_to: int = S):
        """{name: [B,pad_to] fp64 log-probabilities over the VALID set (NEG elsewhere)}, plus valid [B,pad_to].
        The belief axis is m1m3's S = MAX_SPECIES_ID + 1 (nums 1..S-1 valid unless revealed); the op's tables are
        wider (padding rows), which are padded here as INVALID."""
        out, valid = self._logbeliefs(ids, hid)
        if pad_to > S:
            B = ids.shape[0]
            out = {k: torch.cat([v, torch.full((B, pad_to - S), NEG, dtype=v.dtype)], 1) for k, v in out.items()}
            valid = torch.cat([valid, torch.zeros(B, pad_to - S, dtype=torch.bool)], 1)
        return out, valid

    def _logbeliefs(self, ids: torch.Tensor, hid: torch.Tensor):
        valid, prior_log, pool_log, ex, r = self._parts(ids, hid)
        P_log = self._norm(prior_log, valid)
        lam = torch.tensor([self.lams.get(int(x), 0.0) for x in r], dtype=torch.float64)[:, None]
        Mlog = self._norm((1 - lam) * P_log + lam * self._norm(pool_log, valid), valid)
        eps = torch.tensor([self.eps.get(int(x), 1.0) for x in r], dtype=torch.float64)[:, None]
        Elog = self._norm(torch.log(ex + eps * P_log.exp()), valid)
        return {"prior": P_log, "memo_nb": Mlog, "memo_exact": Elog}, valid

    # ---- the fits, exactly as m1m3.py -------------------------------------------------------------
    def _fit(self):
        random.seed(0)
        rows = []
        for t in self.teams:
            for r in range(0, 6):
                for _ in range(3 if r else 1):
                    perm = random.sample(t, 6); rows.append((r, perm[:r], perm[r:]))
        B = len(rows)
        ids = torch.zeros(B, 6, dtype=torch.long); hid = torch.ones(B, 6, dtype=torch.bool)
        truth = torch.zeros(B, S, dtype=torch.float64); rr = torch.zeros(B, dtype=torch.long)
        for i, (r, rev, hd) in enumerate(rows):
            for j, s in enumerate(rev):
                ids[i, j] = s; hid[i, j] = False
            for s in hd:
                truth[i, s] = 1
            rr[i] = r
        valid, prior_log, pool_log, ex, _r = self._parts(ids, hid)
        P_log = self._norm(prior_log, valid)

        def slot_nll(lq, mask):
            v = -(lq * truth).sum(-1) / truth.sum(-1).clamp(min=1)
            return v[mask].mean().item()

        def blend(lam):
            return self._norm((1 - lam) * P_log + lam * self._norm(pool_log, valid), valid)
        lams = {}
        for r_ in range(1, 6):
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if slot_nll(blend(mid), rr == r_) > HEAD[r_]:
                    lo = mid
                else:
                    hi = mid
            lams[r_] = (lo + hi) / 2
        fits = {}
        for r_ in range(1, 6):
            lo, hi = -12.0, 3.0
            for _ in range(50):
                mid = (lo + hi) / 2
                if slot_nll(self._norm(torch.log(ex + (10 ** mid) * P_log.exp()), valid), rr == r_) > HEAD[r_]:
                    hi = mid
                else:
                    lo = mid
            fits[r_] = 10 ** ((lo + hi) / 2)
        self.lams, self.eps = lams, fits

    def check_fits(self):
        got_l = {k: round(v, 3) for k, v in self.lams.items()}
        got_e = {k: f"{v:.3g}" for k, v in self.eps.items()}
        if got_l != M1M3_LAMS or got_e != M1M3_EPS:
            raise RuntimeError(f"belief fits differ from m1m3's: lams {got_l} vs {M1M3_LAMS}; eps {got_e} vs {M1M3_EPS}")
        return {"lams": got_l, "eps": got_e, "n_pool_teams": len(self.teams)}


def fixed_size(a: torch.Tensor, k: torch.Tensor, valid: torch.Tensor, iters: int = 64) -> torch.Tensor:
    """pi = sigmoid(a + tau), sum_V pi = k (design §3.2; m1m3.py's construction, fp64, no Newton step)."""
    a = a.double(); kd = k.double()
    big = torch.tensor(1e30, dtype=torch.float64)
    amax = torch.where(valid, a, -big).max(-1).values; amin = torch.where(valid, a, big).min(-1).values
    nv = valid.sum(-1).double()
    kc = kd.clamp(min=0.5).minimum(nv - 0.5)
    base = torch.log(kc) - torch.log(nv - kc)
    lo = base - amax; hi = base - amin

    def f(tau):
        return torch.where(valid, torch.sigmoid(a + tau[:, None]), torch.zeros_like(a)).sum(-1)
    with torch.no_grad():
        for _ in range(iters):
            mid = (lo + hi) / 2
            up = f(mid) < kd
            lo = torch.where(up, mid, lo); hi = torch.where(up, hi, mid)
        tau = (lo + hi) / 2
    pi = torch.where(valid, torch.sigmoid(a + tau[:, None]), torch.zeros_like(a))
    pi = torch.where((kd == 0)[:, None], torch.zeros_like(pi), pi)
    return pi
