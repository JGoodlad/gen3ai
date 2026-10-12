"""THE MODEL-INPUT INJECTION: push the network's own opponent MOVE belief toward a target set with dose p.

WHERE IT HOOKS, AND WHY EVERY CONSUMER SEES IT
-------------------------------------------------
At both pins (end2 62085750, prod 95d014fa) the move belief is produced in ONE place,
``ExtractorForward._apply_move_belief`` (one call site per forward, PRE-transformer), which calls
``self._typed_hp_posterior(opp_tokens, sctx, raw)`` and receives ``(typed_logits, presence, hp_post, hp_logits)``.
Every forward consumer reads one of those four, after the return:

* ``typed_logits`` [B,6,400] -> ``_publish_belief`` -> ``last_move_belief_logits`` (the published posterior), the
  opponent ACTIVE's move group (``_attach_move_group`` -> ``fixed_mass_moves``), the soft-embed REINJECTION weights
  (end2: ``slot_move_presence(..., graph=True)`` under ``--move-set-closure on``; prod: ``sigmoid(logits)``, closure
  off: the ONLY code difference between the pins on this path), and, through ``last_move_belief_logits``, the damage
  op, the X5 op roster (``build_op_roster``), the E4 seats, the edge cells, the consequence cells;
* ``presence`` [B,6] and ``hp_post`` [B,6,16] -> ``hp_type_belief_head.reinject`` (the presence-gated expected
  Hidden-Power TYPE embedding, revealed slots only). It does NOT read the typed logits, so a hook on the logits
  alone would leave the HP-type reinjection on the old belief. This module therefore rewrites ``presence`` and
  ``hp_post`` CONSISTENTLY with the rewritten typed channels on every targeted slot whose Hidden Power is not
  revealed: presence' = sum_t sigma(l'_HP_t), hp_post' = sigma(l'_HP_t) / presence' (the composition's own identity
  P(HP_t) = presence * P(t), which holds exactly whenever the tracker has not narrowed the type, i.e. whenever HP is
  unrevealed);
* ``hp_logits`` (the HP-type head's logits) -> ``stash.hp_type_logits`` = the aux CE + the prober ONLY (no forward
  consumer): left untouched.

So wrapping the extractor INSTANCE's ``_typed_hp_posterior`` (an instance attribute shadows the class method; the
class and every other instance are untouched) is the narrowest point every consumer reads. ``_apply_move_belief``
is wrapped too, only to capture the REAL ``ctx`` (opp_believed_mask, revealed move ids) and the X5 hypothesis set
``hs`` (``slot_is_hypothesis``, ``slot_species``) of the same forward: ``sctx`` is the hypothesis context, whose
hidden slots carry hypothesis species. Not covered (and not a per-slot posterior): the OTHER-mode roster's
tail-averaged hidden mon (``other_roster`` reads ``self.move_belief``'s prior tables), hidden slots only.

THE REWRITE (per targeted opponent slot s, target set M_s, dose p; brief section 1)
--------------------------------------------------------------------------------------
q = sigma(l); q' = (1 - p) q + p 1[m in M_s]; l' = logit(clip(q', 1e-6, 1 - 1e-6)), on every move num of the slot
EXCEPT its PINNED channels, which keep their exact bytes: the revealed moves, the bookkeeping typeless-HP channel 237,
and, when Hidden Power is revealed (bare 237 or a typed num in the slot's revealed ids), all 16 typed HP channels
(they are then that one revealed move's TYPE distribution, not candidates). One refinement of the clip, so the clip
can never move a channel AGAINST the injection: a non-target channel takes min(l, l'), a target channel max(l, l').
This equals the brief's formula wherever 1e-6 <= q <= 1 - 1e-6; it differs only where the clip would have LIFTED a
hard-off channel (e.g. a move the species cannot learn) up to 1e-6, which the min keeps hard off.
p == 0 selects the original tensor elementwise (``torch.where``), so a p = 0 forward is bit-identical.

TARGETS (per family; computed INSIDE the forward from its own ctx, so a successor row gets its own targets)
---------------------------------------------------------------------------------------------------------------
A slot is SEEN when its species is revealed (``~ctx.opp_believed_mask``); it maps to the TRUE opponent mon whose
species num equals the slot's (a forme shares its base num); its TRUE UNREVEALED set U = true moves - pinned
(Hidden Power as its TYPED num). A slot with U empty gets no target in any family. r = the slot's revealed count.
* TRUE: M_s = U on every seen slot.
* PLACEBO: on the slots TRUE targets, (4 - r) moves drawn without replacement, proportional to the network's own BASE
  sigma(l) (this forward's un-rewritten posterior), from the slot's candidates (the species' legal moves as the
  network's hypothesis builder defines them, ``move_legal & move_valid``) minus the true moves and the pinned
  channels, at most one Hidden Power; RNG = default_rng(sha256("hib_placebo_v1:" + decision_id + ":" + slot)).
* SELFMAP: on the slots TRUE targets, the top-(4 - r) candidates (minus pinned) by base sigma(l), ties by move num,
  at most one Hidden Power (an honest sharpening).
* TRUEHYP: TRUE plus every X5 HYPOTHESIS slot whose hypothesis species is one of the opponent's TRUE never-seen
  species (true team nums minus the row's seen nums) gets that species' true moves.

Use::

    inj = Injector(model.policy.features_extractor)
    with inj.installed():
        inj.spec = Spec("TRUE", 0.95, did, true_sets)     # None = pass-through (the hook still runs, writes nothing)
        out = logits_and_values(model, rows, masks, batch=len(rows))
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

TEAM = 6
HP_BARE = 237
TYPED_LO, TYPED_HI = 355, 370
TYPED = tuple(range(TYPED_LO, TYPED_HI + 1))
CLIP = 1e-6
SET_SIZE = 4


def is_hp(n: int) -> bool:
    return n == HP_BARE or TYPED_LO <= n <= TYPED_HI


# ------------------------------------------------------------------------------------------------
# the true opponent team -> {species num: frozenset(move nums)} (typed Hidden Power)
# ------------------------------------------------------------------------------------------------

class TeamMapper:
    def __init__(self, data_dir: Path):
        import determinize as D  # the pilot's (hp_type_of, parse_set, set_ivs, toid)

        self.D = D
        mv = json.loads((Path(data_dir) / "gen3_moves.json").read_text())
        sp = json.loads((Path(data_dir) / "gen3_species.json").read_text())
        self.move_num = {k: int(v["num"]) for k, v in mv.items() if v.get("num")}
        self.species_num = {k: int(v["num"]) for k, v in sp.items() if v.get("num")}
        for t in D.HP_TYPES:
            n = self.move_num.get("hiddenpower" + t)
            if n is None or not (TYPED_LO <= n <= TYPED_HI):
                raise RuntimeError(f"hiddenpower{t} is not a typed HP num at this pin ({n})")
        if self.move_num.get("hiddenpower") != HP_BARE:
            raise RuntimeError("bare hiddenpower is not 237 at this pin")

    def true_sets(self, packed: str) -> Dict[int, FrozenSet[int]]:
        D = self.D
        out: Dict[int, FrozenSet[int]] = {}
        for entry in packed.split("]"):
            if not entry.strip():
                continue
            f = D.parse_set(entry)
            sid = D.toid(f[1] or f[0])
            if sid not in self.species_num:
                raise RuntimeError(f"unknown species {sid!r}")
            num = self.species_num[sid]
            moves = set()
            for m in f[4].split(","):
                mid = D.toid(m)
                if not mid:
                    continue
                if mid == "hiddenpower":
                    mid = "hiddenpower" + D.hp_type_of(D.set_ivs(f))
                if mid not in self.move_num:
                    raise RuntimeError(f"unknown move {mid!r}")
                moves.add(self.move_num[mid])
            if num in out:
                raise RuntimeError(f"two team members share species num {num}")
            out[num] = frozenset(moves)
        return out


# ------------------------------------------------------------------------------------------------
# the spec of one forward call (every row of the call belongs to ONE decision)
# ------------------------------------------------------------------------------------------------

@dataclass
class Spec:
    family: str                                   # TRUE / PLACEBO / SELFMAP / TRUEHYP
    p: float
    did: str
    true_sets: Dict[int, FrozenSet[int]]          # the opponent's true team: species num -> move nums
    record: bool = False                          # keep the per-forward tensors for the teeth
    stats: dict = field(default_factory=dict)     # filled by the hook (counts of the LAST forward)


def _pinned(rev: List[int]) -> Tuple[set, bool]:
    pins = {int(x) for x in rev if int(x) > 0}
    hp_rev = any(is_hp(x) for x in pins)
    pins.add(HP_BARE)
    if hp_rev:
        pins.update(TYPED)
    return pins, hp_rev


def _pick_selfmap(sig: np.ndarray, cand: np.ndarray, k: int) -> List[int]:
    order = sorted(np.flatnonzero(cand).tolist(), key=lambda n: (-float(sig[n]), n))
    out: List[int] = []
    hp = False
    for n in order:
        if len(out) >= k:
            break
        if is_hp(n):
            if hp:
                continue
            hp = True
        out.append(int(n))
    return out


def _pick_placebo(sig: np.ndarray, cand: np.ndarray, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(cand).tolist()
    # probabilities rounded to 1e-9 so float noise below that (batch composition) cannot move a draw
    w = {n: round(float(sig[n]), 9) for n in pool}
    out: List[int] = []
    while len(out) < k:
        live = [n for n in pool if w[n] > 0]
        if not live:
            break
        tot = sum(w[n] for n in live)
        u = rng.random() * tot
        acc = 0.0
        pick = live[-1]
        for n in live:
            acc += w[n]
            if u < acc:
                pick = n
                break
        out.append(int(pick))
        pool.remove(pick)
        if is_hp(pick):
            pool = [n for n in pool if not is_hp(n)]
    return out


class Injector:
    def __init__(self, ext):
        hb = getattr(ext, "hypothesis_builder", None)
        if hb is None:
            raise RuntimeError("no X5 hypothesis builder on this extractor (the candidate set is undefined)")
        if getattr(ext, "move_belief", None) is None or getattr(ext, "hp_type_belief_head", None) is None:
            raise RuntimeError("this extractor has no move belief / typed HP head")
        self.ext = ext
        self.legal = (hb.move_legal & hb.move_valid.unsqueeze(0)).cpu().numpy()      # [S, M] bool
        self.spec: Optional[Spec] = None
        self.calls = 0
        self._box: dict = {}

    # ---------------------------------------------------------------- install
    @contextmanager
    def installed(self):
        ext = self.ext
        if "_typed_hp_posterior" in ext.__dict__ or "_apply_move_belief" in ext.__dict__:
            raise RuntimeError("the extractor instance is already hooked")
        orig_apply = ext._apply_move_belief          # bound methods of the CLASS
        orig_typed = ext._typed_hp_posterior

        def apply_wrap(opp_tokens, ctx, hctx=None, hs=None):
            self._box = {"ctx": ctx, "hs": hs}
            try:
                return orig_apply(opp_tokens, ctx, hctx, hs)
            finally:
                self._box = {}

        def typed_wrap(opp_tokens, sctx, raw):
            out = orig_typed(opp_tokens, sctx, raw)
            self.calls += 1
            if self.spec is None:
                return out
            if "ctx" not in self._box:
                raise RuntimeError("_typed_hp_posterior called outside _apply_move_belief")
            return self._rewrite(out, self._box["ctx"], self._box["hs"])

        ext._apply_move_belief = apply_wrap
        ext._typed_hp_posterior = typed_wrap
        try:
            yield self
        finally:
            del ext.__dict__["_apply_move_belief"]
            del ext.__dict__["_typed_hp_posterior"]
            self.spec = None

    # ---------------------------------------------------------------- targets
    def targets(self, ctx, hs, base_logits) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """(tgt [B,6,M] bool, slot_on [B,6] bool, pinned [B,6,M] bool, hp_rev [B,6] bool, stats)."""
        sp = self.spec
        assert sp is not None
        B, T, M = base_logits.shape
        species = ctx.species_ids[:, TEAM:].long().cpu().numpy()                     # [B,6] real
        believed = ctx.opp_believed_mask.bool().cpu().numpy()                      # [B,6]
        rev = ctx.all_move_ids[:, TEAM:, :].long().cpu().numpy()                     # [B,6,4]
        if hs is not None:
            hyp = hs.slot_is_hypothesis.bool().cpu().numpy()
            hsp = hs.slot_species.long().cpu().numpy()
        else:
            hyp = np.zeros((B, T), bool)
            hsp = np.zeros((B, T), np.int64)
        sig = None
        if sp.family in ("PLACEBO", "SELFMAP"):
            import torch

            sig = torch.sigmoid(base_logits.detach().double()).cpu().numpy()
        tgt = np.zeros((B, T, M), bool)
        pinned = np.zeros((B, T, M), bool)
        slot_on = np.zeros((B, T), bool)
        hp_rev_a = np.zeros((B, T), bool)
        st = {"rows": B, "seen_slots": 0, "seen_unmatched": 0, "slots": 0, "moves": 0, "true_unrev_lt_k": 0,
              "true_not_cand": 0, "short": 0, "hyp_slots": 0, "hyp_matched": 0, "never_seen": 0,
              "never_seen_covered": 0, "targets": [], "true_slot": [], "k_slot": [], "per_row": []}
        counters = [k for k, v in st.items() if isinstance(v, int)]
        cand_rec = np.zeros((B, T, M), bool) if sp.record else None
        true_nums = set(sp.true_sets)
        for b in range(B):
            snap = {k: st[k] for k in counters}
            seen_nums = {int(species[b, s]) for s in range(T) if not believed[b, s] and species[b, s] > 0}
            never = true_nums - seen_nums
            covered = set()
            row_t = []
            row_true: List[List[int]] = []
            row_k: List[int] = []
            for s in range(T):
                pins, hp_rev = _pinned(rev[b, s].tolist())
                hp_rev_a[b, s] = hp_rev
                for n in pins:
                    if n < M:
                        pinned[b, s, n] = True
                M_s: List[int] = []
                true_s: List[int] = []
                k = 0
                num = int(species[b, s])
                if not believed[b, s] and num > 0:
                    st["seen_slots"] += 1
                    true = sp.true_sets.get(num)
                    if true is None:
                        st["seen_unmatched"] += 1
                    else:
                        U = sorted(n for n in true if n not in pins)
                        r = len({int(x) for x in rev[b, s] if int(x) > 0})
                        k = max(0, SET_SIZE - r)
                        if U:
                            if len(U) < k:
                                st["true_unrev_lt_k"] += 1
                            cand = self.legal[min(num, self.legal.shape[0] - 1)].copy()
                            cand[list(pins)] = False
                            true_s = sorted(true)
                            if cand_rec is not None:
                                cand_rec[b, s] = cand
                            st["true_not_cand"] += sum(1 for n in U if not cand[n])
                            if sp.family in ("TRUE", "TRUEHYP"):
                                M_s = U
                            elif sp.family == "SELFMAP":
                                M_s = _pick_selfmap(sig[b, s], cand, k)
                            elif sp.family == "PLACEBO":
                                c2 = cand.copy()
                                c2[list(true)] = False
                                M_s = _pick_placebo(sig[b, s], c2, k, _seed(sp.did, s))
                            else:
                                raise ValueError(sp.family)
                            if len(M_s) < k and sp.family in ("PLACEBO", "SELFMAP"):
                                st["short"] += 1
                elif sp.family == "TRUEHYP" and hyp[b, s]:
                    st["hyp_slots"] += 1
                    h = int(hsp[b, s])
                    if h in never:
                        st["hyp_matched"] += 1
                        covered.add(h)
                        M_s = sorted(n for n in sp.true_sets[h] if n not in pins)
                if M_s:
                    slot_on[b, s] = True
                    tgt[b, s, M_s] = True
                    st["slots"] += 1
                    st["moves"] += len(M_s)
                row_t.append(M_s)
                row_true.append(true_s)
                row_k.append(k)
            if sp.family == "TRUEHYP":
                st["never_seen"] += len(never)
                st["never_seen_covered"] += len(covered)
            st["targets"].append(row_t)
            st["true_slot"].append(row_true)
            st["k_slot"].append(row_k)
            st["per_row"].append({k: st[k] - snap[k] for k in counters if k != "rows"})
        if cand_rec is not None:
            st["cand"] = cand_rec
        return tgt, slot_on, pinned, hp_rev_a, st

    # ---------------------------------------------------------------- the rewrite
    def _rewrite(self, out, ctx, hs):
        import torch

        logits, presence, hp_post, hp_logits = out
        sp = self.spec
        assert sp is not None
        tgt_np, on_np, pin_np, hprev_np, st = self.targets(ctx, hs, logits)
        dev, dt = logits.device, logits.dtype
        p = float(sp.p)
        sp.stats = st
        if sp.record:
            sp.stats["base_logits"] = logits.detach().float().cpu().numpy().copy()
            sp.stats["tgt"], sp.stats["slot_on"], sp.stats["pinned"] = tgt_np, on_np, pin_np
        if p == 0.0 or not on_np.any():
            # the hook ran (targets computed and counted); nothing is selected: the tensors pass unchanged
            return logits, presence, hp_post, hp_logits
        tgt = torch.from_numpy(tgt_np).to(dev)
        sel = torch.from_numpy(on_np).to(dev).unsqueeze(-1) & ~torch.from_numpy(pin_np).to(dev)
        l64 = logits.double()
        q = torch.sigmoid(l64)
        qn = (1.0 - p) * q + p * tgt.double()
        lc = torch.logit(qn.clamp(CLIP, 1.0 - CLIP))
        new = torch.where(tgt, torch.maximum(l64, lc), torch.minimum(l64, lc))
        logits2 = torch.where(sel, new.to(dt), logits)
        # the HP-type reinjection's inputs, consistent with the rewritten typed channels
        hp_sel = torch.from_numpy(on_np & ~hprev_np).to(dev)                          # [B,6]
        if presence is not None and hp_post is not None and bool(hp_sel.any()):
            qt = torch.sigmoid(logits2[..., TYPED_LO:TYPED_HI + 1].double())          # [B,6,16]
            pres2 = qt.sum(-1)
            post2 = qt / pres2.clamp_min(1e-300).unsqueeze(-1)
            presence = torch.where(hp_sel, pres2.to(presence.dtype), presence)
            hp_post = torch.where(hp_sel.unsqueeze(-1), post2.to(hp_post.dtype), hp_post)
        if sp.record:
            sp.stats["new_logits"] = logits2.detach().float().cpu().numpy().copy()
        return logits2, presence, hp_post, hp_logits


def _seed(did: str, slot: int) -> int:
    from hib_common import placebo_seed

    return placebo_seed(did, slot)
