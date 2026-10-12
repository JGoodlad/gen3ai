"""THE MODEL-INPUT INJECTION: consumer (c) of the information-lever spec. A FROZEN network's own belief inputs about the
opponent's hidden facts are pushed toward a lever's target with its mass p (``designs/endstate/design_information_levers.md``
§8; lifted from the hidden-information study's ``measurements/hidden_info_binding_2026-10-11/inject.py``, its §7.1).

WHERE IT HOOKS (every consumer of the belief reads the rewritten tensor; nothing else is touched):

* MOVES — the extractor INSTANCE's ``_typed_hp_posterior`` (an instance attribute shadows the class method for this
  extractor only): it returns ``(typed_logits, presence, hp_post, hp_logits)``, and every forward consumer of the move
  belief reads one of those (the published ``last_move_belief_logits``, the active's move group, the reinjection
  weights, the damage op, the X5 op roster, the E4 seats, the edges and cells). ``_apply_move_belief`` is wrapped only
  to capture the forward's real ``ctx`` (seen slots, revealed move ids) and the X5 hypothesis set. Per targeted slot:
  q = σ(ℓ); q' = (1 − p)·q + p·1[m ∈ target]; ℓ' = logit(clip(q')), on every move channel EXCEPT the pinned ones (the
  revealed moves, the bookkeeping typeless Hidden Power 237, and all 16 typed Hidden Power channels once Hidden Power
  is revealed); the clip never moves a channel against the injection (a non-target keeps min(ℓ, ℓ'), a target
  max(ℓ, ℓ')); the Hidden-Power presence and type posterior stay consistent with the rewritten typed channels.
* ITEM — a forward hook on ``item_belief_head``: on a SEEN slot whose item the observation does not state, the item
  posterior becomes (1 − p)·softmax + p·one-hot(target) (consumers: the op's P(Choice Band) / P(Quick Claw)).
* SPREAD — a forward hook on ``spread_belief``: on a SEEN slot the believed stats become (1 − p)·believed + p·true
  stats, and the head's token reinjection is recomputed from them, so every reader of the believed spread sees it.
  NOT covered: the nature posterior the op marginalises under ``--spread-belief-nature-marginalize``.

TARGETS (computed inside the forward from its own ctx, so a successor row gets its own): a slot is SEEN when its
species is revealed; it maps to the TRUE opponent mon of that species num (a forme shares its base num). ``truth`` =
the true unrevealed moves / the true item / the true stats; ``placebo`` = 4 − r moves (or one item) drawn ∝ the
network's OWN base belief excluding the truth and the pinned channels (deterministic per decision and slot); ``map``
= the network's own top 4 − r moves (or its top item): an honest sharpening. An ``unseen`` move lever applies on an X5
HYPOTHESIS slot whose hypothesised species is one of the opponent's true never-seen species.

p = 0 / ``off`` is BIT-IDENTICAL to the unhooked forward (the hooks pass the tensors through). Use::

    inj = Injector(model.policy.features_extractor)
    with inj.installed():
        inj.set("TRUE@0.95", decision_id, TrueTeam.of(opponent_packed_team))
        out = model.policy(...)                    # every forward in the block is injected
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

import numpy as np

from agents.info_levers import mixture as MX
from agents.info_levers.consumers import INJECT
from agents.info_levers.spec import InfoReveal

TEAM = 6
HP_BARE = 237
TYPED_LO, TYPED_HI = 355, 370
TYPED = tuple(range(TYPED_LO, TYPED_HI + 1))
CLIP = 1e-6
SET_SIZE = 4
#: the spread belief's stat order (``SpreadBelief``: the 5 derived non-HP stats at L100)
SPREAD_STATS = ("atk", "def", "spa", "spd", "spe")


def is_hp(n: int) -> bool:
    return n == HP_BARE or TYPED_LO <= n <= TYPED_HI


@dataclass(frozen=True)
class TrueMon:
    moves: FrozenSet[int]               # move nums, a Hidden Power typed by its IVs
    item: int                           # item num (0 = no item)
    stats: Tuple[int, ...]              # SPREAD_STATS at L100, exact


@dataclass(frozen=True)
class TrueTeam:
    """The opponent's TRUE team as the network's belief channels index it: species num → :class:`TrueMon`."""

    mons: Dict[int, TrueMon]

    @staticmethod
    def of(packed: str) -> "TrueTeam":
        from agents.gen3_data import items, moves, natures, priors, species
        from agents.info_levers.packed import Spread, parse_set, toid, typed_moves

        mult = natures.multipliers()
        out: Dict[int, TrueMon] = {}
        for entry in packed.split("]"):
            if not entry.strip():
                continue
            f = parse_set(entry)
            sid = toid(f[1] or f[0])
            sd = species.get(sid)
            if sd is None:
                raise ValueError(f"unknown species {sid!r}")
            nums = set()
            for mid in typed_moves(f):
                md = moves.get(mid)
                if md is None:
                    raise ValueError(f"unknown move {mid!r}")
                nums.add(int(md.num))
            it = items.get(toid(f[2])) if f[2] else None
            sp = Spread.of(f)
            nat = mult.get(sp.nature[:1].upper() + sp.nature[1:].lower(), {}) if sp.nature else {}
            base = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), sd.base_stats)) if isinstance(
                sd.base_stats, (list, tuple)) else dict(sd.base_stats)
            stats = tuple(priors.gen3_stat(int(base[s]), int(sp.evs[1 + k]), float(nat.get(s, 1.0)), int(sp.ivs[1 + k]))
                          for k, s in enumerate(SPREAD_STATS))
            if int(sd.num) in out:
                raise ValueError(f"two team members share species num {sd.num}")
            out[int(sd.num)] = TrueMon(frozenset(nums), int(it.num) if it is not None else 0, stats)
        return TrueTeam(out)


@dataclass
class _Call:
    spec: InfoReveal
    did: str
    team: TrueTeam
    salt: str
    stats: dict = field(default_factory=dict)       # the last forward's counts (targeted slots, moves, items, ...)


def placebo_seed(salt: str, did: str, slot: int, what: str = "") -> int:
    return int(hashlib.sha256(f"{salt}{did}:{slot}{what}".encode()).hexdigest()[:16], 16)


def _pins(rev: List[int]) -> Tuple[set, bool]:
    pins = {int(x) for x in rev if int(x) > 0}
    hp_rev = any(is_hp(x) for x in pins)
    pins.add(HP_BARE)
    if hp_rev:
        pins.update(TYPED)
    return pins, hp_rev


class Injector:
    """Installs the hooks on ONE extractor instance (module docstring). ``set(spec, did, team)`` arms them for the
    following forwards; ``set(None, …)`` / spec ``off`` leaves every tensor untouched."""

    def __init__(self, ext, salt: str = "info_levers_placebo_v1:"):
        hb = getattr(ext, "hypothesis_builder", None)
        if hb is None:
            raise RuntimeError("no X5 hypothesis builder on this extractor (the move candidate set is undefined)")
        if getattr(ext, "move_belief", None) is None or getattr(ext, "hp_type_belief_head", None) is None:
            raise RuntimeError("this extractor has no move belief / typed HP head")
        self.ext = ext
        self.salt = salt
        self.legal = (hb.move_legal & hb.move_valid.unsqueeze(0)).cpu().numpy()      # [S, M] bool
        self.call: Optional[_Call] = None
        self.calls = 0
        self._box: dict = {}

    def set(self, spec, did: str = "", team: Optional[TrueTeam] = None) -> None:
        if spec is None:
            self.call = None
            return
        s = INJECT.validate(spec)
        if s.is_off:
            self.call = None
            return
        if team is None:
            raise ValueError("an injected spec needs the opponent's TRUE team (TrueTeam.of(packed))")
        if s.lever("item", "seen") is not None and getattr(self.ext, "item_belief_head", None) is None:
            raise RuntimeError("an item lever on a network with no item belief head")
        if s.lever("spread", "seen") is not None and getattr(self.ext, "spread_belief", None) is None:
            raise RuntimeError("a spread lever on a network with no spread belief")
        self.call = _Call(s, did, team, self.salt)

    @property
    def stats(self) -> dict:
        return self.call.stats if self.call is not None else {}

    # ---------------------------------------------------------------- install
    @contextmanager
    def installed(self) -> Iterator["Injector"]:
        ext = self.ext
        if any(k in ext.__dict__ for k in ("_typed_hp_posterior", "_apply_move_belief", "_spread_hp_damage")):
            raise RuntimeError("the extractor instance is already hooked")
        orig_apply, orig_typed, orig_spread = ext._apply_move_belief, ext._typed_hp_posterior, ext._spread_hp_damage

        def apply_wrap(opp_tokens, ctx, hctx=None, hs=None):
            self._box = {"ctx": ctx, "hs": hs}
            try:
                return orig_apply(opp_tokens, ctx, hctx, hs)
            finally:
                self._box = {}

        def typed_wrap(opp_tokens, sctx, raw):
            out = orig_typed(opp_tokens, sctx, raw)
            self.calls += 1
            if self.call is None or self.call.spec.lever("moves", "seen") is None and \
                    self.call.spec.lever("moves", "unseen") is None:
                return out
            if "ctx" not in self._box:
                raise RuntimeError("_typed_hp_posterior called outside _apply_move_belief")
            return self._rewrite_moves(out, self._box["ctx"], self._box["hs"])

        def spread_wrap(opp_tokens, ctx, hctx=None, hs=None, fm=None, x5r=None):
            self._box_s = {"ctx": ctx}
            try:
                return orig_spread(opp_tokens, ctx, hctx, hs, fm, x5r)
            finally:
                self._box_s = {}

        handles = []
        if getattr(ext, "item_belief_head", None) is not None:
            handles.append(ext.item_belief_head.register_forward_hook(self._item_hook))
        if getattr(ext, "spread_belief", None) is not None:
            handles.append(ext.spread_belief.register_forward_hook(self._spread_hook))
        ext._apply_move_belief = apply_wrap
        ext._typed_hp_posterior = typed_wrap
        ext._spread_hp_damage = spread_wrap
        self._box_s: dict = {}
        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            for k in ("_apply_move_belief", "_typed_hp_posterior", "_spread_hp_damage"):
                del ext.__dict__[k]
            self.call = None

    # ---------------------------------------------------------------- the MOVES rewrite
    def move_targets(self, ctx, hs, base_logits) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """(target [B,6,M] bool, per-slot mass [B,6] (0 = untouched), pinned [B,6,M] bool, hp revealed [B,6], stats)."""
        import torch

        c = self.call
        assert c is not None
        B, T, M = base_logits.shape
        species = ctx.species_ids[:, TEAM:].long().cpu().numpy()
        believed = ctx.opp_believed_mask.bool().cpu().numpy()
        rev = ctx.all_move_ids[:, TEAM:, :].long().cpu().numpy()
        if hs is not None:
            hyp = hs.slot_is_hypothesis.bool().cpu().numpy()
            hsp = hs.slot_species.long().cpu().numpy()
        else:
            hyp, hsp = np.zeros((B, T), bool), np.zeros((B, T), np.int64)
        sig = torch.sigmoid(base_logits.detach().double()).cpu().numpy()
        lv_s, lv_u = c.spec.lever("moves", "seen"), c.spec.lever("moves", "unseen")
        tgt = np.zeros((B, T, M), bool)
        mass = np.zeros((B, T))
        pinned = np.zeros((B, T, M), bool)
        hp_rev_a = np.zeros((B, T), bool)
        st = {"rows": B, "seen_slots": 0, "seen_unmatched": 0, "slots": 0, "moves": 0, "short": 0,
              "hyp_slots": 0, "hyp_matched": 0, "targets": []}
        true_nums = set(c.team.mons)
        for b in range(B):
            seen_nums = {int(species[b, s]) for s in range(T) if not believed[b, s] and species[b, s] > 0}
            never = true_nums - seen_nums
            row = []
            for s in range(T):
                pins, hp_rev = _pins(rev[b, s].tolist())
                hp_rev_a[b, s] = hp_rev
                pinned[b, s, [n for n in pins if n < M]] = True
                M_s: List[int] = []
                num = int(species[b, s])
                p = 0.0
                if not believed[b, s] and num > 0 and lv_s is not None:
                    st["seen_slots"] += 1
                    tm = c.team.mons.get(num)
                    if tm is None:
                        st["seen_unmatched"] += 1
                    else:
                        U = sorted(n for n in tm.moves if n not in pins)
                        k = max(0, SET_SIZE - len({int(x) for x in rev[b, s] if int(x) > 0}))
                        if U:
                            cand = self.legal[min(num, self.legal.shape[0] - 1)].copy()
                            cand[list(pins)] = False
                            w = {int(n): round(float(sig[b, s, n]), 9) for n in np.flatnonzero(cand)}
                            rng = np.random.default_rng(placebo_seed(c.salt, c.did, s))
                            M_s = [int(x) for x in MX.set_target(lv_s.source, U, w, k, rng, exclude=pins,
                                                                 at_most_one=is_hp)] if lv_s.source != "truth" else U
                            if lv_s.source == "map":    # the network's own top k: ties by move num
                                M_s = sorted(w, key=lambda n: (-w[n], n))
                                M_s = _first_k(M_s, k)
                            if len(M_s) < k and lv_s.source != "truth":
                                st["short"] += 1
                            p = lv_s.mass
                elif believed[b, s] and hyp[b, s] and lv_u is not None:
                    st["hyp_slots"] += 1
                    h = int(hsp[b, s])
                    if h in never:
                        st["hyp_matched"] += 1
                        M_s = sorted(n for n in c.team.mons[h].moves if n not in pins)
                        p = lv_u.mass
                if M_s and p > 0:
                    tgt[b, s, M_s] = True
                    mass[b, s] = p
                    st["slots"] += 1
                    st["moves"] += len(M_s)
                row.append(M_s)
            st["targets"].append(row)
        return tgt, mass, pinned, hp_rev_a, st

    def _rewrite_moves(self, out, ctx, hs):
        import torch

        logits, presence, hp_post, hp_logits = out
        tgt_np, mass_np, pin_np, hprev_np, st = self.move_targets(ctx, hs, logits)
        assert self.call is not None
        self.call.stats["moves"] = st
        on_np = mass_np > 0
        if not on_np.any():
            return logits, presence, hp_post, hp_logits
        dev, dt = logits.device, logits.dtype
        tgt = torch.from_numpy(tgt_np).to(dev)
        p = torch.from_numpy(mass_np).to(dev).unsqueeze(-1)
        sel = torch.from_numpy(on_np).to(dev).unsqueeze(-1) & ~torch.from_numpy(pin_np).to(dev)
        l64 = logits.double()
        q = torch.sigmoid(l64)
        qn = (1.0 - p) * q + p * tgt.double()
        lc = torch.logit(qn.clamp(CLIP, 1.0 - CLIP))
        new = torch.where(tgt, torch.maximum(l64, lc), torch.minimum(l64, lc))
        logits2 = torch.where(sel, new.to(dt), logits)
        hp_sel = torch.from_numpy(on_np & ~hprev_np).to(dev)
        if presence is not None and hp_post is not None and bool(hp_sel.any()):
            qt = torch.sigmoid(logits2[..., TYPED_LO:TYPED_HI + 1].double())
            pres2 = qt.sum(-1)
            post2 = qt / pres2.clamp_min(1e-300).unsqueeze(-1)
            presence = torch.where(hp_sel, pres2.to(presence.dtype), presence)
            hp_post = torch.where(hp_sel.unsqueeze(-1), post2.to(hp_post.dtype), hp_post)
        return logits2, presence, hp_post, hp_logits

    # ---------------------------------------------------------------- the ITEM and SPREAD hooks
    def _seen_slots(self) -> Optional[Tuple[np.ndarray, np.ndarray, object]]:
        ctx = self._box_s.get("ctx")
        if ctx is None or self.call is None:
            return None
        species = ctx.species_ids[:, TEAM:].long().cpu().numpy()
        believed = ctx.opp_believed_mask.bool().cpu().numpy()
        return species, ~believed & (species > 0), ctx

    def _item_hook(self, module, inputs, output):
        import torch

        c = self.call
        if c is None or c.spec.lever("item", "seen") is None:
            return output
        got = self._seen_slots()
        if got is None:
            raise RuntimeError("item_belief_head ran outside _spread_hp_damage")
        species, seen, ctx = got
        lv = c.spec.lever("item", "seen")
        assert lv is not None
        logits, post = output
        known = ctx.item_ids[:, TEAM:].long().cpu().numpy() > 0           # the observation states the item
        B, T, N = post.shape
        tgt = np.full((B, T), -1, dtype=np.int64)
        base = post.detach().double().cpu().numpy()
        st = {"slots": 0, "unmatched": 0, "known": 0}
        for b in range(B):
            for s in range(T):
                if not seen[b, s]:
                    continue
                tm = c.team.mons.get(int(species[b, s]))
                if tm is None:
                    st["unmatched"] += 1
                    continue
                if known[b, s]:
                    st["known"] += 1
                    continue
                w = {n: round(float(base[b, s, n]), 9) for n in range(N)}
                rng = np.random.default_rng(placebo_seed(c.salt, c.did, s, ":item"))
                t = MX.categorical_target(lv.source, tm.item, w, rng)
                if t is None or not 0 <= int(t) < N:
                    continue
                tgt[b, s] = int(t)
                st["slots"] += 1
        c.stats["item"] = st
        if (tgt < 0).all():
            return output
        sel = torch.from_numpy(tgt >= 0).to(post.device)
        onehot = torch.nn.functional.one_hot(torch.from_numpy(np.maximum(tgt, 0)).to(post.device), N).double()
        q = (1.0 - lv.mass) * post.double() + lv.mass * onehot
        lg = torch.log(q.clamp_min(1e-300))
        logits2 = torch.where(sel.unsqueeze(-1), lg.to(logits.dtype), logits)
        return logits2, torch.softmax(logits2, dim=-1)

    def _spread_hook(self, module, inputs, output):
        import torch

        c = self.call
        if c is None or c.spec.lever("spread", "seen") is None:
            return output
        got = self._seen_slots()
        if got is None:
            raise RuntimeError("spread_belief ran outside _spread_hp_damage")
        species, seen, _ctx = got
        lv = c.spec.lever("spread", "seen")
        assert lv is not None
        opp_tokens, apply_mask, opp_species = inputs
        enriched, believed, nat_logits, ev = output
        B, T, K = believed.shape
        true = np.zeros((B, T, K))
        sel = np.zeros((B, T), bool)
        st = {"slots": 0, "unmatched": 0}
        for b in range(B):
            for s in range(T):
                if seen[b, s]:
                    tm = c.team.mons.get(int(species[b, s]))
                    if tm is None:
                        st["unmatched"] += 1
                        continue
                    true[b, s] = tm.stats
                    sel[b, s] = True
                    st["slots"] += 1
        c.stats["spread"] = st
        if not sel.any():
            return output
        selt = torch.from_numpy(sel).to(believed.device).unsqueeze(-1)
        b2 = (1.0 - lv.mass) * believed.double() + lv.mass * torch.from_numpy(true).to(believed.device)
        believed2 = torch.where(selt, b2.to(believed.dtype), believed)
        prior = module.spread_prior[opp_species]
        mean, std = prior[..., 0], prior[..., 1]
        delta = (believed2 - mean) / (std.clamp(min=1.0) if module.nature else std)
        pub = (lambda t: t.detach()) if getattr(module, "publish_detach", False) else (lambda t: t)
        enriched2 = module.norm(opp_tokens + apply_mask.unsqueeze(-1) * module.reinject(pub(delta)))
        mask = (selt & apply_mask.bool().unsqueeze(-1))
        return torch.where(mask, enriched2, enriched), believed2, nat_logits, ev


def _first_k(order: List[int], k: int) -> List[int]:
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
