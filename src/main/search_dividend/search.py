"""The depth-1 search itself: branch the current decision, score the successors, pick an action.

**Shape of the search (FIXED depth 1, registered).** For each determinized world *w*, each of OUR
legal actions *a*, each α-weighted OPPONENT candidate *c* and each CRN dice draw *r*, roll the
turn forward once and read the critic on the successor. The action taken is

    argmax_a  Σ_w p(w) · Σ_c α(c) · (1/R) · Σ_r  V(s'(w, a, c, r))

Three properties of that expression are the whole experiment and none is incidental:

* **the opponent is MARGINALIZED, the dice are AVERAGED, and OUR action is MAXIMIZED** — the
  three-axis variance measurement put the behavior-weighted split at OPP 59.7% >> DICE 26.5% >
  OUR 10.0%, so the axis that buys the most per unit of compute is the one α is spent on;
* **the dice are COMMON across arms** (CRN): draw *r* uses ONE seed shared by every (a, c) pair in
  the world, so a difference between two actions is not a difference between two dice streams.
  ⚠️ CRN here shares the dice STREAM, not the roll→event MAPPING — two arms that consume a
  different number of draws desynchronize after the first divergence, which is why a one-seed
  sweep over-reads the OUR×OPP interaction by roughly 2× and why R>1 is on the budget ladder at
  all rather than pinned to 1;
* **a fallback is COUNTED, never silent.** Every path that fails to produce a scored arm returns
  the policy's own action tagged with a reason from :data:`FALLBACK_REASONS`. A search that
  quietly degraded to the policy would report a null dividend that is indistinguishable from a
  real one — the single most important thing this module can get wrong.

**Where the arms come from.** ``SearchSession`` (the warm clone-and-branch search server) opens a
root at the current turn from a LIVE-synthesized reconstruction record and expands every arm in
one round trip. The successor OBS is materialized through :func:`materialize_branches`, which
replays the shared battle prefix ONCE for the whole arm set rather than once per arm — the prefix
is the measured majority of a counterfactual arm's cost and grows linearly in the turn number, so
without prefix sharing the realized widths would collapse in exactly the late-game positions the
probe cares about.

**The one-sided / omniscient wall holds.** ``expand_many`` returns per-side chunks AND an
omniscient ``outcome``. Only the chunks reach the encoder — that is the same wall the re-roll path
keeps. The ORACLE arm's privilege is the true opponent TEAM in the record it searches, nothing
more; it never reads a referee-view board into an observation.

**Threading.** ``materialize_branches`` refuses to run on ``POKE_LOOP`` (it drives a replay player
through that loop and would deadlock on its own result), and ``choose_move`` runs there. So the
whole search executes in a worker thread and the player's ``choose_move`` is an ``async def`` that
awaits it — poke-env awaits an awaitable choice (``player.py``: ``if isinstance(choice, Awaitable)``),
which frees POKE_LOOP for exactly as long as the search holds it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.search_dividend import determinize as dz
from main.search_dividend.alpha import (AlphaPublication, build_candidates,
                                        legal_choices_from_request)
from main.search_dividend.budget import (CostModel, Deadline, RealizedWidths, WidthCaps,
                                         WidthPlan, allocate)

ARMS = ("base", "honest", "oracle")

# Terminal successors, on the WIN-PROB scale. On the value scale the same constants are used and
# the caveat is reported with the row: a shaped-return V and a ±1 terminal are not the same units,
# so `--score win_prob` is the default whenever the head exists.
TERMINAL_VALUE = {"win": 1.0, "loss": 0.0, "tie": 0.5}


class _NoWorld(RuntimeError):
    """The honest arm could not build a single pool-consistent world. Its own exception type so
    the caller reports ``no_world`` rather than the generic ``search_error`` — "the pool has no
    completion for this information set" and "something threw" are different findings."""


@dataclass
class SearchConfig:
    arm: str = "base"
    budget_s: float = 1.0
    caps: WidthCaps = field(default_factory=WidthCaps)
    score: str = "auto"                 # auto | value | win_prob
    search_impl: str = "node"
    honest_swap_moves: bool = False     # axis M — see determinize.swap_unused_moves
    seed: int = 0

    def resolved_caps(self) -> WidthCaps:
        """The ORACLE arm is ONE world by construction (the true state), and the BASE arm has no
        search at all. Encoding that here rather than at every call site means a budget can never
        be spent on a world axis that does not exist."""
        if self.arm == "oracle":
            return WidthCaps(m_opp=self.caps.m_opp, k_worlds=1, r_dice=self.caps.r_dice)
        if self.arm == "base":
            return WidthCaps(m_opp=0, k_worlds=0, r_dice=0)
        return self.caps


@dataclass
class DecisionResult:
    action: int
    fallback: Optional[str]
    widths: RealizedWidths
    diagnostics: dict = field(default_factory=dict)
    scores: Optional[dict] = None       # {action_index: aggregated score}
    policy_action: Optional[int] = None

    @property
    def changed(self) -> bool:
        return (self.policy_action is not None and self.fallback is None
                and int(self.action) != int(self.policy_action))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def batch_scores(model, obs: np.ndarray, masks: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """``(scores, mode_used)`` for a batch of successor observations.

    ``win_prob`` reads the head's stashed logit from the SAME forward that produced the values,
    so the two never come from different states. ``auto`` prefers it and falls back to the shaped
    value when the run trained no win-prob head."""
    import torch

    policy = model.policy
    dev = model.device
    with torch.no_grad():
        inp = {"observation": torch.as_tensor(obs).to(dev),
               "action_mask": torch.as_tensor(masks).to(dev)}
        values = policy.predict_values(inp).squeeze(-1).float().cpu().numpy()
        wp_logits = getattr(getattr(policy, "features_extractor", None),
                            "last_win_prob_logits", None)
        wp = (torch.sigmoid(wp_logits.float()).squeeze(-1).cpu().numpy()
              if wp_logits is not None else None)
    if mode == "value" or (wp is None and mode in ("auto", "win_prob")):
        return np.asarray(values, dtype=np.float64), "value"
    if wp is None:
        return np.asarray(values, dtype=np.float64), "value"
    return np.asarray(wp, dtype=np.float64).reshape(-1), "win_prob"


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


@dataclass
class _Arm:
    world: int
    action: int
    cand: int
    dice: int


class SearchEngine:
    """One engine per battle-playing player. Holds the WARM search session and the running cost
    model, so the ~0.6 s driver spawn is paid once per battery rather than once per decision."""

    def __init__(self, *, model, mappings, cfg: SearchConfig,
                 pool_packed: Optional[Sequence[str]] = None):
        self.model = model
        self.mappings = mappings
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self._session = None
        # An EWMA of the two measured costs. Seeded from the CostModel defaults and updated after
        # every decision, so a battery that starts fast and slows down (the prefix grows with the
        # turn) re-plans its widths instead of missing the deadline by a constant factor.
        self._cost = CostModel()
        self._pool_packed = pool_packed
        self._pool_mons: Optional[List[List[dz.MonSet]]] = None
        self._gender_tbl: Optional[Dict[str, str]] = None
        self._move_bank: Optional[Dict[str, list]] = None

    # -- lifecycle ----------------------------------------------------------

    def session(self):
        from utils.bridge.search_session import SearchSession

        if self._session is None:
            self._session = SearchSession(impl=self.cfg.search_impl)
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _pool(self) -> Tuple[List[List[dz.MonSet]], Dict[str, str], Dict[str, list]]:
        if self._pool_mons is None:
            packed, gender = dz.load_pool(self._pool_packed)
            self._pool_mons = [dz.split_team(p, gender) for p in packed]
            self._gender_tbl = gender
            self._move_bank = dz.species_move_bank(self._pool_mons)
        return self._pool_mons, self._gender_tbl, self._move_bank   # type: ignore[return-value]

    # -- the decision -------------------------------------------------------

    def choose(self, *, record, side: str, turn: int, our_history: Sequence[int],
               our_tokens: Dict[int, str], observed_our_lines: Sequence[str],
               pub: Optional[AlphaPublication], policy_action: int,
               opp_true_packed: Optional[str] = None) -> DecisionResult:
        """Run the depth-1 search for ONE decision and return the action to play.

        ``our_tokens`` maps each legal action INDEX to its sim choice string, built live from the
        real action mapper — the search never re-derives legality, it inherits it.
        ``opp_true_packed`` is the ORACLE arm's privilege and must be ``None`` on every other arm;
        :meth:`_worlds` raises rather than silently searching the truth.
        """
        caps = self.cfg.resolved_caps()
        widths = RealizedWidths(planned={}, n_our_actions=len(our_tokens))
        if self.cfg.arm == "base" or caps.k_worlds <= 0:
            widths.planned = WidthPlan(0, 0, 0).as_dict()
            return DecisionResult(policy_action, "no_search", widths,
                                  policy_action=policy_action)
        if not our_tokens:
            return DecisionResult(policy_action, "not_move_selection", widths,
                                  policy_action=policy_action)

        deadline = Deadline(self.cfg.budget_s)
        plan = allocate(self.cfg.budget_s, len(our_tokens), self._cost, caps)
        widths.planned = plan.as_dict()
        widths.dice = plan.r_dice
        widths.worlds_requested = plan.k_worlds
        try:
            res = self._run(record, side, turn, our_history, our_tokens, observed_our_lines,
                            pub, policy_action, opp_true_packed, plan, widths, deadline)
        except _NoWorld as e:
            widths.elapsed_s = round(deadline.elapsed(), 4)
            return DecisionResult(policy_action, "no_world", widths,
                                  diagnostics={"error": str(e)}, policy_action=policy_action)
        except Exception as e:                       # noqa: BLE001
            # A search failure must never cost the battle: the battery's whole point is a
            # per-arm win rate, and a crashed decision would silently drop a game.
            widths.elapsed_s = round(deadline.elapsed(), 4)
            return DecisionResult(policy_action, "search_error", widths,
                                  diagnostics={"error": f"{type(e).__name__}: {e}"},
                                  policy_action=policy_action)
        widths.elapsed_s = round(deadline.elapsed(), 4)
        self._update_cost(widths)
        return res

    # -- internals ----------------------------------------------------------

    def _run(self, record, side, turn, our_history, our_tokens, observed_our_lines, pub,
             policy_action, opp_true_packed, plan: WidthPlan, widths: RealizedWidths,
             deadline: Deadline) -> DecisionResult:
        other = "p2" if side == "p1" else "p1"
        ss = self.session()

        worlds = self._worlds(record, other, observed_our_lines, plan.k_worlds, opp_true_packed)
        widths.worlds_requested = len(worlds)

        # CRN: ONE seed per dice draw r, SHARED by every (action, candidate) arm — and across
        # worlds too, so two worlds disagree about the TEAM rather than about the dice.
        seeds = ["original"] + [self._crn_seed(turn, r) for r in range(1, plan.r_dice)]

        actions = sorted(our_tokens)
        scores: Dict[int, float] = {a: 0.0 for a in actions}
        weight_sum = 0.0
        arm_diag: List[dict] = []
        alpha_diag: List[dict] = []
        world_diag: List[dict] = []
        # ⚠️ ONE WORLD AT A TIME, opened AND consumed before the next. `open_root` CLEARS the
        # driver's node cache (that is what bounds a warm session's memory across battles), so
        # opening K roots up front invalidates every earlier one and `expand_many` then fails with
        # `unknown node nNN`. Found by the smoke: the oracle arm (K=1) was clean while the honest
        # arm lost 13 of 17 searched decisions to it.
        for wi, (wrec, wmeta) in enumerate(worlds):
            world_diag.append(wmeta)
            batch_cost = (self._cost.world_open_s
                          + self._cost.arm_s * len(actions) * plan.m_opp * len(seeds))
            if wi and not deadline.fits(batch_cost):
                widths.deadline_truncated = True
                break
            try:
                root = ss.open_root(turn, record=wrec)
            except Exception as e:                   # noqa: BLE001
                widths.worlds_open_failed += 1
                wmeta["gate"] = f"open_failed: {type(e).__name__}: {e}"
                continue
            prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
            if not dz.prefix_matches(observed_our_lines, prefix, turn=turn):
                # THE GATE. A world whose replay does not reproduce the protocol we actually saw
                # is a different battle; keeping it would answer about a position we were never
                # in. Dropped WITH A COUNTER (the precedent ran 535/535 and 615/615 clean, so a
                # non-zero count here is news, not noise).
                widths.worlds_gate_failed += 1
                wmeta["gate"] = "prefix_mismatch"
                continue
            widths.worlds_gated_ok += 1
            wmeta["gate"] = "ok"

            # The legal surface differs between worlds (a determinized bench holds different
            # mons), so the marginalization set is rebuilt per world — reusing one would branch
            # on switch targets that do not exist.
            legal = legal_choices_from_request((root.requests or {}).get(other))
            cands, diag = build_candidates(legal, pub, m_opp=plan.m_opp)
            alpha_diag.append(diag)
            if not cands:
                continue
            widths.opp_candidates = max(widths.opp_candidates, len(cands))
            got = self._score_world({"root": root, "meta": wmeta, "cands": cands},
                                    side, other, record, our_history, our_tokens, actions,
                                    seeds, widths)
            if got is None:
                continue
            per_action, adiag = got
            arm_diag.append(adiag)
            for a in actions:
                scores[a] += per_action.get(a, 0.0)
            weight_sum += 1.0
        if weight_sum <= 0:
            if widths.worlds_gated_ok == 0:
                # A broken DRIVER and a wrong WORLD are different diagnoses; report the one that
                # actually happened. Ties go to the driver — if it died, nothing else is trustworthy.
                reason = "root_failed" if widths.worlds_open_failed else "prefix_gate_failed"
            elif not any(d.get("n_legal") for d in alpha_diag):
                reason = "no_candidates"
            else:
                reason = "no_scored_arm"
            return DecisionResult(policy_action, reason, widths,
                                  diagnostics={"worlds": world_diag, "alpha": alpha_diag},
                                  policy_action=policy_action)
        for a in actions:
            scores[a] /= weight_sum

        # Tie-break TOWARD THE POLICY. A search that breaks an exact tie away from the policy's
        # own pick would report a `changed` decision it has no evidence for, and `change_rate` is
        # one of the two headline numbers here.
        best = max(actions, key=lambda a: (scores[a], a == policy_action))
        return DecisionResult(
            best, None, widths,
            diagnostics={"worlds": world_diag, "alpha": alpha_diag, "arms": arm_diag,
                         "score_mode": arm_diag[0].get("score_mode") if arm_diag else None},
            scores={int(a): round(float(v), 5) for a, v in scores.items()},
            policy_action=policy_action)

    def _score_world(self, w: dict, side: str, other: str, record, our_history, our_tokens,
                     actions: Sequence[int], seeds: Sequence[str],
                     widths: RealizedWidths) -> Optional[Tuple[Dict[int, float], dict]]:
        """Expand + score every arm of ONE world. Returns ``({action: E[score]}, diagnostics)``."""
        from agents.training.obs_materializer import Branch, materialize_branches

        root, cands = w["root"], w["cands"]
        arms: List[_Arm] = []
        payload = []
        for ai, a in enumerate(actions):
            for ci, c in enumerate(cands):
                for ri, sd in enumerate(seeds):
                    label = len(arms)
                    arms.append(_Arm(world=0, action=a, cand=ci, dice=ri))
                    payload.append({"node_id": root.node_id,
                                    f"{side}_action": our_tokens[a],
                                    f"{other}_action": c.token,
                                    "seed": sd, "label": label})
        expanded = self.session().expand_many(payload)
        widths.arms_expanded += len(expanded)

        prefix = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
        branches: List[Branch] = []
        branch_of: List[int] = []
        terminal: Dict[int, float] = {}
        username = record.username(side)
        for e in expanded:
            li = int(e.label)
            if e.ended:
                lab = _terminal_label(e.outcome, username)
                terminal[li] = TERMINAL_VALUE[lab]
                widths.arms_terminal += 1
                continue
            suffix = e.p1_chunks if side == "p1" else e.p2_chunks
            branches.append(Branch(chunks=suffix, actions=[arms[li].action], label=li))
            branch_of.append(li)

        vals: Dict[int, float] = dict(terminal)
        score_mode = self.cfg.score
        if branches:
            dec_i = len(our_history) + 1
            traces = materialize_branches(
                prefix, branches, username=username, packed_team=record.packed_team(side),
                side=side, prefix_actions=list(our_history), battle_format=record.format_id,
                battle_tag=record.battle_tag, mappings=self.mappings,
                map_actions_at=dec_i, stop_after_decision=dec_i, encode_only_at={dec_i})
            obs_rows, mask_rows, keys = [], [], []
            for li, mt in zip(branch_of, traces):
                if len(mt.decisions) <= dec_i:
                    continue                 # the successor never produced a request (rare)
                d = mt.decisions[dec_i]
                obs_rows.append(d.obs)
                mask_rows.append(d.mask)
                keys.append(li)
            if obs_rows:
                sc, score_mode = batch_scores(self.model, np.stack(obs_rows),
                                              np.stack(mask_rows), self.cfg.score)
                for li, v in zip(keys, sc):
                    vals[li] = float(v)
        widths.arms_scored += len(vals)
        if not vals:
            return None

        per_action: Dict[int, float] = {}
        for a in actions:
            acc, wsum = 0.0, 0.0
            for li, arm in enumerate(arms):
                if arm.action != a or li not in vals:
                    continue
                acc += cands[arm.cand].weight * vals[li]
                wsum += cands[arm.cand].weight
            per_action[a] = (acc / wsum) if wsum > 0 else 0.0
        return per_action, {"score_mode": score_mode, "n_scored": len(vals),
                            "n_terminal": len(terminal), "tier": w["meta"].get("tier")}

    def _worlds(self, record, opp_side: str, observed_our_lines: Sequence[str], k: int,
                opp_true_packed: Optional[str]) -> List[Tuple[object, dict]]:
        """The records to search. ORACLE = the truth, K=1. HONEST = K pool-consistent
        determinizations of the never-revealed slots."""
        if self.cfg.arm == "oracle":
            if opp_true_packed is None:
                raise ValueError("the oracle arm needs the opponent's true packed team")
            return [(dz.record_with_team(record, opp_side, opp_true_packed, "-oracle"),
                     {"kind": "oracle", "tier": 0})]
        if opp_true_packed is not None:
            # A guard, not politeness: an honest arm handed the truth would produce an oracle
            # result under an honest label, which is the one confusion this experiment cannot
            # survive.
            raise ValueError("the honest arm must not be given the opponent's true team")

        pool, gender, bank = self._pool()
        base_packed = record.packed_team(opp_side)
        base = dz.split_team(base_packed, gender)
        revealed = dz.revealed_species(observed_our_lines, opp_side)
        if self.cfg.honest_swap_moves:
            used = dz.used_moves_by_species(observed_our_lines, opp_side)
            base, _n = dz.swap_unused_moves(base, revealed, used, bank, self.rng)
        dets, stats = dz.build_determinizations(base, revealed, pool, gender, k=k, rng=self.rng)
        out: List[Tuple[object, dict]] = []
        for i, d in enumerate(dets):
            out.append((dz.record_with_team(record, opp_side, d["packed"], f"-w{i}"),
                        {"kind": "honest", "tier": d["tier"],
                         "hidden": d["hidden_species"], "n_hidden": stats["n_hidden"]}))
        if not out:
            # ⚠️ The record's ``>player`` team IS the truth (the live builder read it off the
            # opponent object), so falling back to it is only legitimate when there is nothing
            # hidden to determinize — then that team IS our information set. With hidden slots
            # still open, the same fallback would be an ORACLE search wearing the honest label,
            # which is the one confusion this experiment cannot survive. Decline instead, and let
            # the caller count it.
            if stats["n_hidden"] == 0:
                out.append((dz.record_with_team(record, opp_side, base_packed, "-w0"),
                            {"kind": "honest_complete", "tier": 0, "n_hidden": 0}))
            else:
                raise _NoWorld(
                    f"no pool-consistent completion for {stats['n_hidden']} hidden slot(s) "
                    f"(tier1 donors={stats['tier1_donors']}, tier2={stats['tier2_donors']})")
        return out

    def _crn_seed(self, turn: int, r: int) -> str:
        return f"sodium,{self.rng.getrandbits(128):032x}"

    def _update_cost(self, widths: RealizedWidths) -> None:
        """EWMA the two measured costs so the next decision's plan reflects THIS battle's turn
        depth. A constant cost model would over-plan late game (the prefix replay grows linearly
        in the turn) and under-plan early, which shows up as a budget that is never spent."""
        if widths.arms_scored <= 0 or widths.elapsed_s <= 0:
            return
        opens = max(1, widths.worlds_gated_ok + widths.worlds_gate_failed
                    + widths.worlds_open_failed)
        arm_share = max(0.0, widths.elapsed_s - opens * self._cost.world_open_s)
        arm_s = arm_share / max(1, widths.arms_expanded)
        a = 0.3
        self._cost = CostModel(
            world_open_s=self._cost.world_open_s,
            arm_s=max(1e-4, (1 - a) * self._cost.arm_s + a * arm_s))


def _terminal_label(outcome: dict, username: str) -> str:
    w = (outcome or {}).get("winner")
    return "win" if w == username else ("tie" if not w else "loss")
