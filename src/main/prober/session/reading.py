"""Reading a run and a battle — the MODEL-FREE orientation surface.

`run_summary` / `battles` / `decision_table` / `battle_overview` / `battle_turns`. None of these
loads a checkpoint, which is why they work on every archived run forever (see the model-free vs
model-loading tier table in `main/prober/CLAUDE.md`).
"""

from __future__ import annotations

from main.prober.discovery import list_checkpoints
from main.prober.engine import (_npz_win_prob, _timeline_for, build_board, parse_pct,
    protocol_for_turn, summary_flags, timeline_entry_text)
from main.prober.session.serialize import (_active_str, _choice_dict, _chosen_prob,
    _opp_intent_dict, _r, _short_id, _side_dict)


class _ReadingMixin:
    # -- run orientation -----------------------------------------------------

    def run_summary(self) -> dict:
        """Orient on a run: steps, per-step model identity, opponents with win/loss
        tallies, persisted checkpoints, and γ. The natural first call. Model-free."""
        steps = []
        for sg in self.tree.steps:
            man = self.tree.manifest_for(sg.step)
            opps, wl = [], {"win": 0, "loss": 0}
            for og in sg.opponents:
                w = sum(1 for b in og.battles if b.outcome == "win")
                l = sum(1 for b in og.battles if b.outcome == "loss")
                wl["win"] += w
                wl["loss"] += l
                opps.append({"name": og.name, "win": w, "loss": l, "battles": len(og.battles)})
            steps.append({
                "step": sg.step,
                "identity": None if not man else {
                    "git_hash": man.get("git_hash"),
                    "arch_signature": man.get("arch_signature"),
                    "snapshot_available": self._snapshot_available(sg.step, man),
                },
                "opponents": opps,
                "totals": wl,
            })
        totals = {
            "win": sum(s["totals"]["win"] for s in steps),
            "loss": sum(s["totals"]["loss"] for s in steps),
            "battles": len(self.tree.all_battles()),
        }
        return {
            "run_dir": self.run_dir, "gamma": self._gamma, "n_steps": len(steps),
            "checkpoints": [{"step": s, "path": p} for s, p in list_checkpoints(self.run_dir)],
            "steps": steps, "totals": totals,
        }

    # -- discovery -----------------------------------------------------------

    def battles(self, *, outcome: "str | None" = None, opponent: "str | None" = None,
                step: "int | None" = None) -> "list[dict]":
        """List battles, optionally filtered by outcome / opponent / step."""
        out = []
        for b in self.tree.all_battles():
            if outcome and b.outcome != outcome:
                continue
            if opponent and b.opponent != opponent:
                continue
            if step is not None and b.step != step:
                continue
            out.append({
                "id": b.summary_path, "short_id": _short_id(b), "step": b.step,
                "opponent": b.opponent, "outcome": b.outcome, "index": b.index,
                "has_npz": b.npz_path is not None,
            })
        return out

    # -- consolidated forensic exports (forensics.py) -----------------------

    def decision_table(self, *, steps=None, opponents=None, outcomes=None,
                       categories=None, max_battles=None) -> "list[dict]":
        """MODEL-FREE per-decision forensic table over the matching battles (one row each:
        move-category, our/opp species+HP, policy ``conf``, ``reward``, critic ``dV``,
        incoming-KO ``pko`` belief, faint flags, outcome). The single source for the
        softmax/dV/belief-decode plumbing every behavioural-hypothesis check reuses."""
        from main.prober import forensics
        return forensics.build_decision_table(
            self, steps=steps, opponents=opponents, outcomes=outcomes,
            categories=categories, max_battles=max_battles)

    # -- model-free battle digest -------------------------------------------

    def battle_overview(self, battle_id: str) -> dict:
        """A MODEL-FREE per-decision digest: chosen, top prob, recorded V(s), ΔV to
        the next decision, TD residual (critic surprise), per-step reward + events,
        and flags — plus a `notable` summary and how a deep analyze would resolve the
        model. No checkpoint loaded."""
        b = self._battle(battle_id)
        summary = self._summary(b)
        values = self._values(b)
        invs = summary["invocations"]
        rows = []
        for i, inv in enumerate(invs):
            chosen = inv.get("chosen", "")
            reward = (inv.get("outcome") or {}).get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            v = self._v(values, i)
            v_next = self._v(values, i + 1)
            board = build_board(inv)
            rows.append({
                "inv": i, "turn": inv.get("turn"), "phase": inv.get("phase"),
                "chosen": chosen,
                "our_active": _active_str(board.ours), "opp_active": _active_str(board.opp),
                "top_prob": _chosen_prob(inv),
                "value": v,
                "delta_v": (v_next - v) if (v is not None and v_next is not None) else None,
                "td_residual": self._td(rtotal, v, v_next),
                "reward_total": rtotal,
                "events": (inv.get("outcome") or {}).get("events") or [],
                "flags": list(summary_flags(inv)),
            })
        return {
            "id": b.summary_path, "short_id": _short_id(b), "meta": summary.get("meta", {}),
            "gamma": self._gamma,
            "model_resolution": _choice_dict(self._resolve(b)),
            "notable": self._notable(rows),
            "invocations": rows,
        }

    def battle_turns(self, battle_id: str) -> dict:
        """A MODEL-FREE **turn-by-turn replay** of one battle: the decisions GROUPED BY GAME TURN,
        each carrying the board it was made on, what was chosen, the ordered battle-log timeline of
        what then happened, and the critic's read (V, ΔV, TD δ) + reward.

        `battle_overview` answers "which decision cratered?" as a flat ranked table. This answers
        "how did the game GO?" — you read it top to bottom like a replay. The two are deliberately
        different shapes over the same trace; neither derives from the other.

        Turn grouping matters because a turn is not a decision: a faint makes the same turn carry a
        `move_selection` AND the `forced_switch` that follows it, and a reader tracking a game needs
        those under one heading. `n_decisions` != sum of turns only when a trace records a decision
        with no turn number (older recorders), which is bucketed under its own `turn: None`.

        No checkpoint is loaded, so this is fast enough to open any battle in the run interactively.
        """
        b = self._battle(battle_id)
        summary = self._summary(b)
        # ONE npz read for the whole replay: `_npz` re-opens the file on every call, and this view
        # now wants three of its arrays (values, win_probs, value_dist).
        npz = self._npz(b)
        values = npz.get("values")
        invs = summary.get("invocations", [])
        meta = summary.get("meta", {})
        # "Did it KNOW?" folded once for the battle, then joined onto each decision by INDEX.
        awareness = self._awareness(b, invs, npz=npz)
        aware_by_dec = self._awareness_by_decision(awareness)

        boards = [build_board(inv) for inv in invs]
        protocol = self._protocol_lines(b)      # parsed ONCE, sliced per turn below
        turns: "list[dict]" = []
        by_turn: dict = {}
        for i, inv in enumerate(invs):
            outcome = inv.get("outcome") or {}
            reward = outcome.get("reward")
            rtotal = reward.get("total") if isinstance(reward, dict) else reward
            v, v_next = self._v(values, i), self._v(values, i + 1)
            wp, wp_next = _npz_win_prob(npz, i), _npz_win_prob(npz, i + 1)
            next_board = boards[i + 1] if i + 1 < len(boards) else None
            # This turn's raw protocol slice rides along: when the TurnDelta recorded no
            # `move_order`, the engine reads the real execution order off the log's `|move|` lines
            # rather than declaring it unknown. Sliced once here and reused for the row's
            # `protocol` field below, so the parse is not repeated per decision.
            proto_slice = (tuple(protocol_for_turn(protocol, int(inv.get("turn") or 0)))
                           if protocol else ())
            entries = _timeline_for(inv, next_board, outcome, protocol=proto_slice)
            row = {
                "inv": i,
                "turn": inv.get("turn"),
                "phase": inv.get("phase"),
                "chosen": inv.get("chosen", ""),
                "top_prob": _chosen_prob(inv),
                "our": _side_dict(boards[i].ours),
                "opp": _side_dict(boards[i].opp),
                # What the model expected THEM to do (the v67 α/β heads) — the one per-decision fact
                # that separates "played around the Fire Blast" from "never saw it coming", which
                # the board, the timeline and the critic's numbers all read identically. `text` is
                # the engine's rendering of the same view, so no surface re-derives the sentence
                # (the `timeline` rule). None on any run that did not train the heads.
                "opp_intent": _opp_intent_dict(inv),
                # Each entry keeps its structured fields AND the engine's rendering of them, so a
                # surface prints `text` rather than re-deriving the sentence (see the module rule in
                # `web/app.py`: a view reshapes nothing).
                "timeline": [dict(e, text=timeline_entry_text(e)) for e in entries],
                # When both sides moved and `move_order` wasn't recorded, top-to-bottom is NOT the
                # real sequence — the reader must be told rather than shown a guess.
                "order_certain": all(e.get("order_certain", True) for e in entries
                                     if e.get("kind") == "move"),
                "value": _r(v),
                "delta_v": _r(v_next - v) if (v is not None and v_next is not None) else None,
                "td_residual": _r(self._td(rtotal, v, v_next)),
                # The win-prob head's calibrated P(win) beside V — the interpretable [0,1]
                # complement to a shaped, discounted return whose zero is NOT "even". `analyze`
                # has carried this since the head shipped; the replay is where a reader actually
                # follows the game, so it belongs on the row too. None on a no-win-prob run.
                "win_prob": _r(wp),
                "delta_win_prob": _r(wp_next - wp) if (wp is not None and wp_next is not None)
                                  else None,
                # The distributional head's per-decision read of the SAME question the critic
                # answers with one number: P(return < 0), and the catastrophic-band mass. A scalar
                # V cannot show tail mass piling up while the mean still reads healthy — that is
                # exactly the stall signature — so these ride beside V rather than replacing it.
                # `knew` is true from the sustained-onset decision onward (see `_awareness_by_decision`).
                **(aware_by_dec.get(i)
                   or {"p_loss": None, "p_win": None, "p_tail": None, "knew": False}),
                "reward_total": _r(rtotal),
                # `RewardBreakdown.to_dict()` is `{"total": …}` plus ONE STRING PER GROUP
                # ("base": "pbrs_material=-0.44"), not a nested components dict — so the components
                # are every other key, which is exactly how the TUI reads it too.
                "reward_components": ({k: v for k, v in reward.items() if k != "total"}
                                      if isinstance(reward, dict) else {}),
                "events": outcome.get("events") or [],
                "flags": list(summary_flags(inv)),
                # The FULL recorded action distribution — what else the policy considered, and how
                # close it came to picking it. Already action-index ordered by the recorder (see the
                # "do NOT re-sort move labels" gotcha in this package's CLAUDE.md), so it is passed
                # through in that order and never re-sorted.
                "actions": [{"label": label, "prob": parse_pct(a.get("prob", "0%")),
                             "valid": bool(a.get("valid")), "chosen": label == inv.get("chosen")}
                            for label, a in (inv.get("actions") or {}).items()],
                # The raw Showdown lines for this decision's TURN — the exact mechanics the summary
                # collapses (per-hit damage, a miss, an immunity, the switch-in). Empty when the
                # trace has no `*_replay.html` sibling.
                "protocol": list(proto_slice),
            }
            key = row["turn"]
            if key not in by_turn:
                by_turn[key] = {"turn": key, "decisions": []}
                turns.append(by_turn[key])
            by_turn[key]["decisions"].append(row)

        flat = [d for t in turns for d in t["decisions"]]
        return {
            "id": b.summary_path, "short_id": _short_id(b),
            "step": b.step, "opponent": b.opponent, "outcome": b.outcome,
            "meta": meta, "gamma": self._gamma,
            "n_turns": len(turns), "n_decisions": len(invs),
            # The same `notable` block `battle_overview` returns (faints, switches, the biggest value
            # drops) — a 249-turn replay needs entry points, and re-deriving them per surface is how
            # two views of one battle start disagreeing about where it went wrong. Its entries name
            # DECISIONS, so `decision_turns[inv]` is the turn each one happened on (null when the
            # recorder didn't number it) — a lookup, so a surface can turn "worst drop at decision
            # 37" into a link to a turn without doing arithmetic of its own.
            "notable": self._notable(flat),
            "decision_turns": [d["turn"] for d in flat],
            # "Did it KNOW?" — the dist head's battle-level verdict (awareness.py): sustained
            # P(loss)>0.5 onset (`knew_by_turn`/`lead_time`), `blind_loss`, and the stall
            # signature (`mean_tail_divergence`: tail mass while the MEAN still read positive).
            # Model-free (support from model_config.json, popart denorm fit from the trace's own
            # values). None when the run has no dist head or the trace predates recording. The
            # per-decision `p_loss`/`p_tail`/`knew` on each row above are this same fold, joined
            # by decision index — folded ONCE, above.
            "awareness": awareness,
            "turns": turns,
        }
