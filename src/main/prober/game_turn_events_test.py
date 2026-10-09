"""`engine.turn_events` — the turn story `/game` reads: every state-bearing protocol line becomes a
typed event, and the board after each turn is a fold over the same lines (pure; no torch, no IO)."""

from __future__ import annotations

from main.prober.engine.turn_events import fold_turns

# A real gen-3 shape (from a Rust-eval core trace, 2026-10-08), trimmed: leads, an immune hit, a
# resisted hit with recoil + Leftovers + sand, a Spikes layer, a faint.
LOG = (
    "|init|battle", "|player|p1|rhone||", "|player|p2|rhtwo||", "|teamsize|p1|6", "|teamsize|p2|6",
    "|start",
    "|switch|p1a: Jynx|Jynx, F|271/271",
    "|switch|p2a: Swampert|Swampert, F|100/100",
    "|turn|1",
    "|",
    "|switch|p1a: Gengar|Gengar, M|263/263",
    "|move|p2a: Swampert|Earthquake|p1a: Gengar",
    "|-immune|p1a: Gengar",
    "|upkeep",
    "|turn|2",
    "|",
    "|move|p1a: Gengar|Spikes|p2a: Swampert",
    "|-sidestart|p2: rhtwo|Spikes",
    "|move|p2a: Swampert|Double-Edge|p1a: Gengar",
    "|-resisted|p1a: Gengar",
    "|-damage|p1a: Gengar|200/263",
    "|-damage|p2a: Swampert|92/100|[from] Recoil|[of] p1a: Gengar",
    "|-weather|Sandstorm|[upkeep]",
    "|-heal|p2a: Swampert|98/100|[from] item: Leftovers",
    "|upkeep",
    "|turn|3",
    "|",
    "|move|p2a: Swampert|Earthquake|p1a: Gengar",
    "|-damage|p1a: Gengar|0 fnt",
    "|faint|p1a: Gengar",
    "|win|rhtwo",
)


def test_every_turn_gets_its_events_in_order_with_the_side_from_the_trainees_seat():
    turns = fold_turns(LOG, trainee_side="p1", our_team=["Jynx", "Gengar"])
    assert [t["turn"] for t in turns] == [0, 1, 2, 3]
    assert [e["kind"] for e in turns[1]["events"]] == ["switch", "move", "immune"]
    t2 = [(e["kind"], e["side"]) for e in turns[2]["events"]]
    assert t2 == [("move", "we"), ("hazard", "opp"), ("move", "opp"), ("resisted", "we"),
                  ("damage", "we"), ("damage", "opp"), ("weather", None), ("heal", "opp")]
    # the details the one-line-per-action timeline drops: recoil and Leftovers carry their SOURCE
    rec = turns[2]["events"][5]
    assert rec["source"] == "recoil" and rec["pct"] == -8.0 and rec["hp_after"] == 92.0
    assert turns[2]["events"][7]["source"] == "Leftovers"
    # our HP is exact points: 263 -> 200 is 76.0% of max
    assert turns[2]["events"][4]["hp_after"] == round(100 * 200 / 263, 1)


def test_the_seat_decides_the_side_never_the_line_order():
    """The same log read from p2's seat swaps every we/opp — a trainee on p2 is not a mirror bug."""
    turns = fold_turns(LOG, trainee_side="p2")
    assert turns[2]["events"][0]["side"] == "opp" and turns[2]["events"][1]["side"] == "we"


def test_the_board_after_each_turn_is_the_fold():
    turns = fold_turns(LOG, trainee_side="p1", our_team=["Jynx", "Gengar"])
    b2 = turns[2]["board"]
    assert b2["opp"]["conditions"] == {"spikes": 1}
    assert b2["weather"] == "Sandstorm"
    opp = {m["name"]: m for m in b2["opp"]["mons"]}
    assert opp["Swampert"]["hp_pct"] == 98.0 and opp["Swampert"]["active"]
    assert b2["opp"]["team_size"] == 6
    b3 = turns[3]["board"]
    ours = {m["name"]: m for m in b3["we"]["mons"]}
    assert ours["Gengar"]["fainted"] and ours["Gengar"]["hp_pct"] == 0.0
    assert ours["Jynx"]["hp_pct"] == 100.0 and not ours["Jynx"]["active"]
    assert turns[3]["events"][-1]["kind"] == "end"
    # a move is remembered on the mon that used it
    assert "Earthquake" in opp["Swampert"]["moves"] and "Double-Edge" in opp["Swampert"]["moves"]


def test_an_unknown_line_is_kept_as_other_never_dropped():
    turns = fold_turns(("|turn|1", "|-somethingnew|p2a: Swampert|x"), trainee_side="p1")
    (ev,) = turns[1]["events"]
    assert ev["kind"] == "other" and ev["side"] == "opp" and "somethingnew" in ev["text"]


def test_boosts_reset_on_switch_and_clamp_at_six():
    log = ("|turn|1", "|switch|p2a: Snorlax|Snorlax|100/100", "|-boost|p2a: Snorlax|atk|2",
           "|-boost|p2a: Snorlax|atk|6", "|turn|2", "|switch|p2a: Skarmory|Skarmory|100/100")
    turns = fold_turns(log, trainee_side="p1")
    assert turns[1]["board"]["opp"]["boosts"] == {"atk": 6}
    assert turns[2]["board"]["opp"]["boosts"] == {}
