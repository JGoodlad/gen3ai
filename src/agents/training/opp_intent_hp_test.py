"""The opponent's Hidden Power must reach α as a TYPED num (gen3_typed_hp_intent_label_v1).

THE DEFECT THIS PINS. α's seats come from `refine_candidates`, which masks the bare typeless 237 out
of the candidate axis (`HP_CAND_MASK[237] = 0.0` — it is the PRESENCE channel at BP 0, and the op
reasons only over the typed nums 355-370). The env label, meanwhile, is whatever poke-env saw the
opponent click, and gen 3 NEVER reveals an opponent's HP type, so it arrives as bare `hiddenpower`
→ 237. `match_seats_to_move_num` compares raw ints, so a bare-237 label matches NO seat, EVER, and
every opponent Hidden Power is silently dropped from α's supervision.

Why it is worth a dedicated test rather than a comment:
  * **43.2% of `data/teams/` mons carry a Hidden Power** (2041/4722) — this is not a rounding error
    in `alpha_mask_rate`, it is a large systematic deletion.
  * It deletes precisely the SURPRISE-coverage move α exists to anticipate.
  * It is INVISIBLE in every metric: the rows just vanish into the mask, so `alpha_acc` looks
    healthy while a coin-flip fraction of the phenomenon is unmeasured.
  * `designs/ai_v9/design_opponent_intent.md` asserted the OPPOSITE as settled fact ("under
    canonical-id matching a seat holding bare `hiddenpower` MATCHES a used HP Ice"), and that
    sentence was load-bearing for retiring the soft-target proposal. Neither half is true in code.

The fix types the label from the PRIVILEGED source — agent2's own team keeps the type suffix on
`Move._id` — never from `HPTypeBelief`'s argmax, which would supervise α against the model's own
guess and bake this belief's ~9% error into the target. A label must be ground truth or absent.
"""
import torch

from agents.model.damage_tables import HIDDEN_POWER_NUM, _hp_typed_nums
from agents.model.opp_intent import INTENT_IGNORE, match_seats_to_move_num

_KIND_MOVE = 0
_N_SEATS = 6


def _seats_with_typed_hp(typed_num: int) -> torch.Tensor:
    """A realistic top-6: a typed Hidden Power plus five ordinary moves."""
    return torch.tensor([[typed_num, 94, 89, 157, 191, 57]])


def test_bare_237_label_matches_no_seat_and_is_masked():
    """THE BUG, stated as an executable fact: a bare-237 label supervises NOTHING."""
    seats = _seats_with_typed_hp(360)                      # HP Ice sits at seat 0
    out = match_seats_to_move_num(seats, torch.tensor([HIDDEN_POWER_NUM]),
                                  torch.tensor([_KIND_MOVE]), _N_SEATS)
    assert int(out[0]) == INTENT_IGNORE, (
        "a bare typeless 237 resolved to a seat — either HP_CAND_MASK stopped excluding 237 from "
        "the candidate axis, or the matcher stopped comparing raw nums. Re-derive the label path."
    )


def test_typed_label_resolves_to_its_seat():
    """The fix's premise: type the label and it supervises the seat the op actually built."""
    for slot, typed in ((0, 360), (3, 366)):
        seats = torch.tensor([[94, 89, 157, 191, 57, 1]])
        seats[0, slot] = typed
        out = match_seats_to_move_num(seats, torch.tensor([typed]),
                                      torch.tensor([_KIND_MOVE]), _N_SEATS)
        assert int(out[0]) == slot, f"typed {typed} at seat {slot} did not resolve"


def test_every_typed_hp_num_is_distinct_from_the_bare_presence_num():
    """The property the whole scheme rests on — if a typed num ever collided with 237 the
    exclusion would silently start dropping a real candidate instead of the presence channel."""
    typed = _hp_typed_nums()
    assert len(typed) == 16 and len(set(typed)) == 16
    assert HIDDEN_POWER_NUM not in typed


def test_env_resolver_types_a_bare_hidden_power_from_the_privileged_team():
    """The env-side fix: `_intent_move_num_resolver` maps a bare `hiddenpower` to the TRUE typed num
    using the attacker's own moveset (agent2's team), and leaves everything else untouched."""
    from agents.training.gen3_env import Gen3Env

    class _Move:
        def __init__(self, mid): self.id = mid

    class _Mon:
        def __init__(self, species, move_ids):
            self.species = species
            self.moves = {m: _Move(m) for m in move_ids}

    class _Battle2:
        team = {"a": _Mon("Zapdos", ["thunderbolt", "hiddenpowerice"]),
                "b": _Mon("Skarmory", ["spikes", "drillpeck"])}

    class _Delta:
        opp_prev_active = "Zapdos"

    env = Gen3Env.__new__(Gen3Env)                      # no __init__: this is a pure-method test
    env.battle2 = _Battle2()
    env._move_num = {"hiddenpower": HIDDEN_POWER_NUM, "thunderbolt": 85, "spikes": 191}

    resolve = env._intent_move_num_resolver(_Delta())
    # DERIVE the expected num the same way production does — never hardcode the type's ordinal.
    # (An earlier draft of this test guessed index 2 for ICE and failed against a working fix.)
    from agents.observation.belief_labels import hp_type_idx_from_move_id
    hp_ice = _hp_typed_nums()[hp_type_idx_from_move_id("hiddenpowerice")]
    assert resolve("hiddenpower") == hp_ice, (
        "a bare hiddenpower was not typed from the attacker's privileged moveset — alpha will "
        "silently drop every opponent Hidden Power again (43% of pool mons carry one)."
    )
    assert resolve("thunderbolt") == 85, "a non-HP move must pass through untouched"


def test_resolver_leaves_the_label_bare_when_the_true_type_is_unrecoverable():
    """An unresolvable case must stay MASKED, never become a guessed seat. A wrong seat is worse
    than a missing one: it trains alpha toward a move they did not pick."""
    from agents.training.gen3_env import Gen3Env

    class _Battle2:
        team = {}                                        # attacker not found

    class _Delta:
        opp_prev_active = "Zapdos"

    env = Gen3Env.__new__(Gen3Env)
    env.battle2 = _Battle2()
    env._move_num = {"hiddenpower": HIDDEN_POWER_NUM}
    assert env._intent_move_num_resolver(_Delta())("hiddenpower") == HIDDEN_POWER_NUM
