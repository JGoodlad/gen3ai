"""Exactness gate for the content-keyed belief-block memo (`gen3_belief_block_memo_v1`).

`encode_block` measured **60.0% of `process_turn_reward`** (cProfile, 2026-08-23) — the largest
single item in the per-decision CPU budget — and it now answers from a content-keyed cache. That
cache feeds Φ_belief, which is THE OBJECTIVE: an under-key does not crash, it silently trains a
different policy. So the contract this file pins is **identical inputs ⇒ identical outputs, cached
or not**, and it pins it three independent ways:

  1. **A STRUCTURAL key-coverage gate.** `attacker_state_key` must cover every board read
     `_attacker_threat` performs. Enumerated MECHANICALLY, by AST-walking the function for every
     attribute reached from ``live`` (including through ``getattr``), so a future read that the key
     does not carry fails here rather than in a training run. This is the "enumerate the writers,
     not the reads you noticed" discipline of the `live_view` epoch memo, applied to a pure
     function: the doors board state can enter through are the thing being counted.
  2. **A FIELD gate on the output.** `AttackerThreat`'s field list is pinned, because the coverage
     proof is a claim about that constructor's arguments. Adding a field to it is exactly the
     change that can widen the input set without touching `_attacker_threat`'s reads.
  3. **BEHAVIOURAL equality** over a matrix of boards: memo-on vs a fresh encoder, bit-for-bit,
     including the paths that must BYPASS the memo (an `hp_tracker`) and the states that must
     produce the all-zero block.

Plus the two identities the speed comes from: `dmax_crit == 2·dmax` when no screen is up (an
algebraic identity, checked against the screened branch that cannot use it), and
`compute_mon_row` ≡ the row `compute_team_block` used to write inline.

The real-battle counterpart is `agents/training/reward_skip_parity_fuzz_test.py`, which drives the
same comparison through thousands of live decisions.
"""
import ast
import copy
import inspect
import textwrap
from dataclasses import fields as dc_fields
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agents.enums import Status
from agents.observation import incoming_damage as inc
from agents.observation import incoming_damage_encoder as enc


# --------------------------------------------------------------------------------------------
# 1. STRUCTURAL — the key covers every board read
# --------------------------------------------------------------------------------------------

def _live_rooted_attribute_reads(func) -> set:
    """Every attribute name `func` reads off the `live` argument, transitively.

    Walks the AST tracking which locals are rooted at ``live`` (via attribute chains, ``getattr``,
    and ``or``-defaults), and records the attribute NAME of every read on one of them. Method calls
    (``boosts.get(...)``, ``name.upper()``) are excluded — the receiver's attribute set is what
    matters, not the dict/str protocol used on the value once it is out.
    """
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    fn = tree.body[0]
    rooted = {fn.args.args[0].arg}          # the `live` parameter
    reads: set = set()
    call_funcs: set = set()

    def is_rooted(node) -> bool:
        if isinstance(node, ast.Name):
            return node.id in rooted
        if isinstance(node, ast.Attribute):
            return is_rooted(node.value)
        if isinstance(node, ast.Subscript):
            return is_rooted(node.value)
        if isinstance(node, ast.BoolOp):     # `opp.boosts or {}`
            return any(is_rooted(v) for v in node.values)
        if isinstance(node, ast.Call):       # `getattr(live, "weather", None)`
            return (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                    and bool(node.args) and is_rooted(node.args[0]))
        return False

    # Two passes: bind rooted locals first (source order is enough — Python has no forward refs
    # inside a function body for this purpose), then collect.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_rooted(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    rooted.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    for el in tgt.elts:
                        if isinstance(el, ast.Name):
                            rooted.add(el.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            call_funcs.add(id(node.func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in call_funcs and is_rooted(node.value):
            reads.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and is_rooted(node.args[0]) and isinstance(node.args[1], ast.Constant)):
            reads.add(node.args[1].value)
    return reads


def test_attacker_key_covers_every_board_read_of_attacker_threat():
    """The mechanical coverage proof: no read the key does not carry.

    Fails on a NEW read as well as a renamed one. If this fires, the fix is to put the new input in
    `attacker_state_key` AND in `_ATTACKER_KEY_LIVE_READS` — never just the latter."""
    reads = _live_rooted_attribute_reads(enc._attacker_threat)
    assert reads, "the AST walk found no reads at all — the walker is broken, not the function"
    uncovered = reads - enc._ATTACKER_KEY_LIVE_READS
    assert not uncovered, (
        f"`_attacker_threat` reads {sorted(uncovered)} off the board, and "
        f"`attacker_state_key` does not carry them. An UNDER-KEY is a silently wrong reward: "
        f"add the input to the key, then to `_ATTACKER_KEY_LIVE_READS`.")
    # And the reverse, so the declared set cannot rot into a superset that proves nothing.
    stale = enc._ATTACKER_KEY_LIVE_READS - reads
    assert not stale, f"`_ATTACKER_KEY_LIVE_READS` declares {sorted(stale)}, no longer read"


def test_the_coverage_gate_can_actually_fail():
    """A gate that cannot fail is worth nothing. Run the walker over a function shaped like
    `_attacker_threat` but reading one extra thing through each channel it must cover — a plain
    chain, a rebound local, a `getattr`, and an `or`-default — and demand it sees all four."""
    def _probe(live, hp_tracker=None):
        opp = live.opp.active
        boosts = opp.boosts or {}
        lw = getattr(live, "weather", None)
        sneaky = opp.item_the_key_forgot                      # plain chain off a rebound local
        also = getattr(lw, "turns_active", None)              # through getattr
        via_default = boosts.get("atk", 0)
        return live.ours.trapped, sneaky, also, via_default   # plain chain off the parameter

    reads = _live_rooted_attribute_reads(_probe)
    for expected in ("item_the_key_forgot", "turns_active", "trapped", "boosts"):
        assert expected in reads, f"the walker missed {expected}: {sorted(reads)}"
    assert "get" not in reads, "method calls on a value are not board reads"


def test_attacker_threat_field_list_is_pinned():
    """The coverage proof is a claim about THESE constructor arguments (see the proof comment in
    `incoming_damage_encoder`). A new field is the change most likely to widen the input set."""
    assert [f.name for f in dc_fields(inc.AttackerThreat)] == [
        "types", "atk_tail", "atk_mean", "spa_tail", "spa_mean", "spe_dist", "boost_spe",
        "para", "burn", "phys", "spec", "our_reflect", "our_light_screen", "weather",
        "recovery_rate", "cures_status", "recovery_known",
    ], ("AttackerThreat gained/lost a field — re-derive the key-coverage proof in "
        "`incoming_damage_encoder.py` before updating this list")


def test_defender_is_its_own_key():
    """The row cache keys on the `Defender` VALUE, so its completeness needs no proof — but that
    only holds while `Defender` is a frozen dataclass of hashable primitives, and while `_defender`
    is rebuilt fresh from the board on every call (it is: nothing caches it)."""
    assert inc.Defender.__dataclass_params__.frozen
    d = _defender(hp=100, hp_max=200)
    assert hash(d) == hash(_defender(hp=100, hp_max=200))
    assert d == _defender(hp=100, hp_max=200)


# --------------------------------------------------------------------------------------------
# Board fakes — LiveView-shaped, primitives only (the encoder reads nothing else)
# --------------------------------------------------------------------------------------------

def _defender(hp=150, hp_max=200, **kw):
    base = dict(def_stat=200, spd_stat=180, hp_remaining=hp, hp_max=hp_max, spe=170,
                type1=inc.PokemonType.WATER, type2=None, ability=None, status=None)
    base.update(kw)
    return inc.Defender(**base)


def _mon(species="suicune", *, active=False, fainted=False, hp=150, hp_max=200, status=None,
         boosts=None, types=("water",), sub=False, ability=None, spe=170):
    return NS(species=species, active=active, fainted=fainted, status=status,
              types=types, ability=ability,
              stats={"def": 200, "spd": 180, "spe": spe},
              current_hp=hp, max_hp=hp_max, hp_fraction=hp / hp_max if hp_max else 0.0,
              boosts=boosts or {}, has_volatile=lambda v, _s=sub: _s and v == "substitute",
              move_ids=())


def _opp(species="tyranitar", *, moves=("rockslide", "earthquake"), status=None, boosts=None,
         types=("rock", "dark")):
    return NS(species=species, move_ids=tuple(moves), status=status, boosts=boosts or {},
              types=types, active=True, fainted=False)


_DEFAULT = object()   # distinguishes "not passed" from an explicit `opp=None` (no opp active)


def _live(opp=_DEFAULT, *, mons=None, side=None, weather=None):
    return NS(opp=NS(active=_opp() if opp is _DEFAULT else opp),
              ours=NS(mons=mons if mons is not None else [_mon(active=True), _mon("skarmory")],
                      side_conditions=side or {}),
              weather=NS(weather=weather))


# The board matrix the behavioural tests sweep. Each entry is a DIFFERENT value of some key
# component, so a memo that ignored one would be caught by the sweep as well as by the AST gate.
def _boards():
    yield "plain", _live()
    yield "no-opp", _live(opp=None)
    yield "no-species", _live(opp=_opp(species=""))
    yield "boosted", _live(_opp(boosts={"atk": 2, "spa": 1, "spe": 1}))
    yield "burned", _live(_opp(status="brn"))
    yield "para", _live(_opp(status="par"))
    yield "more-moves", _live(_opp(moves=("rockslide", "earthquake", "hiddenpower", "crunch")))
    yield "move-order", _live(_opp(moves=("earthquake", "rockslide")))
    yield "reflect", _live(side={"reflect": 1})
    yield "lightscreen", _live(side={"light_screen": 1})
    yield "both-screens", _live(side={"reflect": 1, "light_screen": 1})
    yield "sand", _live(weather="sandstorm")
    yield "rain", _live(weather="raindance")
    yield "sun", _live(weather="sunnyday")
    yield "other-species", _live(_opp("gengar", moves=("shadowball",), types=("ghost", "poison")))
    yield "sub", _live(mons=[_mon(active=True, sub=True), _mon("skarmory")])
    yield "hurt", _live(mons=[_mon(active=True, hp=17), _mon("skarmory", hp=3)])
    yield "fainted", _live(mons=[_mon(active=True), _mon("skarmory", fainted=True)])
    yield "statused", _live(mons=[_mon(active=True, status="par"), _mon("skarmory", status="brn")])
    yield "def-boosts", _live(mons=[_mon(active=True, boosts={"def": 2, "spd": -1, "spe": 1})])
    yield "empty-team", _live(mons=[])


# --------------------------------------------------------------------------------------------
# 2. BEHAVIOURAL — the memo returns the identical block
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,live", list(_boards()), ids=[n for n, _ in _boards()])
def test_memo_block_is_bit_identical_to_a_fresh_encode(name, live):
    fresh = enc.encode_block(live)
    memo = enc.IncomingBeliefMemo()
    first = enc.encode_block(live, memo=memo)      # cold
    second = enc.encode_block(live, memo=memo)     # warm — the path under test
    assert np.array_equal(fresh, first), f"{name}: cold memo differs from fresh"
    assert np.array_equal(fresh, second), f"{name}: WARM memo differs from fresh — under-key"
    assert first.dtype == fresh.dtype


def test_one_memo_across_every_board_stays_bit_identical():
    """The realistic shape: ONE memo, many boards in sequence — so a row minted under board A can
    be wrongly served for board B if the key is incomplete. Swept twice, so every board is seen
    once cold and once warm with a fully-populated cache."""
    memo = enc.IncomingBeliefMemo()
    boards = list(_boards())
    for _pass in range(2):
        for name, live in boards:
            assert np.array_equal(enc.encode_block(live), enc.encode_block(live, memo=memo)), name
    st = memo.stats()
    assert st["attacker_hits"] > 0, "the sweep never re-used an attacker — this proved nothing"


def test_no_two_distinct_boards_share_a_key():
    """Every board in the matrix that differs in a key component must produce a DIFFERENT key —
    otherwise the identity test above passes for the wrong reason (a memo that never hits)."""
    keys = {}
    for name, live in _boards():
        k = enc.attacker_state_key(live)
        if k is None:
            continue
        keys.setdefault(k, []).append(name)
    # Boards that differ only on OUR side legitimately share an attacker key — that is the point of
    # keying rows separately. Group and assert the sharers are exactly the our-side variations.
    shared = {tuple(v) for v in keys.values() if len(v) > 1}
    for group in shared:
        assert set(group) <= {"plain", "sub", "hurt", "fainted", "statused", "def-boosts",
                              "empty-team"}, f"unexpected key collision: {group}"


def test_hp_tracker_bypasses_the_memo():
    """The tracker's per-episode narrowing is NOT in the key, so it must never be served from it.
    Priming the memo with the tracker-free answer and then asking WITH a tracker must return the
    tracker's answer, not the cached one."""
    from agents.training.hidden_power_tracker import HiddenPowerTracker
    live = _live(_opp("zapdos", moves=("hiddenpower",), types=("electric", "flying")))
    memo = enc.IncomingBeliefMemo()
    enc.encode_block(live, memo=memo)                       # prime with the prior-typed HP
    tr = HiddenPowerTracker(_priors={"zapdos": {"ice": 1.0}})
    with_tracker = enc.encode_block(live, hp_tracker=tr, memo=memo)
    assert np.array_equal(with_tracker, enc.encode_block(live, hp_tracker=tr))


def test_clearing_at_any_point_changes_nothing():
    """The memo is CONTENT-keyed, not history-carried: dropping it mid-stream must be invisible.
    This is the property `_pbrs_step` refuses to give up by telescoping Φ, stated as a test."""
    memo = enc.IncomingBeliefMemo()
    boards = list(_boards())
    with_clears = []
    for i, (_n, live) in enumerate(boards):
        if i % 3 == 0:
            memo.clear()
        with_clears.append(enc.encode_block(live, memo=memo))
    for (name, live), got in zip(boards, with_clears):
        assert np.array_equal(enc.encode_block(live), got), name


def test_memo_survives_deepcopy_and_serves_the_same_values():
    """The materializer/reward_tracker paths deepcopy state. A content-keyed memo is the SAFE shape
    for that: the copy holds immutable values under content keys, so a hit in either copy is
    correct by construction — there is no arm-to-arm hazard to guard against."""
    memo = enc.IncomingBeliefMemo()
    for _n, live in _boards():
        enc.encode_block(live, memo=memo)
    clone = copy.deepcopy(memo)
    for name, live in _boards():
        assert np.array_equal(enc.encode_block(live), enc.encode_block(live, memo=clone)), name


def test_memo_is_bounded_and_clears_wholesale():
    memo = enc.IncomingBeliefMemo()
    memo.MAX_ATTACKERS = 4
    memo.MAX_ROWS = 8
    for hp in range(1, 200):
        live = _live(_opp(boosts={"atk": hp % 7}),
                     mons=[_mon(active=True, hp=hp), _mon("skarmory", hp=hp)])
        assert np.array_equal(enc.encode_block(live), enc.encode_block(live, memo=memo))
    st = memo.stats()
    assert st["attackers"] <= memo.MAX_ATTACKERS and st["rows"] <= memo.MAX_ROWS
    assert st["clears"] > 0, "the bound never fired — the test did not exercise eviction"


# --------------------------------------------------------------------------------------------
# 3. The two arithmetic identities the speed comes from
# --------------------------------------------------------------------------------------------

def _threat(**kw):
    base = dict(types=(inc.PokemonType.ROCK,), atk_tail=350.0, atk_mean=300.0,
                spa_tail=250.0, spa_mean=220.0, spe_dist=((200, 1.0),),
                phys=(inc.Candidate(inc.PokemonType.ROCK, 75, 1.0),),
                spec=(inc.Candidate(inc.PokemonType.FIRE, 95, 1.0),))
    base.update(kw)
    return inc.AttackerThreat(**base)


@pytest.mark.parametrize("screen", [False, True])
@pytest.mark.parametrize("hp", [1, 40, 90, 140, 199])
def test_crit_branch_is_exactly_twice_the_screenless_damage(screen, hp):
    """`dmax_crit = 2 · gen3_damage_max(..., screen=False, ...)`. With no screen up that inner call
    has the same arguments as the modal one, so the reuse is an algebraic identity. Pinned by
    computing the crit line the long way and demanding the SAME float."""
    d = _defender(hp=hp)
    a = _threat(our_reflect=screen, our_light_screen=screen)
    nc, cr, _exp, _rev = inc._channel_threat(a.phys, d, a.atk_tail, a.atk_mean, a=a,
                                             screen=screen, is_phys=True)
    c = a.phys[0]
    eff = inc.effective_multiplier_by_types(c.move_type, d.type1, d.type2, d.ability, d.status)
    defense = max(1, int(d.def_stat * inc.boost_mult(d.boost_def)))
    dmax = inc.gen3_damage_max(c.power, int(a.atk_tail), defense, stab=True, type_eff=eff,
                               screen=screen, weather=1.0, burned=False)
    dmax_free = inc.gen3_damage_max(c.power, int(a.atk_tail), defense, stab=True, type_eff=eff,
                                    screen=False, weather=1.0, burned=False)
    if not screen:
        assert dmax_free == dmax, "the identity the fast path relies on"
    expect_nc = inc.p_ko(dmax, hp)
    expect_cr = ((1.0 - inc._CRIT_P) * expect_nc
                 + inc._CRIT_P * inc.p_ko(2 * dmax_free, hp))
    assert nc == expect_nc
    assert cr == expect_cr


def test_compute_mon_row_matches_the_block_it_assembles():
    """`compute_mon_row` was factored OUT of `compute_team_block`; the block must still be exactly
    its rows, including the float32 store."""
    a = _threat()
    ds = [_defender(hp=140), _defender(hp=12, has_sub=True), None, _defender(hp=200, boost_spe=2)]
    block = inc.compute_team_block(ds, a, 6)
    for i, d in enumerate(ds):
        row = np.zeros(inc.PER_MON, dtype=np.float32) if d is None else np.array(
            inc.compute_mon_row(d, a), dtype=np.float32)
        assert np.array_equal(block[i * inc.PER_MON:(i + 1) * inc.PER_MON], row), i
    assert np.array_equal(block[6 * inc.PER_MON:],
                          np.array([a.recovery_rate, a.cures_status, a.recovery_known],
                                   dtype=np.float32))


def test_row_cache_needs_both_a_map_and_a_key():
    """A half-supplied memo must fall through to the fresh path, never cache under `None`."""
    a = _threat()
    ds = [_defender()]
    ref = inc.compute_team_block(ds, a, 6)
    cache: dict = {}
    assert np.array_equal(inc.compute_team_block(ds, a, 6, row_cache=cache), ref)
    assert not cache, "a cache with no attacker_key must stay untouched"
    assert np.array_equal(inc.compute_team_block(ds, a, 6, attacker_key=("k",)), ref)
    assert np.array_equal(
        inc.compute_team_block(ds, a, 6, row_cache=cache, attacker_key=("k",)), ref)
    assert cache


def test_paralysed_defender_row_still_keys_on_status():
    """A regression shape for the row key: `status` reaches the row through BOTH the effectiveness
    primitive and P(outspeed), and it lives on the `Defender`, so equal-HP mons that differ only by
    status must not share a row."""
    a = _threat(spe_dist=((100, 1.0),))   # slower than our 170, but faster than a paralysed 42
    cache: dict = {}
    healthy = inc.compute_team_block([_defender()], a, 6, row_cache=cache, attacker_key=("k",))
    para = inc.compute_team_block([_defender(status=Status.PAR)], a, 6,
                                  row_cache=cache, attacker_key=("k",))
    assert not np.array_equal(healthy, para)
    assert para[inc.IDX_OUTSPEED] < healthy[inc.IDX_OUTSPEED]
