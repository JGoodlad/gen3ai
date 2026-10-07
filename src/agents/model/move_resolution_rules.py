"""gen3_move_resolution_v1 — the gen-3 RULES the move-resolution family computes from, torch-free.

`designs/endstate/design_arch_audit.md` F11 + §9 (the owner's 2026-10-06 ruling: FACTS — what will actually happen
if I press this — are kept and consolidated; JUDGMENTS — opinions of the right play — are dropped). This module is
the single declaration of every rule the family applies, each verified against the vendored Showdown
(`deps/pokemon-showdown`, the gen-3 mod resolved through its inheritance chain gen3 → gen4 → … → base, and the
merged dex loaded from `dist/` with ``Dex.mod('gen3')``). Paths below are relative to `deps/pokemon-showdown/`.

**Why a code table and not `data/`.** The move flags this needs (``protect``, ``bypasssub``, ``sound``,
``reflectable``, ``defrost``, ``failencore``) are not in `data/pokemon/gen3_moves.json` (``flags`` is null there),
and `data/` was FROZEN while this was built (pinned runs live). The sets below are therefore declared here, and
`move_resolution_rules_integration_test.py` re-derives every one of them from the gen-3 dex through node and fails
on any drift — so the table cannot silently disagree with the simulator. When `data/` unfreezes the flags belong
in the acquisition layer (`tools/`), read through the facade (a named follow-up, not done here).

Nothing here imports torch: `move_resolution.py` builds its buffers from these declarations and the data facade.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# ----------------------------------------------------------------------------------------------- targets
#: The facade `target` values that aim at the FOE in a singles battle (the rest are self / side / field / ally).
#: A foe-targeting move is the only kind a Substitute, a Protect, a type immunity or a Magic Coat can stop.
FOE_TARGETS: FrozenSet[str] = frozenset({
    "normal", "any", "adjacentFoe", "allAdjacentFoes", "allAdjacent", "randomNormal", "scripted"})

# ------------------------------------------------------------------------------------------ move flags
#: Foe-targeting gen-3 moves WITHOUT the ``protect`` flag (Protect / Detect cannot block them); every other
#: foe-targeting move carries it. `gen4/moves.ts:1014-1033` (Protect's ``onTryHit`` checks ``flags.protect``).
FOE_NO_PROTECT: FrozenSet[str] = frozenset({
    "conversion2", "doomdesire", "sketch", "curse", "futuresight", "psychup", "transform", "roleplay"})

#: Moves with ``bypasssub`` — a Substitute does not stop them (`gen4/moves.ts:1283-1320`, Substitute's
#: ``onTryPrimaryHit``: blocked unless ``target === source`` or ``flags.bypasssub``; there is no ``infiltrates``
#: path in gen 3, and a SOUND move does NOT bypass a substitute until gen 6).
BYPASSSUB: FrozenSet[str] = frozenset({
    "conversion2", "disable", "encore", "foresight", "haze", "mimic", "odorsleuth", "sketch", "spite", "taunt",
    "tickle", "imprison", "psychup", "roar", "snatch", "torment", "transform", "whirlwind", "destinybond",
    "grudge", "attract", "helpinghand", "roleplay", "skillswap"})

#: Moves with ``sound`` — Soundproof blocks them (`gen7/abilities.ts:98`, ``onTryHit``). Soundproof is BANNED in
#: gen3ou (`config/formats.ts:4418-4422`), so its Smogon prior is ~0 and the fact is moot in practice.
SOUND: FrozenSet[str] = frozenset({
    "uproar", "healbell", "perishsong", "roar", "snore", "grasswhistle", "growl", "hypervoice", "metalsound",
    "screech", "sing", "supersonic"})

#: Moves with ``reflectable`` — a Magic Coat bounces them back (the source flag set; the v85 cell's predicate,
#: status AND target 'normal', also counted Taunt / Encore / Disable / Torment / Roar, which carry no flag).
REFLECTABLE: FrozenSet[str] = frozenset({
    "flash", "glare", "hypnosis", "tickle", "cottonspore", "leechseed", "poisongas", "scaryface", "toxic", "yawn",
    "block", "charm", "grasswhistle", "growl", "meanlook", "metalsound", "poisonpowder", "screech", "sing",
    "sleeppowder", "spore", "stringshot", "stunspore", "supersonic", "sweetkiss", "sweetscent", "willowisp",
    "swagger", "thunderwave", "spiderweb", "kinesis", "lovelykiss", "attract", "confuseray", "faketears",
    "featherdance", "flatter", "leer", "sandattack", "smokescreen", "tailwhip"})

#: Moves with ``defrost`` — usable (and thawing) while frozen (`gen3/scripts.ts` beforeMove freeze check).
DEFROST: FrozenSet[str] = frozenset({"flamewheel", "sacredfire"})

#: Moves with ``failencore`` — Encore fails when the target's last move is one of these
#: (`gen3/moves.ts:251-260` over base `data/moves.ts:4724`).
FAILENCORE: FrozenSet[str] = frozenset({"encore", "mimic", "mirrormove", "sketch", "struggle", "transform"})

#: STATUS moves that DO check type immunity in gen 3 (``ignoreImmunity: false``; every other status move skips
#: it, `gen3/scripts.ts:198,308`): Thunder Wave (Electric → a Ground target is immune, `data/moves.ts:19607`)
#: and Glare (Normal → a Ghost target is immune, `gen3/moves.ts:335-338`).
STATUS_TYPE_IMMUNITY: FrozenSet[str] = frozenset({"thunderwave", "glare"})

# ---------------------------------------------------------------------------------- the move KINDS
#: Every move the family gives its own rule, by kind. A kind's ids are verified to exist in the data facade at
#: table build (a missing id raises — a silent never-firing gate reads exactly like a null result).
KINDS: Dict[str, Tuple[str, ...]] = {
    # --- the owner's named mechanics
    "counter": ("counter",),                 # gen3/moves.ts:163-183 — a Physical (type-based) or ANY Hidden Power hit
    "mirrorcoat": ("mirrorcoat",),           # gen3/moves.ts:407-427 — a Special hit that is NOT Hidden Power
    "focuspunch": ("focuspunch",),           # gen4/moves.ts:497-510 — lost focus on a non-Status HIT (not its sub)
    "destinybond": ("destinybond",),         # data/moves.ts:3482-3520 — up until the user's next move
    "beatup": ("beatup",),                   # gen3/moves.ts:31-56 — one hit per alive, unstatused member
    "rapidspin": ("rapidspin",),             # Normal: a Ghost is immune — no damage AND no clearing
    # --- protection
    "protect": ("protect", "detect"),        # +3; share the stall counter with Endure (gen5/conditions.ts:24-47)
    "endure": ("endure",),                   # +4 in Showdown; survives, does not block
    "magiccoat": ("magiccoat",),             # +4; bounces a REFLECTABLE move
    # --- self-target fail conditions
    "substitute": ("substitute",),           # fails at hp <= maxhp/4 or behind a substitute (onTryHit)
    "bellydrum": ("bellydrum",),             # fails at hp <= maxhp/2 or Atk +6
    "rest": ("rest",),                       # fails asleep, at full HP, or with Insomnia / Vital Spirit
    "heal": ("recover", "softboiled", "milkdrink", "slackoff", "moonlight", "morningsun", "synthesis"),
    "refresh": ("refresh",),                 # fails unless psn / tox / par / brn (onHit: '', slp, frz fail)
    "cleric": ("healbell", "aromatherapy"),  # a no-op with no statused party member
    "sleeptalk": ("sleeptalk", "snore"),     # usable only while asleep
    "batonpass": ("batonpass",),             # needs an alive teammate to pass to
    "wish": ("wish",),                       # a slot condition with no onRestart: fails while one is pending
    "focusenergy": ("focusenergy",),         # volatile, no onRestart → fails when up
    "ingrain": ("ingrain",),                 # volatile, no onRestart → fails when up
    # --- side / field
    "reflect": ("reflect",), "lightscreen": ("lightscreen",), "safeguard": ("safeguard",), "mist": ("mist",),
    "spikes": ("spikes",),                   # onSideRestart: fails at 3 layers (data/moves.ts:17517-17518)
    "sunnyday": ("sunnyday",), "raindance": ("raindance",), "sandstorm": ("sandstorm",), "hail": ("hail",),
    # --- foe volatiles (no onRestart → fail when already up; pokemon.ts:1983-1984)
    "taunt": ("taunt",),                     # gen3/moves.ts:591-603: duration 2, bypasssub
    "encore": ("encore",),                   # fails: no last move / encored / failencore last move
    "disable": ("disable",),                 # fails: no last move / disabled
    "torment": ("torment",),
    "yawn": ("yawn",),                       # fails on a statused or drowsy target; Sleep Clause blocks its sleep
    "leechseed": ("leechseed",),             # fails seeded (no onRestart) and on a Grass type
    "confuse": ("confuseray", "supersonic", "sweetkiss", "teeterdance"),   # fail on an already-confused target
    "trap": ("meanlook", "block", "spiderweb"),                             # fail on an already-trapped target
    "attract": ("attract",),                 # fails attracted (the gender rule is a NAMED residual)
    "nightmare": ("nightmare",),             # needs a sleeping target
    "dreameater": ("dreameater",),           # needs a sleeping target
    "phaze": ("roar", "whirlwind"),          # needs an alive bench to drag in; Ingrain anchors
    # --- the rest of the shipped cells' gates
    "boom": ("explosion", "selfdestruct"),   # the user faints anyway (gen3/scripts.ts:201-203)
    "pursuit": ("pursuit",),                 # ×2, never-miss on a switching target (gen4/moves.ts:1050-1078)
}

#: The OPPONENT seats whose ORDER matters to our move: a faster Protect / Detect blocks it, a faster Magic Coat
#: bounces a reflectable one, a faster Substitute stops a status move, a faster Taunt stops a status move.
SEAT_KINDS: Dict[str, Tuple[str, ...]] = {
    "protect": ("protect", "detect"),
    "magiccoat": ("magiccoat",),
    "substitute": ("substitute",),
    "taunt": ("taunt",),
    "pursuit": ("pursuit",),
    "rapidspin": ("rapidspin",),
}

# ------------------------------------------------------------------------------------- move ORDER
# The gen-3 order rule (`p_seat_first`: priority bracket, then P(we act first within it)) lives in
# `agents.model.move_order` with the speed physics it is completed by (gen3_speed_physics_v1, audit F7b) — one module,
# so the bracket and the within-bracket rule can never fork.


#: Every gen-3 move's priority lies in ``[PRIORITY_MIN, PRIORITY_MAX]`` (Roar / Whirlwind −6 … Helping Hand +5,
#: `data/moves.ts`; the gen-3 dex holds −6 −5 −4 −3 −1 0 +1 +3 +4 +5). Under X5 the family splits OTHER_move — a SET
#: of moves — into one seat per integer PRIORITY level (`move_resolution.gather_ops`), so `move_order.p_seat_first`
#: stays the ONE order rule and is exact per level (`gen3_move_resolution_x5_v1`). A move outside the range is refused when
#: the table is built (`move_resolution_tables.build_priority_table`).
PRIORITY_MIN, PRIORITY_MAX = -6, 5


#: Protect / Detect / Endure FAIL when no action follows them in the queue (`data/moves.ts` protect / endure
#: `onPrepareHit: return !!this.queue.willAct() && …`): into a switch (the switch resolved before any move) and
#: when the user moves LAST. Every reader prices them × P(an action follows) — `p_seat_first` gives the order.

# ------------------------------------------------------------------------- the per-turn probabilities
#: Full paralysis (gen 3: 1 in 4, before the move).
P_FULL_PARA = 0.25
#: A frozen mon thaws at its move with probability 1/5 (gen 3; a ``defrost`` move thaws and acts).
P_THAW = 0.20
#: Confusion: a confused mon hits itself instead of moving with probability 1/2 (gen 3). The confusion COUNTER is
#: not observed, so a confusion that ends this turn is a NAMED residual (this rate is then an upper bound).
P_CONFUSION_SELF_HIT = 0.5
#: Infatuation: an attracted mon is immobilised with probability 1/2.
P_INFATUATION = 0.5
#: Substitute's cost: it fails at ``hp <= maxhp/4``.
SUB_HP_FRACTION = 0.25
#: Belly Drum fails at ``hp <= maxhp/2``.
BELLY_DRUM_HP_FRACTION = 0.5
#: Pursuit on a switching target: ×2 base power, and it always hits.
PURSUIT_SWITCH_MULT = 2.0
#: The op's own modal-roll window (`DamageOperator._rolls`: P(KO | hit) = clamp((dmg − hp) / (0.15·dmg))) — a
#: uniform roll over [0.85, 1.0] of the max roll, the gen-3 damage rule, not a tuned threshold.
ROLL_WINDOW = 0.15

#: Abilities whose fact the family reads (revealed exactly; the Smogon species prior otherwise).
ABILITY_INNER_FOCUS = "innerfocus"           # blocks flinch
ABILITY_SOUNDPROOF = "soundproof"            # blocks the SOUND set (banned in gen3ou)
ABILITY_OWN_TEMPO = "owntempo"               # blocks confusion
ABILITY_SLEEP_BLOCK = ("insomnia", "vitalspirit")   # Rest fails; Yawn's sleep is blocked (the op's status table)

# ------------------------------------------------------------------------------- the coordinate CONTRACT
#: The MOVE-cell coordinates, in order — the contract the consumers and the tests read; never re-spell an index.
MOVE_RESOLUTION_MOVE_COORDS: Tuple[str, ...] = (
    # --- resolution (new): P(the action resolves as stated) and its decorrelated factors
    "p_resolve", "p_lands_stay", "p_lands_switch", "p_ko_first", "p_act",
    # --- the owner's Destiny Bond feature and the KO context
    "p_ko_us", "dbond_p_ko",
    # --- consolidated facts from intent_conditional
    "counter_return", "mcoat_return", "p_flinch", "pursuit_trigger", "pursuit_bonus",
    "protect_dmg_avoided", "protect_status_avoided", "protect_attack_mass", "protect_blocked_mass",
    "boom_trade_ko",
    # --- consolidated facts from switch_branch (OA2)
    "e_high_switch", "e_pko_switch", "e_mult_switch", "a_switch",
    # --- consolidated facts from intent_move_cell (what a landed status does)
    "d_their_outspeed", "e_burn", "d_sched", "e_slp", "e_slp_free",
    # --- consolidated facts from pair_outcome_cell (the α-reduced incoming row at our active)
    "in_low", "in_high", "in_crit", "in_ko", "in_acc", "in_is_phys",
    "in_p_par", "in_p_brn", "in_p_frz", "in_p_slp", "in_p_psn", "in_p_tox",
)
MOVE_RESOLUTION_MOVE_IDX: Dict[str, int] = {n: i for i, n in enumerate(MOVE_RESOLUTION_MOVE_COORDS)}

#: The SWITCH-cell coordinates, in order.
MOVE_RESOLUTION_SWITCH_COORDS: Tuple[str, ...] = (
    # --- consolidated facts from pair_outcome_switch (mon j's own α-reduced incoming row; the two judgments gone)
    "low", "high", "crit", "ko_ramp", "acc", "is_phys", "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox",
    # --- the FACT half of spin_denied (their Rapid Spin fails on our Ghost); the hazard stake is dropped
    "p_spin_denied",
    # --- consolidated facts from conditional_threat (e_pko with accuracy counted ONCE)
    "e_pko", "e_type_mult", "margin_high", "margin_crit",
    # --- resolution (new): the switch happens unless their Pursuit KOs the departing mon first
    "p_switch_resolves",
)
MOVE_RESOLUTION_SWITCH_IDX: Dict[str, int] = {n: i for i, n in enumerate(MOVE_RESOLUTION_SWITCH_COORDS)}

#: The pair-outcome coordinates the family keeps (the first 12 of `pair_outcome.PAIR_OUTCOME_COORDS`: the damage
#: prefix and the six status identities). `neutralization` and `tempo_cost` are the two JUDGMENTS it drops.
PAIR_FACT_COORDS: Tuple[str, ...] = (
    "low", "high", "crit", "ko_ramp", "acc", "is_phys", "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox")

#: The flag's legal values (`--move-resolution`); 'off' is production and builds nothing.
MOVE_RESOLUTION_MODES: Tuple[str, ...] = ("off", "on")
