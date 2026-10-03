"""THE TRAINING-LABEL INVENTORY — every label key the trainee's observation can carry, and where each
comes from in the Rust env (M5 Lane C, ``designs/endstate/program_rust_core.md`` §2 M5).

The trainee's Dict obs holds the ``observation`` row + ``action_mask`` (Lane 0's ``obs`` / ``mask``
columns) and the TRAINING-ONLY LABEL keys below — read by a loss, a callback or a diagnostic, never by
the policy forward (``designs/ARCHITECTURE.md`` §7). This table is the ONE list of them.
``src/agents/training/rust_env_label_inventory_test.py`` (routine) fails the day
``agents.training.trainee_spaces`` declares a key this table does not list, a row's dtype / shape / emit
gate drifts, the PRODUCTION surface (``--arch production`` + ``designs/production_config.json``) emits a
different set than the rows marked ``production``, or ARCHITECTURE.md §7 disagrees about which keys
production emits. The human-readable version, with each key's derivation, is
``designs/rust_sim/env_labels.md``.

(Until deletion pass U3 the list was the whole truth about the Python env's ``Gen3Env`` label keys, and each
row's ``producer`` names the ``Gen3Env`` method that wrote it — kept as the DEFINITION's name; that env and its
label parity gates are deleted, and the Rust label columns are pinned by ``rust_env/tests`` and the label
columns test.)

Each row says WHERE THE RUST ENV GETS IT (``rust``):

* ``core`` — a per-decision value the env core computes from what it holds: the TRUTH (the other
  side's own reading of its team and the engine board) and the side's own READING / row. A label
  column (Lane C, ``src/rust_env/src/labels/``).
* ``host_const`` — a PLACEHOLDER: a constant that the collector's back-fill overwrites after the
  game. No column: the host (Lane G's vec env) writes ``const``.
* ``host_episode`` — a per-EPISODE value the host already owns (the opponent choice stays Python,
  program §2 M5 inventory). No column: the host writes it from its per-episode routing state, keyed
  by the core's ``episode`` column.

``gate`` is the ``label_gates`` keywords that turn the key on (``agents.training.trainee_spaces.
trainee_env_kwargs`` derives them from the CLI); ``family`` is the Lane-C build unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

TEAM = 6          # agents.observation.constants.TEAM_SIZE
MOVE_SLOTS = 4    # agents.observation.belief_labels.BELIEF_MOVE_SLOTS
SPREAD = 5        # agents.observation.belief_labels.N_SPREAD_STATS

RUST_KINDS = ("core", "host_const", "host_episode")


@dataclass(frozen=True)
class LabelKey:
    key: str
    dtype: str                     # "i64" | "f32" — the declared Box dtype
    shape: Tuple[int, ...]
    family: str
    gate: Tuple[Tuple[str, object], ...]   # `label_gates` kwargs that emit it (any ONE row of `alt_gates` also does)
    production: bool
    producer: str                  # the (deleted) Python env function that originally wrote it — the definition's name
    consumers: Tuple[str, ...]     # src-relative files that READ it (not the carriers: the fork FILL table, the rollout buffer)
    rust: str
    derivation: str
    const: Optional[float] = None  # host_const: the value
    symbol: Optional[str] = None   # the module constant a consumer spells the key with, if not the literal
    alt_gates: Tuple[Tuple[Tuple[str, object], ...], ...] = ()


_BELIEF = (("emit_belief_labels", True),)
_KNOWN = (("move_belief_mode", "both"),)
_SPREAD = (("emit_spread_labels", True),)
_HP = (("emit_hp_type_labels", True),)
_ITEM = (("emit_item_labels", True),)
_WIN = (("emit_win_target", True),)
_INTENT = (("emit_opp_intent_labels", True),)

_BANK = "agents/training/belief_bank.py"
_PPO = "agents/training/instrumented_ppo/ppo.py"
_SETUP = "agents/training/instrumented_ppo/train_setup.py"
_INTENT_FOLD = "agents/training/instrumented_ppo/intent_fold.py"   # K8: the intent block, inside region R1

_REVEALED = ("the side's REVEALED opponent slots (`species_known` read from the side's own row, the "
             "leading-contiguous block; the reading's `opp` list in encoder order)")

LABELS: Tuple[LabelKey, ...] = (
    # ---------------------------------------------------------------- belief (species / moves)
    LabelKey("belief_species", "i64", (TEAM,), "belief", _BELIEF, True, "Gen3Env._belief_labels",
             (_BANK,), "core",
             "species NUM of the hidden opp mon assigned to each BELIEVED slot (`assign_hidden_to_slots`: "
             "the other side's own team minus the revealed species, sorted by num; the j-th fills the j-th "
             "believed slot); -1 elsewhere",
             alt_gates=(_KNOWN,)),
    LabelKey("belief_moves", "i64", (TEAM, MOVE_SLOTS), "belief", _BELIEF, True, "Gen3Env._belief_labels",
             (_BANK,), "core",
             "that hidden mon's move NUMS in its own set order (typed Hidden Power num), -1 pad",
             alt_gates=(_KNOWN,)),
    LabelKey("known_moves", "i64", (TEAM, MOVE_SLOTS), "belief", _KNOWN, True, "Gen3Env._belief_labels",
             (_BANK,), "core",
             f"at {_REVEALED}: that species' FULL move NUMS from the other side's own team, -1 pad"),
    # ---------------------------------------------------------------- spread / nature / EV
    LabelKey("belief_spread", "f32", (TEAM, SPREAD), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core",
             f"at {_REVEALED}: the TRUE derived stats (atk, def, spa, spd, spe) — the other side's own "
             "reading's `stats` (the request's `baseStoredStats`)"),
    LabelKey("belief_spread_mask", "f32", (TEAM,), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core", "1 where `belief_spread` holds a complete 5-stat tuple"),
    LabelKey("belief_nature", "i64", (TEAM,), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core",
             "the set's TRUE declared nature num (`belief_tables.true_nature_ev_label`, not cached; "
             "raises unless the declared spread at its true IVs reproduces the derived stats)"),
    LabelKey("belief_nature_mask", "f32", (TEAM,), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core", "1 where the truth mon has complete stats and a dex row (== `belief_spread_mask`)"),
    LabelKey("belief_ev", "f32", (TEAM, SPREAD), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core", "the set's TRUE EVs, stat-effective `4*floor(ev/4)` (atk, def, spa, spd, spe)"),
    LabelKey("belief_ev_mask", "f32", (TEAM,), "spread", _SPREAD, True, "Gen3Env._spread_labels",
             (_BANK,), "core", "== `belief_nature_mask`"),
    # ---------------------------------------------------------------- Hidden Power type
    LabelKey("hp_type_label", "i64", (TEAM,), "hp_type", _HP, True, "Gen3Env._hp_type_labels",
             (_BANK,), "core",
             f"at {_REVEALED}: the HP type index 0..15 of that species' typed `hiddenpower<type>` move on "
             "the other side's own team; -1 where it runs none"),
    LabelKey("hp_type_mask", "f32", (TEAM,), "hp_type", _HP, True, "Gen3Env._hp_type_labels",
             (_BANK,), "core", "1 where `hp_type_label` is set"),
    # ---------------------------------------------------------------- item
    LabelKey("item_label", "i64", (TEAM,), "item", _ITEM, True, "Gen3Env._item_labels",
             (_BANK,), "core",
             f"at {_REVEALED}: the item NUM the other side's own reading holds NOW (0 = nothing — a "
             "consumed / knocked-off item reads 0); -1 pad"),
    LabelKey("item_mask", "f32", (TEAM,), "item", _ITEM, True, "Gen3Env._item_labels",
             (_BANK,), "core", "1 where `item_label` is set"),
    # ---------------------------------------------------------------- win-prob
    LabelKey("win_target", "f32", (1,), "winprob", _WIN, True, "Gen3Env._merge_training_keys",
             ("agents/training/win_prob_callback.py", _PPO), "host_const",
             "PLACEHOLDER; the Rust collector back-fills the episode outcome post-collection "
             "(under `--critic winprob` it IS the value target)", const=0.0),
    LabelKey("win_mask", "f32", (1,), "winprob", _WIN, True, "Gen3Env._merge_training_keys",
             ("agents/training/win_prob_callback.py", _PPO), "host_const",
             "PLACEHOLDER; back-filled with the known-outcome mask", const=0.0),
    LabelKey("win_margin", "f32", (1,), "margin", _WIN, True,
             "Gen3RewardManager.process_turn_reward → material_margin.material_margin",
             (_PPO, "agents/training/value_sidecar.py"), "core",
             "the normalized MATERIAL MARGIN of the side's LiveView at this decision (`material_margin.py`: "
             "HP + alive over the declared team size, unrevealed opp mons full-HP-alive); 0.0 at reset"),
    LabelKey("opp_class", "i64", (1,), "opp_class", _WIN, True, "Gen3Env._merge_training_keys / _opp_intent_labels",
             (_PPO, "agents/training/value_sidecar.py"), "host_episode",
             "WHICH kind of opponent this episode faces (bot / pool / stable / exploiter), set by the "
             "wrapper at reset — the host's per-episode routing state",
             alt_gates=(_INTENT,)),
    # ---------------------------------------------------------------- opponent intent (α/β)
    LabelKey("opp_action_kind", "i64", (1,), "intent", _INTENT, True, "Gen3Env._opp_intent_labels",
             (_INTENT_FOLD, _SETUP), "core",
             "what the opponent DID at the PREVIOUS decision (move / switch / unknown) — the α/β label the "
             "port's trackers already fold (`trackers::IntentLabel`, slice T)"),
    LabelKey("opp_action_num", "i64", (1,), "intent", _INTENT, True, "Gen3Env._opp_intent_labels",
             (_INTENT_FOLD, _SETUP), "core",
             "that move's NUM, Hidden Power resolved to the attacker's TRUE typed num from the other side's "
             "own team"),
    LabelKey("opp_switch_slot", "i64", (1,), "intent", _INTENT, True, "Gen3Env._opp_intent_labels",
             (_INTENT_FOLD, _SETUP), "core",
             "the switch-in's REVEALED slot as of the previous decision (`_opp_slot_map_prev`), "
             "SWITCH_SLOT_NONE otherwise"),
    LabelKey("opp_switch_species", "i64", (1,), "intent", _INTENT, True, "Gen3Env._opp_intent_labels",
             (_INTENT_FOLD, _SETUP), "core", "the switch-in's species NUM (content-addressed β)"),
    # ---------------------------------------------------------------- OFF the production surface
    LabelKey("fork_pg_m", "f32", (1,), "fork", (("emit_fork_pg_mask", True),), False,
             "Gen3Env._merge_training_keys", ("agents/training/fork_arm.py",), "host_const",
             "PLACEHOLDER 1.0; only a row the fork arm INJECTS holds anything else", const=1.0,
             symbol="PG_MASK_KEY"),
)


def _check() -> None:
    keys = [r.key for r in LABELS]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"duplicate label keys: {keys}")
    for r in LABELS:
        if r.rust not in RUST_KINDS:
            raise AssertionError(f"{r.key}: rust must be one of {RUST_KINDS}")
        if r.dtype not in ("i64", "f32"):
            raise AssertionError(f"{r.key}: dtype {r.dtype}")
        if (r.rust == "host_const") != (r.const is not None):
            raise AssertionError(f"{r.key}: a host_const row (and only one) carries `const`")


_check()

BY_KEY: Mapping[str, LabelKey] = {r.key: r for r in LABELS}


def production_keys() -> Tuple[str, ...]:
    return tuple(r.key for r in LABELS if r.production)


def families(rust: Optional[str] = None) -> Dict[str, Tuple[LabelKey, ...]]:
    """``{family: rows}`` in table order, optionally only the rows of one ``rust`` kind."""
    out: Dict[str, list] = {}
    for r in LABELS:
        if rust is None or r.rust == rust:
            out.setdefault(r.family, []).append(r)
    return {k: tuple(v) for k, v in out.items()}
