"""The training-label INVENTORY is the whole truth about ``Gen3Env``'s label keys (M5 Lane C).

``utils.rust_env.label_inventory.LABELS`` is the list the Rust env's label columns are built from.
These tests EXECUTE ``Gen3Env`` (no battle, no server: construction alone declares the obs space)
and fail the day the env and the inventory disagree:

* a key ``Gen3Env`` can declare that the inventory does not list (or one it lists that the env
  cannot declare), or a dtype / shape that drifted;
* the PRODUCTION surface (``--arch production`` + ``designs/production_config.json``, resolved the
  way a production launch's env factory resolves it) emitting a different set than the rows marked
  ``production`` — so a flag flip that adds a label to production cannot reach the Rust env unseen;
* a row's emit gate that does not emit it, or an ungated env that does;
* a named consumer that no longer reads the key;
* ``designs/ARCHITECTURE.md`` §7 or ``designs/rust_sim/env_labels.md`` disagreeing with the table.
"""
import re

import numpy as np

from utils.paths import repo_path, src_path
from utils.rust_env import label_inventory as LI

_NON_LABEL = {"observation", "action_mask"}
_OFF = dict(emit_belief_labels=False, move_belief_mode="off", emit_win_target=False,
            emit_fork_pg_mask=False, emit_spread_labels=False, emit_opp_intent_labels=False,
            emit_hp_type_labels=False, emit_item_labels=False, distill_team_species=None)
_ENVS = {}


def _space(**kw):
    """The label part of a ``Gen3Env``'s declared obs space for ``kw`` (on top of every gate OFF)."""
    from poke_env import AccountConfiguration

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env

    args = dict(_OFF, **kw)
    k = repr(sorted(args.items(), key=lambda t: t[0]))
    if k not in _ENVS:
        env = Gen3Env(load_mappings(), battle_format="gen3ou",
                      account_configuration1=AccountConfiguration(f"LInv{len(_ENVS)}", None),
                      start_listening=False, **args)
        _ENVS[k] = {n: s for n, s in env.observation_space.spaces.items() if n not in _NON_LABEL}
    return _ENVS[k]


def _all_on():
    kw = {}
    for r in LI.LABELS:
        kw.update(dict(r.gate))
    return _space(**kw)


_NP = {"i64": np.int64, "f32": np.float32}


def test_every_key_gen3env_can_emit_is_inventoried_with_its_dtype_and_shape():
    space = _all_on()
    assert set(space) == {r.key for r in LI.LABELS}, (
        f"Gen3Env declares {sorted(set(space) - {r.key for r in LI.LABELS})} the inventory does not "
        f"list; the inventory lists {sorted({r.key for r in LI.LABELS} - set(space))} Gen3Env cannot "
        "declare — add / remove the row in utils/rust_env/label_inventory.py (and its Rust status)")
    for r in LI.LABELS:
        assert space[r.key].dtype == _NP[r.dtype], (r.key, space[r.key].dtype)
        assert tuple(space[r.key].shape) == r.shape, (r.key, space[r.key].shape)


def test_the_production_surface_emits_exactly_the_production_rows():
    from main.rust_core_cutover.envs import production_args
    from main.train.env_factory import trainee_env_kwargs

    kw = trainee_env_kwargs(production_args())
    kw.pop("obs_source")
    emitted = set(_space(**kw))
    assert emitted == set(LI.production_keys()), (
        f"production emits {sorted(emitted - set(LI.production_keys()))} not marked production, and "
        f"does not emit {sorted(set(LI.production_keys()) - emitted)} marked production")


def test_every_gate_emits_its_rows_and_nothing_is_emitted_ungated():
    assert _space() == {}
    for r in LI.LABELS:
        for gate in (r.gate,) + r.alt_gates:
            assert r.key in _space(**dict(gate)), (r.key, gate)


def test_every_named_consumer_reads_its_key():
    for r in LI.LABELS:
        for path in r.consumers:
            text = src_path(*path.split("/")).read_text()
            spelled = [f'"{r.key}"', f"'{r.key}'"] + ([r.symbol] if r.symbol else [])
            assert any(s in text for s in spelled), f"{r.key}: {path} no longer names it ({spelled})"


def test_architecture_section_7_marks_production_as_the_inventory_does():
    """Every §7 table row names its keys in backticks and opens its last cell with ✅ / ❌."""
    text = repo_path("designs", "ARCHITECTURE.md").read_text()
    sec = text.split("## 7. Training-only obs keys", 1)[1].split("\n## ", 1)[0]
    seen = {}
    for line in sec.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        on = cells[-1].lstrip("*").startswith("✅")
        for key in re.findall(r"`([a-z_]+)`", cells[0]):
            seen[key] = on
    assert set(seen) == {r.key for r in LI.LABELS}, (
        "ARCHITECTURE.md §7 lists "
        f"{sorted(set(seen) - {r.key for r in LI.LABELS})} extra and misses "
        f"{sorted({r.key for r in LI.LABELS} - set(seen))}")
    wrong = sorted(k for k, on in seen.items() if on != LI.BY_KEY[k].production)
    assert not wrong, f"ARCHITECTURE.md §7's ✅/❌ disagrees with the inventory on {wrong}"


def test_the_design_doc_lists_every_key_with_its_rust_status():
    text = repo_path("designs", "rust_sim", "env_labels.md").read_text()
    for r in LI.LABELS:
        rows = [ln for ln in text.splitlines() if ln.startswith(f"| `{r.key}`")]
        assert len(rows) == 1, f"env_labels.md must have exactly one table row for `{r.key}`"
        assert f"`{r.rust}`" in rows[0], (r.key, r.rust, rows[0])


def test_each_family_is_one_rust_kind():
    """A Lane-C build unit is one family, so a family is wholly built, host-filled or refused."""
    for fam, rows in LI.families().items():
        assert len({r.rust for r in rows}) == 1, (fam, [(r.key, r.rust) for r in rows])


def test_the_rust_hp_type_table_is_the_python_one():
    """`labels/per_slot.rs` spells `belief_labels.HP_TYPE_NAMES` by hand; the index IS the label."""
    from agents.observation.belief_labels import HP_TYPE_NAMES

    text = src_path("rust_env", "src", "labels", "per_slot.rs").read_text()
    body = text.split("pub const HP_TYPE_NAMES: [&str; 16] = [", 1)[1].split("];", 1)[0]
    assert tuple(re.findall(r'"([a-z]+)"', body)) == HP_TYPE_NAMES
