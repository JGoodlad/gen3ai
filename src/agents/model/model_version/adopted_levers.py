"""THE LEGACY + JUDGMENT DELETION's refusal rule (`designs/ops/legacy_deletion_manifest_2026-10-12.md`, unit U0).

After the END STATE's adoption (2026-10-12) every lever of the `--arch endstate` overlay has ONE value that anything
trains with; the deletion units U1-U4 delete each lever's flag, its `ModelVersion` field and its OFF path. This module
is the table those units fill and the ONE predicate both load paths judge a recorded value with:

* **the config** (`model_config.json`, unmigrated): `retired_levers.RETIRED` carries one STRUCTURAL row per deleted
  field (`retired_rows`), so `_migrate_config` refuses it on every load (`refuse_structural`) and a resume / fork is
  refused before the migration pops the evidence (`check_no_retired_levers`, `main.checkargs`);
* **the zip** (the pickled `features_extractor_kwargs` SB3 splats into the constructor):
  `snapshot.sanitize_dead_extractor_kwargs` judges the SAME predicate (`judge_pickled_kwargs`) and pops the inactive
  ones. NOT a `_DEAD_FEK_JUDGED` row: that table holds ONE supported value per kwarg, and the seven per-action blocks
  are recorded True in every adopted checkpoint (inert under move resolution), so a plain `(<block>, False)` row would
  refuse the E arm, the bf16 runs and X26 (manifest F-LD-4).

**THE PREDICATE (D-LD1 + D-LD2 + the X26 readiness doc's vacuity rule, §3.5).** A recorded lever is ACTIVE — the
checkpoint trained a forward this code no longer builds — iff

1. every prerequisite the lever `requires` is ENABLED in the recorded config (an absent prerequisite reads as OFF:
   every prerequisite's registry default is an OFF value). A config that LACKS the prerequisites never built the
   lever, whatever value it recorded: the lever is VACUOUS there, so a TOY checkpoint (`damage_op` off, no move
   resolution, …) still loads after the deletion (`design_x26_readiness.md` §3.5, R-T2);
2. the lever is not INERT in that config (`inert_when`: the seven per-action blocks are constructed and then retired
   under move resolution, so a recorded True beside `move_resolution on` built nothing that survives);
3. the recorded value (an ABSENT field reads as `off_default`, the value every config before the lever's version
   built) differs from the ADOPTED value.

The adopted value of a deleted lever is unconditional once its prerequisites are built (D-LD2 "DERIVED"), so a
config that recorded the adopted surface (the closing test's E arm, the bf16 A/B, X26) keeps loading at HEAD and
every other config is refused, typed, naming the pin tag `pre-legacy-deletion` (D-LD4: any later ablation of an
adopted lever runs PINNED there). No MIGRATION_FLOOR raise and no ARCH_SIGNATURE bump (D-LD1).

**How a unit lands.** Its rows are already in :data:`PLANNED`; a unit adds its name to :data:`LANDED_UNITS` and its
first config version to :data:`UNIT_VERSIONS`, deletes the flag / field / OFF path, and the refusal follows from the
table. Torch-free, and it imports only the flag registry's `is_enabled` (the registry keeps the PREREQUISITE rows,
which are not deleted by this pass).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Tuple

#: The git TAG every refusal names: the last commit whose tree still has EVERY lever below (and its OFF path). Any
#: later ablation of an adopted lever runs pinned here (D-LD4). Created at U0 on the commit before U0's own.
PRE_LEGACY_DELETION_TAG = "pre-legacy-deletion"

#: The commit the tag points at (`0a0d9e1d`, T31's umbrella fix: the tip when U0 branched). Pinned here too, so a
#: refusal still names the fix on a clone that never fetched the tag.
LAST_COMMIT_PRE_LEGACY_DELETION = "0a0d9e1d42ff592daf9abd9b09f0e3e59257cbde"

#: The manifest the rows come from (named in every refusal).
MANIFEST = "designs/ops/legacy_deletion_manifest_2026-10-12.md"


def _block_inert_under_move_resolution(raw: Mapping[str, Any]) -> bool:
    """The seven per-action blocks: `--move-resolution on` CONSTRUCTS them and then RETIRES them
    (`ExtractorApi.retire_superseded_action_cells`), so their recorded True beside it built nothing the
    surviving forward lacks."""
    return str(raw.get("move_resolution", "off")) == "on"


class AdoptedLever(NamedTuple):
    """One lever of the adopted end state whose OFF path a deletion unit removes."""

    field: str                       # the recorded ModelVersion field (== the extractor kwarg)
    flag: str                        # the CLI flag, WITHOUT its leading `--` (the freshness gate's reason)
    adopted: Any                     # the one value anything trains with after adoption
    off_default: Any                 # what an ABSENT field means (the registry default before the deletion)
    requires: Tuple[str, ...]        # the prerequisites the lever needs built (frozen copy of the registry's)
    kind: str                        # the manifest's class: J / AF / DUP / INC / HC / SUP / NF / LEG
    unit: str                        # the deletion unit (U1-U4) that removes it
    what: str                        # what a NON-adopted value meant (the refusal's evidence line)
    inert_when: Optional[Callable[[Mapping[str, Any]], bool]] = None


_OP = ("damage_op",)
_MR = ("opp_intent", "damage_op", "damage_outgoing", "damage_matrices_incoming", "damage_matrices_outgoing")

#: EVERY row of the manifest's §1 inventory, in its order. `adopted` equals `designs/production_config.json`
#: (pinned by `adopted_levers_test` while the field is still in the mirror).
PLANNED: Tuple[AdoptedLever, ...] = (
    # ---- U1, lane OP (all `requires damage_op`) -------------------------------------------------------------
    AdoptedLever("status_facts", "status-facts", "exact", "off", _OP, "J", "U1",
                 "status_facts != 'exact' (the pair-outcome grid's JUDGMENT coordinates `neutralization` / "
                 "`tempo_cost`, P2 / P3)"),
    AdoptedLever("ko_ramp", "ko-ramp", "exact", "ramp", _OP, "AF", "U1",
                 "ko_ramp != 'exact' (the mean-roll 15 %-window KO ramp, no crit)"),
    AdoptedLever("g_ledger", "g-ledger", "eot", "coarse", _OP, "DUP", "U1",
                 "g_ledger != 'eot' (the op's own older end-of-turn ledger, a second end-of-turn rule)"),
    AdoptedLever("op_reduction", "op-reduction", "principled", "max", _OP, "INC", "U1",
                 "op_reduction != 'principled' (per-channel hard maxima, acc / provenance read at another "
                 "channel's argmax)"),
    AdoptedLever("speed_physics", "speed-physics", "on", "off", _OP, "HC", "U1",
                 "speed_physics != 'on' (the hand constant sigmoid(dSpe / 15) for P(outspeed))"),
    AdoptedLever("eot_residual", "eot-residual", "on", "off", _OP, "NF", "U1",
                 "eot_residual != 'on' (the per-mon end-of-turn residual fact not handed to the network)"),
    AdoptedLever("switch_hazard_cost", "switch-hazard-cost", "on", "off", _OP, "NF", "U1",
                 "switch_hazard_cost != 'on' (the switch cell's entry-hazard cost not handed to the network)"),
    AdoptedLever("effective_stats", "effective-stats", "on", "off", _OP, "NF", "U1",
                 "effective_stats != 'on' (the active mons' stage-applied stats not handed to the network)"),
    # ---- U2, lane OBS (model side; the observation itself is untouched) -------------------------------------
    AdoptedLever("obs_facts", "obs-facts", "v1", "off", (), "NF", "U2",
                 "obs_facts != 'v1' (the OBS-FACTS block not read by the model)"),
    AdoptedLever("drop_progress_clock", "drop-progress-clock", "on", "off", (), "J", "U2",
                 "drop_progress_clock != 'on' (the model READ `turns_since_progress`, a judgment clock, O10)"),
    AdoptedLever("move_target_state", "move-target-state", "on", "off", (), "NF", "U2",
                 "move_target_state != 'on' (the move seats' target state not handed to the network)"),
    AdoptedLever("move_set_closure", "move-set-closure", "on", "off", ("opp_belief_slots", "opp_intent"), "NF", "U2",
                 "move_set_closure != 'on' (the opponent slots' MoveBelief reinjection read the raw sigmoid, "
                 "not the four-move closure)"),
    # ---- U3, lanes MR + VTI (the construct-then-retire unit) ------------------------------------------------
    AdoptedLever("move_resolution", "move-resolution", "on", "off", _MR, "SUP", "U3",
                 "move_resolution != 'on' (the seven per-action blocks were the policy's move-side route)"),
    AdoptedLever("move_resolution_facts", "move-resolution-facts", "full", "off", ("move_resolution",), "NF", "U3",
                 "move_resolution_facts != 'full' (move resolution without the restored facts)"),
    AdoptedLever("value_threat_inject", "value-threat-inject", False, False, _OP, "C2", "U3",
                 "value_threat_inject=True (the critic's third route for the threat fact: "
                 "CLSPool.value_threat_proj)"),
    *(AdoptedLever(f, f.replace("_", "-"), False, False, req, "SUP", "U3",
                   f"{f}=True without move resolution (the per-action block was BUILT and read)",
                   _block_inert_under_move_resolution)
      for f, req in (("intent_move_cell", ("opp_intent", "damage_op")),
                     ("intent_threshold", ("opp_intent", "damage_op")),
                     ("intent_conditional", ("opp_intent", "damage_op", "damage_outgoing",
                                             "damage_matrices_outgoing")),
                     ("pair_outcome_cell", _OP),
                     ("pair_outcome_switch", _OP),
                     ("switch_branch_cell", ("opp_intent", "damage_op", "damage_matrices_outgoing")),
                     ("conditional_threat_cell", ("damage_op", "damage_matrices_incoming")))),
    # ---- U4, lane LEG ------------------------------------------------------------------------------------------
    AdoptedLever("token_encoding", "token-encoding", "static", "legacy", (), "LEG", "U4",
                 "token_encoding != 'static' (the legacy PokemonEncoder token encoding)"),
    AdoptedLever("mon_hazard_cost", "mon-hazard-cost", "on", "off", ("token_encoding", "damage_op"), "NF", "U4",
                 "mon_hazard_cost != 'on' (the per-mon Spikes fact not handed to the network)"),
    AdoptedLever("move_actor_state", "move-actor-state", "on", "off", ("token_encoding",), "NF", "U4",
                 "move_actor_state != 'on' (the actor's state on its move seats not handed to the network)"),
)

#: The units that have LANDED (their fields are gone from `ModelVersion`). U0 lands the machinery only.
LANDED_UNITS: FrozenSet[str] = frozenset()

#: unit -> the first `MODEL_CONFIG_VERSION` written without its fields (filled as each unit lands).
UNIT_VERSIONS: Dict[str, int] = {}


def _enabled(value: Any) -> bool:
    from agents.model.flag_registry import is_enabled
    return is_enabled(value)


def _same(recorded: Any, adopted: Any) -> bool:
    """Compare on the ADOPTED value's type: a bool lever by truth value (a pickled 0 / 1 is a bool), a mode string
    or a depth by equality (`bool('off')` is True, which is why truthiness is never the test)."""
    if isinstance(adopted, bool):
        return bool(recorded) is adopted
    return bool(recorded == adopted)


def prerequisites_built(lever: AdoptedLever, raw: Mapping[str, Any]) -> bool:
    """Every prerequisite of `lever` ENABLED in `raw` (an absent one reads as OFF)."""
    return all(_enabled(raw.get(r)) for r in lever.requires)


def lever_active(lever: AdoptedLever, raw: Mapping[str, Any]) -> bool:
    """THE predicate (module docstring): did the config / kwargs `raw` record a NON-adopted value of `lever` in a
    network that BUILT it? False whenever the lever is vacuous (prerequisites lacking) or inert."""
    if not prerequisites_built(lever, raw):
        return False
    if lever.inert_when is not None and lever.inert_when(raw):
        return False
    return not _same(raw.get(lever.field, lever.off_default), lever.adopted)


def landed(units: Optional[FrozenSet[str]] = None) -> Tuple[AdoptedLever, ...]:
    """The rows of the landed units (`units` overrides :data:`LANDED_UNITS`, for tests)."""
    u = LANDED_UNITS if units is None else units
    return tuple(r for r in PLANNED if r.unit in u)


def active_levers(raw: Mapping[str, Any], *, units: Optional[FrozenSet[str]] = None) -> List[AdoptedLever]:
    """The landed levers `raw` recorded ACTIVE, in table order."""
    return [r for r in landed(units) if lever_active(r, raw)]


def refusal_text(levers: List[AdoptedLever], where: str) -> str:
    lines = "\n".join(f"  · {r.what}  (--{r.flag}, deleted in {r.unit})" for r in levers)
    return (f"{where} recorded a lever the LEGACY DELETION removed ({MANIFEST}):\n{lines}\n"
            "The end state was adopted 2026-10-12 and the other value of each lever deleted; this code cannot "
            "rebuild the forward that checkpoint trained, so it is refused rather than loaded as something else.\n"
            f"To re-read it, run PINNED at the tag `{PRE_LEGACY_DELETION_TAG}` "
            f"({LAST_COMMIT_PRE_LEGACY_DELETION[:8]}) or at the git_hash in its own metadata.json.")


def judge_pickled_kwargs(fek: Dict[str, Any], *, units: Optional[FrozenSet[str]] = None) -> bool:
    """The ZIP-side twin of the config refusal: raise `ModelVersionError` when the pickled extractor kwargs record
    an ACTIVE landed lever, else POP every landed lever's kwarg (the constructor no longer takes it). Every row is
    judged BEFORE any is popped — a row's predicate may read another landed row's field (`move_resolution` for the
    seven blocks). True if `fek` changed."""
    from agents.model.model_version.constants import ModelVersionError

    rows = landed(units)
    active = [r for r in rows if lever_active(r, fek)]
    if active:
        raise ModelVersionError(refusal_text(active, "this checkpoint's pickled extractor kwargs"))
    changed = False
    for r in rows:
        if r.field in fek:
            fek.pop(r.field)
            changed = True
    return changed
