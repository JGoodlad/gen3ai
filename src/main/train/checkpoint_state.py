"""Reading a checkpoint's saved architecture, and realigning its optimizer state by NAME.

`_validate_or_reset_optimizer_state` is the guard for the position-keyed-Adam desync described in
its own docstring; the three helpers below it are the pieces it is built from, kept separate so
they can be unit-tested without a checkpoint on disk.
"""
import os
import sys

import torch


def _load_saved_version(model_path: str):
    """Best-effort read of a checkpoint's saved ModelVersion (its model_config.json).

    Returns the ModelVersion, or **None** when the config is missing/unreadable — so a caller can
    distinguish "could not determine" from a real value (rather than silently fail-safe to a default
    and then FATAL at the version check). Used to let a flagless resume INHERIT every version-checked
    structural toggle (use_popart / opp_belief_cls_k / move_belief_mode / damage_op)
    + the belief coef, so the documented `--model … --steps …` resume works uniformly."""
    try:
        from agents.model.snapshot import _resolve_paths
        from agents.model.model_version import ModelVersion
        _, cfg_dir = _resolve_paths(model_path)
        cfg = os.path.join(cfg_dir, "model_config.json")
        if not os.path.exists(cfg):
            # The checkpoint may live in <run>/checkpoints/ while the run-level
            # model_config.json stays at the run root — search the parent too (mirroring
            # load_model_snapshot). Without this, a flagless resume of a toggle-ON run reads
            # no saved version, falls back to OFF defaults, and FATALs at the arch check.
            parent_cfg = os.path.join(os.path.dirname(cfg_dir), "model_config.json")
            if os.path.exists(parent_cfg):
                cfg = parent_cfg
        if os.path.exists(cfg):
            return ModelVersion.from_json_file(cfg)
    except Exception as e:
        print(f"[Resume] WARNING: could not read saved model_config.json from {model_path}: {e}")
    return None


def _read_saved_optimizer_state(checkpoint_path: str, opt_name_set):
    """Read ``(saved_optimizer_state_dict, saved_param_names)`` straight from an SB3 checkpoint zip.

    ``saved_param_names`` is the saved registration ORDER of the parameters the saved optimizer
    indexed (params only), recovered by filtering the saved ``policy.pth`` state_dict keys to
    ``opt_name_set`` (the names of the params in the CURRENT optimizer). The param subsequence of a
    module's ``state_dict()`` keys is exactly its ``named_parameters()`` order, and the optimizer
    indexes those params 0..N-1 in that order, so ``saved_param_names[i]`` is the param that owns the
    saved optimizer state entry ``i``. Raises on any structural surprise (missing members, count
    mismatch) so the caller falls back to the shape-only guard instead of remapping on bad data."""
    import io
    import zipfile
    with zipfile.ZipFile(checkpoint_path) as z:
        members = set(z.namelist())
        if not {"policy.optimizer.pth", "policy.pth"} <= members:
            raise FileNotFoundError("checkpoint missing policy.optimizer.pth / policy.pth")
        saved_opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")),
                               map_location="cpu", weights_only=False)
        saved_policy = torch.load(io.BytesIO(z.read("policy.pth")),
                                  map_location="cpu", weights_only=False)
    saved_param_names = [k for k in saved_policy.keys() if k in opt_name_set]
    n_saved_opt = sum(len(g.get("params", [])) for g in saved_opt.get("param_groups", []))
    if len(saved_param_names) != n_saved_opt:
        raise ValueError(f"saved param-name count ({len(saved_param_names)}) != saved optimizer "
                         f"param count ({n_saved_opt}); cannot safely map momentum by name")
    return saved_opt, saved_param_names


def _remap_optimizer_state_by_name(opt, current_named_params, saved_opt, saved_param_names) -> dict:
    """Rebuild ``opt``'s per-param momentum so each CURRENT param receives the momentum that was saved
    for a param of the SAME NAME, regardless of registration order — closing the same-shape-reorder
    blind spot a shape check cannot see. ``current_named_params`` is ``(name, param)`` in OPTIMIZER
    index order. A name whose shape changed, or that is new, gets fresh zero-init momentum; a vanished
    saved name is ignored. Mutates ``opt`` via ``load_state_dict`` (torch casts each entry to its
    param's device/dtype). Returns a counts dict for logging/tests. Pure given its inputs → unit-tested."""
    saved_state = saved_opt.get("state", {}) or {}
    saved_idx_of = {nm: i for i, nm in enumerate(saved_param_names)}
    corrected, counts = {}, {"carried": 0, "reordered": 0, "dropped_shape": 0, "fresh": 0}
    for j, (name, param) in enumerate(current_named_params):
        si = saved_idx_of.get(name)
        entry = saved_state.get(si) if si is not None else None
        if entry is None:
            counts["fresh"] += 1                          # new param, or one that carried no momentum
            continue
        ea = entry.get("exp_avg")
        if ea is not None and tuple(ea.shape) != tuple(param.shape):
            counts["dropped_shape"] += 1                  # name reused at a different shape → drop
            continue
        corrected[j] = entry
        counts["carried"] += 1
        if si != j:
            counts["reordered"] += 1
    sd = opt.state_dict()                                 # current param_groups (correct indices) ...
    sd["state"] = corrected                               # ... with momentum re-keyed to current positions
    opt.load_state_dict(sd)                               # torch casts each entry to its param's device/dtype
    return counts


def _shape_only_reset_optimizer_state(model) -> None:
    """Fallback guard, used only when the checkpoint zip can't be read for a name-keyed remap: drop
    ALL momentum if ANY param's saved exp_avg/exp_avg_sq shape disagrees with the live param (proof
    the position-keyed state is misaligned). Same-shape permutations are UNDETECTABLE here — which is
    exactly why the name-keyed remap is preferred whenever the zip is readable."""
    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is None:
        return
    name_of = {id(p): n for n, p in model.policy.named_parameters()}
    bad = []
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                t = st.get(key)
                if t is not None and tuple(t.shape) != tuple(p.shape):
                    bad.append(f"{name_of.get(id(p), '?')} param{tuple(p.shape)} {key}{tuple(t.shape)}")
    if bad:
        from collections import defaultdict
        print(f"[Resume] WARNING: optimizer momentum is MISALIGNED with current parameters "
              f"({len(bad)} shape mismatch(es)) — a parameter-reorder refactor since this checkpoint "
              f"was saved desynced the position-keyed Adam state. RESETTING optimizer momentum "
              f"(fresh zero-init; LR/param_groups preserved). Mismatches: " + "; ".join(bad[:8]))
        sys.stdout.flush()
        opt.state = defaultdict(dict)


def _validate_or_reset_optimizer_state(model, checkpoint_path: str = None) -> None:
    """Realign a resumed AdamW optimizer state to the CURRENT parameters BY NAME.

    SB3/torch save+load the optimizer state BY PARAMETER POSITION, not by name, so a refactor that
    REORDERS a module's parameters between the save and the resume (e.g. v40's `SpreadBelief.__init__`
    building `reinject`/`norm` before `stat_head`) silently misassigns the saved per-param momentum
    (`exp_avg`/`exp_avg_sq`) to the WRONG params: a DIFFERENT-shape reorder crashes `AdamW.step()`
    later (the gen3_nature_ev_belief_v1 bug), and — worse, because it is silent — a SAME-shape reorder
    is invisible to a shape check and quietly corrupts momentum.

    Fix: remap BY NAME. We read the saved optimizer state + the saved parameter NAME ORDER straight
    from the checkpoint zip and rebuild `opt.state` so each current param receives exactly the momentum
    that was saved for its name, regardless of registration order (a param whose name is new or whose
    shape changed gets fresh zero-init). This SUPERSEDES a shape-only reset and closes the
    same-shape-reorder blind spot, so "append new params LAST" is no longer load-bearing for optimizer
    correctness. Falls back to the legacy shape-only reset if the zip can't be read (defensive — never
    crash a resume). No-op (momentum carried verbatim) when the saved order already matches current."""
    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is None:
        return
    if checkpoint_path:
        try:
            opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
            named = [(n, p) for n, p in model.policy.named_parameters() if id(p) in opt_param_ids]
            id_to_name = {id(p): n for n, p in named}
            # (name, param) in the EXACT optimizer index order — robust even if some named params
            # are excluded from the optimizer (e.g. a frozen param).
            current = [(id_to_name.get(id(p)), p)
                       for group in opt.param_groups for p in group["params"]]
            saved_opt, saved_param_names = _read_saved_optimizer_state(
                checkpoint_path, {n for n, _ in named})
            if len(saved_param_names) != len(current):
                raise ValueError(f"saved param count ({len(saved_param_names)}) != current "
                                 f"({len(current)})")
            counts = _remap_optimizer_state_by_name(opt, current, saved_opt, saved_param_names)
            if counts["reordered"] or counts["dropped_shape"]:
                print(f"[Resume] Optimizer momentum remapped BY NAME: {counts['carried']} carried "
                      f"({counts['reordered']} were REORDERED since save → corrected), "
                      f"{counts['dropped_shape']} dropped on shape change, {counts['fresh']} fresh. "
                      f"Position-keyed desync prevented.")
                sys.stdout.flush()
            return
        except Exception as e:
            print(f"[Resume] WARNING: name-keyed optimizer remap unavailable ({e}); falling back to "
                  f"the shape-only guard.")
            sys.stdout.flush()
    _shape_only_reset_optimizer_state(model)
