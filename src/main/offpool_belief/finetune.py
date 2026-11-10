"""MF-3 step (b), THE BELIEF-ONLY FINE-TUNE: one checkpoint's three belief deltas trained on the labels of games
against BROAD teams, every other weight frozen.

Why it exists. The species, move and item beliefs are each ``Smogon prior ⊕ a zero-initialised learned delta``. Self-play
trains the deltas on the labels of POOL teams, so off the pool they are confidently wrong (the meter's baseline). This
is the cheapest test of whether that is a fact about the LABELS: keep the network, retrain only the deltas on teams
the pool does not hold, and read the result with the meter. It answers nothing about strength.

**The training bank** (:func:`build_trainbank`): the meter's own game loop (:func:`main.offpool_belief.bank.play`) with
seat 1 piloting a slice of a registered ``--team-mix`` corpus (``utils.team_corpora``; a human-data corpus is refused
by that loader) and seat 2 a ``home72`` team. The slice is ``permutation(seed)[offset : offset + teams]``, so two
shards of one seed are disjoint by construction, and every drawn team is checked against the held-out panels
(``main.heldout_panels.contamination``: an exact copy or a near neighbour REFUSES the build).

**The recipe** (:data:`RECIPE`, nothing else is a knob): eval mode, fp32, CPU or ``--device``; the loss is the sum of
the trainer's own three belief losses (``agents.training.belief_bank`` rows ``hidden_team_set``, ``move_belief``,
``item``, each at weight 1); only :data:`TRAINED` takes gradient; Adam, constant learning rate, gradient-norm clip;
an epoch is one seeded permutation of the training rows. After every epoch the same loss is taken on the VALIDATION
bank (a disjoint shard of the same corpus), and the epoch with the lowest validation loss is exported (ties to the
earlier; epoch 0 is the untouched checkpoint, so "no step helped" exports the input). No panel row is read here.

**The export** is the input zip with ``policy.pth``'s trained tensors replaced (``main.x65.export.write_zip``: every
other member byte for byte), beside a copy of the run's ``model_config.json``; it is re-loaded and its trained
tensors compared bit for bit before the tool reports success. Score it with ``python -m main.offpool_belief score``.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.offpool_belief import HOME_PANEL, OffPoolBeliefError
from main.offpool_belief import bank as BK
from main.offpool_belief import panels as PN

TRAIN_SCHEMA = "gen3_offpool_belief_trainbank_v1"
FINETUNE_SCHEMA = "gen3_offpool_belief_finetune_v1"
RESULT_FILE = "finetune.json"
EXPORT_FILE = "finetuned.zip"
#: The parameters that take gradient, as prefixes under ``policy.features_extractor``: the three deltas and the
#: norms that feed only them. Everything else (trunk, policy, critic, every other belief head) is frozen.
TRAINED: Tuple[str, ...] = (
    "hypothesis_builder.delta_global.", "hypothesis_builder.delta_norm.", "hypothesis_builder.delta_hidden.",
    "hypothesis_builder.delta_out.", "move_belief.norm.", "move_belief.move_head.", "item_belief_head.norm.",
    "item_belief_head.item_head.")
#: Each leg's output bias: the first step must give every one of them a non-zero gradient (a leg the loss does not
#: reach would otherwise train nothing, silently).
OUTPUT_BIASES: Tuple[str, ...] = ("delta_out.bias", "move_head.bias", "item_head.bias")
#: Each leg's output layer: zeroed (weight and bias), the leg is exactly its prior (:func:`prior_losses`).
OUTPUT_LAYERS: Tuple[str, ...] = ("hypothesis_builder.delta_out.", "move_belief.move_head.", "item_belief_head.item_head.")
LABEL_KEYS = ("belief_species", "belief_moves", "known_moves", "item_label", "item_mask")
GATES = {"hidden_team_set": True, "move_belief": True, "item": True}
COEFS = {"opp_belief_aux_coef": 1.0, "move_belief_coef": 1.0, "item_belief_coef": 1.0}
#: ``moves_weight`` 0: the set loss's MOVES term belongs to the frozen BeliefHead, which this fine-tune does not train.
PARAMS = {"moves_weight": 0.0}
RECIPE: Dict[str, Any] = {"lr": 3e-4, "batch": 256, "epochs": 8, "grad_clip": 1.0, "seed": 6801,
                          "optimizer": "Adam, betas (0.9, 0.999), eps 1e-8, no weight decay, constant lr",
                          "loss": "hidden_team_set (moves_weight 0) + move_belief + item (belief_bank rows, weight 1 each)",
                          "pick": "lowest validation loss over epochs 0..E, ties to the earlier"}
DRAW_STREAM = 7


# ------------------------------------------------------------------------------------------------ the training bank
def draw(n_corpus: int, teams: int, offset: int, seed: int) -> np.ndarray:
    """The corpus rows of one shard: ``permutation(seed)[offset : offset + teams]``."""
    if teams < 1 or offset < 0 or offset + teams > n_corpus:
        raise OffPoolBeliefError(f"shard [{offset}, {offset + teams}) does not fit a corpus of {n_corpus} teams")
    return np.random.default_rng([int(seed), DRAW_STREAM]).permutation(n_corpus)[offset:offset + teams]


def resolve_shard(corpus: str, tier: str, out_dir: Path, *, teams: int, offset: int, seed: int) -> Tuple[Any, list]:
    """Everything a shard build resolves before its first game: the output path refusals, the verified corpus, the
    drawn ``(id, text)`` teams and the held-out contamination check (a hit REFUSES). Plays nothing, writes nothing."""
    from main import heldout_panels as HP
    from utils.team_corpora import CorpusError, load_corpus

    out_dir = Path(out_dir)
    PN.refuse_under_models(out_dir, "the training bank directory")
    PN.refuse_inside_repo(out_dir, "the training bank directory")
    if (out_dir / BK.MANIFEST_FILE).exists() or (out_dir / BK.ROWS_FILE).exists():
        raise OffPoolBeliefError(f"{out_dir} already holds a bank: name a new directory")
    try:
        cp = load_corpus(corpus, tier)
    except (CorpusError, KeyError, OSError) as e:
        raise OffPoolBeliefError(f"corpus {corpus!r} tier {tier!r}: {e}") from e
    idx = draw(len(cp), teams, offset, seed)
    items = [(cp.ids[i], cp.texts[i]) for i in idx]
    hit = HP.contamination(items)
    if hit:
        raise OffPoolBeliefError(f"{len(hit)} drawn team(s) of {corpus} hit a held-out panel (first: {hit[0]}): "
                                 "a training bank never holds a panel team or its near neighbour")
    return cp, items


def shard_plan(corpus: str, tier: str, pilot_zip: str, out_dir: Path, *, teams: int, offset: int,
               seed: int = BK.DEFAULT_SEED) -> Dict[str, Any]:
    """``trainbank --check``: the shard's identity, nothing played."""
    from utils.hashing import sha256_file

    cp, items = resolve_shard(corpus, tier, out_dir, teams=teams, offset=offset, seed=seed)
    return {"schema": TRAIN_SCHEMA, "corpus": cp.ident(), "draw": {"seed": int(seed), "stream": DRAW_STREAM,
            "offset": int(offset), "teams": int(teams)}, "first_team": items[0][0], "last_team": items[-1][0],
            "heldout_contamination": 0, "registry_sha256": PN.registry_sha256(), "home_panel": HOME_PANEL,
            "pilot": {"path": str(pilot_zip), "sha256": sha256_file(Path(pilot_zip))}, "n_games": len(items),
            "code": BK.code_stamp(), "out": str(out_dir)}


def build_trainbank(corpus: str, tier: str, pilot_zip: str, out_dir: Path, *, teams: int, offset: int,
                    seed: int = BK.DEFAULT_SEED, pilot_name: Optional[str] = None, n_envs: int = BK.DEFAULT_ENVS,
                    threads: int = 1, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """One shard of TRAINING rows: one game a drawn team, the facing seat's rows and labels."""
    from main.offpool_belief.score import load_checkpoint
    from utils.hashing import sha256_file
    from utils.rust_env import ffi
    from utils.rust_env.build import ensure_built
    from utils.torch_state_guard import torch_globals

    out_dir = Path(out_dir)
    cp, items = resolve_shard(corpus, tier, out_dir, teams=teams, offset=offset, seed=seed)
    home = PN.panel_ref(HOME_PANEL)
    schedule = BK.game_schedule(len(items), len(home.teams), 1, seed + 1 + offset, same_panel=False)
    _child, lib_path = ensure_built("release", emit=log)
    t0 = time.time()
    with torch_globals(num_threads=int(threads)):
        model, ckpt = load_checkpoint(pilot_zip)

        def progress(done: int, total: int) -> None:
            if done % 100 == 0 or done == total:
                log(f"[trainbank {corpus}+{offset}] {done}/{total} games, {time.time() - t0:.0f} s")
        rows = BK.play(BK.pack([t for _i, t in items]), BK.pack([t for _l, t in home.teams]), schedule,
                       BK.greedy_actor(model), lib=ffi.load(str(lib_path)), n_envs=n_envs, progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / BK.ROWS_FILE, **{k: rows[k] for k in BK.ROW_KEYS})
    manifest = {
        "schema": TRAIN_SCHEMA, "use": "TRAINING rows for the belief-only fine-tune; never a panel, never a prior",
        "corpus": cp.ident(), "draw": {"seed": int(seed), "stream": DRAW_STREAM, "offset": int(offset),
                                        "teams": int(teams)},
        "team_ids": [i for i, _t in items], "heldout_contamination": 0, "registry_sha256": PN.registry_sha256(),
        "home_panel": HOME_PANEL, "home_content_digest": home.content_digest, "pilot": {"name": pilot_name, **ckpt},
        "policy": "greedy (argmax of the masked policy), both seats", "n_games": len(schedule),
        "n_rows": int(len(rows["obs"])), "obs_dim": int(rows["obs"].shape[1]),
        "rows_sha256": sha256_file(out_dir / BK.ROWS_FILE), "env_core_sha256": sha256_file(lib_path),
        "code": BK.code_stamp(), "built_s": round(time.time() - t0, 1),
    }
    (out_dir / BK.MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"[trainbank {corpus}+{offset}] {len(schedule)} games, {manifest['n_rows']} rows -> {out_dir}")
    return manifest


def load_trainbank(path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """A training shard, its rows file checked against the manifest. A PANEL bank is refused: its rows are held out."""
    from utils.hashing import sha256_file

    path = Path(path)
    try:
        man = json.loads((path / BK.MANIFEST_FILE).read_text())
    except (OSError, ValueError) as e:
        raise OffPoolBeliefError(f"{path}: no readable training bank manifest ({e})") from e
    if man.get("schema") != TRAIN_SCHEMA:
        raise OffPoolBeliefError(f"{path}: schema {man.get('schema')!r} is not a TRAINING bank ({TRAIN_SCHEMA}); a "
                                 "panel's scoring bank is held-out material and never trains anything")
    if sha256_file(path / BK.ROWS_FILE) != man["rows_sha256"]:
        raise OffPoolBeliefError(f"{path}: rows file does not match its manifest")
    with np.load(path / BK.ROWS_FILE) as z:
        rows = {k: z[k] for k in ("obs", "mask", *LABEL_KEYS)}
    if len(rows["obs"]) != man["n_rows"]:
        raise OffPoolBeliefError(f"{path}: {len(rows['obs'])} rows, manifest says {man['n_rows']}")
    return man, rows


def gather(paths: Sequence[Path], val_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                                                           List[Dict[str, Any]]]:
    """The training shards concatenated, the validation shard, and their identities. Shards must share one corpus,
    draw seed and pilot, and no team may appear twice (across training shards or in validation)."""
    mans, parts = [], []
    for p in paths:
        m, r = load_trainbank(p)
        mans.append(m)
        parts.append(r)
    vman, vrows = load_trainbank(val_path)
    first = mans[0]
    seen: Dict[str, str] = {}
    for p, m in list(zip(paths, mans)) + [(val_path, vman)]:
        same = (m["corpus"] == first["corpus"] and m["draw"]["seed"] == first["draw"]["seed"]
                and m["pilot"]["sha256"] == first["pilot"]["sha256"] and m["obs_dim"] == first["obs_dim"])
        if not same:
            raise OffPoolBeliefError(f"{p}: another corpus, draw seed, pilot or observation width than {paths[0]}")
        for t in m["team_ids"]:
            if t in seen:
                raise OffPoolBeliefError(f"team {t} is in both {seen[t]} and {p}: shards must be disjoint")
            seen[t] = str(p)
    train = {k: np.concatenate([r[k] for r in parts]) for k in parts[0]}
    ident = [{"path": str(p), "role": "train", "rows_sha256": m["rows_sha256"], "n_rows": m["n_rows"],
              "teams": len(m["team_ids"]), "offset": m["draw"]["offset"]} for p, m in zip(paths, mans)]
    ident.append({"path": str(val_path), "role": "validation", "rows_sha256": vman["rows_sha256"],
                  "n_rows": vman["n_rows"], "teams": len(vman["team_ids"]), "offset": vman["draw"]["offset"]})
    return train, vrows, ident


# ------------------------------------------------------------------------------------------------------ the loop
def trained_parameters(model: Any) -> Dict[str, Any]:
    """Freeze the policy, then un-freeze :data:`TRAINED`; ``{name under the extractor: parameter}``."""
    for p in model.policy.parameters():
        p.requires_grad_(False)
    out = {}
    for name, p in model.policy.features_extractor.named_parameters():
        if name.startswith(TRAINED):
            p.requires_grad_(True)
            out[name] = p
    missing = [pre for pre in TRAINED if not any(n.startswith(pre) for n in out)]
    if missing:
        raise OffPoolBeliefError(f"the extractor has no parameter under {missing}: not the three-delta belief")
    return out


def belief_terms(model: Any, rows: Dict[str, np.ndarray], sel: np.ndarray, device: str) -> Dict[str, Any]:
    """The three belief losses on ``rows[sel]`` by leg (the ``belief_bank`` row's gate name -> a graph-carrying
    scalar); a leg with no supervised unit in the batch is absent."""
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.training import belief_bank as BB

    fe = model.policy.features_extractor
    mb = th.as_tensor(rows["mask"][sel], dtype=th.float32, device=device)
    ob = {"observation": th.as_tensor(rows["obs"][sel], dtype=th.float32, device=device), "action_mask": mb}
    ob.update(zero_extra_obs(fe, batch=len(mb), device=device))
    model.policy.get_distribution(ob)
    labels = {k: th.as_tensor(rows[k][sel], device=device) for k in LABEL_KEYS}
    out: Dict[str, Any] = {}
    for site in ("hidden_move", "revealed"):
        for row, term, _m in BB.compute(fe, labels, COEFS, GATES, site, params=PARAMS):
            out[row.gate] = out[row.gate] + term if row.gate in out else term
    return out


def belief_loss(model: Any, rows: Dict[str, np.ndarray], sel: np.ndarray, device: str) -> Any:
    """The sum of the three belief losses on ``rows[sel]`` (a graph-carrying scalar), or None when the batch holds
    no supervised unit at all."""
    terms = list(belief_terms(model, rows, sel, device).values())
    return sum(terms[1:], terms[0]) if terms else None


def leg_losses(model: Any, rows: Dict[str, np.ndarray], device: str, batch: int) -> Dict[str, float]:
    """The row-weighted mean of each leg's batch loss over ``rows`` in file order, no gradient, plus their sum
    under ``"total"`` (a batch in which a leg has no unit counts that leg as 0, as the training loss does)."""
    import torch as th

    tot = {g: 0.0 for g in GATES}
    n = 0
    with th.no_grad():
        for a in range(0, len(rows["obs"]), batch):
            sel = np.arange(a, min(a + batch, len(rows["obs"])))
            terms = belief_terms(model, rows, sel, device)
            if terms:
                for g, t in terms.items():
                    tot[g] += float(t.item()) * len(sel)
                n += len(sel)
    if n == 0:
        raise OffPoolBeliefError("the validation rows hold no supervised unit")
    out = {g: v / n for g, v in tot.items()}
    out["total"] = sum(out.values())
    return out


def mean_loss(model: Any, rows: Dict[str, np.ndarray], device: str, batch: int) -> float:
    """The row-weighted mean of the batch losses over ``rows`` in file order, no gradient."""
    return leg_losses(model, rows, device, batch)["total"]


def prior_losses(model: Any, rows: Dict[str, np.ndarray], device: str, batch: int) -> Dict[str, float]:
    """:func:`leg_losses` of the PRIOR ALONE: the three legs' output layers (:data:`OUTPUT_LAYERS`) zeroed for the
    read and restored after it, so each belief is exactly its Smogon prior. The reference line of the curve."""
    import torch as th

    outs = {n: p for n, p in trained_parameters(model).items() if any(o in n for o in OUTPUT_LAYERS)}
    if len(outs) != 2 * len(OUTPUT_LAYERS):
        raise OffPoolBeliefError(f"expected a weight and a bias for each of {OUTPUT_LAYERS}, found {sorted(outs)}")
    keep = {n: p.detach().clone() for n, p in outs.items()}
    with th.no_grad():
        for p in outs.values():
            p.zero_()
    try:
        return leg_losses(model, rows, device, batch)
    finally:
        with th.no_grad():
            for n, p in outs.items():
                p.copy_(keep[n])


def plan(init_zip: str, train_paths: Sequence[Path], val_path: Path, out_dir: Path, recipe: Dict[str, Any], *,
         banks_optional: bool = False) -> Dict[str, Any]:
    """Everything a run resolves before its first step (``--check`` prints this and stops). ``banks_optional`` (the
    check alone) reports a bank directory that is not built yet as ``"not built"`` instead of refusing, so a
    registration can paste the plan before any game is played."""
    from main.offpool_belief.score import load_checkpoint

    out_dir = Path(out_dir)
    PN.refuse_under_models(out_dir, "the fine-tune directory")
    PN.refuse_inside_repo(out_dir, "the fine-tune directory")
    if (out_dir / RESULT_FILE).exists():
        raise OffPoolBeliefError(f"{out_dir} already holds a fine-tune: name a new directory")
    paths = [Path(p) for p in train_paths] + [Path(val_path)]
    if len({str(p.resolve()) for p in paths}) != len(paths):
        raise OffPoolBeliefError("a bank directory is named twice (training and validation must be different shards)")
    unbuilt = [str(p) for p in paths if not (p / BK.MANIFEST_FILE).exists()]
    model, ckpt = load_checkpoint(init_zip)
    params = trained_parameters(model)
    doc: Dict[str, Any] = {
        "schema": FINETUNE_SCHEMA, "init": ckpt, "recipe": dict(recipe),
        "trained_tensors": {n: list(p.shape) for n, p in params.items()},
        "trained_parameters": int(sum(p.numel() for p in params.values())),
        "frozen_parameters": int(sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)),
        "code": BK.code_stamp(), "out": str(out_dir)}
    if unbuilt and banks_optional:
        doc.update({"banks": "not built", "unbuilt": unbuilt, "train_rows": None, "val_rows": None,
                    "steps_per_epoch": None, "total_steps": None})
        return doc
    train, val, ident = gather(train_paths, val_path)
    steps = int(np.ceil(len(train["obs"]) / recipe["batch"]))
    doc.update({"banks": ident, "train_rows": int(len(train["obs"])), "val_rows": int(len(val["obs"])),
                "steps_per_epoch": steps, "total_steps": steps * int(recipe["epochs"])})
    return doc


def run(init_zip: str, train_paths: Sequence[Path], val_path: Path, out_dir: Path, *, device: str = "cpu",
        threads: int = 1, recipe: Optional[Dict[str, Any]] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train, pick by validation, export, verify the export. Returns the result document (also written)."""
    import torch as th

    from main.offpool_belief.score import load_checkpoint
    from utils.hashing import sha256_file
    from utils.torch_state_guard import torch_globals

    recipe = dict(RECIPE if recipe is None else recipe)
    out_dir = Path(out_dir)
    with torch_globals(num_threads=int(threads)):
        doc = plan(init_zip, train_paths, val_path, out_dir, recipe)
        train, val, _ident = gather(train_paths, val_path)
        model, _ckpt = load_checkpoint(init_zip)
        model.policy.to(device)
        params = trained_parameters(model)
        opt = th.optim.Adam(list(params.values()), lr=float(recipe["lr"]))
        snap = lambda: {n: p.detach().cpu().clone() for n, p in params.items()}  # noqa: E731
        states = [snap()]
        val_prior = prior_losses(model, val, device, recipe["batch"])
        legs = leg_losses(model, val, device, recipe["batch"])
        curve = [{"epoch": 0, "val_loss": legs["total"], "val_legs": legs, "train_loss": None}]
        log(f"[finetune] the prior alone: val {val_prior['total']:.5f} {val_prior}")
        log(f"[finetune] epoch 0 val {curve[0]['val_loss']:.5f} ({doc['train_rows']} train rows)")
        t0, n = time.time(), len(train["obs"])
        for epoch in range(1, int(recipe["epochs"]) + 1):
            perm = np.random.default_rng([int(recipe["seed"]), epoch]).permutation(n)
            tot = cnt = 0
            for a in range(0, n, recipe["batch"]):
                sel = np.sort(perm[a:a + recipe["batch"]])
                loss = belief_loss(model, train, sel, device)
                if loss is None:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if epoch == 1 and a == 0:
                    dead = [k for k, p in params.items() if k.endswith(OUTPUT_BIASES)
                            and (p.grad is None or not bool(p.grad.abs().sum() > 0))]
                    if dead:
                        raise OffPoolBeliefError(f"no gradient reaches {dead}: the loss does not train this leg")
                th.nn.utils.clip_grad_norm_(list(params.values()), float(recipe["grad_clip"]))
                opt.step()
                tot += float(loss.item()) * len(sel)
                cnt += len(sel)
            states.append(snap())
            legs = leg_losses(model, val, device, recipe["batch"])
            curve.append({"epoch": epoch, "val_loss": legs["total"], "val_legs": legs,
                          "train_loss": tot / max(cnt, 1)})
            log(f"[finetune] epoch {epoch} train {curve[-1]['train_loss']:.5f} val {curve[-1]['val_loss']:.5f} "
                f"{time.time() - t0:.0f} s")
        picked = min(range(len(curve)), key=lambda e: (curve[e]["val_loss"], e))
        out_dir.mkdir(parents=True, exist_ok=True)
        export(Path(init_zip), states[picked], out_dir / EXPORT_FILE)
        doc.update({"device": device, "torch_threads": int(threads), "curve": curve, "picked_epoch": picked,
                    "val_prior": val_prior,
                    "export": {"path": str(out_dir / EXPORT_FILE), "sha256": sha256_file(out_dir / EXPORT_FILE)},
                    "train_s": round(time.time() - t0, 1)})
    (out_dir / RESULT_FILE).write_text(json.dumps(doc, indent=1) + "\n")
    log(f"[finetune] picked epoch {picked} -> {out_dir / EXPORT_FILE}")
    return doc


def export(init_zip: Path, state: Dict[str, Any], dst: Path) -> None:
    """``init_zip`` with the trained tensors of ``policy.pth`` replaced by ``state`` (names under the extractor),
    its run's ``model_config.json`` copied beside it, then re-loaded and compared bit for bit."""
    import io
    import zipfile

    import torch as th

    from main.offpool_belief.score import load_checkpoint
    from main.x65.export import run_config, write_zip

    with zipfile.ZipFile(init_zip) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    for name, t in state.items():
        keys = [k for k in sd if k.endswith("features_extractor." + name)]
        if not keys:
            raise OffPoolBeliefError(f"{init_zip}: policy.pth has no tensor for {name}")
        for k in keys:
            if tuple(sd[k].shape) != tuple(t.shape):
                raise OffPoolBeliefError(f"{init_zip}: {k} is {tuple(sd[k].shape)}, the trained tensor {tuple(t.shape)}")
            sd[k] = t.to(sd[k].dtype).clone()
    write_zip(init_zip, sd, dst)
    shutil.copyfile(run_config(init_zip), dst.parent / "model_config.json")
    back, _ = load_checkpoint(str(dst))
    got = dict(back.policy.features_extractor.named_parameters())
    bad = [n for n, t in state.items() if not th.equal(got[n].detach().cpu(), t)]
    if bad:
        raise OffPoolBeliefError(f"{dst}: re-loaded tensors differ from the trained ones ({bad}); the export is VOID")
