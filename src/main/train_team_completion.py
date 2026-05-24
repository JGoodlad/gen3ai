"""
Training script for the Team Completion Model.

Fresh run:
    python -m main.train_team_completion --backbone models/.../checkpoint.zip

Resume from a previous run:
    python -m main.train_team_completion --run-dir models/team_prediction/run_20260523_143012

Models are saved to models/team_prediction/run_<timestamp>/ with a layout that
mirrors the PPO runs (latest.txt, metadata.json, command.txt, per-epoch .pt files).
"""

import argparse
import json
import os
import random
import sys
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.model.team_completion_model import TeamCompletionModel, model_config
from agents.training.team_completion.team_dataset import build_datasets, Mappings
from utils.git import get_git_hash, get_repo_root


console = Console()


# ── Run directory helpers ─────────────────────────────────────────────────────

def _make_run_dir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def _write_latest(run_dir: str, filename: str) -> None:
    with open(os.path.join(run_dir, "latest.txt"), "w") as f:
        f.write(filename)


def _read_latest(run_dir: str) -> Optional[str]:
    p = os.path.join(run_dir, "latest.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read().strip()


def _write_metadata(
    run_dir: str,
    args: argparse.Namespace,
    mappings: Mappings,
    epoch: Optional[int] = None,
    val_metrics: Optional[dict] = None,
) -> None:
    meta = {
        "saved_at":   datetime.utcnow().isoformat() + "Z",
        "git_hash":   get_git_hash(),
        "python_version": sys.version,
        "backbone":   args.backbone,
        "data":       args.data,
        "epochs_total": args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "val_split":  args.val_split,
    }
    if epoch is not None:
        meta["current_epoch"] = epoch
    if val_metrics is not None:
        meta["evals"] = {
            k: ({str(ck): round(cv, 6) for ck, cv in v.items()} if isinstance(v, dict) else round(v, 6))
            for k, v in val_metrics.items()
        }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _write_model_config(run_dir: str, args: argparse.Namespace, mappings: Mappings) -> None:
    cfg = model_config(mappings.num_species, mappings.num_moves, mappings.num_items, args.backbone)
    with open(os.path.join(run_dir, "model_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's dataset RNG so masking patterns are independent."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    underlying = getattr(ds, "dataset", ds)  # unwrap Subset if present
    underlying._rng = random.Random(info.seed + worker_id)


def _model_step(
    model: TeamCompletionModel,
    batch: dict,
    move_pos_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict, dict]:
    """Run forward + loss. Returns (loss, parts, logits)."""
    logits = model(
        batch["species_ids"],
        batch["move_multihot"],
        batch["item_ids"],
        batch["is_masked"],
        batch["is_padding"],
    )
    loss, parts = model.compute_loss(logits, batch, move_pos_weight=move_pos_weight)
    return loss, parts, logits


@torch.no_grad()
def evaluate(
    model: TeamCompletionModel,
    loader: DataLoader,
    device: str,
    move_pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    sp_loss_sum = item_loss_sum = mv_loss_sum = 0.0
    correct_top1 = correct_top5 = total_masked = 0
    item_correct = item_total = 0
    move_recall_sum = 0.0
    move_recall_count = 0
    ctx_correct: dict[int, int] = defaultdict(int)
    ctx_total:   dict[int, int] = defaultdict(int)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, parts, logits = _model_step(model, batch, move_pos_weight)
        total_loss    += loss.item()
        sp_loss_sum   += parts["species_loss"].item()
        item_loss_sum += parts["item_loss"].item()
        mv_loss_sum   += parts["move_loss"].item()

        is_masked  = batch["is_masked"]
        is_padding = batch["is_padding"]
        active = is_masked & ~is_padding               # [B, 6]

        # ── Species top-1 / top-5 ────────────────────────────────────────
        if active.any():
            sp_logits  = logits["species_logits"][active]
            sp_targets = batch["target_species"][active]
            top5 = sp_logits.topk(5, dim=-1).indices
            correct_top1 += (top5[:, 0] == sp_targets).sum().item()
            correct_top5 += (top5 == sp_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_masked += active.sum().item()

        # ── Species top-1 by context size ────────────────────────────────
        ctx_sizes  = (~is_masked & ~is_padding).sum(dim=1)   # [B] — revealed context count
        sp_pred    = logits["species_logits"].argmax(dim=-1) # [B, 6]
        is_correct = (sp_pred == batch["target_species"]) & active
        for ctx in ctx_sizes.unique().tolist():
            ctx = int(ctx)
            mask = (ctx_sizes == ctx).unsqueeze(1) & active   # [B, 6]
            ctx_correct[ctx] += is_correct[mask].sum().item()
            ctx_total[ctx]   += mask.sum().item()

        # ── Item top-1 ───────────────────────────────────────────────────
        item_active = active & batch["item_known"]
        if item_active.any():
            it_pred    = logits["item_logits"][item_active].argmax(dim=-1)
            it_targets = batch["target_item"][item_active]
            item_correct += (it_pred == it_targets).sum().item()
            item_total   += item_active.sum().item()

        # ── Move recall@4 ────────────────────────────────────────────────
        if active.any():
            mv_logits  = logits["move_logits"][active]          # [N, M]
            mv_targets = batch["target_move_multihot"][active]  # [N, M]
            n_revealed = mv_targets.sum(dim=-1)                 # [N]
            has_moves  = n_revealed > 0
            if has_moves.any():
                top4_idx = mv_logits.topk(4, dim=-1).indices    # [N, 4]
                top4_hot = torch.zeros_like(mv_targets, dtype=torch.bool)
                top4_hot.scatter_(1, top4_idx, True)
                hits = (top4_hot & mv_targets.bool()).sum(dim=-1).float()  # [N]
                recall = (hits[has_moves] / n_revealed[has_moves])
                move_recall_sum   += recall.sum().item()
                move_recall_count += has_moves.sum().item()

    n = max(len(loader), 1)
    return {
        "loss":               total_loss / n,
        "species_loss":       sp_loss_sum / n,
        "item_loss":          item_loss_sum / n,
        "move_loss":          mv_loss_sum / n,
        "species_top1":       correct_top1 / max(total_masked, 1),
        "species_top5":       correct_top5 / max(total_masked, 1),
        "item_top1":          item_correct / max(item_total, 1),
        "move_recall_at_4":   move_recall_sum / max(move_recall_count, 1),
        "species_top1_by_ctx": {
            ctx: ctx_correct[ctx] / ctx_total[ctx]
            for ctx in sorted(ctx_total)
        },
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = args.device
    repo_root = get_repo_root()

    # ── Dataset & mappings ────────────────────────────────────────────────
    console.print(f"[bold]Loading dataset:[/] {args.data}")
    train_ds, val_ds, mappings = build_datasets(
        args.data, repo_root, val_split=args.val_split, seed=42
    )
    console.print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}  "
                  f"species: {mappings.num_species}  moves: {mappings.num_moves}  items: {mappings.num_items}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device == "cuda"),
                              worker_init_fn=_worker_init_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"),
                              worker_init_fn=_worker_init_fn)

    # ── Run directory ────────────────────────────────────────────────────
    resuming = args.run_dir is not None
    if resuming:
        run_dir = args.run_dir
        # Read backbone path from metadata if not specified
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path) as f:
            saved_meta = json.load(f)
        if args.backbone is None:
            args.backbone = saved_meta.get("backbone")
        console.print(f"[bold]Resuming run:[/] {run_dir}")
    else:
        run_dir = _make_run_dir(os.path.join(repo_root, "models", "team_prediction"))
        console.print(f"[bold]New run:[/] {run_dir}")
        # Record command
        with open(os.path.join(run_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        _write_metadata(run_dir, args, mappings)
        _write_model_config(run_dir, args, mappings)

    # ── Model ───────────────────────────────────────────────────────────
    model = TeamCompletionModel(
        num_species=mappings.num_species,
        num_moves=mappings.num_moves,
        num_items=mappings.num_items,
        backbone_path=args.backbone,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    start_epoch = 1
    if resuming:
        latest = _read_latest(run_dir)
        if latest:
            ckpt_path = os.path.join(run_dir, latest)
            console.print(f"  Loading checkpoint: {latest}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            console.print(f"  Resuming from epoch {start_epoch}")

    # ── TensorBoard ──────────────────────────────────────────────────────
    tb_dir = os.path.join(repo_root, "tensorboard",
                          f"team_completion_{os.path.basename(run_dir)}")
    writer = SummaryWriter(tb_dir)
    console.print(f"[bold]TensorBoard:[/] {tb_dir}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    console.print(f"  Trainable params: {trainable:,}  |  Frozen (backbone): {frozen:,}")

    # ── Training loop ────────────────────────────────────────────────────
    move_pos_weight = mappings.move_pos_weight
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = sp_loss_sum = item_loss_sum = mv_loss_sum = 0.0
        n_batches = len(train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Epoch {epoch:>3}/{args.epochs}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("train", total=n_batches, loss=0.0)

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss, parts, _ = _model_step(model, batch, move_pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                train_loss    += loss.item()
                sp_loss_sum   += parts["species_loss"].item()
                item_loss_sum += parts["item_loss"].item()
                mv_loss_sum   += parts["move_loss"].item()

                progress.advance(task, advance=1)
                completed = progress.tasks[task].completed
                progress.update(task, loss=train_loss / max(completed, 1))

        train_loss /= n_batches
        sp_loss_sum /= n_batches
        item_loss_sum /= n_batches
        mv_loss_sum /= n_batches

        val_metrics = evaluate(model, val_loader, device, move_pos_weight)
        scheduler.step(val_metrics["loss"])

        # ── TensorBoard logging ───────────────────────────────────────────
        writer.add_scalars("team_completion/loss",
                           {"train": train_loss, "val": val_metrics["loss"]}, epoch)
        for component, train_val, metric_key in [
            ("species", sp_loss_sum,   "species_loss"),
            ("item",    item_loss_sum, "item_loss"),
            ("moves",   mv_loss_sum,   "move_loss"),
        ]:
            writer.add_scalars(f"team_completion/loss_{component}",
                               {"train": train_val, "val": val_metrics[metric_key]}, epoch)
        writer.add_scalar("team_completion/accuracy/species_top1",
                          val_metrics["species_top1"], epoch)
        writer.add_scalar("team_completion/accuracy/species_top5",
                          val_metrics["species_top5"], epoch)
        writer.add_scalar("team_completion/accuracy/item_top1",
                          val_metrics["item_top1"], epoch)
        writer.add_scalar("team_completion/accuracy/move_recall_at_4",
                          val_metrics["move_recall_at_4"], epoch)
        for ctx, acc in val_metrics["species_top1_by_ctx"].items():
            writer.add_scalar(f"team_completion/accuracy/species_top1_by_ctx/ctx_{ctx}",
                              acc, epoch)
        writer.add_scalar("team_completion/lr",
                          optimizer.param_groups[0]["lr"], epoch)

        # ── Terminal summary row ──────────────────────────────────────────
        console.print(
            f"Epoch {epoch:>3}/{args.epochs} │ "
            f"train {train_loss:.4f} │ "
            f"val {val_metrics['loss']:.4f} │ "
            f"sp_top1 [bold green]{val_metrics['species_top1']*100:.1f}%[/] │ "
            f"sp_top5 {val_metrics['species_top5']*100:.1f}% │ "
            f"item {val_metrics['item_top1']*100:.1f}% │ "
            f"mv_rec@4 {val_metrics['move_recall_at_4']*100:.1f}%"
        )

        # ── best_model.pt — updated every epoch, regardless of save_every ──
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_metrics["loss"],
                "val_top1":  val_metrics["species_top1"],
            }, os.path.join(run_dir, "best_model.pt"))

        # ── Periodic checkpoint ───────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_name = f"checkpoint_epoch_{epoch:04d}.pt"
            ckpt_path = os.path.join(run_dir, ckpt_name)
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_metrics["loss"],
                "val_top1":  val_metrics["species_top1"],
            }, ckpt_path)
            _write_latest(run_dir, ckpt_name)
            _write_metadata(run_dir, args, mappings, epoch=epoch, val_metrics=val_metrics)

    writer.close()
    console.print(f"\n[bold green]Training complete.[/] Best val loss: {best_val_loss:.4f}")
    console.print(f"Run dir: {run_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Team Completion Model")
    parser.add_argument("--backbone",  type=str, default=None,
                        help="Path to MaskablePPO .zip checkpoint (required for fresh runs)")
    parser.add_argument("--data",      type=str,
                        default=os.path.join(get_repo_root(), "data", "replay_teams.jsonl"),
                        help="Path to replay_teams.jsonl")
    parser.add_argument("--run-dir",   type=str, default=None,
                        help="Existing run dir to resume from (omit for a fresh run)")
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--batch-size",type=int, default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--save-every",type=int, default=5,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.run_dir is None and args.backbone is None:
        parser.error("Either --backbone (fresh run) or --run-dir (resume) is required")

    train(args)


if __name__ == "__main__":
    main()
