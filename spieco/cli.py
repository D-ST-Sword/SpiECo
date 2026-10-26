"""Portable teacher -> target cache -> consolidation -> AV refinement workflow."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import load_splits, loader
from .masking import erase, GRID, EVAL_SEED
from .metrics import metrics
from .objectives import consolidation_loss, set_refinement_dropout, zero_augmentation
from .student import AVOnly, Student
from .teacher import Teacher
from .utils import seed_all, sha, write, device_batch, load_state

RECIPES = {
    "teacher": dict(epochs=40, batch=32, lr=1e-5, weight_decay=1e-4, eta_min=1e-7, patience=20),
    "student": dict(epochs=40, batch=32, lr=1e-4, weight_decay=1e-4, eta_min=1e-6, patience=15),
    "refine": dict(epochs=40, batch=64, lr=3e-5, weight_decay=.1, eta_min=1e-6, patience=None),
}


def erase_batch(raw, rates, seed, dataset):
    options = dict(special_ids=(0, 101, 102), pad_id=0) if dataset == "sims" else {}
    tokens, retention, _, _ = erase(raw["text"], raw["id"], rates, seed, **options)
    return {**raw, "text": tokens}, retention


@torch.no_grad()
def collect(model, batches, dataset, device, rate=None):
    model.eval()
    rows = {key: [] for key in ("id", "target", "prediction")}
    for raw in batches:
        if rate is not None:
            batch, retention = erase_batch(raw, rate, EVAL_SEED, dataset)
            output = model(device_batch(batch, device), retention.to(device))
        else:
            output = model(device_batch(raw, device))
        rows["id"].extend(raw["id"])
        rows["target"].append(raw["target"].numpy())
        rows["prediction"].append(output["M"].cpu().numpy())
    return {k: np.asarray(v) if k == "id" else np.concatenate(v) for k, v in rows.items()}


def valid_score(model, batches, dataset, device, stage):
    rates = (0., .5, 1.) if stage == "student" else (None,)
    records = [collect(model, batches, dataset, device, r) for r in rates]
    # Historical selection used un-clipped regression errors on CH-SIMS.
    scores = [metrics(dataset, r["target"], r["prediction"], sims_clip=False) for r in records]
    if stage == "refine":
        s = scores[0]
        rank = (s["accuracy"], s["weighted_f1"], -s["selection_classification_loss"])
    else:
        rank = (-float(np.mean([s["mae"] for s in scores])),)
    return rank, scores, records


class TrainingTargets:
    def __init__(self, folder, ids, teacher_path, data_sha256):
        folder = Path(folder)
        info = json.loads((folder / "complete.json").read_text())
        path = folder / "training_targets.npz"
        if (info["split"] != "train_only" or info["teacher_sha256"] != sha(teacher_path)
                or info["data_sha256"] != data_sha256 or info["targets_sha256"] != sha(path)):
            raise ValueError("Teacher cache provenance mismatch")
        with np.load(path, allow_pickle=False) as record:
            source_ids = record["id"].tolist()
            self.prediction = torch.from_numpy(record["prediction"].copy())
            self.latent = torch.from_numpy(record["latent"].copy())
        if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(ids):
            raise ValueError("Teacher cache must contain exactly the training IDs")
        if self.prediction.shape != (len(ids),) or self.latent.shape != (len(ids), 128):
            raise ValueError("Teacher cache has invalid shapes")
        if not torch.isfinite(self.prediction).all() or not torch.isfinite(self.latent).all():
            raise ValueError("Nonfinite teacher targets")
        self.index = {uid: i for i, uid in enumerate(source_ids)}

    def get(self, ids, device):
        indices = torch.tensor([self.index[uid] for uid in ids])
        return self.prediction[indices].to(device), self.latent[indices].to(device)


def initialized_teacher(args, manifest):
    seed_all(args.seed)
    model = Teacher(args.dataset, args.encoder)
    if args.teacher is not None:
        record = load_state(model, args.teacher, args.dataset, manifest["data_sha256"])
        if record.get("kind") != "teacher" or record.get("seed") != args.seed:
            raise ValueError("Expected the matching-seed teacher checkpoint")
        model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def cache_targets(args, splits, manifest):
    teacher = initialized_teacher(args, manifest).to(args.device).eval()
    rows = {k: [] for k in ("id", "prediction", "latent")}
    for raw in loader(splits["train"]):
        out = teacher(device_batch(raw, args.device))
        rows["id"].extend(raw["id"])
        for key in ("prediction", "latent"):
            rows[key].append(out["M" if key == "prediction" else key].cpu().numpy())
    rows = {k: np.asarray(v) if k == "id" else np.concatenate(v) for k, v in rows.items()}
    if rows["id"].tolist() != splits["train"].ids or not all(np.isfinite(rows[k]).all() for k in ("prediction", "latent")):
        raise ValueError("Invalid training target cache")
    path = args.output / "training_targets.npz"
    np.savez_compressed(path, **rows)
    write(args.output / "complete.json", dict(split="train_only", teacher_sha256=sha(args.teacher),
          data_sha256=manifest["data_sha256"], targets_sha256=sha(path), samples=len(rows["id"])))


def train(args, splits, manifest):
    recipe = RECIPES[args.stage]
    targets = None
    if args.stage in ("teacher", "student"):
        model = initialized_teacher(args, manifest)
        if args.stage == "student":
            model = Student(args.dataset, model)
            targets = TrainingTargets(args.cache, splits["train"].ids, args.teacher, manifest["data_sha256"])
    else:
        model = AVOnly(args.dataset)
        saved = load_state(model, args.checkpoint, args.dataset, manifest["data_sha256"])
        if saved.get("kind") != "av" or saved.get("stage") != "student" or saved.get("seed") != args.seed:
            raise ValueError("Refinement requires the same-seed consolidated AV export")
        set_refinement_dropout(model)
    model.to(args.device)
    training = loader(splits["train"], recipe["batch"], train=True, seed=args.seed)
    validation = loader(splits["valid"])
    if args.stage == "teacher":
        groups = [{"params": [p for p in model.core.parameters() if p.requires_grad], "lr": 1e-5},
                  {"params": model.adapters.parameters(), "lr": 1e-4}]
    else:
        groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"], eta_min=recipe["eta_min"])
    amp = args.stage == "teacher" and torch.device(args.device).type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.)
    if args.stage in ("teacher", "student"):
        seed_all(args.seed)
    best, stale, selected = None, 0, None
    for epoch in range(1, recipe["epochs"] + 1):
        model.train()
        total, count, overflows = 0., 0, 0
        rng = np.random.default_rng(args.seed + 100003 * epoch)
        for raw in training:
            optimizer.zero_grad(set_to_none=True)
            if args.stage == "student":
                batch, rho = erase_batch(raw, rng.choice(GRID, size=len(raw["id"])), args.seed + 100003 * epoch, args.dataset)
                batch = device_batch(batch, args.device)
                out = model(batch, rho.to(args.device))
                tp, tz = targets.get(raw["id"], args.device)
                loss, _ = consolidation_loss(out, batch["target"], tp, tz)
            else:
                batch = device_batch(raw, args.device)
                if args.stage == "refine":
                    batch = zero_augmentation(batch)
                loss = F.l1_loss(model(batch)["M"], batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not torch.isfinite(norm):
                overflows += 1
                if overflows > 3 or scaler.get_scale() >= previous_scale or scaler.get_scale() < 1:
                    raise FloatingPointError("Repeated AMP overflow")
            total += float(loss.detach()) * len(raw["id"])
            count += len(raw["id"])
        if args.stage == "refine":
            scheduler.step()
            # Preserve clean training evaluation in the refinement recipe.
            collect(model, loader(splits["train"]), args.dataset, args.device)
        rank, scores, records = valid_score(model, validation, args.dataset, args.device, args.stage)
        if best is None or rank > best:
            best, stale, selected = rank, 0, epoch
            state = dict(model=model.state_dict(), dataset=args.dataset, seed=args.seed,
                         kind="av" if args.stage == "refine" else args.stage, stage=args.stage,
                         data_sha256=manifest["data_sha256"], epoch=epoch, validation=scores)
            torch.save(state, args.output / "selected.pt")
            for index, record in enumerate(records):
                np.savez_compressed(args.output / f"validation_{index}.npz", **record)
        else:
            stale += 1
        row = dict(epoch=epoch, loss=total / count, validation=scores, selected_epoch=selected, amp_skipped=overflows)
        with (args.output / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if args.stage != "refine":
            scheduler.step()
        if recipe["patience"] is not None and epoch >= 20 and stale >= recipe["patience"]:
            break
    del optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    state = load_state(model, args.output / "selected.pt")
    _, _, replay = valid_score(model, validation, args.dataset, args.device, args.stage)
    for index, current in enumerate(replay):
        with np.load(args.output / f"validation_{index}.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["id"], current["id"])
            np.testing.assert_allclose(previous["prediction"], current["prediction"], atol=1e-6, rtol=0)
    if args.stage == "student":
        export = AVOnly(args.dataset).to(args.device).eval()
        export.load_state_dict(model.av.state_dict(), strict=True)
        arrays = collect(export, validation, args.dataset, args.device)
        np.testing.assert_allclose(arrays["prediction"], replay[-1]["prediction"], atol=1e-6, rtol=0)
        torch.save({**state, "model": export.state_dict(), "kind": "av", "requires_text": False}, args.output / "av_only.pt")
    write(args.output / "trained.json", dict(selected_epoch=selected, checkpoint_sha256=sha(args.output / "selected.pt"),
          test_evaluated=False, validation_replay="passed"))


def evaluate(args, splits, manifest):
    model = AVOnly(args.dataset).to(args.device)
    state = load_state(model, args.checkpoint, args.dataset, manifest["data_sha256"])
    if state.get("kind") != "av":
        raise ValueError("The deployment evaluator accepts AV-only checkpoints")
    records = collect(model, loader(splits[args.split]), args.dataset, args.device)
    scores = metrics(args.dataset, records["target"], records["prediction"])
    np.savez_compressed(args.output / "predictions.npz", **records)
    write(args.output / "metrics.json", scores)
    write(args.output / "evaluation.json", dict(split=args.split, checkpoint_sha256=sha(args.checkpoint),
          input_modalities=["audio", "vision"], requires_text=False, selected_epoch=state["epoch"]))
    print(json.dumps(scores, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "cache", "student", "refine", "evaluate"))
    parser.add_argument("--dataset", choices=("urfunny", "mosi", "sims"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", help="Local RoBERTa-large (English) or Chinese RoBERTa-wwm-ext")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    args = parser.parse_args(argv)
    if args.stage in ("teacher", "cache", "student") and not args.encoder:
        parser.error("--encoder is required for training-time text access")
    if args.stage in ("cache", "student") and args.teacher is None:
        parser.error("--teacher is required")
    if args.stage == "student" and args.cache is None:
        parser.error("--cache is required")
    if args.stage in ("refine", "evaluate") and args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.stage == "teacher" and args.teacher is not None:
        parser.error("Teacher training starts from pretrained encoders, not a frozen teacher")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    try:
        splits, manifest = load_splits(args.data, args.dataset, args.encoder if args.stage in ("teacher", "cache", "student") else None)
        write(args.output / "data_manifest.json", manifest)
        write(args.output / "config.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "recipe": RECIPES.get(args.stage), "torch": torch.__version__,
              "source_hashes": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}})
        if args.stage == "cache":
            cache_targets(args, splits, manifest)
        elif args.stage == "evaluate":
            evaluate(args, splits, manifest)
        else:
            train(args, splits, manifest)
    except BaseException as exc:
        write(args.output / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
