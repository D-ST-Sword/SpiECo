import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from spieco import AVOnly, Student, Teacher
from spieco import cli
from spieco.data import FeatureSplit, SIZES, load_splits
from spieco.masking import erase
from spieco.metrics import metrics
from spieco.normalization import normalize_records
from spieco.objectives import consolidation_loss
from spieco.spiking import TemporalAdapter
from spieco.utils import seed_all


class TinyBackbone(nn.Embedding):
    def gradient_checkpointing_enable(self):
        pass


class TinyText(nn.Module):
    def __init__(self, width=1024):
        super().__init__()
        self.model = TinyBackbone(128, width)

    def forward(self, tokens):
        return self.model(tokens[:, 0].long())


def batch(dataset="mosi"):
    dims = {"mosi": (5, 20), "urfunny": (81, 371), "sims": (33, 709)}[dataset]
    tokens = torch.tensor([[[0, 8, 9, 2, 1], [1, 1, 1, 1, 0]]] * 2)
    return {"id": ["a", "b"], "text": tokens, "target": torch.tensor([-1., 1.]),
            "audio": torch.randn(2, 6, dims[0]), "vision": torch.randn(2, 4, dims[1]),
            "audio_mask": torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool),
            "vision_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)}


class ModelsTest(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_three_dataset_av_backward_and_no_text(self):
        for dataset in ("urfunny", "mosi", "sims"):
            model = AVOnly(dataset).eval()
            sample = batch(dataset)
            text_free = {k: v for k, v in sample.items() if k != "text"}
            first = model(text_free)["M"]
            torch.testing.assert_close(first, model({**sample, "text": None})["M"], rtol=0, atol=0)
            first.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            forbidden = {"text", "text_encoder", "text_private", "pretrained_text_encoder"}
            self.assertFalse(any(forbidden.intersection(name.split(".")) for name, _ in model.named_parameters()))

    def test_consolidation_and_export(self):
        teacher = Teacher("mosi", encoder=TinyText()).eval()
        student = Student("mosi", teacher).train()
        self.assertFalse(student.text_encoder.training)
        sample = batch()
        with torch.no_grad():
            targets = teacher(sample)
        output = student(sample, torch.tensor([0., .5]))
        loss, _ = consolidation_loss(output, sample["target"], targets["M"], targets["latent"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in student.text_encoder.parameters()))
        student.eval()
        export = AVOnly("mosi").eval()
        export.load_state_dict(student.av.state_dict(), strict=True)
        with patch.object(student.text_encoder, "forward", side_effect=AssertionError("text accessed")):
            actual = student({k: v for k, v in sample.items() if k != "text"}, torch.zeros(2))["M"]
        torch.testing.assert_close(actual, export(sample)["M"], atol=0, rtol=0)

    def test_spike_padding_and_reset(self):
        layer = TemporalAdapter(dim=8).eval()
        x = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        first = layer(x, mask)
        torch.testing.assert_close(first, layer(x, mask), rtol=0, atol=0)
        padded = torch.cat([x, torch.randn(2, 5, 8)], 1)
        extended = torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], 1)
        torch.testing.assert_close(first, layer(padded, extended)[:, :4], rtol=0, atol=0)
        self.assertEqual(layer(padded, extended)[:, 4:].count_nonzero().item(), 0)

    def test_av_padding(self):
        model = AVOnly("mosi").eval()
        sample = batch()
        expected = model(sample)["M"]
        changed = copy.deepcopy(sample)
        for m in ("audio", "vision"):
            changed[m][~changed[m + "_mask"]] = 12345
        torch.testing.assert_close(expected, model(changed)["M"], atol=0, rtol=0)


class ProtocolTest(unittest.TestCase):
    def test_token_mask_protocols(self):
        for special, pad, content in (((0, 1, 2), 1, [0, 8, 9, 2, 1]), ((0, 101, 102), 0, [101, 8, 9, 102, 0])):
            tokens = torch.tensor([[content, [1, 1, 1, 1, 0]]])
            full, rho, counts, removed = erase(tokens, ["sample"], 1., 42, special, pad)
            self.assertEqual(rho.item(), 0)
            self.assertEqual(counts.item(), removed.item())
            self.assertEqual(full[0, 1].tolist(), [1, 0, 0, 1, 0])
            keep, rho, _, _ = erase(tokens, ["sample"], 0., 42, special, pad)
            torch.testing.assert_close(tokens, keep)
            self.assertEqual(rho.item(), 1)

    def test_metrics_boundaries(self):
        score = metrics("mosi", [-1, 0, 1], [-1, 3, 1])
        self.assertEqual(score["n_samples"], 3)
        self.assertEqual(score["classification_samples"], 2)
        self.assertEqual(score["accuracy"], 1)
        score = metrics("sims", [-1., -.7, -.1, 0, .1, .7, 1.], [-2., -.7, -.1, 0, .1, .7, 2.])
        self.assertEqual(score["mae"], 0)
        self.assertEqual(score["acc_3"], 1)
        self.assertEqual(score["acc_5"], 1)
        self.assertLess(metrics("sims", [.1, -.1], [.1001, -.0999])["acc_3"], 1)
        self.assertEqual(metrics("urfunny", [-1, 1], [0, 1])["accuracy"], 1)

    def test_training_only_normalization(self):
        records = {s: {m: np.array([[[1.], [3.], [999.]]]) for m in ("audio", "vision")} for s in ("train", "valid", "test")}
        for r in records.values():
            r.update(audio_lengths=np.array([2]), vision_lengths=np.array([2]))
        records["test"]["audio"][:] = 500
        stats = normalize_records(records)
        self.assertEqual(stats["audio"]["mean"], [2.])
        np.testing.assert_array_equal(records["train"]["audio"].reshape(-1), [-1, 1, 0])

    def test_reject_wrong_labels(self):
        r = {"id": ["a", "b"], "regression_labels": [0, 1]}
        for m, width in (("audio", 81), ("vision", 371)):
            r[m] = np.ones((2, 2, width))
            r[m + "_lengths"] = [2, 2]
        with self.assertRaisesRegex(ValueError, "signed"):
            FeatureSplit(r, "urfunny")

    def test_cache_refuses_validation_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            teacher = p / "teacher.pt"
            teacher.write_bytes(b"synthetic-checkpoint-not-real")
            np.savez(p / "training_targets.npz", id=np.array(["train-a"]), prediction=np.zeros(1, np.float32), latent=np.zeros((1, 128), np.float32))
            cli.write(p / "complete.json", dict(split="train_only", teacher_sha256=cli.sha(teacher), data_sha256="x", targets_sha256=cli.sha(p / "training_targets.npz")))
            cli.TrainingTargets(p, ["train-a"], teacher, "x")
            with self.assertRaisesRegex(ValueError, "training IDs"):
                cli.TrainingTargets(p, ["valid-a"], teacher, "x")


class WorkflowTest(unittest.TestCase):
    def test_five_stage_cpu_smoke(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = {}
            for split in ("train", "valid", "test"):
                records[split] = dict(id=[split + "-a", split + "-b"], raw_text=["negative", "positive"],
                    regression_labels=np.array([-1., 1.]), audio=np.random.randn(2, 4, 5),
                    vision=np.random.randn(2, 3, 20), audio_lengths=[4, 3], vision_lengths=[3, 2])
            with (root / "features.pkl").open("wb") as stream:
                pickle.dump(records, stream)

            def load(path, dataset, encoder_path=None):
                with patch.dict(SIZES, {"mosi": (2, 2, 2)}):
                    sets, manifest = load_splits(path, dataset)
                if encoder_path:
                    for ds in sets.values():
                        ds.tokens = batch()["text"]
                return sets, manifest

            def teacher(dataset, path):
                return Teacher(dataset, encoder=TinyText())

            recipes = {stage: {**cfg, "epochs": 1} for stage, cfg in cli.RECIPES.items()}
            common = ["--dataset", "mosi", "--data", str(root / "features.pkl"), "--device", "cpu"]
            with patch.object(cli, "Teacher", side_effect=teacher), patch.object(cli, "load_splits", side_effect=load), patch.dict(cli.RECIPES, recipes):
                cli.main(["teacher", *common, "--encoder", "synthetic", "--output", str(root / "teacher")])
                source = ["--encoder", "synthetic", "--teacher", str(root / "teacher/selected.pt")]
                cli.main(["cache", *common, *source, "--output", str(root / "cache")])
                cli.main(["student", *common, *source, "--cache", str(root / "cache"), "--output", str(root / "student")])
                cli.main(["refine", *common, "--checkpoint", str(root / "student/av_only.pt"), "--output", str(root / "refine")])
                cli.main(["evaluate", *common, "--checkpoint", str(root / "refine/selected.pt"), "--output", str(root / "eval")])
            self.assertFalse(json.loads((root / "refine/trained.json").read_text())["test_evaluated"])
            self.assertFalse(json.loads((root / "eval/evaluation.json").read_text())["requires_text"])
            self.assertEqual(json.loads((root / "eval/metrics.json").read_text())["n_samples"], 2)


if __name__ == "__main__":
    unittest.main()
