# SpiECo

**Privileged Knowledge Consolidation through Spiking Evidence Dynamics for
Audiovisual Affective Understanding**

Jiahao Qin and Yiwen Wang

Implementation of SpiECo (Spiking Evidence Consolidation). Complete
text--audio--video observations support training; the deployed student takes
**audio and video only**, with no text encoder or teacher at inference.

This repository is private during manuscript preparation. Public access is
planned after manuscript submission. This initial release contains code and
documentation only: **no datasets, pretrained weights, trained checkpoints,
per-sample predictions, or baseline implementations**.

## Implementation

- Private modality encoders and bidirectional audiovisual attention.
- Residual channel-wise parametric LIF adapters using SpikingJelly.
- Complete-input teacher and training-only prediction/latent target caching.
- Supervised audiovisual consolidation with prediction and latent transfer.
- Final AV-only refinement, validation-based selection, and text-free evaluation.

The training-time text-assisted branch is not the deployed predictor. It is
removed when exporting `av_only.pt`; final AV refinement produces another
AV-only checkpoint. The package does not implement competing methods.

## Setup

Use Python 3.10 or newer in a separate environment. Install a PyTorch build
compatible with your GPU/driver first, then install this repository:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m spieco.cli --help
```

SpikingJelly is pinned to `0.0.0.0.14`; the adapters use its PyTorch backend,
not a custom CUDA kernel. Code packaging was tested on CPU with PyTorch
2.0.1, Transformers 4.35.0, NumPy 1.26.4 and scikit-learn 1.7.2.
This records the software check, not a recommended CUDA build for every GPU.

## Data and Text Encoders

Obtain each dataset and pretrained encoder under its original access conditions.
The input is a **trusted local prepared feature file**, not raw video or audio.
See [the data contract](docs/data.md) before running the commands.

| CLI dataset | Text encoder for teacher/consolidation only | Text / audio / visual widths |
|---|---|---|
| `urfunny` | `roberta-large` | 1024 / 81 / 371 |
| `mosi` | `roberta-large` | 1024 / 5 / 20 |
| `sims` (CH-SIMS) | `hfl/chinese-roberta-wwm-ext` | 768 / 33 / 709 |

Provide the pretrained model and tokenizer in a local directory. Model loading
uses `local_files_only=True`; the scripts do not download assets implicitly.
CH-SIMS uses its Chinese tokenizer and training-frame normalization, not an
English text encoder. Only multimodal targets are used, not unimodal labels.

## Training and Evaluation

Example for MOSI, seed 42. Replace the local paths with your own.
Run each stage in order. Outputs must be new directories; existing runs are
never silently overwritten. The manuscript uses seeds **7, 22, 42, 93, 99**.

```bash
spieco teacher --dataset mosi --seed 42 --data data/mosi.pkl \
  --encoder pretrained/roberta-large --output runs/mosi_s42/teacher

spieco cache --dataset mosi --seed 42 --data data/mosi.pkl \
  --encoder pretrained/roberta-large \
  --teacher runs/mosi_s42/teacher/selected.pt --output runs/mosi_s42/cache

spieco student --dataset mosi --seed 42 --data data/mosi.pkl \
  --encoder pretrained/roberta-large \
  --teacher runs/mosi_s42/teacher/selected.pt --cache runs/mosi_s42/cache \
  --output runs/mosi_s42/student

spieco refine --dataset mosi --seed 42 --data data/mosi.pkl \
  --checkpoint runs/mosi_s42/student/av_only.pt --output runs/mosi_s42/refine

spieco evaluate --dataset mosi --seed 42 --data data/mosi.pkl \
  --checkpoint runs/mosi_s42/refine/selected.pt --split test \
  --output runs/mosi_s42/test
```

Use `--device cuda:0` (default `cuda`) or `--device cpu`. Training text encoders
on CPU is not practical; CPU is useful for the small synthetic tests.
Evaluation and refinement do not require a text encoder or raw text fields.
Teacher targets are cached for **training IDs only** and bound to the teacher
checkpoint and data-file hashes. Selection uses validation data; test evaluation
is a separate command. See [the protocol](docs/protocol.md).

## AV-Only Inference API

```python
import torch
from spieco import AVOnly

model = AVOnly("mosi").eval()
checkpoint = torch.load("runs/mosi_s42/refine/selected.pt",
                        map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
with torch.no_grad():
    output = model({
        "audio": audio_features,       # float tensor [B, La, 5]
        "vision": visual_features,     # float tensor [B, Lv, 20]
        "audio_mask": audio_mask,      # bool tensor [B, La]
        "vision_mask": visual_mask,    # bool tensor [B, Lv]
    })
    prediction = output["M"]          # [B]
```

Use the same feature preprocessing as training, including saved CH-SIMS
normalization statistics. Load only trusted checkpoint and pickle files.

## Verification and Scope

The model computations were extracted from the experiment implementation and
the command-line orchestration was reorganized for portable paths. Tests cover
padding/state reset, consolidation, gradient propagation, text-free export,
Chinese masking, metrics and a synthetic five-stage workflow. Additional local
parity checks compare the released teacher/student against archived computations
on all three feature layouts. See [verification details](docs/verification.md).

This packaging work does **not** constitute a new five-seed benchmark rerun.
Different hardware, software kernels and data preparation may change numerical
results. The prepared datasets and checkpoints are not distributed in this
initial code-only release.
