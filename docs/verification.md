# Verification

The code-only packaging check uses synthetic inputs, not private benchmark
data. Run the portable suite with:

```bash
python -m unittest discover -s tests -v
```

Coverage:

- AV-only forward/backward for all three feature layouts.
- Frozen text backbone during consolidation; no text access at zero retention.
- Exact AV export parity with the consolidated student endpoint.
- Padded-feature invariance and spiking state reset between forwards.
- English and Chinese content-token erasure.
- MOSI Non0 and CH-SIMS clipping/class boundary conventions.
- Training-only normalization and teacher target ID validation.
- Five-stage CPU smoke workflow with a tiny synthetic text encoder and one
  epoch per training stage, followed by explicit held-out evaluation.

The maintainer also checks released teacher and student computations against
the archived implementation on UR-FUNNY, MOSI and CH-SIMS feature dimensions:
initial parameter tensors, forward outputs, consolidation loss and parameter
gradients. On the 2026-10-26 CPU packaging check, all three layouts matched
exactly: maximum output difference 0, matching initialization, loss and
parameter gradients. All ten portable tests passed. This comparison uses
synthetic inputs and is not a claim of rerunning the manuscript experiments.

Data, weights, generated predictions and machine-specific experiment runners
are intentionally absent. The public suite needs no pretrained text download.
PyTorch and SpikingJelly remain runtime dependencies for these checks.
