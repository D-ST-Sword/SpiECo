# SpiECo Training Protocol

The three tasks use one scalar regression predictor with dataset-specific
classification metrics derived from the predicted score. The main method has
three training stages and a separate train-only cache operation.

## Complete-Input Teacher

Text uses a finetuned pretrained encoder, projection/normalization and one
Transformer layer. Audio/video use projection/normalization and bidirectional
LSTMs. All private states have width 128. Audio and video each receive a
channel-wise parametric LIF residual adapter. The TA, TV and AV pair encoders
use gated bidirectional cross-attention. Their concatenation feeds the teacher
fusion layer and one TAV predictor.

PLIF settings: at most eight valid-rank temporal bins; initial tau 10;
threshold 1; subtractive reset; `decay_input=False`; detached reset gradient;
ATan surrogate with alpha 2; residual scale 0.1. State resets between forwards.
The code uses continuous residual features as well as spikes; it is not an
event-only network and does not by itself establish hardware energy savings.

Teacher optimizer: AdamW, backbone lr 1e-5, adapters lr 1e-4, weight decay
1e-4, batch 32, gradient norm limit 1. Cosine decay over 40 epochs to 1e-7;
minimum 20 epochs, patience 20. Loss and selection: L1 and validation MAE.
CUDA mixed precision is scoped to the text backbone; gradient scaling handles
teacher training. Frozen unused predictor slots remain in the teacher state
for historical checkpoint-key and random initialization compatibility; they
never contribute to the teacher output or deployed student.

## Privileged Consolidation

The teacher provides predictions and 128-dimensional fused representations
only for training samples. Its parameters initialize the relevant student
components. The deployed branch combines pooled private A/V states and their
AV interaction into a 128-dimensional latent representation and a scalar
prediction. A training-only text-assisted branch supports consolidation.

Loss:

```
L = L1(assisted_prediction, target)
    + 0.5 * L1(AV_prediction, target)
    + 0.5 * SmoothL1(AV_prediction, teacher_prediction)
    + 0.1 * mean(1 - cosine(AV_latent, teacher_latent))
```

Per-sample content erasure rates are drawn uniformly from 0, 0.1, ..., 1.0.
Masks are deterministic given sample ID, seed and epoch. The auxiliary mixture
uses the realized fraction of retained content tokens. The pretrained text
backbone is frozen and kept in evaluation mode; the auxiliary private/fusion
modules remain trainable. This auxiliary mixture is **not** a deployment
missing-rate-dependent ensemble.

Student optimizer: AdamW lr 1e-4, weight decay 1e-4, batch 32, norm limit 1,
40-epoch cosine decay to 1e-6, minimum 20 epochs and patience 15. Selection
minimizes mean validation MAE at auxiliary erasure rates 0, 0.5 and 1.
The AV export is checked against the fully text-erased student output.

## AV-Only Refinement

Start from the consolidated AV export. Set dropout and attention dropout to
0.5; use L1 supervision only, no teacher targets. AdamW lr 3e-5, weight decay
0.1, batch 64, gradient norm limit 1, 40 epochs with cosine decay to 1e-6.
Select validation accuracy, weighted F1, then negative binary logistic loss
lexicographically. CH-SIMS selection uses raw regression MAE for reporting;
final CH-SIMS benchmark MAE/Corr use clipped [-1, 1] predictions.

Zero-strength channel/frame/noise augmentation calls are retained to preserve
the original recipe's random-number consumption. No effective corruption is
applied by those calls. Test data are not used in checkpoint selection.

## Evaluation

- UR-FUNNY: MAE, Pearson correlation, binary accuracy, weighted F1; raw scalar
  predictions, positive iff score > 0, signed targets -1/+1.
- MOSI: MAE, correlation and Acc-7 on all 686 examples; binary accuracy/F1 on
  the 656 non-neutral examples, positive iff score > 0. Acc-7 rounds scores
  clipped to [-3, 3].
- CH-SIMS: MAE, correlation, Acc-2, weighted F1, Acc-3, Acc-5; scores clipped
  to [-1, 1]. Zero is nonpositive. Acc-3 boundaries are -0.1/0.1; Acc-5
  boundaries are -0.7/-0.1/0.1/0.7. Boundary values go to the lower interval.

Accuracy/F1 values in JSON are fractions. Convert to percentages for tables.
Use all five predeclared seeds (7, 22, 42, 93, 99); report their mean and sample
standard deviation (`ddof=1`), not an independently selected best seed.

## Portable Release Versus Archived Runs

The shared model operations, initialization order, parameter names and loss
formula are preserved and checked against the archived implementation.
The public command-line layer consolidates several historical entrypoints;
its path management, diagnostics and explicit evaluation commands are new.
The package has not been benchmark-retrained across five seeds. It should not
be described as a bitwise replay of every historical run. Full numeric
reproduction additionally depends on identical feature/tokenizer assets,
pretrained weights, hardware, kernels and execution order.
