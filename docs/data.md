# Prepared Feature Contract

The code expects a trusted pickle mapping with three entries: `train`, `valid`,
and `test`. Each entry contains:

| Field | Shape/type | Meaning |
|---|---|---|
| `id` | length N | Unique sample IDs, disjoint across splits |
| `raw_text` | length N | Raw strings; only required for teacher/cache/student stages |
| `regression_labels` | N or N x 1 | Multimodal task targets |
| `audio` | N x La x Da, float | Prepared unaligned audio feature sequences |
| `vision` | N x Lv x Dv, float | Prepared unaligned visual feature sequences |
| `audio_lengths` | N, integer | Number of valid leading audio frames |
| `vision_lengths` | N, integer | Number of valid leading visual frames |

All values must be finite. Lengths must be positive and cannot exceed the
stored sequence length. Padding is trailing; masks are constructed from lengths.
Do not silently substitute feature extractors or shuffle the published splits.

| Dataset | Train | Validation | Test | Targets |
|---|---:|---:|---:|---|
| UR-FUNNY | 7614 | 980 | 994 | -1 (not humorous), +1 (humorous) |
| MOSI | 1284 | 229 | 686 | Sentiment intensity in [-3, 3] |
| CH-SIMS | 1368 | 456 | 457 | Multimodal sentiment intensity in [-1, 1] |

These counts describe the prepared feature protocol used by this project,
not every release/variant of the original datasets. Raw-data download or feature
extraction scripts are not bundled. Users must obtain compatible features under
the dataset providers' access conditions; matching dimensions alone is not
sufficient to establish feature equivalence.

English text is tokenized by the local RoBERTa-large tokenizer. Chinese text is
tokenized by the local Chinese RoBERTa-wwm-ext tokenizer. Both use a maximum
length of 64, truncation and fixed-length padding. Tokens have shape N x 2 x 64
(input IDs, attention mask). Masking preserves English special IDs 0/1/2 and
Chinese special IDs 0/101/102. The Chinese padding token is 0, not 1.

CH-SIMS features are z-scored with per-feature means and standard deviations
computed from **valid training frames only** (standard deviation floor 1e-5).
Padded frames are reset to zero. Do not pre-normalize the file and then apply
this operation a second time. Statistics are saved in each data manifest.
UR-FUNNY and MOSI use their prepared feature values without this extra step.

The AV-only evaluator does not construct text tokens. It reads all split
records to verify IDs and, for CH-SIMS, fit/replay training-only normalization;
it runs the model only on the requested split. For separate deployment data,
reuse the training statistics instead of fitting them on deployment samples.

No data are included in the repository, even synthetic binary fixtures.
Unit tests construct tiny examples in temporary directories at runtime.
