# Initinal Replication of the Abliteration Project 


This is the initial discovery of the abliteration practice. We want to do that in my personal machine.

## Environment

Use `uv sync --locked` to install the versions recorded in `uv.lock`.
The project now uses TransformerLens 4's `TransformerBridge` API with
Transformers 5. NNsight is also installed for comparison.

```python
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens import utilities as utils  # only if helpers are needed

# Replace model_id with the exact checkpoint selected for the experiment.
# model = TransformerBridge.boot_transformers(model_id)
```

`HookedTransformer` and `transformer_lens.utils` were removed in TransformerLens 4.
The new workflow does not use `transformers_stream_generator`: do not import it.
It may still be installed as a transitive dependency, but its legacy generation
API is incompatible with Transformers 5.

## Verify the interpretability tools

```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_interpretability.py
```

This CPU check constructs identical tiny random Llama models without downloading
checkpoints. It compares native Transformers, TransformerBridge, and NNsight on
baseline logits, residual activation capture, and a single-block projection
intervention. It also checks that temporary bridge hooks are removed. This is
a tooling check, not a replication of refusal behavior or an all-layer ablation.

## Replication considerations

TransformerBridge preserves raw Hugging Face weights by default. Its optional
`enable_compatibility_mode()` reproduces older HookedTransformer weight processing
(such as LayerNorm folding and weight centering); it does not automatically
reproduce the original paper's configuration. Extract and apply a refusal
direction under the same weight-processing settings.

For a controlled replication, record the checkpoint and tokenizer revisions,
prompt/chat formatting, dataset splits, activation locations and token positions,
direction-selection procedure, precision, generation settings, and evaluation
metrics. Verify intervention coverage during both prompt processing and decoding.
Using a different small model tests the method on that model; it does not reproduce
the paper's exact model-specific scores.

References: [TransformerLens migration guide](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v4.html)
and [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717).
