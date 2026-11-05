"""Offline smoke test: native Transformers, TransformerBridge, and NNsight.

Uses identical tiny random Llama weights and token IDs. This checks the tooling,
not the paper's refusal results or the complete all-layer ablation procedure.
"""

from copy import deepcopy
from importlib.metadata import version

import nnsight
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformer_lens.model_bridge import TransformerBridge
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def main():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    native = LlamaForCausalLM(config).eval()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"token{i}": i for i in range(3, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        name_or_path="offline-tiny-llama",
        add_bos_token=True,
    )
    bridge = TransformerBridge.boot_transformers(
        "offline-tiny-llama",
        hf_model=deepcopy(native),
        tokenizer=tokenizer,
        device="cpu",
        dtype=torch.float32,
    ).eval()
    traced = nnsight.NNsight(deepcopy(native))
    tokens = torch.tensor([[1, 4, 5, 6]])
    direction = torch.randn(config.hidden_size)
    direction /= direction.norm()

    def project(h):
        return h - (h * direction).sum(dim=-1, keepdim=True) * direction

    native_cache = {}

    def capture(module, inputs, output):
        native_cache["residual"] = output.detach().clone()

    with torch.inference_mode():
        handle = native.model.layers[0].register_forward_hook(capture)
        try:
            baseline = native(tokens, use_cache=False).logits
        finally:
            handle.remove()

        logits, cache = bridge.run_with_cache(
            tokens, names_filter="blocks.0.hook_out"
        )
        torch.testing.assert_close(logits, baseline, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            cache["blocks.0.hook_out"], native_cache["residual"],
            atol=1e-5, rtol=1e-4,
        )

        handle = native.model.layers[0].register_forward_hook(
            lambda module, inputs, output: project(output)
        )
        try:
            expected = native(tokens, use_cache=False).logits
        finally:
            handle.remove()

        modified = bridge.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_out", lambda h, hook: project(h))],
        )
        assert not torch.allclose(expected, baseline), "Intervention had no effect"
        torch.testing.assert_close(modified, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(bridge(tokens), baseline, atol=1e-5, rtol=1e-4)

        with traced.trace(tokens, use_cache=False):
            nn_residual = nnsight.save(traced.model.layers[0].output)
            nn_baseline = nnsight.save(traced.output.logits)
        torch.testing.assert_close(nn_baseline, baseline)
        torch.testing.assert_close(nn_residual, native_cache["residual"])

        with traced.trace(tokens, use_cache=False):
            h = traced.model.layers[0].output
            traced.model.layers[0].output = project(h)
            nn_modified = nnsight.save(traced.output.logits)
        torch.testing.assert_close(nn_modified, expected, atol=1e-5, rtol=1e-4)

    for package in ("transformer-lens", "transformers", "nnsight"):
        print(f"{package}: {version(package)}")
    print("PASS: baseline logits, residual capture, projection intervention, hook cleanup")


if __name__ == "__main__":
    main()
