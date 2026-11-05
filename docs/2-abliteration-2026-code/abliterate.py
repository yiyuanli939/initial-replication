"""abliterate.py - refusal-direction ablation with plain Transformers + PyTorch hooks.

Everything here is a move on the residual stream:
  READ      last_token_residuals      what the stream holds at the last prompt token, every layer
  CHOOSE    refusal_directions        difference of means -> one candidate direction per layer
  SUBTRACT  ablate                    no writer may put r into the stream (runtime hooks)
  ADD       add_at                    push the stream along v at one layer (steering)
  BAKE      orthogonalize_checkpoint  the SUBTRACT move written into the weights, shard by shard

Everywhere, r may be one direction [d] or a few stacked directions [k, d].
"""
from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# A "writer" is any weight whose output is ADDED to the residual stream.
# These names cover Llama / Qwen2-3.5 / Mistral / OLMo / gpt-oss style models.
WRITER_RE = re.compile(r"(embed_tokens|o_proj|out_proj|down_proj)(\.weight|\.bias)?$")
SKIP_RE = re.compile(r"(visual|vision|audio)")  # other towers have their own residual streams


# ---------------------------------------------------------------- 1. prompts
def format_prompts(tok, instructions: list[str]) -> list[str]:
    """Wrap each instruction in the chat template, ending where the reply starts."""
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ignored by templates without a thinking switch
        )
        for x in instructions
    ]


# ---------------------------------------------------------------- 2. read the stream
@torch.inference_mode()
def last_token_residuals(model, tok, prompts: list[str], batch_size: int = 32) -> torch.Tensor:
    """Residual-stream vector at the LAST prompt token, for every prompt and layer.

    Returns float32 CPU tensor [n_layers + 1, n_prompts, d_model]. Row i is the
    input to layer i (row 0 = embeddings; the last row has the final norm applied).
    """
    tok.padding_side = "left"  # so position -1 is always the last real token
    chunks = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        out = model(**batch, output_hidden_states=True)
        chunks.append(torch.stack(out.hidden_states)[:, :, -1, :].float().cpu())
    return torch.cat(chunks, dim=1)


# ---------------------------------------------------------------- 3. directions
def refusal_directions(h_harmful: torch.Tensor, h_harmless: torch.Tensor) -> torch.Tensor:
    """One unit-length candidate per layer: normalize(mean_harmful - mean_harmless). -> [L+1, d]"""
    diff = h_harmful.mean(dim=1) - h_harmless.mean(dim=1)
    return diff / diff.norm(dim=-1, keepdim=True)


def orthonormal(r: torch.Tensor) -> torch.Tensor:
    """[d] or [k, d] -> [k, d] with orthonormal rows spanning the same directions."""
    q, _ = torch.linalg.qr(r.reshape(-1, r.shape[-1]).float().T)
    return q.T


def project_out(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """x - (x . r) r for every orthonormal row r of Q (a single unit vector also works)."""
    Q = Q.to(x.device, torch.float32).reshape(-1, x.shape[-1])
    return x - ((x.float() @ Q.T) @ Q).to(x.dtype)        # project in fp32, keep the stream's dtype


# ---------------------------------------------------------------- 4. test it live
def writer_modules(model):
    for name, mod in model.named_modules():
        if WRITER_RE.search(name) and not SKIP_RE.search(name):
            yield name, mod


@contextlib.contextmanager
def ablate(model, r: torch.Tensor):
    """Inside this block, no writer can put anything along r into the residual stream."""
    Q = orthonormal(r).to(model.device)
    hooks = [
        mod.register_forward_hook(lambda m, inp, out: project_out(out, Q))
        for _, mod in writer_modules(model)
    ]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def decoder_layers(model):
    """The text model's list of decoder layers (works inside multimodal wrappers too)."""
    return next(m for n, m in model.named_modules()
                if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and not SKIP_RE.search(n))


@contextlib.contextmanager
def add_at(model, layer: int, v: torch.Tensor):
    """Inside this block, v is added to the stream at the input of `layer`, at every position."""
    def pre(mod, args, kwargs):
        if args:
            return (args[0] + v.to(args[0].dtype), *args[1:]), kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + v.to(kwargs["hidden_states"].dtype)
        return args, kwargs
    h = decoder_layers(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------- 5. bake it in
def orthogonalize_(name: str, w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Edit one tensor so its contribution to the stream has no r component.

    nn.Linear weight [d_model, d_in]  : W <- W - r (r^T W)
    nn.Embedding weight [vocab, d_model]: E <- E - (E r) r^T
    bias [d_model]                    : b <- b - (b . r) r
    (with Q = orthonormal(r) the same formulas cover k directions at once)
    """
    w32, Q = w.float(), orthonormal(r).to(w.device)
    if name.endswith("embed_tokens.weight") or w.ndim == 1:
        w32 = project_out(w32, Q)
    else:
        w32 = w32 - Q.T @ (Q @ w32)
    return w32.to(w.dtype)


def is_writer(name: str, shape, d_model: int) -> bool:
    if not WRITER_RE.search(name) or SKIP_RE.search(name):
        return False
    if name.endswith("embed_tokens.weight"):
        return shape[-1] == d_model          # [vocab, d_model]
    return shape[0] == d_model               # [d_model, d_in] or bias [d_model]


def orthogonalize_checkpoint(src: str | Path, dst: str | Path, r: torch.Tensor, device="cuda") -> list[str]:
    """Stream every *.safetensors shard from src to dst, editing writer tensors.

    Peak memory = one shard, so this works for checkpoints far larger than the GPU.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    d_model = r.shape[-1]
    edited = []
    for f in sorted(src.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy(f, dst / f.name)  # config, tokenizer, index, template ...
            continue
        tensors = load_file(f, device=device)
        for name, w in tensors.items():
            if is_writer(name, w.shape, d_model):
                tensors[name] = orthogonalize_(name, w, r)
                edited.append(name)
        save_file({k: v.contiguous() for k, v in tensors.items()}, dst / f.name, metadata={"format": "pt"})
        del tensors
    return edited
