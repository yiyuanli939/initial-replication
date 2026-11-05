# Code for "Abliteration in 2026: a dance on the residual stream"

Everything here ran on one NVIDIA RTX PRO 6000 (96 GB) on 2026-11-04/05.
Section 9 of `../2-Abliteration-2026.pdf` lists the run order, the outputs and the measured times.

## Environment

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "torch>=2.9" "transformers>=5" accelerate datasets safetensors matplotlib nnsight reportlab pyphen
# used: torch 2.15.0+cu132, transformers 5.22.1, datasets 5.1.0, nnsight 0.8.2, reportlab 5.1, matplotlib 3.11, Python 3.12
```

The repo's own `.venv` (transformers 4.51.3, TransformerLens 2.x) cannot load Qwen3.5. Use a separate environment.
No TeX installation is needed: all math (in the PDF text and in the figures) is written as TeX and typeset by
matplotlib's mathtext in Computer Modern. `pyphen` is optional; it only enables hyphenation of justified text.

## Library (the moves on the residual stream)

| file | what it does |
|---|---|
| `abliterate.py` | READ (`last_token_residuals`), CHOOSE (`refusal_directions`), SUBTRACT (`ablate`), ADD (`add_at`), BAKE (`orthogonalize_checkpoint`) |
| `common.py` | data splits, generation, keyword check, LLM judge prompt, KL |
| `cosmic.py` | COSMIC direction selection (Siu et al., ACL Findings 2025) |
| `rdo.py` | Refusal Direction Optimization (Wollschläger et al., ICML 2025) |
| `som.py` | SOM multi-direction ablation (Piras et al., AAAI 2026), sequential and span versions |
| `audit.py` | diff a checkpoint against its base model |

## Experiments (run from this folder, in order)

```bash
python experiment.py      # directions, keyword sweep, GSM8K baseline
python experiment0.py     # Section 0: the stream is a sum (bf16 + fp32 check), writes along r22, token x layer grid
python experiment2.py     # judged single-layer sweep
python experiment3.py     # multi-direction candidates, test eval, 3-direction bake, thinking mode
python experiment4.py     # minimal pairs, first tokens, per-item judge labels
python audit.py ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<hash> qwen3.5-9b-abliterated
python experiment5a.py    # padding check, per-direction gaps, COSMIC
python experiment5b.py    # RDO, SOM, WRDO banks
python experiment5c.py    # every method on the same test set
python experiment5d.py    # SOM selected with the span operator; WRDO-family at alpha 2, 3
python nn_demo.py 22      # nnsight cross-check
```

## Figures and the PDF

```bash
python figs.py; python figs2.py; python figs3.py    # figs/*.png (figs3.py also writes results/figs3_stats.json)
python report.py out.pdf                            # two passes: the second fills in the contents page numbers
```

| file | what it does |
|---|---|
| `figstyle.py` | shared figure style; one colour meaning everywhere (orange = harmful / refusal direction, blue = harmless / stream, violet = this guide, aqua = papers, greys = original and 2024 recipe) |
| `figs.py`, `figs2.py`, `figs3.py` | figures for Sections 0-2, 3-4 and 5-6 |
| `labels.py` | readable names and colour families for every method on the test set (stages 5c + 5d) |
| `report_style.py` | page design: fonts, inline/display LaTeX math, boxes, highlighted code, floats, contents |
| `report.py` | build driver; text in `report_body.py` (cover, Sections 0-4), `report_papers.py` (Section 5), `report_tail.py` (6-9, appendices), `report_takeaways.py` (result-dependent sentences) |

`experiment3.py` writes an abliterated checkpoint (`qwen3.5-9b-abliterated/`, 19 GB). It is a model with its refusals
removed. Keep it local, and delete it when you are done.
