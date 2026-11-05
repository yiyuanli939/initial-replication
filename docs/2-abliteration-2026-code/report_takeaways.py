# Result-dependent sentences for Section 5 (exec'd in report.py's namespace). Every number is read from results/.
import os as _os
R5d = J(_os.environ.get("R5D", "results5d.json"))
_t = lambda p: T5[tkey(p)]
_rdo, _stack, _l22, _l23, _cos = _t("rdo"), _t("dim stack"), _t("dim_l22"), _t("dim_l23"), _t("cosmic")
_seq, _span = _t("som sequential"), _t("som span")

RDO_TAKEAWAY = (
    f"<b>Takeaway.</b> One trained direction does what three averaged ones do: {pct(_rdo['harmful_comply'])} compliance "
    f"against {pct(_stack['harmful_comply'])} for the stack, at about a third of the KL ({_rdo['kl']:.2f} vs "
    f"{_stack['kl']:.2f}). Two cautions. First, its answer targets came from our 3-direction model, so part of this "
    "is distillation: RDO compressed a working edit into one cleaner direction rather than finding compliance on its "
    "own. Second, the method is only as good as its targets. With the paper’s recipe on this model, targets from the "
    f"single-direction ablation would mostly be deflections (layer 23: {pct(_l23['harmful_comply'])} real compliance).")

COSMIC_TAKEAWAY = (
    "Its score could not have told us that layer 23 fails. But its search over <i>token positions</i> found a single "
    f"direction that beats every last-token direction we tried ({pct(_cos['harmful_comply'])} vs "
    f"{pct(_l22['harmful_comply'])} for the best, layer 22), with the lowest KL of any method here ({_cos['kl']:.2f}). "
    "<b>Takeaway:</b> use COSMIC to search widely and cheaply, then confirm the winners with a judge.")

_dt = R5d["test"]
_vs = {t["k"]: t for t in R5d["trace"]}
_vbest = max(R5d["trace"], key=lambda t: (t["comply"], -t["kl"]))
SOM_TAKEAWAY = (
    "The set above was chosen with the sequential operator, which favours sequential removal. For a fair comparison we "
    f"also ran the greedy search with the span operator. It picks {R5d['set']} "
    f"({pct(_vbest['comply'])} on validation), which gives <b>{pct(_dt['harmful_comply'])}</b> on test at KL "
    f"{_dt['kl']:.2f}. "
    + ("So on this model the exact span projection did not beat the repo’s sequential version. Removing the full span of "
       "aligned directions also removes more of what harmless prompts need (higher KL). "
       if _dt["harmful_comply"] <= _seq["harmful_comply"] else
       f"Chosen with the matching operator, span removal complies more ({pct(_dt['harmful_comply'])} vs "
       f"{pct(_seq['harmful_comply'])}) but moves harmless outputs more (KL {_dt['kl']:.2f} vs {_seq['kl']:.2f}). Neither "
       "operator dominates. The comparison is only fair when the set is chosen with the operator you will ship. "
       "(For a single direction the two operators are identical. Yet the same neuron scored 59% and 62% on validation in "
       "the two searches, because fp32 rounding changed a few greedy tokens. On 32 prompts, treat ±3 points as noise.) ")
    + "<b>Takeaway:</b> many centres beat one mean. The best single SOM direction reached 59% on validation, against 31% "
      "for the best single mean. Be explicit about which projection you apply, because P<sub>k</sub>⋯P<sub>1</sub> and "
      "span removal are different edits and give different numbers.")

_wa = R5d["wrdo_alpha"]
_wbest = max(_wa.items(), key=lambda kv: kv[1]["harmful_comply"])
WRDO_TAKEAWAY = (
    f"At peak strength 1.0 both operators barely move refusal ({pct(_t('basic rd')['harmful_comply'])} and "
    f"{pct(_t('uniform wrd')['harmful_comply'])}). Most layers get only a partial projection, and the first 13 of 32 "
    "layers (and the embedding) are not edited at all. The paper also reports stronger peaks, where α &gt; 1 flips the component instead of "
    "removing it: "
    + "; ".join(f"{k.replace('alpha', 'α')}: {pct(v['harmful_comply'])} (KL {v['kl']:.2f})" for k, v in _wa.items())
    + f". The best of these, {_wbest[0].replace('alpha', 'α')} at {pct(_wbest[1]['harmful_comply'])}, is far below the "
      f"3-direction stack ({pct(_stack['harmful_comply'])}). Meanwhile the keyword check reports almost no “I can’t” for "
      "all of them. <b>Takeaway:</b> on Qwen3.5-9B the per-layer, partial-strength operator "
      "is weaker than removing a few directions everywhere. Learning prompt weights on top of it (full WRDO) may close "
      "the gap, but its released code needs the dataset fix above before its numbers can be trusted.")

HEAD_TO_HEAD = (
    "<b>What the table says.</b> (1) A single last-token mean direction tops out below 40% on this model. COSMIC’s "
    f"position search reaches {pct(_cos['harmful_comply'])}. (2) Getting to about 90% took either more than one "
    f"direction (our stack, KL {_stack['kl']:.2f}) or a trained direction (RDO, KL {_rdo['kl']:.2f}). (3) No method "
    "hurt harmless answers (99–100% everywhere). Use the KL column to see how much each method moves the model on "
    "normal prompts. (4) The keyword column does not track the judge in either direction: it shows 0% “I can’t” for "
    f"the 2024 recipe, which still fails to comply {pct(1 - _l23['harmful_comply'])} of the time, and "
    f"{pct(_cos['kw_refusal'])} for COSMIC, which complies {pct(_cos['harmful_comply'])} of the time.")
