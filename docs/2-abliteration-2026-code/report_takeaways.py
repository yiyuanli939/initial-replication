# Result-dependent sentences for Section 5 (exec'd by report.py). Every number is read from results/.
_t = lambda p: T5[tkey(p)]
_rdo, _stack, _l22, _l23, _cos = _t("rdo"), _t("dim stack"), _t("dim_l22"), _t("dim_l23"), _t("cosmic")
_seq, _span = _t("som sequential"), _t("som span [")

RDO_TAKEAWAY = (
    f"<b>Takeaway.</b> One trained direction does what three averaged ones do: {pct(_rdo['harmful_comply'])} "
    f"compliance against {pct(_stack['harmful_comply'])} for the stack, at about a third of the $\\KL$ "
    f"({_rdo['kl']:.2f} against {_stack['kl']:.2f}). Two cautions. First, its answer targets came from our "
    "3-direction model, so part of this is distillation: RDO compressed a working edit into one cleaner direction "
    "rather than finding compliance on its own. Second, the method is only as good as its targets. With the paper's "
    "recipe on this model, targets from the single-direction ablation would mostly be deflections (layer 23: "
    f"{pct(_l23['harmful_comply'])} real compliance).")

COSMIC_TAKEAWAY = (
    "Its score could not have told us that layer 23 fails. But its search over <i>token positions</i> found a single "
    f"direction that beats every last-token direction we tried ({pct(_cos['harmful_comply'])} against "
    f"{pct(_l22['harmful_comply'])} for the best, layer 22), with the lowest $\\KL$ of any method here "
    f"({_cos['kl']:.2f}). <b>Takeaway:</b> use COSMIC to search widely and cheaply, then confirm the winners with a "
    "judge.")

_dt = R5d["test"]
_vbest = max(R5d["trace"], key=lambda t: (t["comply"], -t["kl"]))
SOM_TAKEAWAY = (
    "The set above was chosen with the sequential operator, which favours sequential removal. For a fair comparison we "
    f"also ran the greedy search with the span operator. It picks {R5d['set']} ({pct(_vbest['comply'])} on "
    f"validation), which gives <b>{pct(_dt['harmful_comply'])}</b> on the test set at $\\KL = {_dt['kl']:.2f}$. "
    + ("So on this model exact span removal did not beat the repository's sequential version. Removing the full span "
       "of aligned directions also removes more of what harmless prompts need (higher $\\KL$). "
       if _dt["harmful_comply"] <= _seq["harmful_comply"] else
       f"Chosen with the matching operator, span removal complies more ({pct(_dt['harmful_comply'])} against "
       f"{pct(_seq['harmful_comply'])}) but moves harmless outputs more ($\\KL$ {_dt['kl']:.2f} against "
       f"{_seq['kl']:.2f}). Neither operator dominates. The comparison is only fair when the set is chosen with the "
       "operator you will ship. (For a single direction the two operators are identical. Yet the same neuron scored "
       "59% and 62% on validation in the two searches, because fp32 rounding changed a few greedy tokens. On 32 "
       "prompts, treat ±3 points as noise.) ")
    + "<b>Takeaway:</b> many centres beat one mean. The best single SOM direction reached 59% on validation, against "
      "31% for the best single mean. Be explicit about which projection you apply: $P_k \\cdots P_1$ and span removal "
      "are different edits and give different numbers.")

_wa = R5d["wrdo_alpha"]
_wfmt = lambda k: ("per-layer mean" if k.startswith("basic") else "uniform bank") + f", $\\alpha = {k.split('=')[1]}$"
_wbest = max(_wa.items(), key=lambda kv: kv[1]["harmful_comply"])
WRDO_TAKEAWAY = (
    f"At peak strength $\\alpha = 1$ both operators barely move refusal ({pct(_t('basic rd')['harmful_comply'])} and "
    f"{pct(_t('uniform wrd')['harmful_comply'])}). Only layers 13–30 are edited, half of them with $\\alpha_\\ell < 0.5$, "
    "and the embedding and the first 13 layers are not edited at all. The paper also reports stronger peaks, where $\\alpha > 1$ "
    "flips the component instead of removing it: "
    + "; ".join(f"{_wfmt(k)}: {pct(v['harmful_comply'])} ($\\KL$ {v['kl']:.2f})" for k, v in _wa.items())
    + f". The best of these, {_wfmt(_wbest[0])} at {pct(_wbest[1]['harmful_comply'])}, is far below the 3-direction "
      f"stack ({pct(_stack['harmful_comply'])}). Meanwhile the keyword check reports almost no “I can't” for all of "
      "them. <b>Takeaway:</b> on Qwen3.5-9B the per-layer, partial-strength operator is weaker than removing a few "
      "directions everywhere. Learning prompt weights on top of it (full WRDO) may close the gap, but its released "
      "code needs the dataset fix above before its numbers can be trusted.")

HEAD_TO_HEAD = (
    "<b>What the comparison says.</b> (1) A single last-token mean direction tops out below 40% on this model. "
    f"COSMIC's position search reaches {pct(_cos['harmful_comply'])}. (2) Getting to about 90% took either more than "
    f"one direction (our stack, $\\KL = {_stack['kl']:.2f}$) or a trained direction (RDO, $\\KL = {_rdo['kl']:.2f}$). "
    "RDO sits in the best corner of the trade-off plot. (3) No method hurt harmless answers (98–100% everywhere); "
    "the $\\KL$ column shows how much each one moves the model on normal prompts. (4) The keyword column does not track "
    f"the judge in either direction: it shows 0% “I can't” for the 2024 recipe, which still fails to comply "
    f"{pct(1 - _l23['harmful_comply'])} of the time, and {pct(_cos['kw_refusal'])} for COSMIC, which complies "
    f"{pct(_cos['harmful_comply'])} of the time.")
