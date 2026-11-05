# Starter Notes 

We want to start the whole project from the beginning. The beginning paper that we need to replicate in a small model is [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717). 

The first note I need to review is huggingface's note. 

Let $r \in \mathbb{R}^d$ be the refusal direction and normalize it 
$$
\hat{r} = \frac{r}{||r||_2}
$$
For any vector $x \in \mathbb{R}^d$, its component along this direction is 

$$
\operatorname{proj}_{\hat r}(x)
=
\underbrace{(\hat r^\top x)}_{\text{scalar amount}}
\underbrace{\hat r}_{\text{direction}}.
$$
Removing this component gives 
$$
\operatorname{proj}_{\hat r}(x)
=
\underbrace{(\hat r^\top x)}_{\text{scalar amount}}
\underbrace{\hat r}_{\text{direction}}.
$$



## Reference

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction*. Advances in Neural Information Processing Systems 37 (NeurIPS 2024). arXiv:2406.11717.

- arXiv: https://arxiv.org/abs/2406.11717
- NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract.html
- Code: https://github.com/andyrdt/refusal_direction

Note: the paper calls the technique "directional ablation" / weight orthogonalization; the name "abliteration" was popularized by the community afterwards.

```bibtex
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11717}
}
```



