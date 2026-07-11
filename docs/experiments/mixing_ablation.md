# Domain-mixture ablation

Uniform, observed-source, and 4x code-heavy mixtures each receive 20,000 training
bytes. A proxy split selects an arm; final numbers use a disjoint held-out split to
avoid selection leakage. Mean held-out loss was 2.658 (uniform), 2.652 (observed),
and 2.804 (code-heavy). Code-heavy is consistently worse; uniform and observed are
not distinguishable with three seeds. Proxy selection chose uniform for one seed and
observed for two, so it did not discover a new mixture beyond the grid.

This is a deliberately constrained analogue of DoReMi: Xie et al.,
[DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429).
The per-domain loss columns in the JSON/CSV make the tradeoff auditable. We make no
claim that the selected mixture transfers beyond this synthetic generator.

Source: `configs/experiments/mixture_ablation.yaml`; config `008655194a79`;
result `7355b35958c6`.
