# Literature Survey: Concept-Aligned and Concept-Supervised SAEs

Methods for learning concept-aligned or concept-supervised sparse autoencoders for
mechanistic interpretability of language models.

---

## 1. AlignSAE — Two-Phase Concept-Supervised Training

**[arXiv:2512.02004](https://arxiv.org/abs/2512.02004)** · Dec 2025

The most directly relevant paper. AlignSAE attaches a standard SAE to a frozen LLM
and trains in two phases:

- **Phase 1**: Unsupervised reconstruction loss (standard SAE training) — lets
  features self-organise
- **Phase 2**: Supervised binding — specific latent slots are bound to named concepts
  while reconstruction capacity is preserved

This produces an SAE where you can say "feature 42 is the age feature" and causally
intervene by swapping its activation. The key insight is that the unsupervised phase
runs first so the SAE develops a reasonable geometry before concept supervision is
applied — this avoids supervision fighting against reconstruction from the start.

**Relation to the fixed-PCA-features idea:** this is essentially the best version of
the concept-initialized-free-features approach, with the addition of a formal
two-phase training protocol.

---

## 2. CB-SAE — Concept Bottleneck Sparse Autoencoders

**[arXiv:2512.10805](https://arxiv.org/abs/2512.10805)** · CVPR 2026

A post-hoc framework with two steps:

1. Train a standard SAE unsupervised
2. Prune low-utility neurons, then augment the latent space with a lightweight
   concept bottleneck layer aligned to a user-defined concept set

Reports +32.1% interpretability and +14.5% steerability over baseline SAEs. The
concept bottleneck is a thin supervised layer sitting on top of the existing SAE
latent space — it does not retrain the SAE, it adds a structured projection on top.
This means you can add concept alignment to an already-trained SAE without retraining
from scratch.

**Application:** train the four SAE types normally, then add a concept bottleneck
layer on top for age, temperature, days, colors. The bottleneck learns to route SAE
features into named concept axes.

---

## 3. Sparse Shift Autoencoders (SSAEs)

**[arXiv:2502.12179](https://arxiv.org/abs/2502.12179)** · ICML 2025

A fundamentally different input to the SAE: instead of encoding raw activations `x`,
SSAEs encode **differences between paired observations** — e.g. the activation
difference between "they are 20 years old" and "they are 80 years old". The sparse
representation of this difference is claimed to recover the concept direction, giving
an identifiability guarantee that standard SAEs lack.

For labelled manifolds this is very natural — you have pairs you can construct:

- age: prompt at age 20 vs age 80
- temperature: −30°F vs 119°F
- colors: high-hue vs low-hue

Training on these differences forces the SAE to learn what varies between paired
inputs, which is exactly the concept. The supervision is weak — you only need paired
observations, not explicit labels on what dimension to learn.

---

## 4. SAE-SSV — Supervised Subspace Steering

**[arXiv:2505.16188](https://arxiv.org/abs/2505.16188)** · EMNLP 2025

Uses linear probes trained on labelled data (e.g. age labels) to identify which SAE
features are most predictive of a concept. Those features define a concept subspace
within the SAE latent space. Steering is then constrained to that subspace, preventing
interference with unrelated features.

This is a post-hoc method — train SAE normally, then use probes to discover the
concept subspace. It demonstrates that a small supervised subspace within an otherwise
unsupervised SAE suffices for targeted interventions.

**Application:** train SAEs normally, then fit probes for each manifold concept to
identify which features span each concept's subspace. This gives subspace capture
analysis almost for free.

---

## 5. "Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts"

**[arXiv:2506.23845](https://arxiv.org/abs/2506.23845)** · 2025/2026

An important negative result that directly constrains the design space. The paper
empirically shows that when the concept is already known (age, temperature, color),
simple baselines like prompting or fine-tuning outperform SAE-based methods for
steering and detection. SAEs are most valuable for enumerating unknown monosemantic
features.

**Implication:** if the goal is to capture and use the subspace for a known concept
like age, a simple linear probe or activation steering vector may be more effective.
The SAE comparison becomes most valuable as a research question about whether SAEs
*naturally* capture these known subspaces — which is exactly what the subspace capture
evaluation measures.

---

## 6. SDCV — SAE-Denoised Concept Vectors

**[arXiv:2505.15038](https://arxiv.org/abs/2505.15038)** · 2025

A hybrid approach: train concept vectors using standard supervised probing (contrastive
pairs or regression on labels), then decompose them through an SAE to remove noise and
identify which SAE features compose the concept. The result is a denoised concept
vector that steers more reliably.

This is the inverse of building a concept prior into the SAE — it uses an SAE post-hoc
to refine a supervised concept vector. Useful if reliable probe directions for
manifolds are already available.

---

## 7. Supervised Sparse Auto-encoders for Semantic Composition

**[arXiv:2602.00924](https://arxiv.org/pdf/2602.00924)** · Jan 2026

Adapts neural collapse theory to SAE training: learns a sparse latent space aligned
with a known concept dictionary where non-zero coefficients directly indicate which
concepts are present in a given representation. Trains a decoder-only model from
concept-sparse vectors — the concept dictionary is fixed and the model learns to
reconstruct activations from sparse concept combinations.

Closest to the original idea of fixing features to known concept directions, but done
more rigorously — the concept dictionary is the supervision signal rather than
pre-computed PCA directions.

---

## 8. ProtSAE — Semantically-Guided SAEs

**[arXiv:2509.05309](https://arxiv.org/abs/2509.05309)** · AAAI 2026

Applies concept-guided SAE training to protein language models, incorporating domain
knowledge (secondary structure, binding sites) as supervision. Demonstrates the
approach generalises beyond text LLMs and shows that domain concept supervision
disentangles representations that are inseparable under standard unsupervised training.

Useful as a proof of concept that semantically-guided SAEs work for structured,
domain-specific concepts analogous to age/temperature/color manifolds.

---

## 9. Supervised Dictionary Learning (Foundational)

**[Mairal et al., NeurIPS 2008](https://arxiv.org/pdf/0809.3083)** ·
**[Review: arXiv:1502.05928](https://arxiv.org/pdf/1502.05928)**

The foundational ML literature on jointly optimising a discriminative dictionary and
sparse codes using class labels. All modern concept-supervised SAE methods are variants
of this idea. Key techniques from this lineage:

- **LC-KSVD**: adds a label-consistency loss term so dictionary atoms are
  class-discriminative
- **Graph-regularised SDL**: adds a manifold smoothness term so nearby points in label
  space have similar codes — directly applicable to continuous manifolds (age as a
  line, colors as a paraboloid)
- **Structured sparsity priors**: block-diagonal constraints forcing features to be
  used in concept-specific groups

---

## 10. From Superposition to Sparse Codes

**[arXiv:2503.01824](https://arxiv.org/pdf/2503.01824)** · 2025

Provides a theoretical framework connecting superposition, sparse coding, and
disentanglement via compressed sensing principles. Shows quantitatively when concept
bottleneck constraints can be integrated with sparse autoencoders to recover
human-interpretable features, and what conditions (concept dimensionality, overlap,
signal strength) determine whether recovery is possible.

---

## Summary Comparison

| Method | Supervision needed | Modifies SAE training? | Best for |
|---|---|---|---|
| AlignSAE | Concept labels | Yes (phase 2) | Named concept slots |
| CB-SAE | Concept labels | No (post-hoc) | Adding concept layer to existing SAE |
| SSAEs | Paired observations | Yes (different input) | Manifolds with natural pairs |
| SAE-SSV | Concept labels | No (post-hoc) | Finding concept subspace in trained SAE |
| SDCV | Contrastive pairs | No (post-hoc) | Denoising probe directions |
| ProtSAE | Domain annotations | Yes (joint loss) | Structured domain-specific concepts |
| Supervised SDL | Class labels | Yes (jointly) | Discriminative concept atoms |

---

## Recommended Reading Order

1. [arXiv:2506.23845](https://arxiv.org/abs/2506.23845) — frame the research question
   correctly first
2. [arXiv:2512.02004](https://arxiv.org/abs/2512.02004) (AlignSAE) — most directly
   relevant architecture
3. [arXiv:2512.10805](https://arxiv.org/abs/2512.10805) (CB-SAE) — easiest to
   implement on top of existing SAEs
4. [arXiv:2505.16188](https://arxiv.org/abs/2505.16188) (SAE-SSV) — post-hoc subspace
   identification, complements the existing evaluation
5. [arXiv:2509.05309](https://arxiv.org/abs/2509.05309) (ProtSAE) — best concrete loss
   formulation for annotation-supervised training
