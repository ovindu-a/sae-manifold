# Verified Literature Survey: Concept-Aligned SAEs

Adversarially verified report. 102 agents · 20 sources fetched · 25 claims verified ·
**13 confirmed · 12 refuted**.

Each finding carries its adversarial vote (e.g. 3-0 = all three verifier agents
confirmed; 0-3 = all three refuted). Claims that did not survive verification are
listed separately at the end.

---

## Context: Why Standard SAEs Fail Here (verified 3-0)

Sources: [arXiv:2502.12179](https://arxiv.org/abs/2502.12179),
[arXiv:2602.00924](https://arxiv.org/pdf/2602.00924)

Standard unsupervised SAEs are **non-identifiable**: latent dimensions can entangle
multiple underlying concepts, making steering interventions unreliable or harmful. Two
separable technical problems drive this:

1. The non-smoothness of the L1 penalty (hurting reconstruction and scalability)
2. The absence of any alignment between learned features and human semantic concepts

Independently corroborated by the Gated SAE paper and OpenAI TopK SAE (ICLR 2025).

---

## Finding 1 — AlignSAE: Latent Partition + Decorrelation Loss (3-0)

**[arXiv:2512.02004](https://arxiv.org/abs/2512.02004)** · Dec 2025

AlignSAE partitions the SAE latent code into two parts:

```
z = [z_concept | z_rest]
```

`z_concept` has `|R|` slots, one per named concept relation. `z_rest` is the free
unsupervised part. The full training objective is:

```
L = L_SAE + λ_align·L_align + λ_perp·L_perp + λ_val·L_val
```

- `L_SAE`: standard reconstruction loss
- `L_align`: cross-entropy loss aligning concept slots to known concept labels
- `L_perp = ||corr(z_concept, z_rest)||²_F`: Frobenius-norm decorrelation penalty
  forcing the supervised and free subspaces apart
- `L_val`: value-prediction loss

The decorrelation penalty is the key innovation — it actively prevents the concept
slots from absorbing unrelated information, and prevents free features from
duplicating the concept information.

**Application to age/temperature/color/days/years manifolds:** assign one slot per
manifold. The decorrelation penalty ensures each slot stays focused on its concept.

> **Refuted (0-3):** the specific claim that AlignSAE achieves a concept-swap success
> rate of 0.847 vs 0.040 for a baseline did not survive verification. The qualitative
> method is confirmed; specific numbers should be taken directly from the paper.

> **Refuted (1-2):** the claim that AlignSAE forces each concept into approximately
> one effective feature (EffFeat ≈ 1) was not verified.

---

## Finding 2 — CB-SAE: Post-Hoc Concept Bottleneck (3-0 on method, 2-1 on numbers)

**[arXiv:2512.10805](https://arxiv.org/abs/2512.10805)** · CVPR 2026

Post-hoc approach: train a standard SAE first, then augment with a lightweight concept
bottleneck aligned to a user-defined concept set. Three training objectives (3-0):

- **(A) Reconstruction loss** — recovers capacity lost by pruning low-utility neurons
- **(B) Interpretability loss** — aligns features with the user-specified concept set
- **(C) Cyclic reconstruction loss** — promotes steerability: encode → concept
  bottleneck → decode → re-encode should be stable

Reports +32.1% interpretability and +14.5% steerability over standard SAEs (2-1).

> **Caveat (2-1 vote on numbers):** gains use automated proxy metrics (CLIP-Dissect,
> cosine similarity), not human evaluation. Steerability gain varies substantially
> across models: +2% for UnCLIP vs +14.5% average.

> **Refuted (1-2):** the claim that standard SAEs fail to capture 27–45% of
> user-desired concepts was not verified. Do not cite this figure.

---

## Finding 3 — ProtSAE: Annotation + Ontology Supervision in the Training Loss (3-0)

**[arXiv:2509.05309](https://arxiv.org/abs/2509.05309)** · AAAI 2026

Applied to protein language models but the training approach transfers directly.
Composite loss (3-0):

```
L = ||x̂ − x||²₂ + λ_annot·L_annot + λ_axiom·L_axiom
```

- `L_annot`: concept prediction loss — specific neurons are supervised to predict
  known concept labels
- `L_axiom`: ontology regularisation — enforces consistency with a concept hierarchy

Also implements weight tying between the concept predictor and reconstruction decoder
so the decoder direction for a supervised neuron is anchored to the direction that
best predicts its concept label (2-1 vote — mild terminological ambiguity in the
formulation `W_def = W_pred^detach * exp(r_pred)`).

Achieves F_max 0.579 vs 0.444 and AUC 0.797 vs 0.565 over a TopK SAE baseline (3-0).

**Simplified version for continuous manifold labels (no ontology needed):**

```
L = L_recon + λ·Σ_m MSE(W_concept_m · z, label_m)
```

where `W_concept_m` is a linear head predicting manifold `m`'s label from the sparse
codes.

> **Refuted (1-2):** the specific claim about a forced activation bias term
> `z_bias = 1_{π_pred>0.5} · ReLU(mean(z_unk) − ẑ_def)` for supervised neurons was
> not verified.

---

## Finding 4 — SAE-SSV: Post-Hoc Concept Subspace Identification (3-0 on results)

**[arXiv:2505.16188](https://arxiv.org/abs/2505.16188)** · EMNLP 2025

Train a standard SAE, then rank latent dimensions by a separation score derived from
concept-labelled data. Select top k=128 dimensions to form the concept steering
subspace. Train supervised steering vectors constrained to that subspace with L1
regularisation.

Key empirical result (3-0): **only a small number of highest-ranked dimensions
achieves substantial concept separation**, with diminishing returns beyond the minimal
subspace.

Quantitative results on LLaMA-3.1-8B sentiment steering (3-0):

| Method | Success rate |
|---|---|
| SAE-SSV | **63.2%** |
| CAA | 45.6% |
| ITI | 41.1% |
| Top PC | 28.4% |
| RePe | 24.7% |

**Application:** post-hoc subspace identification for already-trained SAEs —
identifies which features are most responsible for each manifold. Directly
complements the subspace capture evaluation already in this repo.

---

## Finding 5 — Supervised Dictionary Learning (Foundational) (3-0)

**[Mairal et al., NeurIPS 2008](https://arxiv.org/pdf/0809.3083)** ·
**[Review: arXiv:1502.05928](https://arxiv.org/pdf/1502.05928)**

The foundational paper: jointly trains a shared dictionary with class-decision
functions so that sparse codes are discriminative, not just reconstructive. All modern
concept-supervised SAE methods are descendants of this idea.

> **Refuted (0-3):** "unsupervised dictionary learning is suboptimal for
> classification; supervised methods are needed for discriminative features." This
> general claim did not survive verification — the relationship between supervision and
> feature quality is more nuanced.

> **Refuted (1-2):** "adding a discriminative objective to dictionary learning yields
> features that are more task-aligned than reconstruction-only sparse coding." Also not
> verified as a universal result.

---

## Critical Negative Result

**[arXiv:2506.23845](https://arxiv.org/abs/2506.23845)** — *"Use SAEs to Discover
Unknown Concepts, Not to Act on Known Concepts"* · 2025/2026

When the concept is **already known**, simple baselines (prompting, fine-tuning,
linear probing) outperform SAE-based methods for steering and detection. SAEs are most
valuable for enumerating unknown monosemantic features.

**Implication for research framing:** the subspace capture evaluation in this repo
asks the right question — not "can we use SAEs to steer on known concepts?" but "do
SAEs *naturally* discover the subspaces these known concepts live in?" The latter is a
descriptive/diagnostic question about SAE geometry, which is the appropriate use of
SAEs here. This distinction is worth making explicit in any writeup.

---

## Full Refuted Claims

Claims killed by 2/3 or 3/3 adversarial votes. Do not cite these.

| Claim | Vote | Source |
|---|---|---|
| AlignSAE achieves concept-swap success 0.847 vs 0.040 baseline | 0-3 | [2512.02004](https://arxiv.org/abs/2512.02004) |
| AlignSAE forces each concept into EffFeat ≈ 1 via binding loss | 1-2 | [2512.02004](https://arxiv.org/abs/2512.02004) |
| Standard SAEs fail to capture 27–45% of user-desired concepts | 1-2 | [2512.10805](https://arxiv.org/abs/2512.10805) |
| SSAEs use paired observations to constrain features to causal concept subspaces | 0-3 | [2502.12179](https://arxiv.org/abs/2502.12179) |
| SSAEs provide provable identifiability guarantees for causal factor recovery | 1-2 | [2502.12179](https://arxiv.org/abs/2502.12179) |
| Supervised SAEs with decoder-only architecture address the semantic gap | 0-3 | [2602.00924](https://arxiv.org/pdf/2602.00924) |
| ProtSAE forced activation bias term for supervised neurons | 1-2 | [2509.05309](https://arxiv.org/abs/2509.05309) |
| SAEs as CBM concept layers outperform human-specified concepts | 0-3 | [2603.07343](https://arxiv.org/abs/2603.07343) |
| CBMs with human-specified concepts lack sufficient predictive power | 0-3 | [2603.07343](https://arxiv.org/abs/2603.07343) |
| SAE neurons each learn a single distinct concept under sparsity training | 0-3 | [2603.07343](https://arxiv.org/abs/2603.07343) |
| Discriminative objective makes features more task-aligned than reconstruction-only | 1-2 | [0809.3083](https://arxiv.org/pdf/0809.3083) |
| Unsupervised dictionary learning is suboptimal for classification | 0-3 | [1502.05928](https://arxiv.org/pdf/1502.05928) |

---

## Open Questions Surfaced by the Research

1. Do concept-supervised SAE methods (AlignSAE, CB-SAE, ProtSAE) preserve
   reconstruction fidelity across diverse activation layers and model scales, or do
   supervision terms systematically hurt reconstruction quality at larger scales?

2. Can the SAE-SSV post-hoc subspace identification approach be combined with
   in-training supervision (AlignSAE or ProtSAE) to achieve both better concept
   alignment at training time and more precise steering subspace selection at
   inference time?

3. What theoretical conditions are sufficient to guarantee that supervised SAE
   training recovers the true concept-generating factors rather than spurious
   correlates of concept labels?

4. How do these methods compare when the concept vocabulary is large, open-ended, or
   hierarchical — as in real ontologies — rather than the small curated concept sets
   used in evaluated papers?

---

## Caveats

- Several key papers (AlignSAE Dec 2024, CB-SAE CVPR 2026, arXiv:2602.00924 Jan 2026)
  are very recent preprints or newly accepted papers; results have not yet been widely
  replicated by independent groups.
- Multiple SSAE identifiability claims were refuted (0-3 and 1-2), suggesting the
  theoretical foundations for paired-observation SAE variants are weaker than initially
  claimed.
- ProtSAE is applied to protein language models; direct transferability to general-
  purpose LLM mechanistic interpretability remains undemonstrated.
- The paper arXiv:2603.07343 (Learning CBMs from Mechanistic Explanations) had all
  three of its extracted claims refuted (0-3); treat it with caution.
