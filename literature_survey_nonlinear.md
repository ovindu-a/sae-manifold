# Literature Survey: Nonlinear SAEs and Nonlinear Interpretability/Steering

Methods covering nonlinear sparse autoencoders, nonlinear activation steering, critiques
and extensions of the linear representation hypothesis, curved feature geometry in LLMs,
and nonlinear dictionary learning from ML/signal processing.

---

## 1. Nonlinear SAE Architectures

### 1.1 Not All Language Model Features Are One-Dimensionally Linear

**[arXiv:2405.14860](https://arxiv.org/abs/2405.14860)** · ICLR 2025 ·
Engels, Michaud, Liao, Gurnee, Tegmark

Proposes a rigorous definition of "irreducible multi-dimensional features" and builds a
scalable SAE-based method to automatically detect them in GPT-2 and Mistral 7B. Finds
striking examples of **circular features** encoding days of the week and months of the
year, and confirms these circles are causally used for modular-arithmetic computation in
Mistral 7B and Llama 3 8B.

**Why it matters:** the strongest direct challenge to the assumption that SAEs need only
find 1-D directions. Standard linear SAEs systematically miss multi-dimensional curved
features. Directly relevant to the `days` and `years` manifolds in this repo — both
have known circular/helical structure.

---

### 1.2 Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

**[arXiv:2503.01822](https://arxiv.org/abs/2503.01822)** · NeurIPS 2025 ·
Hindupur, Lubana, Fel, Ba

Recasts SAEs as solutions to a bilevel optimisation problem, showing that each SAE
architecture **imposes implicit structural assumptions about concept geometry**. Identifies
two real-world concept properties that standard SAEs fail to handle: heterogeneous
intrinsic dimensionality and nonlinear separability. Designs a new SAE variant that
explicitly incorporates both, discovering concepts hidden to standard architectures.

**Why it matters:** provides a theoretical framework for building SAEs with nonlinear
concept geometry awareness, and empirically demonstrates recovery of features that are
invisible to standard linear-decoder SAEs.

---

### 1.3 Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds

**[arXiv:2509.02565](https://arxiv.org/abs/2509.02565)** · NeurIPS 2025 MechInterp Workshop ·
Michaud, Gorton, McGrath

Adapts neural-scaling capacity-allocation theory to SAEs, focusing on what happens when
LLM activations contain multi-dimensional manifold features rather than discrete
directions. Identifies a **pathological scaling regime** where feature manifolds cause
SAEs to learn far fewer features than the number of latents, wasting capacity.

**Why it matters:** the manifold structure of LLM features is a practical engineering
problem. Standard SAE training implicitly assumes 0-dimensional point features; continuous
manifold features break scaling predictions and cause systematic capacity waste.

---

### 1.4 KronSAE — Kronecker Factorization and the mAND Activation

**[arXiv:2505.22255](https://arxiv.org/abs/2505.22255)** ·
Kurochkin, Aksenov, Laptev, Gavrilov, Balagansky

Introduces KronSAE, which factorises SAE encoder matrices via Kronecker products, and
the **mAND activation function** — a differentiable approximation of the binary AND
operation. Achieves large parameter count reductions while improving feature atomicity
(reduced absorption).

**Why it matters:** the mAND gate is a departure from purely linear feature detection,
introducing multiplicative nonlinear feature composition inside the SAE encoder. An early
architectural step toward SAEs that can represent conjunctive/compositional concepts.

---

## 2. Nonlinear Activation Steering

### 2.1 Curveball Steering: The Right Direction To Steer Isn't Always Linear

**[arXiv:2603.09313](https://arxiv.org/abs/2603.09313)** · March 2026 ·
Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah

Measures geodesic-to-Euclidean distance ratios in LLM activation spaces across multiple
concepts and models, finding **substantial concept-dependent geometric distortion** — the
activation space is not globally flat. Proposes "Curveball steering" via polynomial
kernel PCA in a lifted feature space, better respecting the learned curved geometry.
Consistently outperforms standard linear PCA steering, especially in high-distortion
regimes.

**Why it matters:** the most direct nonlinear alternative to linear activation steering.
Replaces the "add a direction vector" paradigm with a kernel-based curved intervention.
Provides a prototype for nonlinear SAE-based steering.

---

### 2.2 Steered LLM Activations are Non-Surjective

**[arXiv:2604.09839](https://arxiv.org/abs/2604.09839)** · 2026 ·
Mishra, Khashabi, Liu

Proves under practical assumptions that **linear activation steering pushes the residual
stream off the natural manifold** of states reachable from discrete prompts. Demonstrates
empirically across three major LLMs that steered behavioural states almost certainly lack
prompt-equivalent inputs.

**Why it matters:** provides theoretical motivation for manifold-respecting (i.e.
nonlinear/curved) steering. Off-manifold steered states may not generalise properly,
making nonlinear methods a correctness requirement rather than just a performance
improvement.

---

### 2.3 Local Linearity of LLMs Enables Activation Steering via Linear Optimal Control

**[arXiv:2604.19018](https://arxiv.org/abs/2604.19018)** ·
Skifstad, Yang, Chou

Models LLM layer-to-layer dynamics as a **linear time-varying system** and applies
linear quadratic regulator (LQR) theory to steer activations without offline training.
Argues that despite transformer nonlinearity, layer-wise dynamics are well-approximated
locally by linear models.

**Why it matters:** serves as a principled counter-point — argues for *local* linearity
even if global geometry is curved. The existence of a local-linear regime is itself
evidence that global geometry is nonlinear (a patchwork of linear neighbourhoods), and
LQR is one practical way to navigate it.

---

## 3. Critiques and Extensions of the Linear Representation Hypothesis

### 3.1 The Linear Representation Hypothesis and the Geometry of Large Language Models

**[arXiv:2311.03658](https://arxiv.org/abs/2311.03658)** · ICML 2024 ·
Park, Choe, Veitch

Formalises the linear representation hypothesis using counterfactual pairs, introducing
a **"causal inner product"** that identifies the non-Euclidean metric on activation space
under which representations are truly linear. Connects linear probing and steering
vectors to this causal geometry; validated on LLaMA-2.

**Why it matters:** establishes the formal benchmark against which nonlinear alternatives
are compared. The paper's own conclusion — that a non-Euclidean inner product is required
— implicitly shows that naive Euclidean-linear assumptions are wrong, opening space for
Riemannian or kernel-based alternatives.

---

### 3.2 The Lattice Representation Hypothesis of Large Language Models

**[arXiv:2603.01227](https://arxiv.org/abs/2603.01227)** · ICLR 2026 ·
Xiong

Proposes that LLM embeddings encode not just linear directions but a **symbolic backbone
of concept lattices** — hierarchical logical relationships derived from Formal Concept
Analysis. Linear attribute directions with thresholds create concept lattices through
half-space intersections; validated on WordNet hierarchies.

**Why it matters:** extends the linear representation hypothesis into a
combinatorial/logical geometry: concepts are regions (intersections of half-spaces), not
just directions. A form of nonlinear representation lying between linear vectors and
fully curved manifolds.

---

### 3.3 Token Embeddings Violate the Manifold Hypothesis

**[arXiv:2504.01002](https://arxiv.org/abs/2504.01002)** · NeurIPS 2025 ·
Robinson, Dey, Chiang

Develops a statistical test (the "fibre bundle hypothesis" test) that checks whether
local neighbourhoods around token embeddings are flat and smooth. Applies it to multiple
open-source LLMs and finds that the null hypothesis of smooth manifold structure is
**frequently and significantly rejected**.

**Why it matters:** rigorous empirical evidence that LLM token representations are not
low-curvature manifolds. Directly motivates interpretability methods that do not
presuppose smooth manifold geometry; suggests standard SAEs (which assume
sparse-linear reconstruction) may face fundamental geometric mismatches.

---

## 4. Manifold and Curved Feature Geometry in LLMs

### 4.1 The Origins of Representation Manifolds in Large Language Models

**[arXiv:2505.18235](https://arxiv.org/abs/2505.18235)** · May 2025 ·
Modell, Rubin-Delanchy, Whiteley

Investigates how LLM representations encode features as **continuous multidimensional
manifolds** rather than isolated directions. Shows that cosine similarity in
representation space encodes intrinsic on-manifold geometry — validated on color/date
embeddings from OpenAI text-embedding-large-3 and GPT-2 layer-7 activations for years
of the 20th century.

**Why it matters:** provides theoretical account of why curved manifold structures arise
in LLM representations and how standard similarity metrics relate to them. Directly
motivates manifold-aware SAE decoders. Closely related to the `colors` and `years`
manifolds in this repo.

---

### 4.2 Language Models Use Trigonometry to Do Addition

**[arXiv:2502.00873](https://arxiv.org/abs/2502.00873)** ·
Kantamneni, Tegmark

Reverse-engineers how GPT-J, Pythia-6.9B, and Llama 3.1-8B perform addition, finding
that numbers are encoded as **generalised helices** in the residual stream. The "Clock"
algorithm operates by rotating and combining helices; the helix representation is
causally implicated for addition, subtraction, multiplication, and modular arithmetic.

**Why it matters:** a concrete example where a nonlinear SAE with a curved-manifold
decoder would be needed to properly recover the feature structure. Directly relevant to
the `years` manifold in this repo (known helix structure).

---

### 4.3 REMA: A Unified Reasoning Manifold Framework for Interpreting LLMs

**[arXiv:2509.22518](https://arxiv.org/abs/2509.22518)** · submitted to ICLR 2026 ·
Li et al.

Defines the **"Reasoning Manifold"** as a latent low-dimensional geometric structure
formed by internal representations of all correctly-reasoned generations. Measures
deviation from this manifold to identify divergence points where reasoning chains go
off-track; validated across Qwen3, Llama3.2, and LLaVA-OneVision (3B–90B).

**Why it matters:** uses manifold geometry directly as an interpretability and
failure-detection tool. Demonstrates that correct LLM computation traces a nonlinear
low-dimensional manifold in activation space, and errors correspond to departures from
it.

---

### 4.4 Neural Feature Geometry Evolves as Discrete Ricci Flow

**[arXiv:2509.22362](https://arxiv.org/abs/2509.22362)** · September 2025 ·
Hehl, von Renesse, Weber

Approximates data manifolds with geometric graphs and shows that nonlinear activations
shape feature geometry in ways that closely track **discrete Ricci flow dynamics** — a
geometric process that smooths out curvature. Class separability emerges as community
structure in the associated graphs, driven by Ricci flow geometry.

**Why it matters:** provides a theoretical Riemannian/discrete-geometric account of why
feedforward networks produce the curved feature geometry observed empirically in LLMs.
Establishes that curvature evolution is a principled consequence of nonlinear
activations, not an artefact.

---

## 5. Nonlinear Dictionary Learning from ML/Signal Processing

### 5.1 iVAE: Variational Autoencoders and Nonlinear ICA — A Unifying Framework

**[AISTATS 2020](https://proceedings.mlr.press/v108/khemakhem20a.html)** ·
Khemakhem, Kingma, Monti, Hyvärinen

Unifies VAEs with nonlinear ICA under a single maximum-likelihood framework. Shows that
with a factorised prior conditioned on an **auxiliary observed variable** (e.g. class
labels or segment index), the true latent factors are recoverable up to simple
permutation/scaling — the **first identifiability result for deep nonlinear generative
models**.

**Why it matters:** provides the theoretical foundation for provably disentangled
nonlinear representation learning. For LLMs, iVAE-style identifiability theory could
be used to design probes or sparse coding methods that recover true independent concept
dimensions rather than arbitrary linear mixtures.

---

### 5.2 Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning

**[arXiv:1805.08651](https://arxiv.org/abs/1805.08651)** · AISTATS 2019 ·
Hyvärinen, Sasaki, Turner

Proves that nonlinear ICA is identifiable when an **auxiliary variable** (time index,
history, segment label) is available, implemented via discriminative contrastive
learning with a neural network. Provides consistency proofs and a general framework
unifying prior nonlinear ICA models.

**Why it matters:** the contrastive auxiliary-variable approach maps directly onto LLM
interpretability — context tokens or task labels could serve as auxiliaries to identify
disentangled feature directions in activation space. Provides a theoretical alternative
to linear probing for recovering independent concept axes.

---

## Summary Table

| # | Paper | URL | Angle |
|---|---|---|---|
| 1 | Not All Features Are 1D Linear | [2405.14860](https://arxiv.org/abs/2405.14860) | Nonlinear SAE |
| 2 | SAE/Concept Geometry Duality | [2503.01822](https://arxiv.org/abs/2503.01822) | Nonlinear SAE |
| 3 | SAE Scaling + Feature Manifolds | [2509.02565](https://arxiv.org/abs/2509.02565) | Nonlinear SAE |
| 4 | KronSAE (mAND activation) | [2505.22255](https://arxiv.org/abs/2505.22255) | Nonlinear SAE |
| 5 | Curveball Steering | [2603.09313](https://arxiv.org/abs/2603.09313) | Nonlinear Steering |
| 6 | Steered Activations are Non-Surjective | [2604.09839](https://arxiv.org/abs/2604.09839) | Nonlinear Steering |
| 7 | Local Linearity + LQR Steering | [2604.19018](https://arxiv.org/abs/2604.19018) | Steering (local-linear) |
| 8 | Linear Rep. Hypothesis + LLM Geometry | [2311.03658](https://arxiv.org/abs/2311.03658) | LRH Critique |
| 9 | Lattice Representation Hypothesis | [2603.01227](https://arxiv.org/abs/2603.01227) | LRH Extension |
| 10 | Token Embeddings Violate Manifold Hypothesis | [2504.01002](https://arxiv.org/abs/2504.01002) | LRH Critique |
| 11 | Origins of Representation Manifolds | [2505.18235](https://arxiv.org/abs/2505.18235) | Manifold Geometry |
| 12 | LMs Use Trigonometry for Addition | [2502.00873](https://arxiv.org/abs/2502.00873) | Curved Features |
| 13 | REMA Reasoning Manifold Framework | [2509.22518](https://arxiv.org/abs/2509.22518) | Manifold Geometry |
| 14 | Neural Feature Geometry as Ricci Flow | [2509.22362](https://arxiv.org/abs/2509.22362) | Curved Features |
| 15 | iVAE: VAEs + Nonlinear ICA | [AISTATS 2020](https://proceedings.mlr.press/v108/khemakhem20a.html) | Nonlinear Dict. Learning |
| 16 | Nonlinear ICA + Contrastive Learning | [1805.08651](https://arxiv.org/abs/1805.08651) | Nonlinear Dict. Learning |

---

## Key Synthesis

The consistent picture across all five angles: LLM representations are not globally
linear, but exhibit locally-linear patches, curved low-dimensional manifolds (helices
for numbers, circles for cyclical concepts, lattice structures for conceptual
hierarchies), and token embeddings that actively violate smooth manifold assumptions.

Standard linear SAEs are structurally blind to these nonlinear features (papers 2, 3),
and linear steering vectors leave the natural data manifold (paper 6). The most
actionable directions for nonlinear interpretability are:

1. **Multi-dimensional feature-aware SAEs** (papers 1, 2) — extend the SAE decoder to
   handle low-dimensional curved manifolds rather than single directions
2. **Polynomial kernel PCA steering** (paper 5) — replace direction-addition steering
   with kernel-based curved interventions
3. **Riemannian/manifold-geometry-based probing** (papers 11, 13) — use the intrinsic
   on-manifold geometry for probing rather than Euclidean projections
4. **Provably-identifiable nonlinear ICA** (papers 15, 16) — principled foundation for
   recovering independent concept dimensions beyond what linear probes can find

**Relation to this repo:** papers 1, 2, 3, 11, and 12 are the most directly relevant.
The `days` manifold (known circular/grid structure) and `years` manifold (known helix)
are exactly the multi-dimensional features that papers 1 and 12 show standard SAEs miss.
The subspace capture evaluation could be extended to test whether any SAE architecture
recovers these curved structures, not just linear variance.
