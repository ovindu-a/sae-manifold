# Related Papers: SAEs, Neural Geometry, and Manifold Steering

This document contains research papers related to sparse autoencoders (SAEs), neural geometry, manifold structures, and steering in language models.

## Core Papers on SAEs and Neural Geometry

### 1. Do Sparse Autoencoders Capture Concept Manifolds? (2024)
- **URL**: https://arxiv.org/html/2604.28119v1
- **Focus**: Examines whether SAEs can capture the manifold structure of concepts in neural networks
- **Key Insight**: Explores the relationship between sparse coding and concept manifolds

### 2. Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (March 2025)
- **URL**: https://arxiv.org/html/2503.01822v1
- **Focus**: Directly addresses the duality between SAE representations and geometric concept structure
- **Key Insight**: Provides theoretical framework connecting SAE features to concept geometry

### 3. Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability (2026)
- **URL**: https://arxiv.org/html/2606.06333
- **Focus**: Introduces SASA (Subspace-Aware Sparse Autoencoders)
- **Key Insight**: Incorporates multi-dimensional subspace structure directly into SAE objectives through group sparsity, removing structural pressure that splits multi-dimensional features

## Manifold Steering & Feature Control

### 4. A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering (2025)
- **URL**: https://arxiv.org/html/2510.01246v1
- **Focus**: Compares SAE-based steering with activation difference methods
- **Key Insight**: Evaluates effectiveness of different steering approaches

### 5. Spherical Steering: Geometry-Aware Activation Rotation for Language Models (2026)
- **URL**: https://arxiv.org/html/2602.08169v1
- **Focus**: Uses geometric methods for activation steering
- **Key Insight**: Rotates activations along geodesics toward target directions while preserving signal integrity

### 6. Understanding (Un)Reliability of Steering Vectors in Language Models (2025)
- **URL**: https://arxiv.org/html/2505.22637v1
- **Focus**: Studies the influence of prompt types and geometry on steering reliability
- **Key Insight**: Directional consistency and activation separability predict steering performance

### 7. Steering Large Language Models using Conceptors: Improving Addition-Based Activation Engineering (2024)
- **URL**: https://arxiv.org/html/2410.16314v2
- **Focus**: Uses conceptors for more sophisticated activation engineering
- **Key Insight**: Improves upon simple addition-based steering methods

### 8. In-Distribution Steering: Balancing Control and Coherence in Language Model Generation (2025)
- **URL**: https://arxiv.org/pdf/2510.13285
- **Focus**: Ensures steering maintains in-distribution coherence
- **Key Insight**: Balances behavioral control with output quality

## Linear Representation & Interpretability

### 9. Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints (2026)
- **URL**: https://arxiv.org/html/2602.09783
- **Focus**: Theoretical foundation for linear representations in transformers
- **Key Insight**: Proves that linear interpretability is a structural consequence of transformer architecture through the Invariant Subspace Necessity theorem

### 10. Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models (2024)
- **URL**: https://arxiv.org/html/2504.02821v1
- **Focus**: Demonstrates monosemanticity in multimodal models
- **Key Insight**: Shows SAEs can learn interpretable, single-concept features in vision-language contexts

### 11. Polysemanticity and Capacity in Neural Networks (2022)
- **URL**: https://arxiv.org/abs/2210.01892
- **Focus**: Foundational work on polysemanticity and the superposition hypothesis
- **Key Insight**: Explains how networks represent more features than dimensions through superposition

### 12. A Primer on the Inner Workings of Transformer-based Language Models (2024)
- **URL**: https://arxiv.org/pdf/2405.00208
- **Focus**: Comprehensive overview of transformer internals
- **Key Insight**: Provides foundational understanding of activation spaces and representations

## Application-Focused Papers

### 13. SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following (2025)
- **URL**: https://arxiv.org/pdf/2502.11356
- **Focus**: Framework for using SAEs in instruction-following contexts
- **Key Insight**: Demonstrates practical applications of SAE-based steering

### 14. Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders (2024)
- **URL**: https://arxiv.org/pdf/2410.06981
- **Focus**: Studies feature universality across different LLMs
- **Key Insight**: Examines whether learned features generalize across model architectures

### 15. Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models (2025)
- **URL**: https://arxiv.org/pdf/2505.15634
- **Focus**: Applies feature steering to improve reasoning capabilities
- **Key Insight**: Shows steering can enhance specific cognitive capabilities

### 16. Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups (2024)
- **URL**: https://arxiv.org/pdf/2410.21508
- **Focus**: Improves SAE training efficiency through layer grouping
- **Key Insight**: Reduces computational costs while maintaining interpretability

### 17. Dense SAE Latents Are Features, Not Bugs (2025)
- **URL**: https://arxiv.org/pdf/2506.15679
- **Focus**: Reexamines the role of dense (non-sparse) latents in SAEs
- **Key Insight**: Challenges assumptions about sparsity requirements

## Additional Relevant Work

### 18. Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors (2025)
- **URL**: https://arxiv.org/pdf/2506.01247
- **Focus**: Applies sparse steering concepts to vision models
- **Key Insight**: Extends steering methods beyond language models

### 19. Physics Steering: Causal Control of Cross-Domain Concepts in a Physics Foundation Model (2025)
- **URL**: https://arxiv.org/pdf/2511.20798
- **Focus**: Steering in specialized scientific domains
- **Key Insight**: Shows domain-specific applications of steering concepts

### 20. Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control (2025)
- **URL**: https://arxiv.org/html/2507.12202v1
- **Focus**: SAE applications in recommendation systems
- **Key Insight**: Demonstrates interpretability benefits beyond language models

## Key Concepts & Hypotheses

### Linear Representation Hypothesis
High-level concepts and behavioral properties in neural networks are encoded as directions in activation space. Moving along these directions induces predictable behavioral changes.

### Superposition Hypothesis
Neural networks represent more features than they have dimensions by encoding them as overlapping linear combinations. This explains polysemanticity (neurons responding to multiple unrelated concepts).

### Manifold Structure
Continuous concepts (age, temperature, color) form smooth geometric manifolds in activation space rather than isolated directions. Neural manifold geometry encodes feature fields.

### Monosemanticity via SAEs
SAEs can decompose polysemantic representations into larger sets of monosemantic features, where each feature corresponds to a single interpretable concept.

## Research Directions

1. **Multi-dimensional features**: Moving beyond one-dimensional feature hypotheses to subspace-based representations
2. **Geometry-aware methods**: Leveraging manifold structure and geometric properties for better steering
3. **Theoretical foundations**: Understanding why linear interpretability emerges from architectural constraints
4. **Cross-domain applications**: Extending SAE and steering methods to vision, physics, and other domains
5. **Reliability and robustness**: Improving consistency and predictability of steering interventions

---

*Document created: 2026-06-11*
*Research spans 2022-2026, with concentration in 2024-2025*
