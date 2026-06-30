# Enhancing Data Efficiency with a Trustworthy Counterfactual Generative Model

**Zhaoan Ye, Dezong Zhao** *(Senior Member, IEEE)*, **Li Zhang** *(Senior Member, IEEE)*, **Xidong Yan, Qinglin Bi, Wenjing Zhao, David Flynn** *(Senior Member, IEEE)*

---

## Abstract

Leveraging limited data to synthesize an additional training set is essential for robotic vision, particularly in dynamic environments where collecting large datasets is impractical. Traditional robotic vision systems rely on extensive training data for object recognition and scene understanding but struggle to generalize to real-world variations such as lighting conditions, occlusions, and sensor noise. This paper proposes **Causal DiffuseVAE**, a novel method integrating causal inference with high-fidelity image synthesis to generate counterfactual images. By combining the disentanglement properties of Variational Autoencoders (VAEs) with the generative capabilities of Diffusion Models (DMs), Causal DiffuseVAE produces realistic, interpretable simulations of variations like shadows and occlusions. This combination enables data-efficient generative modeling by learning from small subsets and synthesizing missing or unseen samples. Additionally, causal inference ensures that generated data follow real-world dependencies, making it robust and interpretable for deployment in unpredictable environments. Four baseline approaches are evaluated across six different datasets, demonstrating that Causal DiffuseVAE consistently outperforms the four baseline approaches.

**Project URL:** https://github.com/yza2542/Causal-DiffuseVAE

**Index Terms:** Counterfactual Image Generation, Causal Inference, Variational Autoencoder, Diffusion Models

---

## I. Introduction

Efficiently processing and learning from data is becoming ever more crucial for perception and decision-making in robotics and other industrial applications. Conventional robotic vision systems heavily rely on extensive, task-specific datasets for training. Therefore, their deployment in dynamic real-world environments is limited because collecting large volumes of labeled data is impractical or cost-prohibitive. Recent advances in generative modeling, such as Variational Autoencoders (VAEs) and Diffusion Models, have made synthetic data generation more feasible. However, these models fall short when it comes to capturing and manipulating the causal factors underlying complex environmental variations. Real-world images often suffer from degradations such as motion and blur in the defocus, which can further undermine perception and generative modeling.

Causal inference, proposed to make the model interpretable, has gained popularity in machine learning in recent years. Within causal inference, interventions and counterfactuals are deployed to examine how an event would develop under different conditions. An *intervention* explicitly alters one or more conditions of an event, while a *counterfactual* asks what would have happened under that intervention, given the actual observations.

Consider the example where a robotic dog must detect and grasp a stick hidden in shadow. A purely data-driven vision system may fail under such occlusions or lighting changes, leading to unsafe or incorrect decisions. This highlights the need not only for high-quality generative models but also for causal counterfactual generation. Causal counterfactual generation enables asking *"What if the shadow had been removed?"* and then synthesizes the corresponding scene, providing targeted and realistic variations that improve model interpretability under challenging conditions.

This paper proposes the **Causal Diffuse Variational Autoencoder (Causal DiffuseVAE)**, which integrates causal inference and high-fidelity image synthesis to enhance data efficiency by generating realistic and interpretable counterfactual images. The main contributions are:

1. A structured latent space guided by a Structural Causal Model (SCM) is introduced, in which each latent dimension corresponds to a distinct causal factor, capturing causal relationships between factors to produce semantically meaningful counterfactuals.
2. The interpretable latent encoding of a VAE is combined with the high-fidelity synthesis of a diffusion decoder, resulting in a single pipeline that benefits from both under causal conditioning.
3. Identifiability of the causal latent is verified in experiments. Evaluations on synthetic and real-world benchmarks show much stronger robustness against factor changes, and the proposed method requires fewer annotated datasets.

---

## II. Preliminaries and Problem Formulation

### A. Intervention and Counterfactual in Structural Causal Models

An SCM is commonly defined with a triple ⟨U, V, E⟩, where **U** represents exogenous variables, **V** indicates the system's internal variables, and **E** represents unexplained variation not captured by the model's deterministic part. Causal inference is performed through interventions, typically represented by the **do-operator**, denoted as `do(X = x)`, which implies actively setting X to x and observing the resultant changes in other variables.

### B. Mask Layer for Causal Inference

A mask causal layer achieves causal disentanglement by learning causal latent variables **z** with an adjacency matrix **A** derived from a causal graph:

$$z = A^T z + \epsilon = (I - A^T)^{-1} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{1}$$

where **I** is the identity matrix, ε indicates independent Gaussian exogenous factors, and T denotes the matrix transpose.

### C. Variational Autoencoders

VAEs learn a generative mapping from a simple prior over a low-dimensional code to the data space using an approximate inference network and the Evidence Lower Bound (ELBO) objective. The target distribution is:

$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I) \tag{2}$$

The conditional likelihood is:

$$p_\theta(\mathbf{x} \mid \mathbf{z}) = f_\theta(\mathbf{z}) \tag{3}$$

The approximate posterior is:

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\!\left(\boldsymbol{\mu}_\phi(\mathbf{x}),\, \mathrm{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))\right) \tag{4}$$

The ELBO loss function is:

$$\mathcal{L}_\text{VAE} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right] - \mathrm{KL}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\right) \tag{5}$$

### D. Diffusion Models

The Denoising Diffusion Probabilistic Model (DDPM) models the forward process of transforming structured data into noisy data:

$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \tag{6}$$

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t I\right) \tag{7}$$

The reverse process reconstructs structured data from noise:

$$p(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^{T} p(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \tag{8}$$

$$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\boldsymbol{\mu}(\mathbf{x}_t),\, \boldsymbol{\Sigma}(\mathbf{x}_t)\right) \tag{9}$$

---

## III. Causal Diffuse Variational Autoencoder

Current causal generative models (e.g., CausalVAE, CDRM) cannot generate high-quality 3D images due to the VAE's smooth loss function, which results in overly blurry reconstructions. Causal DiffuseVAE addresses this by combining causal inference with both VAE and Diffusion Model components.

### A. Causal Mechanism

The causal mechanism transforms unstructured data into a structured latent space that explicitly captures causal relationships. This operates under the mild assumption that all relevant causal variables are observed and that no confounders exist. The mechanism is represented as:

$$z_i = g_i(A_i \circ \mathbf{z};\, \eta_i) + \epsilon_i \tag{10}$$

where $z_i$ is the $i$-th element of **z**, $g_i$ is a mild nonlinear and invertible function, $A_i$ is the $i$-th column vector of the adjacency matrix **A**, ∘ is the elementwise product, and $\eta_i$ is the learnable parameter of **g**.

By enforcing $A_i \circ \mathbf{z}$, only the true parent nodes influence each $z_i$, yielding a disentangled, interpretable causal code. Changes to a "cause" latent automatically propagate through all downstream nodes following the causal arrows in the SCM.

**Example — Shadow Formation:**
Let $\mathbf{z} = (z_\text{light},\, z_\text{object\_size},\, z_\text{shadow})$. Increasing $z_\text{object\_size}$ causes the causal layer to recompute:

$$z_\text{shadow} = g_\text{shadow}\!\left(A_\text{shadow} \circ \mathbf{z}\right) \tag{11}$$

As the object size increases, the corresponding shadow extends accordingly, and the diffusion decoder generates an image where the shadow shifts and enlarges.

### B. Model Learning

The combined loss function is:

$$\mathcal{L} = \mathcal{L}_\text{VAE} + \mathcal{L}_\text{DDPM} \tag{12}$$

**1) VAE Learning Strategy:**

The generative process is:

$$p_\theta(\mathbf{x}, \mathbf{z}, \epsilon \mid u) = p_\theta(\mathbf{x} \mid \mathbf{z}, \epsilon, u)\, p_\theta(\epsilon, \mathbf{z} \mid u) \tag{13}$$

The ELBO for the causal layer is redefined as:

$$\text{ELBO} = \mathbb{E}_X\!\left\{\underbrace{\mathbb{E}_{\epsilon,\mathbf{z} \sim q_\phi}\!\left[\log p_\theta(\mathbf{x} \mid \mathbf{z}, \epsilon, u)\right]}_{\text{Reconstruction Loss}} - \underbrace{D_\text{KL}\!\left[q_\phi(\epsilon, \mathbf{z} \mid \mathbf{x}, u) \,\|\, p_\theta(\epsilon, \mathbf{z} \mid u)\right]}_{\text{KL Divergence Regularization}}\right\} \tag{15}$$

To balance quality and accuracy, the reconstruction loss is defined as:

$$\mathcal{L}_\text{recon} = \alpha \cdot \text{BCE}(\mathbf{x}, \hat{\mathbf{x}}) + \nu \cdot \text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) \tag{17}$$

where α = 0.7 and ν = 0.3.

To ensure identifiability of the adjacency matrix **A**, two regularization losses are added:

$$l_u = \mathbb{E}_X \|u - \sigma(A^T u)\|_2^2 \tag{19}$$

$$l_z = \mathbb{E}_{\mathbf{z} \sim q_\phi}\!\left[\sum_{i=1}^n \|z_i - g_i(A_i \circ \mathbf{z};\, \eta_i)\|_2^2\right] \tag{20}$$

The full VAE loss is:

$$\mathcal{L}_\text{VAE} = -\text{ELBO} + \gamma\, l_u + \lambda\, l_z \tag{21}$$

where γ = 0.01 and λ = 0.1.

**2) Diffusion Model Learning Strategy:**

$$\mathcal{L}_\text{DDPM} = \mathbb{E}_{\mathbf{z} \sim q_\psi(\mathbf{z}|y,x_0)}\!\left[\mathbb{E}_{q_\phi(x_{1:T}|y,\mathbf{z},u,x_0)} \log\frac{q_\phi(x_{1:T} \mid y, \mathbf{z}, u, x_0)}{p_\psi(x_{0:T} \mid y, \mathbf{z}, u)}\right] \tag{23}$$

**Algorithm 1: Causal DiffuseVAE Inference**

```
Input:  (image, label) pairs (x₀, u), number of concepts n
Output: Generated counterfactual x₀^DM

1:  Sample x₀ ~ q(x₀)
2:  for i = 1 to n do
3:      if i == intervention variable index then
4:          z = desired value
5:      else
6:          z = gᵢ(Aᵢ ∘ z; ηᵢ) + εᵢ
7:      end if
8:  end for
9:  x̂₀ ← pθ(x₀ | z, ε, u)
10: Add the condition: x_T^DM = x_T^DM + x̂₀
11: for t = T to 1 do
12:     Sample noise εₜ ~ N(0, I)
13:     Compute: x_{t-1}^DM = (x_t^DM - √βₜ · εₜ) / √(1 - βₜ)
14: end for
15: Return x₀^DM
```

---

## IV. Experiments

Experiments were run on a server with Ubuntu 20.04 and two NVIDIA RTX A6000 GPUs.

### A. Experimental Datasets

Six datasets were used:

- **Cube Shadow & Polyhedron Shadow** — Two synthetic shadow datasets (10,000 images each; 7,500 train / 1,000 val / 1,500 test), created with Blender. Each image includes a light source, object, and shadow.
- **Pendulum Dataset** — 7,000 RGBA images (5,950 / 700 / 350 split) capturing causal interactions between pendulum angle, light angle, and shadow characteristics.
- **Flow Dataset** — 7,000 RGBA images (5,100 / 900 / 1,000 split) simulating fluid dynamics of a ball interacting with liquid in a broken vessel.
- **CelebA (Gender & Age)** — 20,000 images (70% / 10% / 20% split) for facial attribute analysis.
- **Causal Circuit** — 512×512 RGB images of a robot arm interacting with a causally connected circuit (80% / 20% split), capturing four causal variables.

### B. Experimental Setting

Baselines: **CausalVAE**, **CDRM**, **CDAE**, and **Conditional Diffusion Models (CDM)**.

Evaluation metrics:
- **MAE** (Mean Absolute Error) — measures accuracy of control over latent factors.
- **LPIPS** (Learned Perceptual Image Patch Similarity) — measures perceptual similarity via feature representations; lower is better.

### C. Experimental Results

**Capability Comparison (Table I)**

| Capability | Causal DiffuseVAE | CausalVAE | CDRM | CDAE | CDM |
|---|:---:|:---:|:---:|:---:|:---:|
| Explicit low-dimensional causal latent | ✓ | ✓ | ✓ | ✓ | ✗ |
| High-fidelity image synthesis | ✓ | ✗ | ✗ | ✓ | ✓ |
| Stable training via reparameterization | ✓ | ✓ | ✓ | ✗ | ✗ |
| Trustworthy causal counterfactuals | ✓ | ✓ | ✓ | ✓ | ✗ |
| Diffusion-based architecture | ✓ | ✗ | ✗ | ✓ | ✗ |

**Ablation Study — LPIPS (Table II)**

| Experiment Setting | LPIPS |
|---|---|
| Causal DiffuseVAE (full) | **0.0185 ± 0.0140** |
| Without Diffusion Decoder | 0.0483 ± 0.0230 |
| Without Causal Layer | 0.0503 ± 0.0240 |
| Without CausalVAE Module | 0.1410 ± 0.1070 |

**MAE Comparison (Table III)**

| Dataset | Causal DiffuseVAE | CausalVAE | CDRM | CDAE | CDM |
|---|---|---|---|---|---|
| Cube Shadow | **0.0098 ± 0.0050** | 0.0930 ± 0.0300 | 0.5400 ± 0.1300 | 0.2930 ± 0.0150 | 1.2100 ± 0.4590 |
| Polyhedron Shadow | **0.0103 ± 0.0040** | 0.1450 ± 0.0350 | 0.6600 ± 0.1320 | 0.4360 ± 0.0460 | 2.4500 ± 0.6770 |
| Pendulum | **0.0190 ± 0.0100** | 21.302 ± 3.4400 | 17.959 ± 2.5430 | 0.2980 ± 0.0100 | 0.8570 ± 0.3050 |
| Flow | **0.0230 ± 0.0100** | 16.650 ± 3.3700 | 14.590 ± 2.3220 | 0.3150 ± 0.0100 | 1.0100 ± 0.4680 |
| CelebA (Gender) | **0.0850 ± 0.0140** | 0.6500 ± 0.2400 | 0.7800 ± 0.4110 | 0.1340 ± 0.0369 | 0.9460 ± 0.3880 |

**LPIPS Quality Comparison (Table IV)**

| Dataset | Causal DiffuseVAE | CausalVAE | CDRM | CDAE | CDM |
|---|---|---|---|---|---|
| Cube Shadow | **0.0185 ± 0.0140** | 0.1059 ± 0.0980 | 0.0590 ± 0.0400 | 0.0482 ± 0.0210 | 0.1410 ± 0.1070 |
| Polyhedron Shadow | **0.0292 ± 0.0160** | 0.1982 ± 0.1200 | 0.0680 ± 0.0400 | 0.0295 ± 0.0200 | 0.1976 ± 0.1300 |

**Data Efficiency — LPIPS under Reduced Training Data (Table V)**

| Training Data | Causal DiffuseVAE | CausalVAE | CDRM | CDAE | CDM |
|---|---|---|---|---|---|
| Full dataset | **0.0185 ± 0.0108** | 0.1059 ± 0.0980 | 0.0590 ± 0.0400 | 0.0482 ± 0.0217 | 0.1410 ± 0.1070 |
| 50% dataset | **0.0193 ± 0.0113** | — | 0.0800 ± 0.0645 | 0.0735 ± 0.0483 | 0.1802 ± 0.1413 |
| 30% dataset | **0.0235 ± 0.0150** | — | 0.1200 ± 0.0967 | 0.0917 ± 0.0657 | 0.2490 ± 0.1650 |

**Sampling Efficiency — LPIPS at Different Steps (Table VI)**

| Sampling Steps | Causal DiffuseVAE | CDAE | CDM |
|---|---|---|---|
| 50 | 0.2156 ± 0.0580 | 0.3018 ± 0.0890 | 0.3566 ± 0.1930 |
| 100 | 0.0440 ± 0.0295 | 0.1486 ± 0.0470 | 0.2044 ± 0.1544 |
| 1000 | **0.0185 ± 0.0108** | 0.0482 ± 0.0217 | 0.1410 ± 0.1070 |

**Training and Inference Time (Table VII)**

| Model | Training Time (h) | Inference Time (s/image) |
|---|---|---|
| Causal DiffuseVAE | 17.3 | 0.68 |
| CausalVAE | 2.0 | 0.66 |
| CDRM | 2.5 | 0.75 |
| CDAE | 55.2 | 1.8 |
| CDM | 17.4 | 6.6 |

Key findings: Causal DiffuseVAE achieves equivalent quality to CDM in 50 steps instead of 1,000 (20× fewer iterations) and to CDAE in 100 steps instead of 1,000 (10× fewer iterations).

### Discussion: Domain Gap and Future Directions

The model demonstrates consistent advantages on synthetic benchmarks. However, a domain gap remains between controlled synthetic settings and real-world deployment. Real robotic perception must handle additional uncertainties including sensor noise, lighting variation, calibration errors, and dynamic environmental changes. The causal framework also assumes all relevant causal factors are observed with no unmeasured confounders — an assumption more valid in synthetic data than in real-world scenarios.

Strategies to narrow the domain gap include:
- Fine-tuning the pre-trained encoder–decoder on small sets of real sensor data via transfer learning.
- Applying domain randomization and photometric augmentation during synthetic data generation.
- Introducing noise-robust or physics-aware causal encoders to model measurement uncertainty.
- Hardware-aware optimization (weight quantization, pruning) for embedded robotic platforms.

---

## V. Conclusion

This paper proposes **Causal DiffuseVAE**, a novel generative method that integrates causal reasoning into the VAE and Diffusion Model framework. Experimental results demonstrate that it consistently outperforms conventional methods (CausalVAE, CDRM, CDAE, CDM) in both reconstruction accuracy and perceptual similarity across multiple datasets. The method effectively preserves structural details while ensuring causal consistency, and produces identifiable, disentangled causal latents under mild assumptions.

Future work will focus on:
- Developing fully learnable causal graphs that update online with new data.
- Testing Causal DiffuseVAE in real-world vision tasks such as shadow removal to improve perceptual robustness in complex settings.

---

## Acknowledgements

This work was supported in part by the Royal Academy of Engineering/Leverhulme Trust Research Fellowship (Grant LTRF-2425-21-151), the EPSRC Innovation Fellowship (Grant EP/S001956/2), and the Royal Society-Newton Advanced Fellowship (Grant NAF\R1\201213).

---

## References

1. X. Li et al., "Data Mode-Related Generative Adversarial Network for Industrial Soft Sensor Application," *IEEE Transactions on Industrial Informatics*, vol. 20, no. 3, pp. 4198–4205, 2024.
2. S. Neupane et al., "Security Considerations in AI-Robotics: A Survey of Current Methods, Challenges, and Opportunities," *IEEE Access*, vol. 12, pp. 22072–22097, 2024.
3. Y.-T. Chen et al., "On the Private Data Synthesis Through Deep Generative Models for Data Scarcity of Industrial Internet of Things," *IEEE Transactions on Industrial Informatics*, vol. 19, no. 1, pp. 551–560, 2023.
4. R. Hoque et al., "IntervenGen: Interventional Data Generation for Robust and Data-Efficient Robot Imitation Learning," *IEEE/RSJ IROS*, 2024, pp. 2840–2846.
5. K. Zhang et al., "MC-Blur: A Comprehensive Benchmark for Image Deblurring," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 34, no. 5, pp. 3755–3767, 2024.
6. K. Zhang et al., "Adversarial Spatio-Temporal Learning for Video Deblurring," *IEEE Transactions on Image Processing*, vol. 28, no. 1, pp. 291–301, 2019.
7. K. Zhang et al., "Deep Image Deblurring: A Survey," *International Journal of Computer Vision*, vol. 130, no. 9, 2022.
8. Y. Jung et al., "Unified covariate adjustment for causal inference," *NeurIPS*, vol. 37, pp. 6448–6499, 2024.
9. L. Cai et al., "Counterfactual causal-effect intervention for interpretable medical visual question answering," *IEEE Transactions on Medical Imaging*, vol. 43, no. 12, pp. 4430–4441, 2024.
10. Z. Hau et al., "Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception," *IEEE SPW*, 2022, pp. 229–235.
11. A. Dhir et al., "A Meta-Learning Approach to Bayesian Causal Discovery," *ICLR*, 2025.
12. M. Yang et al., "CausalVAE: Structured Causal Disentanglement in Variational Autoencoder," *arXiv preprint*, 2020.
13. K. Sueyoshi and T. Matsubara, "Predicated diffusion: predicate logic-based attention guidance for text-to-image diffusion models," *CVPR*, 2024, pp. 8651–8660.
14. K. Preechakul et al., "Diffusion Autoencoders: Toward a Meaningful and Decodable Representation," *CVPR*, 2022.
15. P. Sanchez and S. A. Tsaftaris, "Diffusion Causal Models for Counterfactual Estimation," *Conference on Causal Learning and Reasoning*, 2022.
16. A. Komanduri et al., "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models," 2024.
17. H. Aetesam and S. K. Maji, "Deep variational magnetic resonance image denoising via network conditioning," *Biomedical Signal Processing and Control*, vol. 95, 2024.
18. C. Deng et al., "Causal Diffusion Transformers for Generative Modeling," *arXiv:2412.12095*, 2024.
19. Z. Lin et al., "CCDiff: Causal Compositional Diffusion Model for Closed-loop Traffic Generation," *CVPR*, 2025.
20. X. Tao et al., "Counterfactual Reasoning and Cognitive Intelligence for Rational Robots," *ICARCV*, 2024, pp. 25–30.
21. P. Gupta et al., "Object Importance Estimation Using Counterfactual Reasoning for Intelligent Driving," *IEEE Robotics and Automation Letters*, vol. 9, no. 4, pp. 3648–3655, 2024.
22. J. Dörfler et al., "On the Complexity of Identification in Linear Structural Causal Models," *NeurIPS*, 2024.
23. G. Van Goffrier et al., "Estimating long-term causal effects from short-term experiments and long-term observational data with unobserved confounding," *arXiv:2302.10625*, 2023.
24. J. Chung et al., "Learning distribution-free anchored linear structural equation models in the presence of measurement error," *Journal of the Korean Statistical Society*, vol. 54, pp. 361–385, 2025.
25. S. Zhao et al., "CV-VAE: A Compatible Video VAE for Latent Generative Video Models," *arXiv:2405.20279*, 2024.
26. S. Sadat et al., "LiteVAE: Lightweight and efficient variational autoencoders for latent diffusion models," *NeurIPS*, vol. 37, 2024.
27. J. Ho et al., "Denoising Diffusion Probabilistic Models," *arXiv:2006.11239*, 2020.
28. M. Chen et al., "CDRM: Causal disentangled representation learning for missing data," *Knowledge-Based Systems*, vol. 299, 2024.
29. J. Zhu et al., "Shadow Datasets, New challenging datasets for Causal Representation Learning," 2023.
30. Z. Liu et al., "Deep Learning Face Attributes in the Wild," *ICCV*, 2015.
31. J. Brehmer et al., "Weakly supervised causal representation learning," *NeurIPS*, vol. 35, 2022.
32. P. Dhariwal and A. Nichol, "Diffusion Models Beat GANs on Image Synthesis," *NeurIPS*, vol. 34, pp. 8780–8794, 2021.

---

## Author Affiliations

**Zhaoan Ye, Dezong Zhao, Xidong Yan, Qinglin Bi, Wenjing Zhao, and David Flynn** are with the James Watt School of Engineering, University of Glasgow, G12 8QQ Glasgow, U.K.

**Li Zhang** is with the Department of Computer Science, Royal Holloway, University of London, TW20 0EX Egham, U.K.

---

*Published in IEEE Transactions on Industrial Informatics. DOI: 10.1109/TII.2026.3652484*  
*Deposited: 14 January 2026 — https://eprints.gla.ac.uk/376652/*  
*© 2026 IEEE. Reproduced under a Creative Commons Attribution 4.0 International License.*
