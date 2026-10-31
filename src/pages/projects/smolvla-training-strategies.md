---
layout: ../../layouts/Article.astro
title: Evaluating SmolVLA Training Strategies
subtitle: Fine-tuning, visual alignment, and robotic task success.
description: Comparing SmolVLA fine-tuning strategies on LIBERO and LIBERO-Plus, and investigating how visual alignment changes representations and robot performance.
category: ROBOT LEARNING
kind: Research article
date: Oct 31, 2026
image: /images/smolvla-training-strategies/task-0.gif
imageAlt: SmolVLA All-LoRA putting alphabet soup and tomato sauce in a basket
articleTitle: "Evaluating SmolVLA Training Strategies: An Empirical Study on LIBERO and LIBERO-Plus"
code: https://github.com/NickS0kolov/lerobot/tree/feat/smolvla-alignment
huggingface: https://huggingface.co/NickSokolov/SmolVLA-LIBERO10-Experiments
---
<figure class="article-hook"><img src="/images/smolvla-training-strategies/task-0.gif" alt="SmolVLA All-LoRA putting alphabet soup and tomato sauce in a basket" width="360" height="360" loading="eager" /><figcaption>SmolVLA All-LoRA in action / LIBERO-10, Task 0.</figcaption></figure>

## Overview

Vision-Language-Action (VLA) models combine visual observations, language instructions, and robot control. Their pretrained vision-language backbones provide broadly useful representations, but adapting these models to predict actions may change the structure of those representations.

In this study, I fine-tuned **SmolVLA** using several strategies: full fine-tuning of the Action Expert, LoRA applied only to the Action Expert, and LoRA applied to both the Vision-Language Model (VLM) and the Action Expert. I then investigated **Visual Representation Alignment**, inspired by [*Don't Blind Your VLA*](https://arxiv.org/abs/2510.25616), using DINOv3 and NVIDIA RADIO as frozen visual teachers.

I evaluated robotic success on **LIBERO-10** and robustness to **Background Textures** and **Objects Layout** shifts on **LIBERO-Plus**. To examine what happened inside the VLM, I compared attention maps, t-SNE visualizations, teacher–student patch similarities, linear probing results, and the eigenspectra of image representations. Finally, I ran controlled synthetic experiments to investigate a possible mechanism behind the observed spectral concentration.

### Key findings

- **All-LoRA achieved the highest LIBERO-10 success rate among the tested configurations:** **78.5%**, compared with **53.0%** for Expert-Only FT and **33.0%** for Expert-Only LoRA.
- **Visual alignment did not improve task success in these runs.** DINOv3 and RADIO alignment reached **66.5%** and **69.0%** on LIBERO-10; the unaligned All-LoRA model also performed better in both evaluated LIBERO-Plus shift categories.
- **Visual representations changed substantially during training.** At VLM Layer 15, ImageNet-50 linear-probe accuracy fell from **87.92%** for the base model to **52.92%** after All-LoRA. Alignment did not restore this accuracy.
- **The alignment variants exhibited strong spectral concentration:** PC1 explained **93.21%** of the variance at the DINOv3 target layer and **83.78%** at the RADIO target layer.

These are empirical observations about the tested SmolVLA configurations, rather than evidence that representation alignment is ineffective in general.

The implementation is available on [GitHub](https://github.com/NickS0kolov/lerobot/tree/feat/smolvla-alignment), and the model checkpoints are available on [Hugging Face](https://huggingface.co/NickSokolov/SmolVLA-LIBERO10-Experiments).

---

## 1. Experimental Setup

### 1.1 Model and benchmarks

I used [SmolVLA](https://arxiv.org/abs/2506.01844), which combines a pretrained vision-language backbone with an Action Expert for robot action prediction. The experiments were initialized from the SmolVLA base policy.

The training and in-distribution evaluation focused on **LIBERO-10**, the long-horizon task suite of [LIBERO](https://arxiv.org/abs/2306.03310). For evaluation, each of the 10 tasks was attempted **20 times**, producing **200 episodes per model/checkpoint**. The primary metric is **Success Rate (SR)**, the proportion of episodes in which the task was successfully completed. The evaluation episode limit was 800 steps.

For out-of-distribution (OOD) evaluation, I used two perturbation categories from [LIBERO-Plus](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html):

- **Background Textures:** changes to the visual appearance of the environment.
- **Objects Layout:** changes to the spatial arrangement of objects.

Each category used a fixed set of **100 selected task variations**, with **one rollout per variation**. The same selected variations and initial conditions were used for the three All-LoRA models.

**Scope of the comparison.** All reported headline LIBERO-10 results use the **200,000-step checkpoint**. The study compares training strategies under a consistent evaluation budget, not fully tuned performance ceilings for each strategy.

### 1.2 Training configurations

All training runs used a micro-batch size of **1** and gradient accumulation over **32** steps, for an **effective batch size of 32**, and ran for **200,000 optimizer steps**.

| Configuration | Trainable components | LR schedule (start → end) | LoRA rank / alpha | Alignment |
|---|---|---:|---:|---|
| **Expert-Only FT** | Action Expert; VLM frozen | 5 × 10⁻⁴ → 5 × 10⁻⁵ | — | None |
| **Expert-Only LoRA** | LoRA adapters in Action Expert; VLM frozen | 1 × 10⁻³ → 1 × 10⁻⁴ | 64 / 64 | None |
| **All-LoRA** | LoRA adapters across VLM and Action Expert | 5 × 10⁻⁴ → 5 × 10⁻⁵ | 64 / 64 | None |
| **All-LoRA + DINOv3** | Same All-LoRA setup | 5 × 10⁻⁴ → 5 × 10⁻⁵ | 64 / 64 | Layer 15, weight 0.1 |
| **All-LoRA + RADIO** | Same All-LoRA setup | 5 × 10⁻⁴ → 5 × 10⁻⁵ | 64 / 64 | Layer 16 post-norm, weight 0.2 |
| **All-LoRA + RADIO Layer8** | Same All-LoRA setup | 5 × 10⁻⁴ → 5 × 10⁻⁵ | 64 / 64 | Layer 8, weight 0.2 |

The LoRA configurations targeted the corresponding linear layers. In the alignment variants, the pretrained visual teacher and the alignment projector were **frozen**. Further implementation details are available in the [SmolVLA alignment branch](https://github.com/NickS0kolov/lerobot/tree/feat/smolvla-alignment).

The alignment configurations are **not** a controlled comparison of teacher encoders alone: teacher model, target layer, and loss weight all change simultaneously. Their relative performance therefore cannot be attributed to the choice of DINOv3 versus RADIO in isolation.

---

## 2. Comparing SmolVLA Training Strategies

### 2.1 Expert-Only Fine-Tuning

The first configuration follows the default idea of adapting the **Action Expert** to demonstration data while keeping the VLM frozen. This updates the action-prediction component without changing the pretrained vision-language backbone.

After 200,000 steps, **Expert-Only FT reached 53.0% SR** on LIBERO-10 (106 successful episodes out of 200).

<figure><a href="/images/smolvla-training-strategies/expert_only_ft_overall_sr_site_theme.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/expert_only_ft_overall_sr_site_theme.png" alt="Expert-Only FT success rate over training" width="2332" height="1243" loading="lazy" /></a></figure>

*Figure 1. LIBERO-10 success rate across evaluated Expert-Only FT checkpoints.*

### 2.2 Expert-Only LoRA

To test a more parameter-efficient alternative, I applied LoRA only to the Action Expert while keeping the VLM frozen. This configuration trains considerably fewer parameters than full Action Expert fine-tuning.

**Expert-Only LoRA reached 33.0% SR** (66/200 episodes) after 200,000 steps—**20 percentage points below Expert-Only FT** under the tested settings.

<figure><a href="/images/smolvla-training-strategies/expert_only_lora_overall_sr_site_theme_complete.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/expert_only_lora_overall_sr_site_theme_complete.png" alt="Expert-Only LoRA success rate over training" width="2400" height="1280" loading="lazy" /></a></figure>

*Figure 2. LIBERO-10 success rate across evaluated Expert-Only LoRA checkpoints.*

Under the selected training configuration, this comparison favors full fine-tuning of the Action Expert. However, it does not establish that Expert-Only LoRA is inherently inferior, as the hyperparameters of neither method were exhaustively optimized.

### 2.3 All-LoRA: Adapting the VLM and Action Expert

In the third configuration, LoRA was applied to the VLM as well as the Action Expert. Unlike the two preceding approaches, **All-LoRA allows the VLM representations themselves to change during robot-policy training**.

At 200,000 steps, **All-LoRA achieved 78.5% SR** (157/200)—the highest score of the three initial training strategies.

<figure><a href="/images/smolvla-training-strategies/all_lora_overall_sr_site_theme_50k_200k.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/all_lora_overall_sr_site_theme_50k_200k.png" alt="All-LoRA success rate over training" width="2332" height="1243" loading="lazy" /></a></figure>

*Figure 3. LIBERO-10 success rate across evaluated All-LoRA checkpoints.*

| Training strategy | LIBERO-10 successful episodes | Success rate |
|---|---:|---:|
| Expert-Only FT | 106 / 200 | 53.0% |
| Expert-Only LoRA | 66 / 200 | 33.0% |
| **All-LoRA** | **157 / 200** | **78.5%** |

### All-LoRA in action

Explore the All-LoRA rollouts across the LIBERO-10 tasks.

<section class="libero-demos" aria-label="All-LoRA LIBERO-10 videos" id="all-lora-videos">
<div class="libero-task-picker" role="group" aria-label="Select a LIBERO-10 task">
<button type="button" data-task="0" data-src="/videos/smolvla-all-lora/0.mp4" data-title="Put both the alphabet soup and the tomato sauce in the basket" aria-label="Task 0: Put both the alphabet soup and the tomato sauce in the basket" aria-pressed="true" aria-controls="libero-video">Task 0</button>
<button type="button" data-task="1" data-src="/videos/smolvla-all-lora/1.mp4" data-title="Put both the cream cheese box and the butter in the basket" aria-label="Task 1: Put both the cream cheese box and the butter in the basket" aria-pressed="false" aria-controls="libero-video">Task 1</button>
<button type="button" data-task="2" data-src="/videos/smolvla-all-lora/2.mp4" data-title="Turn on the stove and put the moka pot on it" aria-label="Task 2: Turn on the stove and put the moka pot on it" aria-pressed="false" aria-controls="libero-video">Task 2</button>
<button type="button" data-task="3" data-src="/videos/smolvla-all-lora/3.mp4" data-title="Put the black bowl in the bottom drawer of the cabinet and close it" aria-label="Task 3: Put the black bowl in the bottom drawer of the cabinet and close it" aria-pressed="false" aria-controls="libero-video">Task 3</button>
<button type="button" data-task="4" data-src="/videos/smolvla-all-lora/4.mp4" data-title="Put the white mug on the left plate and put the yellow and white mug on the right plate" aria-label="Task 4: Put the white mug on the left plate and put the yellow and white mug on the right plate" aria-pressed="false" aria-controls="libero-video">Task 4</button>
<button type="button" data-task="5" data-src="/videos/smolvla-all-lora/5.mp4" data-title="Pick up the book and place it in the back compartment of the caddy" aria-label="Task 5: Pick up the book and place it in the back compartment of the caddy" aria-pressed="false" aria-controls="libero-video">Task 5</button>
<button type="button" data-task="6" data-src="/videos/smolvla-all-lora/6.mp4" data-title="Put the white mug on the plate and put the chocolate pudding to the right of the plate" aria-label="Task 6: Put the white mug on the plate and put the chocolate pudding to the right of the plate" aria-pressed="false" aria-controls="libero-video">Task 6</button>
<button type="button" data-task="7" data-src="/videos/smolvla-all-lora/7.mp4" data-title="Put both the alphabet soup and the cream cheese box in the basket" aria-label="Task 7: Put both the alphabet soup and the cream cheese box in the basket" aria-pressed="false" aria-controls="libero-video">Task 7</button>
<button type="button" data-task="8" data-src="/videos/smolvla-all-lora/8.mp4" data-title="Put both moka pots on the stove" aria-label="Task 8: Put both moka pots on the stove" aria-pressed="false" aria-controls="libero-video">Task 8</button>
<button type="button" data-task="9" data-src="/videos/smolvla-all-lora/9.mp4" data-title="Put the yellow and white mug in the microwave and close it" aria-label="Task 9: Put the yellow and white mug in the microwave and close it" aria-pressed="false" aria-controls="libero-video">Task 9</button>
</div>
<p class="libero-task-title" id="libero-task-title" aria-live="polite">Task 0 — Put both the alphabet soup and the tomato sauce in the basket</p>
<video id="libero-video" controls playsinline muted loop preload="metadata" src="/videos/smolvla-all-lora/0.mp4" aria-labelledby="libero-task-title">Your browser does not support embedded video. <a href="/videos/smolvla-all-lora/0.mp4">Download the video</a>.</video>
</section>

---

## 3. What Changes in the VLM After All-LoRA?

Robotic success measures policy behavior, but it does not reveal how internal visual representations change. I therefore compared the original VLM with the VLM after All-LoRA training.

### 3.1 Attention to a Task-Relevant Object

I examined two LIBERO images using the question **“Do you see a basket?”**, visualizing attention weights from the text token corresponding to *basket* to image tokens. The analysis used **VLM Layer 15 (zero-based index 14)**, selected because the inspected object-token attention maps at this layer highlighted relevant objects more clearly than maps from the other examined layers.

<figure class="figure-wide"><a href="/images/smolvla-training-strategies/vlm_attention_before_after_all_lora_english.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/vlm_attention_before_after_all_lora_english.png" alt="VLM attention before and after All-LoRA" width="1640" height="976" loading="lazy" /></a></figure>

*Figure 4. An illustrative token-to-image attention comparison before and after All-LoRA.*

In this comparison, the **attention mass allocated to image tokens decreased from approximately 0.20 to 0.06**. Some localization around the basket remained visible, but the attention distribution changed.

Attention maps show which visual tokens a model attends to at a particular layer, but they do not indicate how much these tokens influence its actions. Therefore, attention maps are useful for analysis, but they alone cannot establish whether a model understands visual information or has lost its visual capabilities.

### 3.2 Visual Embedding Separability

I then extracted hidden states for visual tokens from **Layer 15**, averaged the tokens to obtain one vector per image, and visualized embeddings of CIFAR-10 images using t-SNE.

<figure><a href="/images/smolvla-training-strategies/tsne_before_after_all_lora_english.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/tsne_before_after_all_lora_english.png" alt="CIFAR-10 t-SNE before and after All-LoRA" width="1480" height="846" loading="lazy" /></a></figure>

*Figure 5. Qualitative change in visual embedding structure on the same CIFAR-10 image set.*

Before fine-tuning, the classes formed more clearly distinguishable clusters in the projection. After All-LoRA fine-tuning, the classes appeared more mixed, suggesting a change in the geometry of the extracted features.

These observations motivated a more controlled evaluation using linear probing, discussed in the following section.

---

## 4. Visual Representation Alignment with DINOv3 and RADIO

The paper [*Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization*](https://arxiv.org/abs/2510.25616) proposes preserving visual representations during VLA fine-tuning by matching VLA patch features to features from a **frozen pretrained vision encoder**.

I evaluated this idea in SmolVLA using two visual teachers:

- **DINOv3:** `facebook/dinov3-vitl16-pretrain-lvd1689m`
- **NVIDIA RADIO:** `c-radio_v3-l`

For both camera views (agent and wrist), visual features were brought to an **8 × 8 spatial grid**, giving 64 patch locations per view. The student features were passed through a **frozen MLP projector** to match the teacher feature space. Alignment used patch-wise cosine similarity. Up to an additive constant, the objective can be written as

$$
\mathcal{L}_{\mathrm{align}} = 1 - \frac{1}{N}\sum_{j=1}^{N}
\operatorname{cos}\!\left(P(\mathbf{h}_j),\mathbf{t}_j\right),
$$

where $\mathbf{h}_j$ is a student patch representation, $P$ is the frozen projector, $\mathbf{t}_j$ is the corresponding frozen teacher feature, and $N$ counts the aligned patches across the relevant views. Training combines this term with the policy's flow-matching objective:

$$
\mathcal{L}_{\mathrm{total}} =
\mathcal{L}_{\mathrm{flow}} + \lambda\mathcal{L}_{\mathrm{align}}.
$$

The student receives alignment gradients; the teacher and projector are not updated.

| Alignment configuration | Student target | Alignment weight $\lambda$ |
|:---:|:---:|:---:|
| **DINOv3** | Layer 15 (index 14) | 0.1 |
| **RADIO** | Layer 16, after final normalization | 0.2 |
| **RADIO Layer 8** | Layer 8 (index 7) | 0.2 |


### 4.1 Training Losses

<figure class="figure-wide"><a href="/images/smolvla-training-strategies/all_lora_dino_radio_action_alignment_losses_endpoints_fixed.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/all_lora_dino_radio_action_alignment_losses_endpoints_fixed.png" alt="All-LoRA training action and alignment losses" width="3451" height="1309" loading="lazy" /></a></figure>

*Figure 6. Flow matching loss (action prediction) and auxiliary alignment loss during All-LoRA fine-tuning for the DINOv3 and RADIO Layer 8 configurations.*

The action-loss curves showed broadly similar convergence patterns across the unaligned and aligned variants. Adding the alignment loss did not accelerate the convergence of the action prediction loss.

---

## 5. LIBERO-10 and LIBERO-Plus Results

### 5.1 In-Distribution Performance

At the 200,000-step checkpoint, both alignment variants underperformed the unaligned All-LoRA model on LIBERO-10.

| Model | Layer | Successful episodes | LIBERO-10 SR |
|---|---|---:|---:|
| **All-LoRA** | — | **157 / 200** | **78.5%** |
| All-LoRA + DINOv3 | 15 | 133 / 200 | 66.5% |
| All-LoRA + RADIO | 16 (post-norm) | 138 / 200 | 69.0% |
| All-LoRA + RADIO | 8 | 110 / 200 | 55.0% |

<figure><a href="/images/smolvla-training-strategies/dino_radio_alignment_sr_comparison_site_theme_fixed.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/dino_radio_alignment_sr_comparison_site_theme_fixed.png" alt="LIBERO-10 alignment success rate during training" width="2527" height="1429" loading="lazy" /></a></figure>

*Figure 7. Success-rate progression for the two alignment configurations.*

The gap between All-LoRA and the DINOv3 configuration was 12 percentage points, while the gap for RADIO Layer 16 was 9.5 percentage points. RADIO Layer 8 showed a larger gap of 23.5 percentage points.

Interestingly, alignment at Layer 16, closer to the Action Expert, achieved a higher success rate (69.0%) than alignment at Layer 8 (55.0%). 
Alignment at Layer 8 also showed no improvement in subsequent experiments. Therefore, the following analysis focuses on the RADIO Layer 16 configuration, which demonstrated better performance.

### 5.2 Out-of-Distribution Robustness

To test whether alignment helped under visual distribution shifts despite the lower in-distribution SR, I evaluated the three All-LoRA configurations on LIBERO-Plus.

| Model | Background Textures (100 rollouts) | Objects Layout (100 rollouts) | Overall OOD (200 rollouts) |
|---|---:|---:|---:|
| **All-LoRA** | **42.0%** | **36.0%** | **39.0%** |
| All-LoRA + DINOv3 | 14.0% | 21.0% | 17.5% |
| All-LoRA + RADIO | 26.0% | 26.0% | 26.0% |

*Table 1. Comparison of the unaligned and aligned All-LoRA variants across the two evaluated LIBERO-Plus categories.*

**All-LoRA without visual alignment had the highest observed SR in both OOD categories.** In these experiments, alignment therefore did not compensate for the in-distribution performance gap with improved robustness to the two selected shifts.

---

## 6. Did Visual Alignment Preserve Useful Representations?

### 6.1 Attention and t-SNE After Alignment

I repeated the qualitative visual analyses for the aligned models, focusing on the layers used for alignment and the neighboring layers.

<figure class="figure-extra-wide"><a href="/images/smolvla-training-strategies/dino_radio_attention_three_panels.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/dino_radio_attention_three_panels.png" alt="Visual attention for DINOv3 and RADIO alignment" width="2420" height="928" loading="lazy" /></a></figure>

*Figure 8. Illustrative attention maps for DINOv3 alignment at Layer 15 and RADIO alignment at Layers 15 and 16.*

<figure class="figure-extra-wide"><a href="/images/smolvla-training-strategies/dino_radio_tsne_three_panels.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/dino_radio_tsne_three_panels.png" alt="t-SNE with DINOv3 and RADIO alignment" width="2178" height="823" loading="lazy" /></a></figure>

*Figure 9. Visual-feature t-SNE projections for the aligned variants and relevant layer comparisons.*

Neither the attention maps nor the t-SNE visualizations gave clear evidence that the original visual representation structure had been restored. 

### 6.2 Patch-Level Teacher–Student Similarity

I examined cosine similarity between projected student features and their corresponding teacher features on the **8 × 8** patch grid.

<figure class="figure-extra-wide alignment-comparison" data-teacher="dino" id="figure-10">
<div class="libero-task-picker" role="group" aria-label="Select visual teacher">
<button type="button" data-teacher="dino" aria-pressed="true" aria-controls="patch-similarity">DINOv3</button>
<button type="button" data-teacher="radio" aria-pressed="false" aria-controls="patch-similarity">RADIO</button>
</div>
<a class="alignment-crop" id="patch-similarity" href="/images/smolvla-training-strategies/dino_radio_patch_cosine_combined.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/dino_radio_patch_cosine_combined.png" alt="DINOv3 patch-level cosine similarity maps" width="2984" height="1340" loading="lazy" /></a>
</figure>

*Figure 10. Example 8 × 8 teacher–student cosine-similarity maps, shown alongside the corresponding LIBERO images.*

The **DINOv3** alignment maps often showed lower similarity around the robot gripper and manipulated objects than in background regions. The **RADIO** maps were more spatially uniform in these examples.

One interpretation is that background features were easier to align than task-relevant object and gripper features, particularly for DINOv3. The maps alone do not quantify the overall contribution of these regions to the loss or explain the difference in Success Rate.

### 6.3 Matched and Shuffled Correspondence Checks

To test whether the observed similarities were specific to the correct image and spatial configuration, I compared them with **frame-shuffled** and **patch-shuffled** baselines.

| Teacher configuration | Model | Correct pairs | Frame shuffled | Patch shuffled |
|---|---|---:|---:|---:|
| DINOv3 | All-LoRA + DINOv3 alignment | **0.635** | 0.535 | 0.451 |
| DINOv3 | All-LoRA without alignment | 0.009 | 0.009 | 0.009 |
| DINOv3 | `smolvla_base` | 0.010 | 0.010 | 0.010 |
| RADIO | All-LoRA + RADIO alignment | **0.723** | 0.613 | 0.557 |
| RADIO | All-LoRA without alignment | 0.002 | 0.002 | 0.002 |
| RADIO | `smolvla_base` | 0.003 | 0.003 | 0.003 |

The results confirm that visual feature alignment did occur. For both teachers, the aligned models showed higher similarity for correctly matched pairs than for shuffled pairs. Furthermore, patch shuffling reduced similarity more than frame shuffling.

---

## 7. Linear Probing: Measuring Linear Separability

To obtain a quantitative measure of information accessible in the visual features, I trained linear classifiers on embeddings from **50 ImageNet classes**. Each class provided **200 training images and 50 test images**, for a total of **10,000 training** and **2,500 test** images.

The same image set and fixed train/test split were used across all four models. Image-level features were obtained from the VLM visual hidden states at two extraction points:

- **Layer 15:** the hidden-state extraction point before the final layer used in this analysis.
- **Layer 16 (post-normalization):** the final normalized hidden-state extraction point.

| Model | Layer 15 accuracy | Layer 16 accuracy |
|---|---:|---:|
| **SmolVLA Base** | **87.92%** | **87.48%** |
| All-LoRA | 52.92% | 51.60% |
| All-LoRA + DINOv3 | 51.88% | 47.84% |
| All-LoRA + RADIO | 57.24% | 45.68% |

The drop in probe accuracy after All-LoRA was substantial: **87.92% → 52.92% at Layer 15** and **87.48% → 51.60% at Layer 16**.

Alignment did not restore base-model separability. At the **DINOv3 target layer**, accuracy was **51.88%**, slightly below unaligned All-LoRA (**52.92%**). At the **RADIO target layer**, accuracy was **45.68%**, below unaligned All-LoRA (**51.60%**).

However, this does not necessarily mean that the VLM no longer encodes useful visual information. The representations may still contain task-relevant information that cannot be extracted by the particular linear classifier used in this evaluation.

---

## 8. PCA, Effective Rank, and Spectral Concentration

To understand how the geometry of visual representations changes after All-LoRA fine-tuning and additional alignment with DINOv3/RADIO, I analyzed the **spectrum of the centered covariance matrix** of the extracted ImageNet-50 visual embeddings.

**1. PC1 Explained Variance Ratio**

This metric measures the proportion of total variance explained by the first principal component (PC1), which represents the direction along which the embeddings vary the most.

For example, a PC1 ratio of 90% means that a single direction accounts for 90% of the total variance in the embeddings. This indicates that most of the variation is concentrated along one direction rather than being distributed across many dimensions.

**2. Effective Rank (ER)**

Effective Rank measures how many directions effectively contribute to the total variance of the embeddings, taking into account how evenly that variance is distributed.

Unlike the actual dimensionality of the embeddings, Effective Rank reflects their effective dimensionality.

For example, an Effective Rank of 2 means that the variance is distributed as if approximately two equally important directions contributed to it, even if the embeddings have hundreds of dimensions.

**Interpretation**

A high PC1 ratio combined with a low Effective Rank indicates spectral concentration: most of the variance is concentrated in a small number of directions.

However, this does not necessarily mean that the representation has lost useful visual information. Information needed to distinguish objects or perform tasks may still be encoded in directions that explain relatively little variance.

| Model | PC1 (%) L15 | Effective rank L15 | PC1 (%) L16 | Effective rank L16 |
|---|---:|---:|---:|---:|
| SmolVLA Base | 16.43 | 41.32 | 20.70 | 34.65 |
| All-LoRA | 28.25 | 25.32 | 33.31 | 21.94 |
| **All-LoRA + DINOv3** | **93.21** | **1.41** | 89.54 | 1.73 |
| **All-LoRA + RADIO** | 33.97 | 20.52 | **83.78** | **2.01** |

The most striking changes occurred at the **respective alignment targets**:

- **DINOv3, Layer 15:** PC1 explained **93.21%** of total variance, and effective rank fell to **1.41**.
- **RADIO, Layer 16 post-normalization:** PC1 explained **83.78%** of total variance, and effective rank fell to **2.01**.

At Layer 15, the base model needed **57 principal components** to capture 90% of the variance, versus **42** for All-LoRA and **1** for DINOv3 alignment. At Layer 16, RADIO alignment needed only **2 principal components** for the same threshold.

<figure class="figure-extra-wide pca-comparison" data-layer="15" id="figure-11">
<div class="libero-task-picker" role="group" aria-label="Select VLM layer">
<button type="button" data-layer="15" aria-pressed="true" aria-controls="pca-layer-image">Layer 15</button>
<button type="button" data-layer="16" aria-pressed="false" aria-controls="pca-layer-image">Layer 16</button>
</div>
<a class="pca-crop" id="pca-layer-image" href="/images/smolvla-training-strategies/pca_layers_15_16_vertical.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/pca_layers_15_16_vertical.png" alt="PCA visual embeddings at Layer 15" width="2048" height="1545" loading="lazy" /></a>
</figure>

*Figure 11. Image embeddings projected onto the first two PCA directions. The alignment variants show pronounced concentration along dominant axes.*

This effect can be described as **spectral concentration** — a situation in which most of the variance in visual representations is concentrated along a small number of principal directions.

However, **spectral concentration does not necessarily imply representation collapse or the loss of useful information**. For example, if the first principal component explains 93.21% of the variance, this does not mean that all embeddings are nearly identical or that the remaining directions contain no information useful for distinguishing objects.

Furthermore, the results do not show a direct relationship between spectral concentration and linear probing accuracy. At Layer 15, DINOv3 alignment increased PC1 from **28.25% to 93.21%**, while accuracy decreased only slightly, from **52.92% to 51.88%**. Thus, even strong spectral concentration does not necessarily lead to a comparable degradation in linear separability and, by itself, cannot determine the success of robotic tasks.

---

## 9. A Possible Mechanism: Synthetic Alignment Experiments

The real-model results revealed *where* spectral concentration was strongest, but not *why* it occurred. To investigate possible mechanisms without the complexity of a full VLA, I ran controlled synthetic experiments using trainable student representations and small alignment projectors, including frozen MLPs with LayerNorm, frozen linear projectors, and trainable-projector controls.

These experiments are **mechanistic probes**, not numerical replications of the SmolVLA results.

### 9.1 Concentration Can Accompany Variance Expansion

For the frozen LN→MLP, AdamW, alignment-only condition, averaging over five initialization/data-order seeds produced the following changes:

| Metric | Before alignment | After alignment |
|---|---:|---:|
| PC1 explained variance | 4.74% | 14.89% |
| Effective rank | 42.40 | 17.38 |
| Total variance | 87.05 | 719.69 |
| Largest covariance eigenvalue | 4.13 | 107.66 |
| Variance outside PC1 | 82.92 | 612.03 |
| Linear CKA (relative to initial features) | 1.00 | 0.612 |

PC1's share of the variance increased, and effective rank fell. But **total variance increased by more than eightfold**, and **variance outside PC1 also increased**. This is not an example of all features shrinking into a single direction. Rather, the distribution **expanded unevenly**, with its leading direction growing disproportionately.

Per-vector L2 normalization reduced some of the spectral concentration in additional ablations but did not eliminate it. Thus, changing vector norms was not the sole source of the observed spectral changes.

### 9.2 Projector Architecture and Freezing

I tested whether the phenomenon depended specifically on LayerNorm. It did not: **frozen linear projectors without LayerNorm** could also produce strong spectral concentration. In the synthetic ablations, **trainable projectors** generally preserved the student's original covariance geometry better than frozen projectors.

<figure class="figure-wide"><a href="/images/smolvla-training-strategies/representation_expansion_overview.png" target="_blank" rel="noopener"><img src="/images/smolvla-training-strategies/representation_expansion_overview.png" alt="Synthetic representation alignment ablations" width="2150" height="1250" loading="lazy" /></a></figure>

*Figure 12. Five-seed synthetic ablations. The upper panel shows PC1 explained variance and the lower panel shows effective rank before and after per-vector L2 normalization. These are separate controlled runs, not the ImageNet-50 analysis above.*

One possible explanation is that, when the projector is frozen, the student must adapt its hidden representations to satisfy the teacher-space objective. If the projector is trainable, some adaptation can instead occur inside the projector. This mechanism is plausible and supported by the synthetic comparisons, but the experiments do not establish that it alone caused the SmolVLA results.

### 10. Potential Solutions

**1. Trainable Projector**

In the synthetic experiments, trainable projectors generally preserved the geometry of the student's original representations better than frozen projectors.

However, there is a trade-off: a trainable projector may adapt to the teacher's features without substantially improving the VLM's underlying representations.

**2. Representation Regularization**

Another approach is to introduce constraints directly on the distribution of student embeddings.

For example, variance and covariance regularization could be used to prevent an excessive reduction in representation diversity.

However, the goal should not necessarily be to achieve a uniform spectrum. Anisotropic representations may still contain useful information, and excessive regularization could limit the model's ability to adapt to robotic tasks.

**3. Task-Relevant Alignment**

The patch-level cosine similarity analysis showed that, with DINOv3 alignment, features corresponding to manipulated objects and the robot gripper often exhibited lower similarity than background regions.

One possible approach is to introduce a weighted Alignment Loss that assigns greater importance to regions containing manipulated objects and the robot gripper.

This could help focus training on visual features most relevant to task execution, rather than applying equal alignment pressure across all spatial regions.


---

## 11. Discussion

### 11.1 Better General-Purpose Probes Did Not Predict Better Robot Control

All-LoRA achieved higher LIBERO-10 SR than the expert-only strategies even though its ImageNet-50 linear-probe accuracy was substantially lower than that of the original VLM. Preserving the initial VLM parameters was therefore **not sufficient to obtain the highest policy SR** under these training configurations.

This should not be interpreted as evidence that worse visual features *help* robot control. Image classification and robotic manipulation impose different requirements, and Expert-Only FT versus All-LoRA is not a single-variable ablation. The Action Expert can benefit from a representation that is highly adapted to the demonstrations even when a general-purpose linear probe performs worse.

### 11.2 Alignment Was Real, but Was Not Enough

The matched-versus-shuffled tests indicated meaningful image-specific and spatial correspondence between projected student features and the teacher targets. Nevertheless, neither alignment configuration recovered base-model probe accuracy or improved SR on LIBERO-10 or the two tested OOD shifts.

An alignment objective constrains similarity in a projected feature space; it does not directly guarantee preservation of all useful properties in the unprojected student space.

### 11.3 What Spectral Concentration Does—and Does Not—Explain

Both DINOv3 and RADIO alignment produced their strongest observed spectral concentration at the chosen target layers. That is a compelling association, especially alongside the controlled synthetic results. However, there is **no demonstrated causal link** from the measured PC1 increase to the observed decrease in SR.

The synthetic findings suggest a possible route through **frozen-projector alignment and anisotropic representation expansion**. Determining whether the same mechanism operates in SmolVLA would require measuring the absolute covariance eigenvalues, total variance, and variance outside PC1 in the real embeddings—not just the normalized variance ratios—and ideally performing controlled ablations of projector freezing and optimizer settings.

### 11.4 Relation to *Don't Blind Your VLA*

The original [*Don't Blind Your VLA*](https://arxiv.org/abs/2510.25616) work reported benefits from aligning VLA representations with pretrained visual teachers. My results do not reproduce an SR improvement for the **specific SmolVLA All-LoRA configurations** evaluated here.

This does not invalidate the original method. The architecture, training setup, alignment targets, and teacher choices are not all identical. The present results instead suggest that **the effects of visual alignment may depend strongly on how the objective is integrated with the model and training procedure**.

### 11.5 Limitations

This study has several important limitations:

1. **Limited statistical replication.** Reported SR values are based on the specified evaluation episodes and available training runs, rather than independently repeated training seeds with confidence intervals.
2. **Confounded alignment comparison.** DINOv3 and RADIO runs used different teachers, target layers, and loss weights.
3. **Mechanism not established in SmolVLA.** The synthetic experiments demonstrate plausible behavior in simpler systems, not a proven explanation for the real-model results.

---

## 12. Conclusion

I compared three SmolVLA adaptation strategies and evaluated two additional variants using visual representation alignment. **All-LoRA achieved the highest observed LIBERO-10 SR (78.5%) and the strongest performance on the two tested LIBERO-Plus OOD categories** among the evaluated alignment variants. DINOv3 and RADIO alignment did not improve those results.

At the same time, All-LoRA substantially reduced ImageNet-50 linear-probe accuracy relative to the original VLM, and alignment did not restore that separability. Instead, the aligned models showed pronounced **spectral concentration at their respective alignment layers**. Synthetic controls suggest that alignment through a frozen projector can be accompanied by **anisotropic variance expansion**, rather than necessarily by complete representation collapse.

The central finding is that **preserving general-purpose linear separability, increasing teacher–student cosine similarity, and improving robotic task success are distinct objectives**. Evaluating VLA adaptation methods requires measuring all three rather than assuming that improvements in one guarantee improvements in the others.

## References and Reproducibility

- [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844) — base VLA architecture.
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310) — in-distribution robot-manipulation benchmark.
- [LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html) — OOD robustness evaluation.
- [Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization](https://arxiv.org/abs/2510.25616) — visual alignment method motivating the alignment experiments.
- [SmolVLA alignment implementation](https://github.com/NickS0kolov/lerobot/tree/feat/smolvla-alignment) — project implementation and code.
- [SmolVLA LIBERO-10 experiments on Hugging Face](https://huggingface.co/NickSokolov/SmolVLA-LIBERO10-Experiments).
