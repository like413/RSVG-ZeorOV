# RSVG-ZeorOV
[AAAI 2026] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images.

**[📄 [arXiv]](https://arxiv.org/abs/2509.18711)**  &emsp; 

This repository is the official implementation:
> [RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](https://arxiv.org/abs/2509.18711)  
> Ke Li, Di Wang, Ting Wang, Fuyu Dong, Yiming Zhang, Luyao Zhang, Xiangyu Wang, Shaofeng Li, Quan Wang

## 📢 Update
- **(2025/11/8)** The benchmark method [VisTA](https://github.com/like413/RSVG-ZeorOV/edit/main) is released.

## Abstract
Remote sensing visual grounding (RSVG) aims to localize objects in remote sensing images based on free-form natural language expressions. Existing approaches are typically constrained to closed-set vocabularies, limiting their applicability in open-world scenarios. While recent attempts to leverage generic foundation models for open-vocabulary RSVG, they overly rely on expensive high-quality datasets and time-consuming fine-tuning. To address these limitations, we propose RSVG-ZeroOV, a training-free framework that aims to explore the potential of frozen generic foundation models for zero-shot open-vocabulary RSVG. Specifically, RSVG-ZeroOV comprises three key stages:
(i) Overview: We utilize a vision-language model (VLM) to obtain cross-attention maps that capture semantic correlations between text queries and visual regions.
(ii) Focus: By leveraging the fine-grained modeling priors of a diffusion model (DM), we fill in gaps in structural and shape information of objects, which are often overlooked by VLM.
(iii) Evolve: A simple yet effective attention evolution module is introduced to suppress irrelevant activations, yielding purified segmentation masks over the referred objects.
Without cumbersome task-specific training, RSVG-ZeroOV offers an efficient and scalable solution. Extensive experiments demonstrate that the proposed framework consistently outperforms existing weakly-supervised and zero-shot methods.

<div align="center">
  <img src="https://github.com/like413/VisTA/blob/main/fig/1.png?raw=true" width="100%" height="100%"/>
</div><br/>
