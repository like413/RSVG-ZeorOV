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
  <img src="https://github.com/like413/RSVG-ZeorOV/blob/main/assets/total.jpg?raw=true" width="100%" height="100%"/>
</div><br/>

## 🌟 Installation

### Requirements

- Python 3.8.20
- Numpy
- Pytorch 2.1.0

1. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```
2. Please set up the Qwen model and its runtime environment following the official Qwen documentation:https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

3. Please configure the SAM model and its runtime environment following the official SAM documentation:https://github.com/facebookresearch/segment-anything, and put SAM pretrained model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth into ./sam_checkpoint

## 🌈 Datasets

1. Please download the corresponding dataset from the official link.
   RRSIS-D:https://github.com/Lsan2401/RMSIN   RISBench:https://github.com/HIT-SIRS/CroBIM

## 🔥 Inference

This project consists of three core modules for extracting cross-modal attention, modeling structural information, and refining predictions through attention fusion. The key components are as follows:
1. This module extracts cross-attention maps from the Vision-Language Model (VLM), enabling an initial localization of the referred object.
    ```
        bash llmattn.sh
    ```
2. This module obtains self-attention maps from the Diffusion Model (DM) to capture the structural features of the referred object.
    ```
        bash generate_diffusion.sh
    ```
3. This module fuses the cross- and self-attention maps and utilizes the Segment Anything Model (SAM) for refinement to produce more accurate segmentation results.
    ```
        python rs_evolve.py
    ```
    
```bibtex
@article{li2025rsvg,
  title={RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images},
  author={Li, Ke and Wang, Di and Wang, Ting and Dong, Fuyu and Zhang, Yiming and Zhang, Luyao and Wang, Xiangyu and Li, Shaofeng and Wang, Quan},
  journal={arXiv preprint arXiv:2509.18711},
  year={2025}
}
}
```    
