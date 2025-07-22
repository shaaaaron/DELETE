<div align='center'>
 
# Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks

[![preprint](https://img.shields.io/badge/arXiv-2503.23751-B31B1B)](https://arxiv.org/abs/2503.23751)
[![GitHub top language](https://img.shields.io/github/languages/top/shaaaaron/DELETE)](https://github.com/shaaaaron/DELETE)
[![GitHub repo size](https://img.shields.io/github/repo-size/shaaaaron/DELETE)](https://github.com/shaaaaron/DELETE)
</div>

This is the official code repository for the **CVPR 2025 Highlight paper** <img src="./assets/badage.png" width="40" style="vertical-align: text-bottom;"> 

## Poster

[Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks](https://arxiv.org/abs/2503.23751).

![GitHub repo size](./assets/CVPR_POSTER.png)

## Abstract

In this work, we present **DE**coup**LE**d Distillation **T**o **E**rase (**DELETE**), a general and strong unlearning method for any class-centric tasks. To derive this, we first propose a theoretical framework to analyze the general form of unlearning loss and decompose it into forgetting and retention terms. Through the theoretical framework, we point out that a class of previous methods could be mainly formulated as a loss that implicitly optimizes the forgetting term while lacking supervision for the retention term, disturbing the distribution of pre-trained model and struggling to adequately preserve knowledge of the remaining classes.
To address it, we refine the retention term using "dark knowledge" and propose a mask distillation unlearning method. By applying a mask to separate forgetting logits from retention logits, our approach optimizes both the forgetting and refined retention components simultaneously, retaining knowledge of the remaining classes while ensuring thorough forgetting of the target class.
Without access to the remaining data or intervention (*i.e.*, used in some works), we achieve state-of-the-art performance across various benchmarks. What's more, DELETE is a general solution that can be applied to various downstream tasks, including face recognition, backdoor defense, and semantic segmentation with great performance.

## Run the Code

This code has no special environment requirements.

For running experiments on CIFAR-10, please refer to `scripts.sh`. 

In the script, Most methods are configured with their recommended (or optimal) settings. If the performance is not as expected, you may need to adjust the parameters as detailed in the Supplementary Material.

> The `unlearn_rate` in `config` is just as default setting. Generally, different methods may require different parameter settings.

## Cite This Work
```
@inproceedings{zhou2025decoupled,
  title={Decoupled distillation to erase: A general unlearning method for any class-centric tasks},
  author={Zhou, Yu and Zheng, Dian and Mo, Qijie and Lu, Renjie and Lin, Kun-Yu and Zheng, Wei-Shi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20350--20359},
  year={2025}
}
```
