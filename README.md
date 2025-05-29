<div align='center'>
 
# Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks

[![preprint](https://img.shields.io/badge/arXiv-2503.23751-B31B1B)](https://arxiv.org/abs/2503.23751)



[![Venue:ICLR 2024](https://img.shields.io/badge/Venue-ICLR%202024%20(Spotlight)-blue)](https://iclr.cc/virtual/2024/poster/18130)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/Unlearn-Saliency?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)
</div>



This is the official code repository for the CVPR 2025 Highlight paper [Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks](https://arxiv.org/abs/2503.23751).

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/transition_new.gif" alt="Examples" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Example comparison of pre/after unlearning by SalUn. <br/> (Left) Concept "Nudity"; (Middle) Object "Dog"; (Right) Style "Sketch".</em>
    </td>
  </tr>
</table>


## Abstract

In this work, we present **DE**coup**LE**d Distillation **T**o **E**rase (**DELETE**), a general and strong unlearning method for any class-centric tasks. To derive this, we first propose a theoretical framework to analyze the general form of unlearning loss and decompose it into forgetting and retention terms. Through the theoretical framework, we point out that a class of previous methods could be mainly formulated as a loss that implicitly optimizes the forgetting term while lacking supervision for the retention term, disturbing the distribution of pre-trained model and struggling to adequately preserve knowledge of the remaining classes.
To address it, we refine the retention term using "dark knowledge" and propose a mask distillation unlearning method. By applying a mask to separate forgetting logits from retention logits, our approach optimizes both the forgetting and refined retention components simultaneously, retaining knowledge of the remaining classes while ensuring thorough forgetting of the target class.
Without access to the remaining data or intervention (*i.e.*, used in some works), we achieve state-of-the-art performance across various benchmarks. What's more, DELETE is a general solution that can be applied to various downstream tasks, including face recognition, backdoor defense, and semantic segmentation with great performance.


## Cite This Work
```
@article{zhou2025decoupled,
  title={Decoupled distillation to erase: A general unlearning method for any class-centric tasks},
  author={Zhou, Yu and Zheng, Dian and Mo, Qijie and Lu, Renjie and Lin, Kun-Yu and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2503.23751},
  year={2025}
}
```