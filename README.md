# PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference
Repository for the paper “PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference” (Yang et al., 2025).

Code coming soon — currently under preparation.

---

## Abstract
> Amortized simulator-based inference offers a powerful framework for tackling Bayesian inference in computational fields such as engineering or neuroscience, increasingly leveraging modern generative methods like diffusion models to map observed data to model parameters or future predictions. These approaches yield posterior or posterior-predictive samples for new datasets without requiring further simulator calls after training on simulated parameter-data pairs. 
However, their applicability is often limited by the prior distribution(s) used to generate model parameters during this training phase. To overcome this constraint, we introduce *PriorGuide*, a technique specifically designed for diffusion-based amortized inference methods. PriorGuide leverages a novel guidance approximation that enables flexible adaptation of the trained diffusion model to new priors at test time, crucially without costly retraining. This allows users to readily incorporate updated information or expert knowledge post-training, enhancing the versatility of pre-trained inference models.

---
