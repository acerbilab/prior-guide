# PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference
Repository for the paper “[PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference](https://openreview.net/forum?id=G4I23g5Ugh)” (Yang et al., ICLR 2026).

---

## Installation
Clone the repository. Create an virtual environment and install the dependencies. This codebase has been verified to run on Python 3.10.
```bash
git clone https://github.com/acerbilab/prior-guide.git
cd prior-guide
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
To train a base diffusion model:
```bash
python -m priorg.train task.name=<task> task.num_simulations=10000 seed=<seed>
```

Replace `<task>` with one of the following supported task names:

- `oup`
- `two_moons`
- `turin`
- `gaussian_linear`
- `gaussian_linear_high`
- `bav` (for the BCI task in the paper)

For example, to train the OUP model with a seed value of `0`, run:

```bash
python -m priorg.train task.name=oup task.num_simulations=10000 seed=0
```
The training configuration can be found in `priorg/cfg/train.yaml`

## Experiments
To replicate our experiments, check the scripts under `experiments`.
#### Test-Time Prior Generations
```bash
# Uniform training priors
python experiments/data/priors/gen_priors_uniform.py --task <task>
# Gaussian training priors
python experiments/data/priors/gen_priors_gaussian.py --task <task>
```
#### Posterior Inference
```bash
# Uniform training priors
python experiments/posterior/run_prior_guide_uniform.py --task <task>
# Gaussian training priors
python experiments/posterior/run_prior_guide_gaussian.py --task <task>
```
#### Posterior Predictive Inference
```bash
python experiments/posterior_predictive/run_prior_guide.py --task <task>
```

## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{yang2026priorguide,
  title={PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference},
  author={Yang Yang and Severi Rissanen and Paul Edmund Chang and Nasrulloh Ratu Bagus Satrio Loka and Daolang Huang and Arno Solin and Markus Heinonen and Luigi Acerbi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=G4I23g5Ugh}
}
```


---
