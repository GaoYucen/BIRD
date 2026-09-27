# BIRD

Official code release for the ICDE 2023 paper:

**Online Shipping Container Pricing Strategy Achieving Vanishing Regret with Limited Inventory**

BIRD (Balancing Inventory and Revenue with an epsilon-chasing Decider) combines multiple online pricing strategies using online strategy selection and a chasing mechanism under limited inventory.

## Citation

If you find this work useful, please cite:

> **Yucen Gao**, Xikai Wei, Xi Jing, Yangguang Shi, Xiaofeng Gao, Guihai Chen.  
> *Online Shipping Container Pricing Strategy Achieving Vanishing Regret with Limited Inventory.*  
> IEEE International Conference on Data Engineering (ICDE), 2023, pp. 1719–1731.

```bibtex
@inproceedings{gao2023bird,
  author    = {Yucen Gao and Xikai Wei and Xi Jing and Yangguang Shi and Xiaofeng Gao and Guihai Chen},
  title     = {Online Shipping Container Pricing Strategy Achieving Vanishing Regret with Limited Inventory},
  booktitle = {2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages     = {1719--1731},
  year      = {2023},
  doi       = {10.1109/ICDE55515.2023.00392}
}
```

## Repository structure

```text
DP.py                     Dynamic-programming pricing baseline
chasing.py                Main BIRD / chasing experiment
RL-DDPG/DDPG/             Legacy TensorFlow DDPG implementation
RL-DDPG/DataGen/          Data-generation utilities
RL-DDPG/preprocessing/    Preprocessing utilities
data/README.md            Data availability and expected inputs
```

## Environment

The original implementation was developed with Python 3.6 and TensorFlow 1.x.

```bash
python -m pip install -r requirements.txt
```

## Usage

The dynamic-programming component can be inspected with `python DP.py`.
The full historical BIRD pipeline starts from `python chasing.py` and expects the historical DDPG artifacts and data interfaces described in `data/README.md`.

## Data

The paper experiments use historical shipping data provided by COSCO. Raw commercial shipping records and derived training files are not redistributed in this public repository.

## PyTorch implementation

A maintained PyTorch reproduction is available on the `pytorch-repro` branch and is kept separate from this conference-code release.
