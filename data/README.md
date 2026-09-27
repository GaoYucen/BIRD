# Data availability

The ICDE 2023 experiments use historical shipping-container transaction data provided by COSCO.

Raw commercial records and derived training files are not redistributed in this public repository.

The legacy source expects demand/price time-series data, container-count distributions, DDPG pretraining arrays, and route-specific historical pricing records. Researchers using authorized data can adapt the relative paths in `RL-DDPG/DDPG/config.py` and the preprocessing utilities.
