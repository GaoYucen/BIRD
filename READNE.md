# BIRD

Online Shipping Container Pricing Strategy Achieving Vanishing Regret with Limited Inventory

# Usage

Use the following command to run the codes: python chasing_RL.py

# Repo Structure

The structure of our code and description of important files are given as follows:
├────model/
│    ├────DP/:  code of dynamic pricing
│    ├────RL/:  code to get price through RL
│    └────RL-DDPG/: code of RL  
├────data/  
│    ├────YIK_QZH_COMPLETE_WBL/:  historical data of shipping waybill
│    └────YIK_QZH_COMPLETE_CNTR_FRT/: historical data of shipping cost
└────chasing_RL.py: Main function of this project, used to test the performance of BIRD and other baseline strategies

