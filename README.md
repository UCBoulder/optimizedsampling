# Optimized Sampling

usage: train.py

optional arguments:
  -h, --help            show this help message and exit
  --labels              Labels to run
  --method METHOD       Sampling method: random, image, satclip, greedycost, clusters, invsize
  --cost COST           Cost function: unif, lin, lin+rad, state, cluster
  --budgets             Budget(s)
  -a, --alpha           Value of alpha in alpha*dist+beta
  -b, --beta            Value of beta in alpha*dist+beta
  --gamma               Value of constant valued in rad or unif
  --radius              Radius around city (if cost is lin+rad)
  --states              States to sample from
  --avg AVG             Average results (Boolean)
  --l L                 Hyperparameter lambda in optimization problem
  --cluster_type        Cluster type
  --test_split          Test region