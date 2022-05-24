[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/SplitPlace/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FSplitPlace&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![SplitPlace-Benchmarks](https://github.com/imperial-qore/SplitPlace/workflows/SplitPlace-Benchmarks/badge.svg)
<br>
![Docker pulls mnist_layer](https://img.shields.io/docker/pulls/shreshthtuli/mnist_layer?label=mnist_layer)
![Docker pulls fashionmnist_layer](https://img.shields.io/docker/pulls/shreshthtuli/fashionmnist_layer?label=fashionmnist_layer)
![Docker pulls cifar100_layer](https://img.shields.io/docker/pulls/shreshthtuli/cifar100_layer?label=cifar100_layer)
![Docker pulls mnist_semantic](https://img.shields.io/docker/pulls/shreshthtuli/mnist_semantic?label=mnist_semantic)
![Docker pulls fashionmnist_semantic](https://img.shields.io/docker/pulls/shreshthtuli/fashionmnist_semantic?label=fashionmnist_semantic)
![Docker pulls cifar100_semantic](https://img.shields.io/docker/pulls/shreshthtuli/cifar100_semantic?label=cifar100_semantic)

# SplitPlace Framework

SplitPlace is a container orchestration framework for dynamic scheduling and decision making in resource constrained edge environments. SplitPlace decides whether to use semantic or layer wise splits of neural network applications with latency and accuracy critical user requirements on distributed setups with low memory legacy devices.

## Quick Start Guide

SplitPlace is based on the [COSCO Framework](https://github.com/imperial-qore/COSCO) and uses the co-simulation and surrogate optimization primitives of COSCO. To run the framework, install required packages using
```bash
python3 install.py
```

To run the code with the required scheduler, modify lines 81 and 85 of `main.py` to one of the several options.
```python
decider = MABDecider()
...
scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))
```

To run the simulator, use the following command
```bash
python3 main.py
```

## Wiki 
Access the [wiki](https://github.com/imperial-qore/COSCO/wiki) for installation instructions and replication of results.


## Links
| Items | Contents | 
| --- | --- |
| **Paper** | (coming soon) |
| **Pre-print** | https://arxiv.org/pdf/2205.10635.pdf |
| **Documentation** | https://github.com/imperial-qore/COSCO/wiki |
| **Video** | (coming soon) |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship, H2020-825040 (RADON) |


## Cite this work
Our work is published in IEEE TMC journal. Cite using the following bibtex entry.
```bibtex
@article{tuli2021splitplace,
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R.},
  journal={IEEE Transactions on Mobile Computing}, 
  title={{SplitPlace: AI Augmented Splitting and Placement of Large-Scale Neural Networks in Mobile Edge Environments}}, 
  year={2022}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2020, Shreshth Tuli.
All rights reserved.

See License file for more details.
