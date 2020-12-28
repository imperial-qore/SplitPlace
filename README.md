[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/COSCO/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FCOSCO&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/SimpleFogSim/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=aeneas)
<br>
![Docker pulls mnist_layer](https://img.shields.io/docker/pulls/shreshthtuli/mnist_layer?label=mnist_layer)
![Docker pulls fashionmnist_layer](https://img.shields.io/docker/pulls/shreshthtuli/fashionmnist_layer?label=fashionmnist_layer)
![Docker pulls cifar100_layer](https://img.shields.io/docker/pulls/shreshthtuli/cifar100_layer?label=cifar100_layer)
<br>
![Docker pulls mnist_semantic](https://img.shields.io/docker/pulls/shreshthtuli/mnist_semantic?label=mnist_semantic)
![Docker pulls fashionmnist_semantic](https://img.shields.io/docker/pulls/shreshthtuli/fashionmnist_semantic?label=fashionmnist_semantic)
![Docker pulls cifar100_semantic](https://img.shields.io/docker/pulls/shreshthtuli/cifar100_semantic?label=cifar100_semantic)

# COSCO Framework

COSCO is an AI based coupled-simulation and container orchestration framework for integrated Edge, Fog and Cloud Computing Environments.  It's a simple python based software solution, where academics or industrialists  can develop, simulate, test and deploy their scheduling policies. 

<img src="https://github.com/imperial-qore/COSCO/blob/master/wiki/COSCO.jpg" width="900" align="middle">


## Advantages of COSCO
1. Hassle free development of AI based scheduling algorithms in integrated edge, fog and cloud infrastructures.
2. Provides seamless integration of scheduling policies with simulated back-end for enhanced decision making.
3. Supports container migration physical deployments (not supported by other frameworks) using CRIU utility.
4. Multiple deployment support as per needs of the developers. (Vagrant VM testbed, VLAN Fog environment, Cloud based deployment using Azure/AWS/OpenStack)
5. Equipped with a smart real-time graph generation of utilization metrics using InfluxDB and Grafana.
6. Real time metrics monitoring, logging and consolidated graph generation using custom Stats logger.

The basic architecture of COSCO has two main packages: <br>
**Simulator:** It's a discrete event simulator and runs in a standalone system. <br>
**Framework:** It’s a kind of tool to test the scheduling algorithms in a physical(real time) fog/cloud environment with real world applications.

## Wiki
Access the [wiki](https://github.com/imperial-qore/COSCO/wiki) for installation instructions and replication of results.


## License

BSD-3-Clause. 
Copyright (c) 2020, Shreshth Tuli.
All rights reserved.

See License file for more details.
