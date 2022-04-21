# [GM-TOuNN: Graded Multiscale Topology Optimization using  Neural Networks](https://arxiv.org/abs/2204.06682)

[Aaditya Chandrasekhar*](https://aadityacs.github.io/), [Saketh Sridhara*](https://sakethsridhara.github.io/), [Krishnan Suresh](https://directory.engr.wisc.edu/me/faculty/suresh_krishnan)  
University of Wisconsin-Madison 


## Abstract

Multiscale topology optimization (M-TO) entails generating an optimal global topology,and an optimal set of microstructures at a smaller scale, for a physics-constrained problem. With the advent of additive manufacturing, M-TO has gained significant prominence. However, generating  distinct and optimal microstructures  at various  locations can be computationally very expensive. As an alternate, graded multiscale topology optimization (GM-TO) has been proposed where one or more pre-selected and graded (parameterized) microstructural topologies are used to fill the domain optimally. This leads to a significant reduction in computation while retaining many of the benefits of M-TO.
	
A successful GM-TO framework must: (1) be capable of efficiently handling numerous pre-selected microstructures, (2) be able to continuously switch between these  microstructures (during optimization), (3) ensure that the partition of unity is satisfied, and (4) discourage microstructure mixing at termination.
	
In this paper, we propose to meet these requirements by exploiting the unique classification capacity of neural networks. Specifically, we propose a graded multiscale topology optimization using neural-network (GM-TOuNN) framework with the following features: (1) the number of design variables is only weakly dependent on the number of pre-selected microstructures, (2) it guarantees partition of unity while discouraging microstructure mixing, and (3) it supports automatic differentiation, thereby  eliminating manual sensitivity analysis. The proposed framework is illustrated through several examples.

## Citation

```
@article{chand2022GMTOuNN,
author = {Chandrasekhar, Aaditya and Sridhara, Saketh and Suresh, Krishnan},
title = {GM-TOuNN: Graded Multiscale Topology Optimization using  Neural Networks},
journal = {arXiv preprint arXiv:2204.06682},
year={2022}
}
```

*contributed equally
