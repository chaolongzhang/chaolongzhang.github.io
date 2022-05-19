---
title: 'A Fuzzy Neural Network Based Dynamic Data Allocation Model on Heterogeneous Multi-GPUs for Large-scale Computations'
author: Chaolong Zhang
status: Published
type: published
citation: "<b>Zhang, C.L.</b>, Xu, Y.P., Xu, Z.J., He, J., Wang, J., & Adu, J.H. (2018). A Fuzzy Neural Network Based Dynamic Data Allocation Model on Heterogeneous Multi-GPUs for Large-scale Computations. International Journal of Automation and Computing, 15(2), 181-193."
tag: GPU
subjects: CUDA
comments: no
fave: true
doi: 10.1007/s11633-018-1120-4
date: 2018-04-01
publishdate: 2018-04-01
---

The parallel computation capabilities of modern graphics processing units (GPUs) have attracted increasing attention from researchers and engineers who have been conducting high computational throughput studies. However, current single GPU based engineering solutions are often struggling to fulfill their real-time requirements. Thus, the multi-GPU-based approach has become a popular and cost-effective choice for tackling the demands. In those cases, the computational load balancing over multiple GPU “nodes” is often the key and bottleneck that affect the quality and performance of the real-time system. The existing load balancing approaches are mainly based on the assumption that all GPU nodes in the same computer framework are of equal computational performance, which is often not the case due to cluster design and other legacy issues. This paper presents a novel dynamic load balancing (DLB) model for rapid data division and allocation on heterogeneous GPU nodes based on an innovative fuzzy neural network (FNN). In this research, a 5-state parameter feedback mechanism defining the overall cluster and node performance is proposed. The corresponding FNN-based DLB model will be capable of monitoring and predicting individual node performance under different workload scenarios. A real-time adaptive scheduler has been devised to reorganize the data inputs to each node when necessary to maintain their runtime computational performance. The devised model has been implemented on two dimensional (2D) discrete wavelet transform (DWT) applications for evaluation. Experiment results show that this DLB model enables a high computational throughput while ensuring real-time and precision requirements from complex computational tasks.