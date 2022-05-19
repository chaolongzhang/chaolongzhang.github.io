---
title: 'Dynamic load balancing on multi-GPUs system for big data processing'
author: Chaolong Zhang
status: Published
type: published
citation: "<b>Zhang, C.</b>, Xu, Y., Zhou, J., Xu, Z., Lu, L., & Lu, J. (2017). Dynamic load balancing on multi-GPUs system for big data processing. 23rd International Conference on Automation and Computing (ICAC)."
tag: GPU
subjects: CUDA
comments: no
fave: true
doi: 10.23919/IConAC.2017.8082085
date: 2017-09-07
publishdate: 2017-09-07
---

The powerful parallel computing capability of modern GPU (Graphics Processing Unit) processors has attracted increasing attentions of researchers and engineers who had conducted a large number of GPU-based acceleration research projects. However, current single GPU based solutions are still incapable of fulfilling the real-time computational requirements from the latest big data applications. Thus, the multi-GPU solution has become a trend for many real-time application attempts. In those cases, the computational load balancing over the multiple GPU nodes is often the key bottleneck that needs to be further studied to ensure the best possible performance. The existing load balancing approaches are mainly based on the assumption that all GPUs in the same system provide equal computational performance, and had fallen short to address the situations from heterogeneous multi-GPU systems. This paper presents a novel dynamic load balancing model for heterogeneous multi-GPU systems based on the fuzzy neural network (FNN) framework. The devised model has been implemented and demonstrated in a case study for improving the computational performance of a two dimensional (2D) discrete wavelet transform (DWT). Experiment results show that this dynamic load balancing model has enabled a high computational throughput that can satisfy the real-time and accuracy requirements from many big data processing applications.