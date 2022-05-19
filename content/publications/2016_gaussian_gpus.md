---
title: 'Multi-GPUs Gaussian filtering for real-time big data processing'
author: Chaolong Zhang
status: Published
type: published
citation: "<b>Zhang, C.L.</b>, Xu, Y., He, J., Lu, J., Lu, L., & Xu, Z. (2016). Multi-GPUs Gaussian filtering for real-time big data processing. 2016 10th International Conference on Software, Knowledge, Information Management & Applications (SKIMA),  231-236."
tag: GPU
subjects: CUDA
comments: no
fave: true
doi: 10.1109/SKIMA.2016.7916225
date: 2016-12-15
publishdate: 2016-12-15
---

Gaussian filtering has been extensively used in the field of surface metrology. However, the computing performance becomes a core bottleneck for Gaussian filtering algorithm based applications when facing large-scale and/or real-time data processing. Although researchers tried to accelerate Gaussian filtering algorithm by using GPU (Graphics Processing Unit), single GPU still fail to meet the large-scale and real-time requirements of surface texture micro- and nano-measurements. Therefore, to solve this bottleneck problem, this paper proposes a single node multi-GPUs based computing framework to accelerate the 2D Gaussian filtering algorithm. This paper presents that the devised framework seamlessly integrated the multi-level spatial domain decomposition method and the CUDA stream mechanism to overlap the two main time consuming steps, i.e., the data transfer and GPU kernel execution, such that it can increase concurrency and reduce the overall running time. This paper also tests and evaluates the proposed computing framework with other three conventional solutions by using large-scale measured data extracted from real mechanical surfaces, and the final results show that the proposed framework achieved higher efficiency. It also proved that this framework satisfies the real-time and big data requirements in micro- and nano-surface texture measurement.