---
title: 'A generic parallel computational framework of lifting wavelet transform for online engineering surface filtration'
author: Chaolong Zhang
status: Published
type: published
citation: "Xu, Y., <b>Zhang, C.</b>, Xu, Z., Zhou, J., Wang, K., & Huang, J. (2019). A generic parallel computational framework of lifting wavelet transform for online engineering surface filtration. Signal Processing, 165, 37-56."
tag: GPU
subjects: CUDA
comments: no
fave: true
doi: 10.1016/j.sigpro.2019.06.019
date: 2019-07-02
publishdate: 2019-07-02
---

Nowadays, complex geometrical surface texture measurement and evaluation require advanced filtration techniques. Discrete wavelet transform (DWT), especially the second-generation wavelet (Lifting Wavelet Transform – LWT), is the most adopted one due to its unified and abundant characteristics in measured data processing, geometrical feature extraction, manufacturing process planning, and production monitoring. However, when dealing with varied complex functional surfaces, the computational payload for performing DWT in real-time often becomes a core bottleneck in the context of massive measured data and limited computational capacities. It is a more prominent problem for the areal surface texture filtration by using 2D DWT. To address the issue, this paper presents a generic parallel computational framework for lifting wavelet transform (GPCF-LWT) based on Graphics Process Unit (GPU) clusters and the Compute Unified Device Architecture (CUDA). Due to its cost-effective hardware design and the powerful parallel computing capacity, the proposed framework can support online (or near real-time) engineering surface filtration for micro- and nano-scale surface metrology through exploring a novel parallel method named LBB model, the improved algorithms of lifting scheme and three implementation optimizations on the heterogeneous multi-GPU systems. The innovative approach enables optimizations on individual GPU node through an overarching framework that is capable of data-oriented dynamic load balancing (DLB) driven by a fuzzy neural network (FNN). The paper concludes with a case study on filtering and extracting manufactured surface topographical characteristics from real surfaces. The experimental results have demonstrated substantial improvements on the GPCF-LWT implementation in terms of computational efficiency, operational robustness, and task generalization.