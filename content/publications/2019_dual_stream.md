---
title: 'Dual-channel CNN for efficient abnormal behavior identification through crowd feature engineering'
author: Chaolong Zhang
status: Published
type: published
citation: "Xu, Y., Lu, L., Xu, Z., He, J., Zhou, J., & <b>Zhang, C</b>. (2019). Dual-channel CNN for efficient abnormal behavior identification through crowd feature engineering. Machine Vision and Applications, 30(5), 945-958."
tag: Human Action Recogntion
subjects: Human Action Recogntion
comments: no
fave: false
doi: 10.1007/s00138-018-0971-6
date: 2019-07-01
publishdate: 2019-07-01
---

This research has been investigating an automatic and online crowd anomaly detection model by exploring a novel compound image descriptor generated from live video streams. A dual-channel convolutional neural network (DCCNN) has been set up for efficiently processing scene-related and motion-related crowd information inherited from raw frames and the compound descriptor instances. The novelty of the work stemmed from the creation of the spatio-temporal cuboids in online (or near real-time) manner through dynamically extracting local feature tracklets within the temporal space while handling the foreground region-of-interests (i.e., moving targets) through the exploration of Gaussian Mixture Model in the spatial space. Hence, the extracted foreground blocks can effectively eliminate irrelevant backgrounds and noises from the live streams for reducing the computational costs in the subsequent detecting phases. The devised compound feature descriptor, named as spatio-temporal feature descriptor (STFD), is capable of characterizing the crowd attributes through the measures such as collectiveness, stability, conflict and density in each online generated spatio-temporal cuboid. A STFD instance registers not only the dynamic variation of the targeted crowd over time based on local feature tracklets, but also the interaction information of neighborhoods within a crowd, e.g., the interaction force through the K-nearest neighbor (K-NN) analysis. The DCCNN developed in this research enables online identification of suspicious crowd behaviors based on analyzing the live-feed images and their STFD instances. The proposed model has been developed and evaluated against benchmarking techniques and databases. Experimental results have shown substantial improvements in terms of detection accuracy and efficiency for online crowd abnormal behavior identification.