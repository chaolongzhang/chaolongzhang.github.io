---
title: 'Fall Detection in Elevator Cages Based on XGBoost and LSTM'
author: Chaolong Zhang
status: Published
type: published
citation: "Xu, C., Xu, Y., Xu, Z., Guo, B., <b>Zhang, C.</b>, Huang, J., & Deng, X. (2021). Fall Detection in Elevator Cages Based on XGBoost and LSTM. 26th International Conference on Automation and Computing (ICAC)."
tag: Human Action Recogntion
subjects: Human Action Recogntion
comments: no
fave: false
doi: 10.23919/ICAC50006.2021.9594123
date: 2021-09-02
publishdate: 2021-09-02
---

Fall detection has always been challenging. There are many current works that have achieved good results in fall detections, but most of the existing works only consider single feature dimension (i.e., temporal features or spatial features). This study proposes a method that combines LSTM and XGBoost model to detect human body falls through fusing both temporal and spatial features. It starts with extracting bond points on human body in each frame of the video by using AlphaPose. In the second step, three features of the human body (i.e., vertical height, aspect ratio of the human external rectangle and angle of the knee joint) are calculated as spatial features through the extracted bone point information. Then use LSTM to learn these three kinds of features on both the temporal and spatial dimension. Furthermore, this model integrates the XGBoost to learn multiple features to improve the recognition rate. Finally, various human body fall detections in elevator cabs are applied to test the usability and validity of this study. The experimental results of the whole model reach 92.11% recognition accuracy, and 93.33% on the the F1-measure index.