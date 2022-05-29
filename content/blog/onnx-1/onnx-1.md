---
title: AI模型部署(1) - ONNX
author: Chaolong
summary: AI模型部署
date: '2022-05-16'
draft: false
categories:
  - 模型部署
---

## ONNX简介

模型部署是指让训练好的模型在特定环中运行的过程。目前已经有很多成熟的深度学习和机器学习框架，如[PyTorch](https://pytorch.org/)，[TensorFlow](https://www.tensorflow.org/)，[scikit-learn](https://scikit-learn.org/)和[XGBoost](https://xgboost.ai/)等，但是工业界的开发者往往专注于一种框架和平台，如OpenVINO，TensorRT，CoreML，ARM，NPU等。框架和平台的多样性导致从算法开发和训练到算法部署与应用存在较大的困难，尤其是在边缘计算（Edge Computing）领域，因此需要一种通用的、可交互的平台和工具来简化模型的部署。

[ONNX](https://onnx.ai/)（开放神经网络交换格式，Open Neural Network Exchange）是一种用于表示深度学习和机器学习模型的标准。ONNX提供标准的算子、方法和数据类型，用于表示计算图模型。算法模型可以表示为有向无环图，其中节点（Node）代表算子，边代表数据的流向。同时，ONNX也支持算子扩展，以支持自定义的计算方法。使用[Netron](https://github.com/lutzroeder/Netron)、[Netdrawer](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md)和[Zetane](https://github.com/zetane/viewer)工具也可以方便地查看模型的结构。

此外，ONNX可使模型在不同的框架之间进行转移，允许多种框架和工具之间的交互。目前主流的深度学习和机器学习框架都支持导出ONNX模型，适合用于存储训练好的模型。同时也可以把ONNX模型转换为特定的软件和硬件计算平台，以部署AI应用。

## ONNX历史与发展

ONNX最早由Facebook和Microsoft发起的社区项目，之后IBM，华为，Intel，AMD，Arm和高通等公司纷纷加入，目前官网列出的合作伙伴（Partners）就有43个。众多公司和开发者的加入也保证了ONNX的稳定性和可靠性，同时快速迭代也保证了ONNX可以支持最新的算子和方法。

![Partners](/files/misc/onnx-logo-partners.jpg)

目前支持导出和转换ONNX模型的框架和工具有：

![Frameworks&Tools](/files/misc/onnx-logo-frameworks.jpg)

支持模型推理的平台有：

![Deploy](/files/misc/onnx-logo-inference.jpg)

## 总结

相对于其它的框架（[ncnn](https://github.com/Tencent/ncnn), [Tengine's tmfile](https://github.com/OAID/Tengine), [RKNN](https://github.com/rockchip-linux/rknn-toolkit)），使用ONNX具有如下优点：

1. ONNX模型是平台无关的，可以直接在多个平台部署运行；
2. ONNX的发展很快，支持的算子更多；
3. 由大公司主导研发，可靠性更高。

因此，ONNX更值得去学习和研究，同时也建议其它推理和（硬件）加速平台直接使用ONNX来表示模型，开发兼容ONNX的后端(backend)，并复用ONNX生态系统的工具（如模型转换、优化、量化和[ONNX Runtime](https://onnxruntime.ai/)等）。这种方式可有效减少研发工作量，同时也保证了平台的可用性和通用性。

## 参考

1. ONNX， [https://onnx.ai/](https://onnx.ai/%E2%80%B8)
