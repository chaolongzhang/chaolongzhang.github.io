---
title: AI模型部署(4) - 量化实例
summary: AI模型部署
date: '2022-05-19'
authors: Chaolong
categories:
  - 模型部署
---

## 前言

通过前面的案例，我们已经实现把当前主流深度学习框架的模型转换为ONNX模型。本文将介绍模型在部署中轻量化和加速的问题。模型量化是把32位的浮点型模型转化为低比特的整型计算模型，如常见的int8和uint8，甚至是int4。理论上，把32位转化到8位，模型文件的大小可以减少为1/4。并且目前的深度学习硬件加速器（如NPU、VPU和TensorRT等）大都是以低比特整型计算为基础。因此，模型量化在模型部署和加速具有重要的作用。更多关于模型量化的理论可以参考Google和高通的两份白皮<sup>1,2</sup>。

## ONNX量化

虽然PyTorch和TensFlow框架也已经支持模型量化了，但是ONNX Runtime的量化具有更好的性能。本文使用ONNX Runtime进行模型量化。首先import相关库。


```python
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
```

下载ResNet-50预训练模型，将其转化为ONNX模型，然后进行量化：


```python
# 模型下载
torch_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# 导出ONNX模型
dummy_input = torch.randn(1, 3, 224, 224)
onnx_model_fp32 = 'resnet50.onnx'
torch.onnx.export(torch_model, dummy_input, onnx_model_fp32, opset_version=11, verbose=False)

# 检查模型
model = onnx.load(onnx_model_fp32)
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

# 模型量化
onnx_model_uint8 = 'resnet50_uint8.onnx'
quantize_dynamic(onnx_model_fp32, onnx_model_uint8, weight_type=QuantType.QUInt8)

# 检查量化模型
model = onnx.load(onnx_model_uint8)
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))
```

对比ONNX模型和量化后的模型进行推理，可以看到两种模型对于同一张图片所得到的TOP-5分类结果是一致的。


```python
from PIL import Image
import numpy as np
from scipy.special import softmax
from torchvision import transforms
import onnxruntime as ort

# 准备数据
filename = 'assets/dog.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
input_batch_np = input_batch.numpy()


# ONNX推理函数
def inference(model_name, in_data):
    ort_session = ort.InferenceSession("resnet50_torch.onnx")

    outputs = ort_session.run(
        None,
        { "input.1": in_data },
    )

    probabilities = softmax(outputs[0])[0]
    top5_catid = np.argsort(-probabilities)[:5]
    top5_prob = probabilities[top5_catid]
    
    return top5_catid, top5_prob



# fp32模型推理结果
fp32_top5_catid, fp32_top5_prob = inference(onnx_model_fp32, input_batch_np)
uint8_top5_catid, uint8_top5_prob = inference(onnx_model_uint8, input_batch_np)

print("FP32 result: ", fp32_top5_catid, fp32_top5_prob)
print("UINT8 result: ", uint8_top5_catid, uint8_top5_prob)
print("FP32 == UINT8: ", fp32_top5_catid == uint8_top5_catid)
```

    FP32 result:  [258 259 270 261 248] [0.8732967  0.03027085 0.01967113 0.01107353 0.00920425]
    UINT8 result:  [258 259 270 261 248] [0.8732967  0.03027085 0.01967113 0.01107353 0.00920425]
    FP32 == UINT8:  [ True  True  True  True  True]
    

## 结语

本文简单地介绍了ONNX Runtime的使用，不过由于模型量化目前还处于早期发展阶段，模型量化技术碎片化问题严重，每个硬件和软件平台都使用自己实现的量化算法和推理流程，各平台之间的差异也较大。比如笔者曾经使用了Rockchip的RKNN和Tengine平台，这两个平台都需要对输入数据进行量化，然后在输入到模型进行推理，输出的结果也是量化后的结果，需要对输出进行反量化转化为32为浮点型。而ONNX Runtime的量化模型把输入输出数据的量化也集成了，实现端到端的推理。从上面的例子也可以看到，FP32和UNIT8模型的推理流程完全一致，这种模式简化了开发和调试的难度。笔者认为以ONNX Runtime为基础，统一各平台的量化标准和流程，实现一致的推理过程是解决模型部署碎片化问题的一个方向。

此外，本文使用的是Post-Training Quantization (PTQ)量化方法，模型量化过程中也会导致精度降低。Quantization-Aware-Training （QAT）量化方法在网络训练过程去模拟量化，让网络越来越向量化后的权重靠近，从而获得更准确的量化权重，但是还是存在精度下降的情况。如何提保证量化后模型的精度还需要进一步研究。

## 参考

1. Quantizing deep convolutional networks for efficient inference: A whitepaper，<https://arxiv.org/abs/1806.08342>
2. A White Paper on Neural Network Quantization, <https://arxiv.org/abs/2106.08295>
3. PyTorch Quantization, <https://pytorch.org/docs/stable/quantization.html>
4. Quantize ONNX Models, <https://onnxruntime.ai/docs/performance/quantization.html>
