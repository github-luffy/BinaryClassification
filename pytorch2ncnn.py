# -*- coding: utf-8 -*-

import torch
from model2 import PNet, ONet, MobileNetV2

model = MobileNetV2(nums_class=2)
path = './mouth_mobilenet_32.pth'
dst = './mouth_mobilenet_32.onnx'
model = torch.load(path)
x = torch.rand(1, 3, 32, 32).cuda()

torch_out = torch.onnx._export(model, x, dst, export_params=True)

import onnx
print("==> Loading and checking exported model from '{}'".format(dst))
onnx_model = onnx.load(dst)
onnx.checker.check_model(onnx_model)  # assuming throw on error
print("==> Passed")