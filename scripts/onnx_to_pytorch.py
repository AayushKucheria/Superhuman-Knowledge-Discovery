import torch
import torch.onnx
import onnx
from onnx2torch import convert

onnx_model = onnx.load("t3-512x15x16h-distill-swa-2767500.onnx")
pytorch_model = convert(onnx_model)

torch.save(pytorch_model, "t3-512x15x16h-distill-swa-2767500.pth")

