import torch
import torch.onnx
import onnx
from onnx2torch import convert

# Load the ONNX Model
onnx_model = onnx.load("t3-512x15x16h-distill-swa-2767500.onnx")

# Convert to pytorch
pytorch_model = convert(onnx_model)

print(type(pytorch_model))

torch.save(pytorch_model, "t3-512x15x16h-distill-swa-2767500.pth")

