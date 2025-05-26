from onnx2torch import convert
import onnx

onnx_model_path = "path/to/your/model.onnx"
onnx_model = onnx.load(onnx_model_path)
torch_model = convert(onnx_model)