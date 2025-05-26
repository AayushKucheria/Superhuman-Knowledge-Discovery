import torch
import onnx
from torchinfo import summary

def find_input_size_from_onnx(onnx_path):
    """
    Extract input size from the original ONNX model
    """
    print("🔍 Checking ONNX model for input size...")
    
    model = onnx.load(onnx_path)
    
    for input_tensor in model.graph.input:
        print(f"Input name: {input_tensor.name}")
        if input_tensor.type.tensor_type.shape:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(f"dynamic({dim.dim_param})")
                else:
                    shape.append("unknown")
            print(f"Input shape: {shape}")
            return tuple(shape)
    
    return None

def test_input_sizes(model, possible_shapes):
    """
    Test different input shapes to find the correct one
    """
    print("🧪 Testing possible input shapes...")
    
    for shape in possible_shapes:
        try:
            print(f"Trying shape: {shape}")
            dummy_input = torch.randn(*shape)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✅ Success! Input shape: {shape}")
            if isinstance(output, (list, tuple)):
                print(f"   Outputs: {len(output)} tensors")
                for i, out in enumerate(output):
                    print(f"     Output {i}: {out.shape}")
            else:
                print(f"   Output shape: {output.shape}")
            
            return shape
            
        except Exception as e:
            print(f"❌ Failed with shape {shape}: {str(e)[:100]}...")
            continue
    
    return None

# Main function to find and test input size
def analyze_model_inputs(pytorch_model_path, onnx_model_path=None):
    """
    Comprehensive analysis to find the correct input size
    """
    print("=" * 60)
    print("🔎 ANALYZING MODEL INPUT REQUIREMENTS")
    print("=" * 60)
    
    # Load the PyTorch model
    model = torch.load(pytorch_model_path, weights_only=False)
    model.eval()
    
    # Method 1: Check ONNX if available
    input_shape = None
    if onnx_model_path:
        try:
            input_shape = find_input_size_from_onnx(onnx_model_path)
        except Exception as e:
            print(f"Could not read ONNX: {e}")
    
    # Method 2: Based on LC0 knowledge, try common shapes
    # LC0 typically uses 112 input planes for classical format
    possible_shapes = [
        (1, 112, 8, 8),    # Standard LC0: batch=1, 112 planes, 8x8 board
        (2, 112, 8, 8),    # Batch size 2
        (1, 104, 8, 8),    # Some LC0 variants use 104 planes
        (1, 120, 8, 8),    # Some variants use 120 planes
        (1, 112),          # Flattened version
        (1, 7168),         # 112 * 8 * 8 flattened
    ]
    
    # If we found a shape from ONNX, try it first
    if input_shape:
        # Convert dynamic dimensions to 1 for testing
        test_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_shape)
        possible_shapes.insert(0, test_shape)
    
    # Test shapes
    working_shape = test_input_sizes(model, possible_shapes)
    
    if working_shape:
        print(f"\n🎯 FOUND WORKING INPUT SHAPE: {working_shape}")
        print("\n📊 Running torchinfo summary...")
        
        try:
            summary(model, input_size=working_shape)
        except Exception as e:
            print(f"Summary failed: {e}")
            print("But the model works with this input shape!")
        
        return working_shape
    else:
        print("❌ Could not determine input shape automatically")
        print("Try manually with different shapes based on your specific model")
        return None

# Usage
if __name__ == "__main__":
    pytorch_model_path = "t3-512x15x16h-distill-swa-2767500.pth"
    onnx_model_path = "t3-512x15x16h-distill-swa-2767500.onnx"  # Optional
    
    working_shape = analyze_model_inputs(pytorch_model_path, onnx_model_path)
    
    if working_shape:
        print(f"\n🚀 To use this model:")
        print(f"   input_tensor = torch.randn{working_shape}")
        print(f"   output = model(input_tensor)")
        print(f"\n📋 For torchinfo summary:")
        print(f"   summary(model, input_size={working_shape})")