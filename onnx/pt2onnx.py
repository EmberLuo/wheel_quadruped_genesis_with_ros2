import torch
import torch.onnx
model_path = "../logs/wheel-legged-walking/policy.pt"
model = torch.jit.load(model_path)
model.eval()
dummy_input = torch.randn(1, 174)
onnx_model_path = 'model.onnx'
torch.onnx.export(model,
                  dummy_input,
                  onnx_model_path,  
                  input_names=['input'],  
                  output_names=['output'], 
                  opset_version=11) 
print(f"模型已成功导出为 {onnx_model_path}")
