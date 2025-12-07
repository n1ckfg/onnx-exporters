import torch
from diffusers import StableDiffusionPipeline
import subprocess

pipe = StableDiffusionPipeline.from_single_file(
    "./v1-5-pruned-emaonly-fp16.safetensors",
    torch_dtype=torch.float16, 
    use_safetensors=True,
    load_safety_checker=False  
)

pipe.save_pretrained("./intermediate_diffusers_model")

print("Conversion to Diffusers format complete.")

#optimum-cli export onnx --model ./intermediate_diffusers_model --task stable-diffusion --fp16 ./onnx_output_folder
#subprocess.run(["optimum-cli", "export", "onnx", "--model", "./intermediate_diffusers_model", "--task", "stable-diffusion", "--fp16", "./onnx_output_folder"])
subprocess.run(["optimum-cli", "export", "onnx", "--model", "./intermediate_diffusers_model", "--task", "stable-diffusion", "./onnx_output_folder"])