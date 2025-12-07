import torch
from diffusers import StableDiffusionPipeline
import subprocess

INPUT_MODEL_PATH = "./v1-5-pruned-emaonly-fp16.safetensors"  
TEMP_DIFFUSERS_DIR = "./intermediate_diffusers_model" 
OUTPUT_ONNX_DIR = "./onnx_output"              

pipe = StableDiffusionPipeline.from_single_file(
    INPUT_MODEL_PATH,
    torch_dtype=torch.float16, 
    use_safetensors=True,
    load_safety_checker=False  
)

pipe.save_pretrained(TEMP_DIFFUSERS_DIR)

print("Conversion to Diffusers format complete.")

#optimum-cli export onnx --model ./intermediate_diffusers_model --task stable-diffusion --fp16 ./onnx_output_folder
#subprocess.run(["optimum-cli", "export", "onnx", "--model", "./intermediate_diffusers_model", "--task", "stable-diffusion", "--fp16", "./onnx_output_folder"])
subprocess.run(["optimum-cli", "export", "onnx", "--model", TEMP_DIFFUSERS_DIR, "--task", "stable-diffusion", OUTPUT_ONNX_DIR])