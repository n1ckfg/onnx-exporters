import torch
import shutil
import os
from pathlib import Path
from diffusers import StableDiffusionPipeline
from optimum.exporters.onnx import main_export

INPUT_MODEL_PATH = "./v1-5-pruned-emaonly-fp16.safetensors"  
OUTPUT_ONNX_DIR = "./onnx_output"              
TEMP_DIFFUSERS_DIR = "./intermediate_diffusers_model" 

def convert_safetensors_to_onnx():
    print(f"--- Step 1: Loading '{INPUT_MODEL_PATH}' ---")
    
    pipe = StableDiffusionPipeline.from_single_file(
        INPUT_MODEL_PATH,
        torch_dtype=torch.float16, 
        use_safetensors=True,
        load_safety_checker=False
    )
    
    print(f"--- Step 2: Saving intermediate format to '{TEMP_DIFFUSERS_DIR}' ---")
    pipe.save_pretrained(TEMP_DIFFUSERS_DIR)
    
    del pipe
    torch.cuda.empty_cache()

    print("--- Step 3: Exporting to ONNX (FP16) ---")
    
    main_export(
        model_name_or_path=TEMP_DIFFUSERS_DIR,
        output=Path(OUTPUT_ONNX_DIR),
        task="stable-diffusion",
        fp16=True,         
        device="cuda",     
        opset=14,          
        no_post_process=True 
    )
    
    print(f"ONNX export saved to: {OUTPUT_ONNX_DIR}")

    print("--- Step 4: Cleaning up temporary files ---")
    if os.path.exists(TEMP_DIFFUSERS_DIR):
        shutil.rmtree(TEMP_DIFFUSERS_DIR)
        print("Temporary directory deleted.")

    print("=== Conversion Complete ===")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. FP16 export might fail on CPU.")
    
    convert_safetensors_to_onnx()