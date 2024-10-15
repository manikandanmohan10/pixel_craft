import os
import time
import torch
from dotenv import load_dotenv
from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForInpainting

load_dotenv()

inpainting_model_name = os.getenv("INPAINTING_MODEL_NAME", "stabilityai/stable-diffusion-2-inpainting")
upscale_model_name = os.getenv("UPSCALE_MODEL_NAME", "stabilityai/stable-diffusion-x4-upscaler")

try:
    computation_device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()

    upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        upscale_model_name, revision="fp16", torch_dtype=torch.float16
    ).to(computation_device)

    inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
        inpainting_model_name, revision="fp16", torch_dtype=torch.float16
    ).to(computation_device)

    end_time = time.time()
    print(f"Time taken to load pipelines: {end_time - start_time} seconds")

except Exception as error:
    print(f"Error loading pipelines: {error}")
