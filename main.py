import os
import base64
import uuid
import logging
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from dotenv import load_dotenv
from diffusers import StableDiffusionUpscalePipeline, AutoPipelineForInpainting


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

image_output_dir = "./static/images"
os.makedirs(image_output_dir, exist_ok=True)
logger.info(f"Ensured {image_output_dir} directory exists.")

inpainting_model = os.getenv("INPAINTING_MODEL_NAME", "stabilityai/stable-diffusion-2-inpainting")
upscale_model = os.getenv("UPSCALE_MODEL_NAME", "stabilityai/stable-diffusion-x4-upscaler")

logger.info(f"Loading models: Inpainting - {inpainting_model}, Upscale - {upscale_model}")

execution_device = "cuda" if torch.cuda.is_available() else "cpu"
print(execution_device)
logger.info(f"Using device: {execution_device}")

try:
    upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        upscale_model, revision="fp16", torch_dtype=torch.float16
    ).to(execution_device)

    inpainting_pipeline = AutoPipelineForInpainting.from_pretrained(
        inpainting_model, revision="fp16", torch_dtype=torch.float16
    ).to(execution_device)

    logger.info("Both models loaded successfully.")
except Exception as load_error:
    logger.error(f"Failed to load models: {load_error}", exc_info=True)
    raise ValueError("Model loading failed. Please check model files or connection.")


router = APIRouter(
    prefix="/api/v1",
    responses={404: {"description": "Not found"}},
)


@app.get("/", tags=["HealthCheck"])
async def root_endpoint():
    logger.info("Root endpoint accessed.")
    return {"message": "PixelCraft is running!"}


@router.post("/inpaint/", tags=["Inpaint"])
async def inpaint_image_v1(
    guidance_text: str,
    return_b64: bool = False,
    base_image: UploadFile = File(...),
    mask_image: UploadFile = File(...)
):
    """
    API to inpaint an image using a mask.

    Parameters:
    ----------
    `base_image`: UploadFile
        The image to inpaint.
    `mask_image`: UploadFile
        Mask to guide the inpainting process.
    `guidance_text`: str, optional
        A prompt to guide inpainting.
    `return_b64`: bool, optional
        If True, returns the inpainted image as a base64 string. Otherwise, returns a file response.
    """
    try:
        logger.info("Received request to inpaint image.")
        img_to_inpaint = Image.open(BytesIO(await base_image.read())).convert("RGB")
        inpainting_mask = Image.open(BytesIO(await mask_image.read())).convert("RGB")
        logger.info("Image and mask loaded successfully.")

        final_inpainted_img = inpainting_pipeline(
            prompt=guidance_text, image=img_to_inpaint, mask_image=inpainting_mask
        ).images[0]

        result_filename = f"{uuid.uuid4()}.png"
        result_filepath = f"{image_output_dir}/{result_filename}"
        final_inpainted_img.save(result_filepath)
        logger.info(f"Inpainted image saved at {result_filepath}.")

        if not return_b64:
            return FileResponse(result_filepath)

        buffer = BytesIO()
        final_inpainted_img.save(buffer, format="PNG")
        buffer.seek(0)
        inpainted_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info("Image converted to base64 successfully.")
        return JSONResponse({"image": inpainted_b64}, status_code=status.HTTP_200_OK)

    except Exception as inpaint_error:
        logger.error(f"Error while inpainting image: {inpaint_error}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image processing failed.")


@router.post("/upscale/", tags=["Upscale"])
async def upscale_image_v1(
    image_file: UploadFile = File(...),
    max_width: int = 512,
    max_height: int = 512,
    description: Optional[str] = "",
    return_base64: bool = False
):
    """
    API to upscale an image.

    Parameters:
    ----------
    `image_file`: UploadFile
        The image to upscale.
    `max_width`: int, optional
        Desired width of the upscaled image, default is 512 pixels.
    `max_height`: int, optional
        Desired height of the upscaled image, default is 512 pixels.
    `description`: str, optional
        A text prompt to guide upscaling, default is an empty string.
    `return_base64`: bool, optional
        If True, returns the upscaled image as a base64 string. Otherwise, returns a file response.
    """
    try:
        if max_height > 1024 or max_width > 1024:
            return JSONResponse(
                content={"error": "The image size exceeds the allowed limit of 1024px in either width or height."},
                status_code=status.HTTP_400_BAD_REQUEST
            )

        logger.info("Received request to upscale image.")
        original_image = Image.open(BytesIO(await image_file.read())).convert("RGB")
        logger.info("Image loaded successfully.")

        scaling_factor = 4
        new_width = max(max_width // scaling_factor, 1)
        new_height = max(max_height // scaling_factor, 1)
        original_image = original_image.resize((new_width, new_height))
        logger.info(f"Image resized to {new_width}x{new_height}.")

        upscaled_img = upscale_pipeline(prompt=description, image=original_image).images[0]

        output_filename = f"{uuid.uuid4()}.png"
        output_filepath = f"{image_output_dir}/{output_filename}"
        upscaled_img.save(output_filepath)
        logger.info(f"Upscaled image saved at {output_filepath}.")

        if not return_base64:
            return FileResponse(output_filepath)

        buffered_image = BytesIO()
        upscaled_img.save(buffered_image, format="PNG")
        buffered_image.seek(0)
        image_b64 = base64.b64encode(buffered_image.getvalue()).decode("utf-8")

        logger.info("Image converted to base64 successfully.")
        return JSONResponse({"image": image_b64}, status_code=status.HTTP_200_OK)

    except Exception as processing_error:
        logger.error(f"Error while upscaling image: {processing_error}", exc_info=True)
        raise HTTPException(status_code=500, detail="Image processing failed.")


app.include_router(router)
