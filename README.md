
# Pixel Craft

This project provides a FastAPI-based service for both image inpainting and upscaling using pre-trained Stable Diffusion models. It allows users to upload low-resolution images for upscaling, or submit images with masks for inpainting. NVIDIA GPU acceleration is utilized for optimal performance.


## Key Features
- Image Upload & Upscaling: Upload an image and get an upscaled version.
- Image Inpainting: Upload an image and a mask to fill in missing parts.
- CUDA GPU Support: Leverages local GPUs for faster model inference.
- File or Base64 Response: Choose to receive the result as a file or Base64-encoded image.
- Logging: Logs key operations such as model loading and image processing.

## Project Structure
- `main.py`: FastAPI application handling image uploads and processing.
- `load.py`: Script to preload the model and test the load time.
- `Dockerfile`: Configuration for Docker with CUDA and PyTorch.
- `docker-compose.yml`: Manages the app's deployment with GPU support.
- `sample_images/`: Test images.
- `static/images/`: Stores processed images.

## Setup Instructions

### Clone the Repository
Clone the repository and navigate into the project directory:

```cmd
git clone https://github.com/manikandanmohan10/pixel_craft.git
cd pixel_craft
```

### Docker Setup (Recommended)

#### Prerequisites
- Docker and Docker Compose
- NVIDIA drivers with CUDA support
- Python 3.10+
- Compatible NVIDIA GPU (minimum T4 required)

#### Environment Variables
Create a `.env` file with your Hugging Face model names:

```env
INPAINTING_MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
UPSCALE_MODEL_NAME="stabilityai/stable-diffusion-x4-upscaler"
```

#### Build & Run the App (Docker)
Build the Docker image:

```cmd
docker-compose build
```

Start the service:

```cmd
docker-compose up -d
```

Access the API at `http://localhost:8000`.
For **Swagger**: `http://localhost:8000/docs`

### Manual Setup (Without Docker)

#### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA and cuDNN installed
- PyTorch with CUDA support
- FastAPI, Uvicorn, and other dependencies from `requirements.txt`

#### Environment Variables
Create a `.env` file in the project root directory:

```env
INPAINTING_MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
UPSCALE_MODEL_NAME="stabilityai/stable-diffusion-x4-upscaler"
```

#### Steps

1. **Set up a virtual environment**:

   ```cmd
   python3 -m venv venv
   ```

   **Windows Powershell**
   ```cmd
   .\venv\Scripts\activate
   ```

   **Linux Ubuntu**
   ```cmd
   source venv/bin/activate
   ```

2. **Install dependencies**:

   ```cmd
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

3. **Load the models**:

   Run the `load.py` script to ensure the models are loaded correctly:

   ```cmd
   python load_model.py
   ```

4. **Run the FastAPI app**:

   Use Uvicorn to start the server:

   ```cmd
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

Access the API at `http://localhost:8000`.

For **Swagger**: `http://localhost:8000/docs`

## API Endpoints
- `GET /`: Health check for the API.
- `POST /api/v1/upscale/`: Upload an image and get the upscaled version.
    - Parameters:
        - `image_file`: The image file to be processed (required).
        - `max_width`: Target width of the upscaled image (default 512px, max 1024px).
        - `max_height`: Target height of the upscaled image (default 512px, max 1024px).
        - `description`: Optional text prompt to guide the upscaling process.
        - `return_base64`: Boolean to indicate whether the image should be returned in Base64 format (default False).

- `POST /api/v1/inpaint/`: Inpaint an image by uploading an image and a mask.
    - Parameters:
        - `base_image`: The image file to be processed (required).
        - `mask_image`: The mask image to guide the inpainting process (required).
        - `guidance_text`: Optional text prompt to guide the inpainting process.
        - `return_b64`: Boolean to indicate whether the image should be returned in Base64 format (default False).

## Notes
- The API assumes the presence of a CUDA-enabled GPU. If CUDA is unavailable, the pipeline will fallback to CPU, but performance may be slower.
- Ensure that input dimensions do not exceed the maximum limit of 1024 pixels for both width and height.
- File cleanup is not automatically handled. Consider adding a cron job or manual cleanup routine to manage storage.
