# app.py
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

# --- Hugging Face Login with Token (from environment variable) ---
# This is a best practice for handling secrets.
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    # Raise a clear error if the token is not set
    raise RuntimeError("Hugging Face token not set. Please export HUGGINGFACE_HUB_TOKEN before running.")
login(token=hf_token)

# --- Load processor from the Hugging Face Hub ---
base_model_path = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(base_model_path)

# --- Load the base model and apply the adapter weights ---
local_model_path = "./pali-gemma-model"
try:
    print(f"Loading base model from Hugging Face Hub: {base_model_path}")
    # Use 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    ).eval()

    print(f"Loading adapter weights from local path: {local_model_path}")
    # Load the LoRA adapter onto the base model
    model = PeftModel.from_pretrained(model, local_model_path)

    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU.")
    else:
        print("GPU not available, using CPU.")

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Re-raise the error to prevent the server from starting with a broken model
    raise RuntimeError("Failed to load local model and processor.")

# Define a root endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "PaliGemma API is running"}

# Define the new endpoint for image analysis using file upload
@app.post("/analyze_image")
async def analyze_image(image: UploadFile = File(...), prompt: str = Form(...)):
    """
    Analyzes an uploaded image and answers a question based on a voice command.
    The image is received as a file upload and the prompt as form data.
    """
    try:
        # Read the uploaded image file's bytes
        image_bytes = await image.read()
        
        # Open the image using PIL (Pillow)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prepare inputs for the model using the image and prompt
        inputs = processor(text=prompt, images=img, return_tensors="pt")

        # Move inputs to the same device as the model (GPU or CPU)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate the model's output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        # Decode the output to a human-readable string
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}

    except Exception as e:
        # Catch and handle any errors during the process
        raise HTTPException(status_code=500, detail=str(e))
