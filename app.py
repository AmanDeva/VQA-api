# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from PIL import Image
import base64
import io

app = FastAPI()

# --- Hugging Face Login with Token ---
hf_token = "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"
login(token=hf_token)

# --- Load processor from the Hugging Face Hub ---
base_model_path = "google/paligemma-3b-pt-224"
processor = AutoProcessor.from_pretrained(base_model_path)

# --- Load the base model and apply the adapter weights ---
local_model_path = "./pali-gemma-model"
try:
    print(f"Loading base model from Hugging Face Hub: {base_model_path}")
    # Add 4-bit quantization for memory efficiency on t4g.medium (4 GB RAM)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    ).eval()

    print(f"Loading adapter weights from local path: {local_model_path}")
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, local_model_path)

    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU.")
    else:
        print("GPU not available, using CPU.")

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load local model and processor.")

# Define the request body for the API endpoint
class InferenceRequest(BaseModel):
    image_base64: str
    prompt: str

@app.get("/")
def read_root():
    return {"message": "PaliGemma API is running"}

@app.post("/analyze_image")
async def analyze_image(request: InferenceRequest):
    """
    Analyzes an uploaded image and answers a question based on a voice command.
    """
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Prepare inputs for the model using the image and prompt
        inputs = processor(text=request.prompt, images=image, return_tensors="pt")

        # Move inputs to the same device as the model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate model output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        # Decode the output and return the response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))