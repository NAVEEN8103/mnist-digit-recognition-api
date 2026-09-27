from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import io
import numpy as np

from api.model_loader import model
from api.preprocessing import preprocess_image


app = FastAPI(
    title="MNIST Digit Recognition API",
    description="CNN-based handwritten digit recognition API",
    version="1.0.0"
)


# -------------------------
# Response Model
# -------------------------

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    filename: str


# -------------------------
# Root Endpoint
# -------------------------

@app.get("/")
def root():
    return {
        "message": "MNIST Digit Recognition API is running"
    }


# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "MNIST CNN"
    }


# -------------------------
# Model Information
# -------------------------

@app.get("/model-info")
def model_info():

    return {
        "model_name": "MNIST CNN",
        "framework": "TensorFlow / Keras",
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "total_parameters": model.count_params(),
        "test_accuracy": 0.9913
    }


# -------------------------
# Prediction Endpoint
# -------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    # Check whether uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        # Read uploaded image
        image_bytes = await file.read()

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(
            processed_image,
            verbose=0
        )

        # Get predicted digit
        predicted_digit = int(np.argmax(predictions[0]))

        # Get confidence
        confidence = float(np.max(predictions[0])) * 100

        return {
            "prediction": predicted_digit,
            "confidence": round(confidence, 2),
            "filename": file.filename
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )