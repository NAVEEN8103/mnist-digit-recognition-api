from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np

from api.model_loader import model
from api.preprocessing import preprocess_image


app = FastAPI(
    title="MNIST Digit Recognition API",
    description="CNN-based handwritten digit recognition API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "MNIST Digit Recognition API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "MNIST CNN"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)

    # Get predicted digit
    predicted_digit = int(np.argmax(predictions[0]))

    # Get confidence
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": predicted_digit,
        "confidence": confidence
    }