from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def create_test_image():
    """
    Create a simple handwritten-style test image
    for the /predict endpoint.
    """

    # White background
    image = Image.new(
        "RGB",
        (280, 280),
        "white"
    )

    draw = ImageDraw.Draw(image)

    # Draw a simple digit-like shape
    # This does not need to be classified correctly
    # for the API test. We are testing the endpoint.
    draw.line(
        [(80, 60), (200, 60)],
        fill="black",
        width=18
    )

    draw.line(
        [(80, 60), (80, 140)],
        fill="black",
        width=18
    )

    draw.line(
        [(80, 140), (200, 140)],
        fill="black",
        width=18
    )

    draw.line(
        [(200, 140), (200, 220)],
        fill="black",
        width=18
    )

    # Save image in memory
    image_bytes = BytesIO()

    image.save(
        image_bytes,
        format="PNG"
    )

    image_bytes.seek(0)

    return image_bytes


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data

    assert (
        data["message"]
        == "MNIST Digit Recognition API is running"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model"] == "MNIST CNN"


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "MNIST CNN"

    assert data["framework"] == "TensorFlow / Keras"

    assert data["input_shape"] == [28, 28, 1]

    assert data["num_classes"] == 10

    assert data["total_parameters"] == 113386

    assert data["test_accuracy"] == 0.9913


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

def test_predict():

    image_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test_digit.png",
                image_bytes,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    # Response fields
    assert "prediction" in data

    assert "confidence" in data

    assert "filename" in data

    # Prediction should be one of the 10 MNIST classes
    assert data["prediction"] in range(10)

    # Confidence is returned as a percentage
    assert 0 <= data["confidence"] <= 100

    # Filename should match uploaded filename
    assert data["filename"] == "test_digit.png"


# ============================================================
# INVALID FILE TEST
# ============================================================

def test_predict_invalid_file():

    invalid_file = BytesIO(
        b"This is not an image."
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                invalid_file,
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        data["detail"]
        == "Please upload a valid image file."
    )