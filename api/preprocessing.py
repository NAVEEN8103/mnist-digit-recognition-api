import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image):

    # Convert image to grayscale
    image = image.convert("L")

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Convert image to NumPy array
    image = np.array(image)

    # Invert colors
    # Black digit on white background
    #        ↓
    # White digit on black background
    image = 255 - image

    # Normalize pixel values to 0-1
    image = image.astype("float32") / 255.0

    # Add batch and channel dimensions
    image = image.reshape(1, 28, 28, 1)

    return image