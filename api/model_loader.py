from tensorflow import keras
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "Model" / "mnist_cnn.keras"

model = keras.models.load_model(MODEL_PATH)

print("MNIST CNN model loaded successfully.")