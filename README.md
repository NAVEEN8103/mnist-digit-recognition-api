MNIST Digit Recognition API

An end-to-end handwritten digit recognition system built with TensorFlow/Keras, FastAPI, Streamlit, and Docker.

The project trains a Convolutional Neural Network (CNN) on the MNIST dataset, exposes the trained model through a REST API, and provides a Streamlit web interface where users can either draw a digit or upload an image for prediction.

🚀 Features
🧠 CNN-based MNIST digit classification
🎯 99.13% test accuracy
⚡ FastAPI REST API for model inference
🎨 Streamlit interactive frontend
✍️ Draw digits directly using the mouse
📤 Upload handwritten digit images
🖼️ Image preprocessing and normalization
📊 Prediction confidence
❤️ Health-check endpoint
ℹ️ Model information endpoint
🧪 Automated API tests with Pytest
🐳 Dockerized backend
🐳 Docker Compose for full-stack deployment
🏗️ Architecture
                    ┌──────────────────────┐
                    │       User           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Frontend   │
                    │      Port 8501       │
                    └──────────┬───────────┘
                               │
                               │ HTTP Request
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    │      Port 8000       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ • Grayscale          │
                    │ • Crop & Resize      │
                    │ • Centering          │
                    │ • Normalization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     MNIST CNN        │
                    │   TensorFlow/Keras   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Digit + Confidence   │
                    └──────────────────────┘
🧠 Machine Learning Model

The model is a Convolutional Neural Network trained on the MNIST handwritten digit dataset.

Model Architecture
Input: 28 × 28 × 1

        ↓

Conv2D
32 filters
3 × 3 kernel
ReLU

        ↓

MaxPooling2D
2 × 2

        ↓

Conv2D
32 filters
3 × 3 kernel
ReLU

        ↓

MaxPooling2D
2 × 2

        ↓

Flatten

        ↓

Dense
128 neurons
ReLU

        ↓

Dense
10 neurons
Softmax

        ↓

Digit Prediction
Model Statistics
Metric	Value
Training samples	55,000
Validation samples	5,000
Test samples	10,000
Input size	28 × 28 × 1
Output classes	10
Total parameters	113,386
Test accuracy	99.13%

The trained model is stored at:

Model/mnist_cnn.keras
📊 Model Performance

The CNN achieved:

99.13% test accuracy

Classification performance:

Digit	Precision	Recall	F1-score
0	0.9899	0.9969	0.9934
1	0.9965	0.9912	0.9938
2	0.9971	0.9922	0.9947
3	0.9950	0.9901	0.9926
4	0.9939	0.9908	0.9924
5	0.9866	0.9899	0.9882
6	0.9896	0.9916	0.9906
7	0.9865	0.9951	0.9908
8	0.9897	0.9867	0.9882
9	0.9871	0.9881	0.9876
🔧 Image Preprocessing

Uploaded or drawn images are processed before being passed to the CNN.

The preprocessing pipeline includes:

Convert image to grayscale
Detect the background
Convert to MNIST-style white digit on black background
Remove weak background pixels
Detect the digit bounding box
Crop the digit
Add a margin around the digit
Resize while maintaining the aspect ratio
Center the digit in a 28 × 28 image
Center using the digit's center of mass
Normalize pixel values
Reshape into the model's expected input shape

Final model input:

(1, 28, 28, 1)
⚡ FastAPI Backend

The backend provides REST API endpoints for interacting with the trained model.

Base URL

When running locally:

http://127.0.0.1:8001
Available Endpoints
GET /

Basic API information.

GET /health

Checks whether the API and model are available.

Example:

{
  "status": "healthy"
}
GET /model-info

Returns information about the trained model.

Example:

{
  "model_name": "MNIST CNN",
  "framework": "TensorFlow / Keras",
  "input_shape": [28, 28, 1],
  "num_classes": 10,
  "total_parameters": 113386,
  "test_accuracy": 0.9913
}
POST /predict

Accepts an image and returns the predicted digit and confidence.

Example response:

{
  "prediction": 9,
  "confidence": 99.13,
  "filename": "digit.png"
}

Interactive API documentation is available through FastAPI's Swagger UI:

http://127.0.0.1:8001/docs
🎨 Streamlit Frontend

The Streamlit application provides two ways to submit a digit:

1. Draw a Digit

Users can draw a handwritten digit directly on the canvas using their mouse.

2. Upload an Image

Users can upload an image containing a handwritten digit.

The frontend sends the image to the FastAPI backend and displays the model's prediction and confidence.

When running locally:

http://127.0.0.1:8501
🧪 Testing

The project includes automated API tests using Pytest.

Tests cover:

Root endpoint
Health endpoint
Model information endpoint
Prediction endpoint
Invalid file handling

Run the tests with:

python -m pytest -v

Current test result:

5 passed
🐳 Docker

The backend is containerized using Docker.

Build the API image
docker build -t mnist-digit-api .
Run the API
docker run -d --name mnist-api -p 8001:8000 mnist-digit-api

The API will then be available at:

http://127.0.0.1:8001
🐳 Docker Compose

The complete application can be started using Docker Compose.

The Compose configuration runs:

Frontend → Streamlit
Backend  → FastAPI

Start the application:

docker compose up --build

The services are exposed as:

Frontend: http://127.0.0.1:8501
API:      http://127.0.0.1:8001

Stop the services:

docker compose down
📁 Project Structure
mnist-digit-recognition-api/
│
├── .streamlit/
│   └── config.toml
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   └── preprocessing.py
│
├── Model/
│   └── mnist_cnn.keras
│
├── Training/
│   └── MNIST_CNN.ipynb
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
⚙️ Local Setup
1. Clone the repository
git clone https://github.com/NAVEEN8103/mnist-digit-recognition-api.git
cd mnist-digit-recognition-api
2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Start FastAPI
uvicorn api.main:app --reload --port 8000

The API will be available at:

http://127.0.0.1:8000
5. Start Streamlit

In another terminal:

streamlit run frontend/app.py

The frontend will be available at:

http://127.0.0.1:8501
🛠️ Technologies Used
Machine Learning
Python
TensorFlow
Keras
NumPy
Pillow
Backend
FastAPI
Uvicorn
Python Multipart
Frontend
Streamlit
Streamlit Drawable Canvas
Testing
Pytest
HTTPX
Deployment
Docker
Docker Compose
🔮 Future Improvements

Possible future improvements include:

Deploying the application to a cloud platform
Adding model versioning
Adding request logging and monitoring
Adding batch prediction support
Improving handwritten digit robustness for non-MNIST-style images
Adding CI/CD with GitHub Actions
Adding API authentication
Adding prediction history
Adding a production-grade frontend
Adding model performance monitoring
👨‍💻 Project Goal

This project demonstrates an end-to-end machine learning deployment workflow:

Dataset
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Image Preprocessing
   ↓
FastAPI Inference API
   ↓
Streamlit Interface
   ↓
Automated Testing
   ↓
Docker
   ↓
Docker Compose

It combines machine learning, backend API development, frontend development, testing, and containerization into a single deployable application.

📌 License

This project is intended for educational and portfolio purposes.
