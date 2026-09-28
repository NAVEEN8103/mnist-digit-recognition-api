import streamlit as st
import requests
import io
import os
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# API CONFIG
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")


# ============================================================
# SESSION STATE
# ============================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "last_input_method" not in st.session_state:
    st.session_state.last_input_method = "Draw Digit"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background-color: #FAF8F4;
    color: #302A25;
}

.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Remove Streamlit header */

header[data-testid="stHeader"] {
    background-color: transparent;
}


/* ============================================================
   TITLE
   ============================================================ */

.app-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #3D342D;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.app-subtitle {
    text-align: center;
    font-size: 16px;
    color: #776C62;
    margin-bottom: 24px;
}


/* ============================================================
   API STATUS
   ============================================================ */

.api-status {
    background-color: #F3EEE8;
    border: 1px solid #DDD2C6;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    color: #655A51;
    font-size: 14px;
    margin-bottom: 24px;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 20px;
    font-weight: 650;
    color: #3D342D;
    margin-top: 14px;
    margin-bottom: 11px;
}


/* ============================================================
   INFORMATION BOX
   ============================================================ */

.info-box {
    background-color: #F3EEE8;
    border: 1px solid #E1D7CC;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    color: #6D6259;
    margin-bottom: 16px;
    font-size: 14px;
}


/* ============================================================
   RADIO BUTTONS
   ============================================================ */

div[role="radiogroup"] {
    background-color: #FFFFFF;
    border: 1px solid #DED5CA;
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 20px;
}

div[role="radiogroup"] label {
    color: #51483F !important;
}

div[role="radiogroup"] label p {
    color: #51483F !important;
}


/* ============================================================
   CANVAS
   ============================================================ */

.canvas-label {
    text-align: center;
    color: #7A7068;
    font-size: 14px;
    margin-bottom: 8px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    width: 100%;
    min-height: 44px;

    border-radius: 9px;

    background-color: #8F7964;
    border: 1px solid #8F7964;

    color: #FFFFFF !important;

    font-size: 15px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #786452;
    border-color: #786452;
    color: #FFFFFF !important;
}


/* ============================================================
   PREDICTION AREA
   ============================================================ */

.prediction-title {
    text-align: center;
    color: #776C62;
    font-size: 15px;
    margin-top: 25px;
    margin-bottom: 10px;
}


/* ============================================================
   FIX STREAMLIT METRIC COLORS
   ============================================================ */

/* Metric container */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E1D8CE !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 3px 12px rgba(70, 50, 30, 0.05);
}


/* Metric label */

div[data-testid="stMetricLabel"] {
    color: #776C62 !important;
}


/* Metric label text */

div[data-testid="stMetricLabel"] p {
    color: #776C62 !important;
    font-size: 14px !important;
}


/* Metric value */

div[data-testid="stMetricValue"] {
    color: #4F463F !important;
}


/* Metric value text */

div[data-testid="stMetricValue"] div {
    color: #4F463F !important;
}


/* ============================================================
   RESULT METRIC
   ============================================================ */

.result-metric div[data-testid="stMetric"] {
    text-align: center;
    padding: 18px !important;
}

.result-metric div[data-testid="stMetricLabel"] {
    justify-content: center;
}

.result-metric div[data-testid="stMetricLabel"] p {
    text-align: center !important;
}

.result-metric div[data-testid="stMetricValue"] {
    justify-content: center;
    font-size: 72px !important;
    line-height: 1 !important;
    color: #8A7560 !important;
}

.result-metric div[data-testid="stMetricValue"] div {
    color: #8A7560 !important;
    font-size: 72px !important;
}


/* ============================================================
   PROGRESS BAR
   ============================================================ */

div[data-testid="stProgress"] {
    margin-top: 8px;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px solid #E1D8CE;
    border-radius: 10px;
    padding: 10px;
}

[data-testid="stFileUploader"] label {
    color: #51483F !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #958B82;
    font-size: 12px;
    margin-top: 32px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clear_prediction():
    """Clear previous prediction."""

    st.session_state.prediction = None
    st.session_state.confidence = None


def send_prediction(image_bytes, filename):
    """Send image to FastAPI."""

    files = {
        "file": (
            filename,
            image_bytes,
            "image/png"
        )
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            return (
                result["prediction"],
                result["confidence"]
            )

        try:
            detail = response.json().get(
                "detail",
                "Prediction failed."
            )
        except Exception:
            detail = "Prediction failed."

        st.error(detail)

        return None, None

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the backend is running."
        )

        return None, None

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

        return None, None

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )

        return None, None


def show_prediction(prediction, confidence):
    """
    Display prediction using native Streamlit components.
    This avoids raw HTML/code appearing in the browser.
    """

    st.markdown(
        '<div class="prediction-title">Prediction</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = st.columns(
        [1, 2, 1]
    )

    with result_col2:

        st.markdown(
            '<div class="result-metric">',
            unsafe_allow_html=True
        )

        st.metric(
            "Predicted Digit",
            str(prediction)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(confidence / 100.0, 1.0)
        )

        st.caption(
            f"Model confidence: {confidence:.2f}%"
        )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="app-title">'
    '🔢 MNIST Digit Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Draw a digit or upload an image for CNN-based recognition.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API STATUS
# ============================================================

api_online = False

try:

    response = requests.get(
        f"{API_URL}/health",
        timeout=3
    )

    if response.status_code == 200:

        api_online = True

        st.markdown(
            '<div class="api-status">'
            '● Model API is online and ready'
            '</div>',
            unsafe_allow_html=True
        )

except requests.exceptions.RequestException:

    st.error(
        "FastAPI is offline."
    )

    st.code(
        "uvicorn api.main:app --reload"
    )


# ============================================================
# INPUT METHOD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Choose your input method'
    '</div>',
    unsafe_allow_html=True
)

input_method = st.radio(
    "Input method",
    [
        "Draw Digit",
        "Upload Image"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# RESET WHEN SWITCHING
# ============================================================

if input_method != st.session_state.last_input_method:

    clear_prediction()

    st.session_state.last_input_method = input_method


# ============================================================
# DRAW DIGIT
# ============================================================

if input_method == "Draw Digit":

    st.markdown(
        '<div class="info-box">'
        'Use your mouse to draw one digit from '
        '<b>0 to 9</b>.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="canvas-label">'
        'Write inside the white square'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CENTER CANVAS
    # --------------------------------------------------------

    left, center, right = st.columns(
        [1, 2, 1]
    )

    with center:

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=14,
            stroke_color="#000000",
            background_color="#FFFFFF",

            height=280,
            width=280,

            drawing_mode="freedraw",

            update_streamlit=True,

            display_toolbar=False,

            key=f"canvas_{st.session_state.canvas_key}"
        )


    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        clear_button = st.button(
            "Clear",
            use_container_width=True
        )

    with col2:

        predict_button = st.button(
            "Predict Digit",
            use_container_width=True,
            disabled=not api_online
        )


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    if clear_button:

        st.session_state.canvas_key += 1

        clear_prediction()

        st.rerun()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if predict_button:

        if canvas_result.image_data is None:

            st.warning(
                "Please draw a digit first."
            )

        else:

            image_data = (
                canvas_result.image_data
                .astype("uint8")
            )


            # Check for dark pixels.

            grayscale = image_data[
                :, :, :3
            ].mean(axis=2)


            has_drawing = np.any(
                grayscale < 245
            )


            if not has_drawing:

                st.warning(
                    "Please draw a digit inside the canvas."
                )

            else:

                image = Image.fromarray(
                    image_data,
                    "RGBA"
                ).convert("RGB")


                image_bytes = io.BytesIO()

                image.save(
                    image_bytes,
                    format="PNG"
                )

                image_bytes.seek(0)


                prediction, confidence = send_prediction(
                    image_bytes,
                    "drawn_digit.png"
                )


                if prediction is not None:

                    st.session_state.prediction = (
                        prediction
                    )

                    st.session_state.confidence = (
                        confidence
                    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.prediction is not None:

        show_prediction(
            st.session_state.prediction,
            st.session_state.confidence
        )


# ============================================================
# UPLOAD IMAGE
# ============================================================

else:

    st.markdown(
        '<div class="info-box">'
        'Upload an image containing one handwritten digit.'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        left, center, right = st.columns(
            [1, 2, 1]
        )

        with center:

            st.image(
                image,
                width=280
            )


        st.write("")


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predict_upload = st.button(
            "Predict Digit",
            use_container_width=True,
            disabled=not api_online
        )


        if predict_upload:

            image_bytes = io.BytesIO()

            image.save(
                image_bytes,
                format="PNG"
            )

            image_bytes.seek(0)


            prediction, confidence = send_prediction(
                image_bytes,
                uploaded_file.name
            )


            if prediction is not None:

                st.session_state.prediction = prediction

                st.session_state.confidence = confidence


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if st.session_state.prediction is not None:

            show_prediction(
                st.session_state.prediction,
                st.session_state.confidence
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    'Model Information'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model",
        "CNN"
    )


with col2:

    st.metric(
        "Test Accuracy",
        "99.13%"
    )


with col3:

    st.metric(
        "Dataset",
        "MNIST"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'TensorFlow • FastAPI • Streamlit • MNIST'
    '</div>',
    unsafe_allow_html=True
)