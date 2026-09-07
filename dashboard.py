"""
Automated Pothole Detection System
Streamlit Dashboard

Run:
    streamlit run dashboard.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ultralytics import YOLO


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
COMPARISON_PATH = BASE_DIR / "full_model_comparison.csv"

st.set_page_config(
    page_title="Automated Pothole Detection System",
    page_icon="🚧",
    layout="wide",
)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained YOLO model."""
    if not MODEL_PATH.exists():
        return None

    try:
        return YOLO(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Unable to load the YOLO model: {exc}")
        return None


@st.cache_data
def load_comparison_data():
    """Load model-comparison results."""
    if not COMPARISON_PATH.exists():
        return None

    try:
        df = pd.read_csv(COMPARISON_PATH)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as exc:
        st.error(f"Unable to load model comparison data: {exc}")
        return None


def classify_severity(area_ratio: float) -> str:
    """
    Estimate pothole severity from bounding-box area ratio.

    This is a simple project-level heuristic, not a road-engineering
    severity standard.
    """
    if area_ratio < 0.02:
        return "Low"
    if area_ratio < 0.06:
        return "Medium"
    return "High"


def run_detection(image: np.ndarray, model: YOLO, conf: float):
    """Run YOLO detection and return an annotated image plus statistics."""
    results = model.predict(
        image,
        conf=conf,
        imgsz=640,
        verbose=False,
    )

    if not results:
        return image.copy(), []

    annotated = image.copy()
    height, width = image.shape[:2]
    image_area = max(height * width, 1)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_ratio = box_area / image_area
        severity = classify_severity(area_ratio)

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "severity": severity,
            }
        )

        # OpenCV uses BGR
        severity_color = {
            "Low": (0, 200, 0),
            "Medium": (0, 140, 255),
            "High": (0, 0, 220),
        }[severity]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            severity_color,
            2,
        )

        label = f"{severity} | {confidence:.2f}"
        text_y = max(y1 - 8, 18)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            severity_color,
            2,
            cv2.LINE_AA,
        )

    return annotated, detections


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("🚧 Pothole Detection System")
st.sidebar.markdown(
    "**BS Software Engineering**  \n"
    "Iqra National University, Peshawar"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Model Comparison",
        "🔍 Detect Potholes",
    ],
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

model = load_model()
comparison_df = load_comparison_data()


# -------------------------------------------------
# OVERVIEW
# -------------------------------------------------

if page == "🏠 Overview":
    st.title("🚧 Automated Pothole Detection System")
    st.markdown("### Smart City Road Maintenance")

    st.write(
        "An AI-based computer vision application that detects potholes "
        "in road images using a trained YOLO object-detection model."
    )

    if model is None:
        st.error(
            "The trained model `best.pt` was not found. "
            "Make sure it is in the same folder as `dashboard.py`."
        )
    else:
        st.success("Trained YOLO model loaded successfully.")

    if comparison_df is not None and not comparison_df.empty:
        required = {"Model", "mAP50", "mAP50-95", "Precision", "Recall"}
        if required.issubset(comparison_df.columns):
            best_row = comparison_df.loc[
                comparison_df["mAP50-95"].idxmax()
            ]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Top Model", str(best_row["Model"]))
            col2.metric("Best mAP50", f'{best_row["mAP50"]:.3f}')
            col3.metric("Best mAP50-95", f'{best_row["mAP50-95"]:.3f}')
            col4.metric("Models Evaluated", len(comparison_df))

            st.divider()
            st.subheader("Model Performance Summary")
            st.dataframe(
                comparison_df,
                use_container_width=True,
            )
        else:
            st.warning(
                "The comparison CSV does not contain the expected metric columns."
            )


# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

elif page == "📊 Model Comparison":
    st.title("📊 YOLO Model Comparison")

    if comparison_df is None or comparison_df.empty:
        st.warning("`full_model_comparison.csv` was not found.")
        st.stop()

    available_metrics = [
        metric
        for metric in ["mAP50", "mAP50-95", "Precision", "Recall"]
        if metric in comparison_df.columns
    ]

    if not available_metrics:
        st.warning("No supported comparison metrics were found.")
        st.stop()

    metric = st.selectbox("Select Metric", available_metrics)

    chart_df = comparison_df.sort_values(metric, ascending=False)

    fig = px.bar(
        chart_df,
        x="Model",
        y=metric,
        text_auto=".3f",
        title=f"{metric} — All Evaluated Models",
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparison Table")
    st.dataframe(
        chart_df,
        use_container_width=True,
    )


# -------------------------------------------------
# DETECTION
# -------------------------------------------------

elif page == "🔍 Detect Potholes":
    st.title("🔍 Detect Potholes in Images")

    if model is None:
        st.error(
            "Model file `best.pt` was not found. "
            "Place it in the same directory as `dashboard.py`."
        )
        st.stop()

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    uploaded = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        file_bytes = np.frombuffer(
            uploaded.read(),
            np.uint8,
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            st.error("The uploaded file could not be read as an image.")
            st.stop()

        annotated, detections = run_detection(
            image,
            model,
            conf,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with col2:
            st.subheader(
                f"Detection Result ({len(detections)} pothole(s))"
            )
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        if detections:
            st.divider()
            st.subheader("Detection Details")

            details_df = pd.DataFrame(detections)[
                ["severity", "confidence"]
            ].copy()

            details_df.columns = ["Severity", "Confidence"]

            st.dataframe(
                details_df,
                use_container_width=True,
            )

            severity_counts = (
                details_df["Severity"]
                .value_counts()
                .reset_index()
            )
            severity_counts.columns = ["Severity", "Count"]

            col3, col4 = st.columns(2)

            with col3:
                pie_fig = px.pie(
                    severity_counts,
                    names="Severity",
                    values="Count",
                    title="Severity Distribution",
                )
                st.plotly_chart(
                    pie_fig,
                    use_container_width=True,
                )

            with col4:
                conf_fig = px.histogram(
                    details_df,
                    x="Confidence",
                    nbins=10,
                    title="Confidence Distribution",
                )
                st.plotly_chart(
                    conf_fig,
                    use_container_width=True,
                )
        else:
            st.success(
                "No potholes were detected above the selected confidence threshold."
            )
    else:
        st.info("Upload a road image to start pothole detection.")
