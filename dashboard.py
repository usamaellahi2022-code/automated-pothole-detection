"""
=============================================================
  Pothole Detection – Streamlit Dashboard
  BS Project – IQRA National University, Peshawar
=============================================================
  Run:  streamlit run dashboard.py
=============================================================
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RESULTS_DIR = r"D:\Saqib\work\pothole_results"

st.set_page_config(
    page_title = "Pothole Detection System",
    page_icon  = "🚧",
    layout     = "wide",
)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/c/c9/"
    "Iqra_National_University_Peshawar_Logo.png/200px-"
    "Iqra_National_University_Peshawar_Logo.png",
    width=150,
)
st.sidebar.title("Pothole Detection System")
st.sidebar.markdown("**BS Computer Science**\nIQRA National University, Peshawar")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["🏠 Overview", "📊 Model Comparison",
                                      "🔍 Detect Potholes", "📁 View Results"])

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

@st.cache_data
def load_report():
    path = Path(RESULTS_DIR) / "report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison_df():
    path = Path(RESULTS_DIR) / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def find_best_weights():
    report = load_report()
    if report:
        w = Path(RESULTS_DIR) / report["best_model"] / "weights" / "best.pt"
        if w.exists():
            return str(w)
    for pt in Path(RESULTS_DIR).rglob("best.pt"):
        return str(pt)
    return None


def classify_severity(area_ratio: float) -> str:
    if area_ratio < 0.02:
        return "Low"
    elif area_ratio < 0.06:
        return "Medium"
    return "High"


def run_detection(image: np.ndarray, weights: str, conf: float):
    """Run YOLO on a numpy image, return annotated image + stats."""
    from ultralytics import YOLO
    model   = YOLO(weights)
    results = model.predict(image, conf=conf, imgsz=640, verbose=False)

    h, w = image.shape[:2]
    annotated  = image.copy()
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence      = float(box.conf[0])
        area_ratio      = ((x2 - x1) * (y2 - y1)) / (w * h)
        severity        = classify_severity(area_ratio)
        detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "confidence": confidence, "severity": severity})

        color = {"Low": (0, 200, 0), "Medium": (0, 140, 255), "High": (0, 0, 220)}[severity]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{severity} {confidence:.2f}",
                    (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated, detections


# ──────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────

# ── Overview ──────────────────────────────────
if page == "🏠 Overview":
    st.title("🚧 Automated Pothole Detection System")
    st.markdown("### Smart City Road Maintenance | BS Project")

    report = load_report()
    df     = load_comparison_df()

    if report and df is not None:
        col1, col2, col3, col4 = st.columns(4)
        best   = report["best_model"]
        row    = df[df["Model"] == best].iloc[0]

        col1.metric("Best Model",       best)
        col2.metric("Best mAP50",       f"{row['mAP50']:.3f}")
        col3.metric("Best mAP50-95",    f"{row['mAP50-95']:.3f}")
        col4.metric("Models Evaluated", len(df))

        st.divider()
        st.subheader("Models Evaluated")
        st.dataframe(df.style.highlight_max(
            subset=["mAP50", "mAP50-95", "Precision", "Recall"],
            color="#c6efce"), use_container_width=True)
    else:
        st.warning("No results found. Please run `train_all_models.py` first.")
        st.code("python train_all_models.py", language="bash")


# ── Model Comparison ──────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    df = load_comparison_df()

    if df is None:
        st.warning("Run training first: `python train_all_models.py`")
        st.stop()

    metric = st.selectbox("Select Metric", ["mAP50", "mAP50-95", "Precision",
                                             "Recall", "Training_Time"])

    # Bar chart
    fig = px.bar(df.sort_values(metric, ascending=False),
                 x="Model", y=metric,
                 color="Model",
                 text_auto=".3f",
                 title=f"{metric} – All Models",
                 template="plotly_white")
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.subheader("Multi-Metric Radar")
    categories = ["mAP50", "mAP50-95", "Precision", "Recall"]
    radar_fig  = go.Figure()
    for _, row in df.iterrows():
        vals = [row[c] for c in categories]
        radar_fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=row["Model"],
            opacity=0.6,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar – All Metrics", template="plotly_white")
    st.plotly_chart(radar_fig, use_container_width=True)


# ── Detect Potholes ───────────────────────────
elif page == "🔍 Detect Potholes":
    st.title("🔍 Detect Potholes in Images")

    weights = find_best_weights()
    if weights is None:
        st.error("No trained model found. Run `train_all_models.py` first.")
        st.stop()

    report  = load_report()
    df      = load_comparison_df()
    if df is not None:
        all_weights = []
        for _, row in df.iterrows():
            w = Path(RESULTS_DIR) / row["Model"] / "weights" / "best.pt"
            if w.exists():
                all_weights.append((row["Model"], str(w)))
        if all_weights:
            choice  = st.selectbox("Model", [m for m, _ in all_weights],
                                   index=0,
                                   help="Auto-ranked best → worst by mAP50")
            weights = dict(all_weights)[choice]

    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        col1.subheader("Original")
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with st.spinner("Detecting potholes…"):
            annotated, detections = run_detection(img, weights, conf)

        col2.subheader(f"Detected ({len(detections)} potholes)")
        col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

        if detections:
            st.divider()
            st.subheader("Detection Details")
            det_df = pd.DataFrame(detections)[["severity", "confidence"]]
            counts = det_df["severity"].value_counts().reset_index()
            counts.columns = ["Severity", "Count"]

            c1, c2 = st.columns(2)
            fig_pie = px.pie(counts, names="Severity", values="Count",
                             color="Severity",
                             color_discrete_map={"Low":"green","Medium":"orange","High":"red"},
                             title="Severity Distribution")
            c1.plotly_chart(fig_pie, use_container_width=True)

            fig_conf = px.histogram(det_df, x="confidence", nbins=10,
                                    title="Confidence Distribution",
                                    template="plotly_white")
            c2.plotly_chart(fig_conf, use_container_width=True)

            st.dataframe(det_df.rename(columns={"severity": "Severity",
                                                  "confidence": "Confidence"}),
                         use_container_width=True)
        else:
            st.success("✅ No potholes detected above the confidence threshold.")


# ── View Results ──────────────────────────────
elif page == "📁 View Results":
    st.title("📁 Saved Training Results")

    result_dir = Path(RESULTS_DIR)
    if not result_dir.exists():
        st.warning(f"Results directory not found: {RESULTS_DIR}")
        st.stop()

    model_dirs = [d for d in result_dir.iterdir()
                  if d.is_dir() and (d / "results.csv").exists()]

    if not model_dirs:
        st.info("No trained models found yet.")
        st.stop()

    selected = st.selectbox("Select Model", [d.name for d in model_dirs])
    model_d  = result_dir / selected

    tab1, tab2 = st.tabs(["📈 Training Curves", "🖼️ Sample Predictions"])

    with tab1:
        csv_path = model_d / "results.csv"
        if csv_path.exists():
            df_r = pd.read_csv(csv_path)
            df_r.columns = df_r.columns.str.strip()

            map_col = next((c for c in df_r.columns
                            if "mAP" in c and "0.5" in c and "95" not in c), None)
            if map_col:
                fig = px.line(df_r, x=df_r.index, y=map_col,
                              title=f"{selected} – mAP50 Training Curve",
                              template="plotly_white",
                              labels={"x": "Epoch", map_col: "mAP50"})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pred_dir = model_d / "val_predictions"
        images   = list(pred_dir.glob("*.jpg"))[:6] if pred_dir.exists() else []
        if images:
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                cols[i % 3].image(str(img_path), use_column_width=True)
        else:
            st.info("No prediction images saved for this model.")