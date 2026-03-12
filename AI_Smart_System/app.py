"""Streamlit dashboard for Smart Environment & Energy Monitor"""

import streamlit as st
import pandas as pd
from motion_detection import run_motion_detection
from environment_monitor import run_monitor, load_environment_data, detect_anomalies
from energy_prediction import run_prediction, load_energy_data, train_model, predict_future
from predictive_maintenance import run_maintenance, detect_failure, load_sensor_data
import ai_assistant

st.title("AI Powered Smart Monitoring System for Motion, Environment & Energy")

menu = ["Home", "Motion Detection", "Environmental Monitoring", "Energy Prediction", "Predictive Maintenance", "AI Assistant"]
choice = st.sidebar.selectbox("Select Module", menu)

if choice == "Home":
    st.write("### Problem")
    st.write("Use AI to detect motion, monitor environment, predict energy consumption and maintenance needs.")
    st.write("### Features")
    st.write("- Motion detection with webcam\n- Environmental data analytics and anomaly detection\n- Energy usage forecasting\n- Predictive maintenance of machines")

elif choice == "Motion Detection":
    import threading

    st.write("#### Motion Detection Module")
    if st.button("Start Webcam Motion Detection"):
        st.write("Opening webcam – a separate window will appear.")
        st.write("To stop the camera either press 'q' or 's' inside that window, or use the window's close button.")
        # run detection in background so Streamlit remains responsive
        def _background():
            stats = run_motion_detection()
            if stats is not None:
                # when finished, write results to a temp file so we can display them
                with open("motion_stats.txt", "w") as f:
                    f.write(f"{stats['frames']},{stats['events']},{stats['accuracy']}\n")
        threading.Thread(target=_background, daemon=True).start()
        st.write("Webcam started in background; you can switch tabs.")
    # if stats file exists, read and display
    try:
        with open("motion_stats.txt") as f:
            line = f.readline().strip()
            if line:
                frames, events, acc = line.split(",")
                st.write(f"Frames processed: {frames}")
                st.write(f"Motion events: {events}")
                st.write(f"Detection accuracy: {float(acc):.2%}")
    except FileNotFoundError:
        pass
    st.write("Log file will be saved as motion_times.txt")

elif choice == "Environmental Monitoring":
    st.write("#### Environmental Monitoring Module")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            df = load_environment_data(uploaded)
        except Exception as e:
            st.error(f"Failed to load uploaded file: {e}")
            df = None
        if df is not None:
            df = df.dropna()
            st.write(df.head())
            if st.button("Run Monitor"):
                try:
                    metrics = run_monitor(df)
                    st.write("Metrics:", metrics)
                except Exception as e:
                    st.error(f"Monitoring failed: {e}")
            if st.button("Detect Anomalies"):
                df = df.dropna()
                try:
                    anomalies = detect_anomalies(df)
                    st.write(anomalies)
                    rate = len(anomalies) / len(df) if len(df) else 0
                    st.write(f"Anomaly rate: {rate:.2%}")
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")
    else:
        st.write("Using default dataset.")
        if st.button("Show default monitor"):
            try:
                metrics = run_monitor()
                st.write("Metrics:", metrics)
            except Exception as e:
                st.error(f"Default monitoring failed: {e}")

elif choice == "Energy Prediction":
    st.write("#### Energy Prediction Module")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        # read once for preview; ``load_energy_data`` will reset file position
        df = load_energy_data(uploaded)
        df = df.dropna()
        st.write(df.head())
        if st.button("Run Prediction"):
            # pass the DataFrame directly so we don't re-read the stream
            result = run_prediction(df)
            if result is not None:
                st.write(f"Model score (R^2): {result['score']:.2f}")
    else:
        st.write("Using default dataset.")
        if st.button("Run default prediction"):
            result = run_prediction()
            if result is not None:
                st.write(f"Model score (R^2): {result['score']:.2f}")

elif choice == "Predictive Maintenance":
    st.write("#### Predictive Maintenance Module")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = load_sensor_data(uploaded)
        df = df.dropna()
        st.write(df.head())
        if st.button("Train and Evaluate"):
            result = run_maintenance(df)
            if result is not None:
                st.write(f"Model accuracy: {result['accuracy']:.2%}")
    else:
        st.write("Using default dataset.")
        if st.button("Run default maintenance"):
            result = run_maintenance()
            if result is not None:
                st.write(f"Model accuracy: {result['accuracy']:.2%}")

elif choice == "AI Assistant":
    st.write("#### AI Assistant Module")
    st.write("Upload a CSV and ask simple questions about the data. The assistant can
             describe, compute means, correlations, detect missing values, outliers,
             or even plot a column.")
    uploaded = st.file_uploader("Upload CSV for AI", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Data preview:")
        st.write(df.head())
        # show automatic insights
        insights = ai_assistant.dataset_insights(df)
        st.write("Dataset summary:")
        st.json(insights)
        question = st.text_input("Ask a question (columns, mean, max, describe, etc.)")
        if question:
            result = ai_assistant.analyze_query(df, question)
            if isinstance(result, tuple) and result[0] == "plot":
                st.pyplot(result[1])
            else:
                st.write(result)

