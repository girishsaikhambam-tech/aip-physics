# AI Powered Smart Monitoring System for Motion, Environment & Energy

This Python application demonstrates real-world AI solutions for motion detection, environmental monitoring, energy prediction, and predictive maintenance. It uses OpenCV, scikit-learn, Streamlit and other libraries to provide a simple dashboard interface.

## Folder Structure

```
AI_Smart_System/
│
├── motion_detection.py
├── environment_monitor.py
├── energy_prediction.py
├── predictive_maintenance.py
├── app.py
├── datasets/
│   ├── environment.csv
│   ├── energy.csv
│   └── machine_sensor.csv
└── requirements.txt
```

## Installation

1. Clone or download the repository.
2. Navigate to the `AI_Smart_System` directory.
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate # macOS/Linux
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Module Descriptions

### 1. Motion Detection Module
- Uses OpenCV to capture webcam video.
- Detects moving objects by frame differencing.
- Draws bounding boxes around motion, prints status and saves timestamps.
- When run from the Streamlit UI, the webcam loop now executes in a
  background thread so you can switch to other modules while the camera is
  running.  Results (frames/events/accuracy) are displayed once the window is
  closed.
- Stop the camera by pressing `q` or `s` inside the window or by closing it.
- Real-world use: security and smart surveillance.

### 2. Environmental Monitoring Module
- Loads environmental dataset (`environment.csv`).
  - The loader now automatically handles a variety of air-quality column names
    (e.g. `air_quality_index`, `AQI`, `Air Quality`) and will even compute a
    proxy column from pollutant metrics like `pm2_5`/`pm10` if necessary.
- All date/timestamp columns are parsed with `format="mixed"` and
    `errors="coerce"`, so you may upload files that use differing
    datetime styles (e.g. `2026-01-01 00:00`, `01/01/2026 00:00`,
    `2026/01/01T00:00Z`).  Unparseable rows are dropped and a warning printed.
- Preprocesses data and plots temperature, humidity, and air quality.
- Uses Isolation Forest to flag anomalies.
- Applications include climate monitoring and air pollution tracking.

### 3. Smart Energy Prediction Module
- Loads energy usage data (`energy.csv`).
- Trains a linear regression or random forest model on hour-of-day.
- Predicts future energy consumption and plots historical usage.
- Reports a model score (R²) which is displayed as an accuracy-like metric in
  the Streamlit UI.
- Useful for energy forecasting in smart buildings.

### 4. Predictive Maintenance Module
- Loads sensor data (`machine_sensor.csv`) containing vibration and temperature readings.
- Trains a decision tree or logistic regression model to predict failures.
- Reports model accuracy on a held-out test set; this accuracy is shown in the
  Streamlit dashboard after training.
- Detects abnormal machine behavior to warn of possible failures.
- Applicable to industrial equipment and turbines.

### 5. User Interface
- The `app.py` file uses Streamlit to present a simple dashboard.
- Users can upload their own CSV files or use the default datasets.
  - When uploading, make sure the data contains a timestamp/date column and
    either an `air_quality` column or one of the recognised variants; if your
    file uses a different name just rename it or rely on the loader's
    auto‑renaming logic.  The timestamp column can be in almost any
    reasonable format—the system will convert or drop problematic rows and
    report them in the server log.
- Each module can be executed from the sidebar.
- After running a module, an accuracy/performance metric is displayed
  (motion detection shows detection rate; monitoring shows non‑anomaly
  accuracy; energy prediction shows R² score; maintenance shows classifier
  accuracy).

## Visualization
- Each module generates plots using Matplotlib which are displayed via Streamlit.
- Motion detection shows real-time video with overlays.

## Datasets
- Example CSV files are provided in the `datasets/` folder with synthetic data to illustrate the functionality.
  The loaders tolerate mixed datetime formats and will indicate if any
  entries were dropped during parsing.

## Running Individual Modules
You can also run each Python script directly from the command line:
```bash
python motion_detection.py
python environment_monitor.py
python energy_prediction.py
python predictive_maintenance.py
```

## Notes
- For motion detection the script will open a window; press `q` to quit.
- Customize models or datasets as needed for real applications.

---

This application demonstrates how AI techniques can be combined into a smart monitoring system for smart cities, energy systems, and environmental monitoring.
