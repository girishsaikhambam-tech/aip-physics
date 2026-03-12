"""Environmental monitoring and anomaly detection"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


import os


def load_environment_data(path="datasets/environment.csv"):
    # allow callers to pass a file-like object (e.g. Streamlit UploadedFile) or
    # bytes; pandas can handle those directly.  If we see an object with a
    # ``read`` method we treat it as file-like and skip all path checks.
    if hasattr(path, "read"):
        df = pd.read_csv(path)
    else:
        # resolve default relative path to module directory if not an absolute path
        if not os.path.isabs(path):
            base = os.path.dirname(__file__)
            path = os.path.join(base, path)
        # ensure file exists before attempting read
        if not os.path.exists(path):
            raise FileNotFoundError(f"Environment data file not found: {path}")
        # read file without assuming a specific date column name
        df = pd.read_csv(path)
    # if read_csv returns empty frame or has no columns, raise immediately
    if df.empty or len(df.columns) == 0:
        raise ValueError("Environment file is empty or could not be parsed")

    # normalize column names: strip whitespace (keep case for now)
    df.columns = df.columns.str.strip()

    # detect and standardize date column name
    date_col = None
    for col in df.columns:
        if col.lower() in ("timestamp", "date", "time"):
            date_col = col
            break
    if date_col is not None:
        # allow mixed input formats and coerce invalid entries so we can
        # provide useful feedback instead of crashing.
        df[date_col] = pd.to_datetime(
            df[date_col],
            format="mixed",
            errors="coerce",
            dayfirst=False,
        )
        if df[date_col].isna().any():
            bad = df[df[date_col].isna()]
            print("Warning: dropped rows with unparseable timestamps:")
            print(bad.head())
            df = df.dropna(subset=[date_col])
        # standardize name to 'timestamp' for downstream code
        if date_col != "timestamp":
            df = df.rename(columns={date_col: "timestamp"})
    else:
        raise ValueError("No date/timestamp column found in environment data")

    # normalize known environmental column variations
    # air quality index is a common alternate name
    if "air_quality_index" in df.columns and "air_quality" not in df.columns:
        df = df.rename(columns={"air_quality_index": "air_quality"})

    # also accept other common variants such as "AQI", "Air Quality", or
    # simply "airquality" (case‑insensitive).  We'll look through the
    # original column names and rename the first match we find.
    if "air_quality" not in df.columns:
        lowered = {c.lower().strip(): c for c in df.columns}
        for variant in ("aqi", "air quality", "airquality"):
            if variant in lowered:
                df = df.rename(columns={lowered[variant]: "air_quality"})
                break

    # if we still don't have air_quality but do have pollutant metrics, make a
    # simple proxy by averaging them (user can always replace this with a real
    # AQI calculation later).
    if "air_quality" not in df.columns:
        pollutants = [c for c in df.columns if c.lower().strip() in
                      ("pm2_5", "pm10", "co", "no2")]
        if pollutants:
            df["air_quality"] = df[pollutants].mean(axis=1)
            print(f"Computed 'air_quality' as mean of {pollutants}")

    return df


def preprocess(df):
    # simple preprocessing: drop NaNs, ensure types
    df = df.dropna()
    return df


def visualize(df):
    # ensure expected columns exist (after any preprocessing/renaming)
    required = ["timestamp", "temperature", "humidity", "air_quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot visualize data, missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["temperature"], label="Temperature")
    plt.plot(df["timestamp"], df["humidity"], label="Humidity")
    plt.plot(df["timestamp"], df["air_quality"], label="Air Quality")
    plt.legend()
    plt.title("Environmental Trends")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def detect_anomalies(df):
    # verify that all required numeric columns exist before fitting
    required = ["temperature", "humidity", "air_quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot detect anomalies, missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # use Isolation Forest on numeric features
    features = df[required]
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(features)
    anomalous = df[df['anomaly'] == -1]
    return anomalous


def run_monitor(path_or_df="datasets/environment.csv"):
    # run monitoring workflow on a file path, file-like object or an existing
    # DataFrame.
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = load_environment_data(path_or_df)
    df = preprocess(df)
    # debug: show where data was loaded from (if path given as string)
    # note: path_or_df may be DataFrame so we only print for str
    if isinstance(path_or_df, str):
        print(f"Loaded environment data from: {path_or_df}")
    print("Data loaded:")
    print(df.head())
    print("Columns:", list(df.columns))
    # verify necessary columns exist before proceeding
    for col in ["temperature", "humidity", "air_quality"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data."
                             " Please check your input file.")
    visualize(df)
    anomalies = detect_anomalies(df)
    print("Detected anomalies:")
    print(anomalies)
    total = len(df)
    acc = (total - len(anomalies)) / total if total else 0
    print(f"Data accuracy (non-anomalies): {acc:.2%}")
    return {"total": total, "anomalies": len(anomalies), "accuracy": acc}


if __name__ == '__main__':
    run_monitor()
