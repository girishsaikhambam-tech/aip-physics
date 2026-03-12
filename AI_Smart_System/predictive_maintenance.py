"""Predictive maintenance using sensor data"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_sensor_data(path="datasets/machine_sensor.csv"):
    # support both filesystem paths and file-like objects (such as
    # Streamlit UploadedFile). pandas.read_csv handles those natively; if we
    # receive a seekable stream we reset it so we can read multiple times
    # without getting an empty result.
    if hasattr(path, "read"):
        try:
            path.seek(0)
        except Exception:
            pass
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    date_col = None
    for col in df.columns:
        if col.lower() in ("timestamp", "date", "time"):
            date_col = col
            break
    if date_col:
        # convert with mixed-format support and coerce errors so a single
        # malformed timestamp doesn't blow up the entire upload.
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
        if date_col != "timestamp":
            df = df.rename(columns={date_col: "timestamp"})
    else:
        raise ValueError("No timestamp/date column found in sensor data")
    return df


def preprocess(df):
    df = df.dropna()
    return df


def train_model(df, algorithm="decision_tree"):
    X = df[['vibration', 'temperature']]
    y = df['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if algorithm == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    score = model.score(X_test, y_test)
    return model, score


def detect_failure(model, vibration, temperature):
    import numpy as np
    pred = model.predict(np.array([[vibration, temperature]]))
    return pred[0]


def run_maintenance(path_or_df="datasets/machine_sensor.csv"):
    # accept either a path/stream or a DataFrame to avoid re-reading streams
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = load_sensor_data(path_or_df)
    df = preprocess(df)
    print(df.head())
    model, score = train_model(df)
    # example prediction
    sample = df.iloc[0]
    print("Sample prediction," , "failure:" , detect_failure(model, sample.vibration, sample.temperature))
    print(f"Model accuracy: {score:.2%}")
    return {"accuracy": score}


if __name__ == '__main__':
    run_maintenance()
