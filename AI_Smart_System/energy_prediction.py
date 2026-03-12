"""Energy consumption prediction"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def load_energy_data(path="datasets/energy.csv"):
    # support both filesystem paths and file-like objects (UploadedFile, bytes,
    # etc.). pandas.read_csv handles file-like objects directly.  When reading
    # from a file-like object we reset the stream position so repeated calls
    # work (Streamlit reuses the same UploadedFile between preview and run).
    if hasattr(path, "read"):
        try:
            path.seek(0)
        except Exception:
            pass
        df = pd.read_csv(path)
    else:
        # path ought to be a string; preserve existing behaviour.
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    date_col = None
    for col in df.columns:
        if col.lower() in ("timestamp", "date", "time"):
            date_col = col
            break
    if date_col:
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
        raise ValueError("No timestamp/date column found in energy data")
    return df


def preprocess(df):
    df = df.dropna()
    # create numeric features such as hour of day
    df['hour'] = df['timestamp'].dt.hour
    return df


def train_model(df, algorithm="linear"):
    X = df[['hour']]
    y = df['energy_usage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if algorithm == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def predict_future(model, hours):
    import numpy as np
    future = pd.DataFrame({'hour': hours})
    preds = model.predict(future)
    return preds


def visualize(df):
    plt.figure(figsize=(8,5))
    plt.plot(df.timestamp, df.energy_usage, marker='o')
    plt.title('Energy Usage over Time')
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.tight_layout()
    plt.show()


def run_prediction(path_or_df="datasets/energy.csv"):
    # allow callers to supply either a path/file-like object or a DataFrame
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = load_energy_data(path_or_df)
    df = preprocess(df)
    print(df.head())
    visualize(df)
    model, score = train_model(df, algorithm="linear")
    print(f"Model score (R^2): {score:.2f}")
    future_hours = list(range(24))
    preds = predict_future(model, future_hours)
    print("Predicted energy usage for next 24 hours:")
    print(preds)
    return {"score": score}


if __name__ == '__main__':
    run_prediction()
