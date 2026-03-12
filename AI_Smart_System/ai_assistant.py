"""Simple AI assistant for dataset exploration and natural-language queries."""

import pandas as pd


def summarise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for a DataFrame."""
    return df.describe(include='all')


def dataset_insights(df: pd.DataFrame) -> dict:
    """Return a collection of simple automatic insights."""
    info = {}
    info["rows"] = df.shape[0]
    info["columns"] = df.shape[1]
    info["missing"] = df.isnull().sum().to_dict()
    # correlations only for numeric columns
    info["correlation"] = df.corr(numeric_only=True).to_dict()
    return info


def analyze_query(df: pd.DataFrame, query: str):
    """Answer a user query with a statistic, insight or plot command.

    Returns either a printable object (str, DataFrame, dict) or a tuple
    ("plot", fig) when the query requests a visualization.
    """
    q = query.lower()
    if "columns" in q or "fields" in q:
        return f"Columns: {', '.join(df.columns)}"
    if "mean" in q:
        return df.mean(numeric_only=True)
    if "describe" in q or "summary" in q:
        return summarise_df(df)
    if "correlation" in q:
        return df.corr(numeric_only=True)
    if "missing" in q or "null" in q:
        return df.isnull().sum()
    if "max" in q:
        return df.max(numeric_only=True)
    if "min" in q or "smallest" in q:
        return df.min(numeric_only=True)
    if "outlier" in q:
        # simple z-score based detection for numeric columns
        import numpy as np
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return "No numeric columns to inspect for outliers."
        z = np.abs((numeric - numeric.mean()) / numeric.std())
        outliers = (z > 3).any(axis=1)
        return df[outliers]
    if "plot" in q:
        # extract column name
        for col in df.columns:
            if col.lower() in q:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                try:
                    df[col].plot(ax=ax)
                    ax.set_title(col + " Trend")
                    return ("plot", fig)
                except Exception as e:
                    return f"Could not plot column {col}: {e}"
        return "No matching column found to plot."
    # integration with other modules
    if "predict energy" in q or "energy prediction" in q:
        from energy_prediction import load_energy_data, run_prediction
        # if dataset in context, use it else run default
        try:
            res = run_prediction(df)
            return res
        except Exception as e:
            return f"Energy prediction failed: {e}"
    if "detect failure" in q or "maintenance" in q:
        from predictive_maintenance import load_sensor_data, run_maintenance
        try:
            res = run_maintenance(df)
            return res
        except Exception as e:
            return f"Maintenance prediction failed: {e}"
    return "Sorry, I don't know how to answer that. Try asking about columns, mean, max, correlation, missing, describe, outliers, plot, energy prediction, or maintenance."
