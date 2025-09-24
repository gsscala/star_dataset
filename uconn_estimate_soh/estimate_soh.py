# %% [markdown]
# # Battery SoH Prediction using Linear Regression and Random Forest
# 
# ## 1. Introduction
# This notebook/script estimates the **State of Health (SoH)** of battery cells using cycle data.
# Features are extracted from current, voltage, and duration metrics of different statuses: CC Chg, CC DChg, CCCV Chg, Rest.
# 
# ## 2. Dataset
# - **UCONN Battery Dataset**  
# - Each `CHECKUP.h5` file corresponds to one cell and contains multiple cycles  
# - Metrics per status per cycle:
#   - `avg_current`, `std_current`
#   - `avg_voltage`, `std_voltage`
#   - `duration`  
# - `cell_id` ensures that training and testing sets do not share the same cells.

# %% [Cell 0: Functions]

import os
import json
import h5py
import numpy as np
import pandas as pd

def load_data(src):
    with h5py.File(src, "r") as f:
        keys = list(f.keys())
        raw = f[keys[0]][:]
    text = raw.tobytes().decode("utf-8")
    return pd.DataFrame(json.loads(text))

def calculate_metrics(df):
    df["time"] = df["time"].astype(float)
    df["voltage"] = df["voltage"].astype(float)
    df["current"] = df["current"].astype(float)
    df.loc[:, "state_change"] = (df["State"] != df["State"].shift()).cumsum()

    grouped = df.groupby("state_change")
    results = []

    for _, g in grouped:
        subset = g.iloc[1:-1]
        if subset.empty or subset["State"].isna().all():
            continue
        status = subset["State"].iloc[0]
        start_time = subset["time"].iloc[0]
        end_time = subset["time"].iloc[-1]
        duration = end_time - start_time
        avg_current = subset["current"].mean()
        std_current = subset["current"].std()
        avg_tension = subset["voltage"].mean()
        std_tension = subset["voltage"].std()

        results.append({
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "avg_current": avg_current,
            "std_current": std_current,
            "avg_tension": avg_tension,
            "std_tension": std_tension
        })

    return pd.DataFrame(results)

def status_to_int(status):
    mapping = {"Rest":0, "CCCV Chg":1, "CC Chg":2, "CC DChg":3}
    return mapping.get(status, 4)

def extract_cycles_features(src_folder):
    checkups_paths = []
    for cell in sorted(os.listdir(src_folder)):
        cell_path = os.path.join(src_folder, cell)
        for file in sorted(os.listdir(cell_path)):
            if "CHECKUP" in file:
                checkups_paths.append(os.path.join(cell_path, file))

    all_cycles = []

    for checkup in checkups_paths:
        df = load_data(checkup)
        idx = np.where(df["time"].astype(float) == 0)[0]
        idx = np.append(idx, len(df))

        for i in range(len(idx)-1):
            df_cycle_i = df.iloc[idx[i]:idx[i+1]]
            results = calculate_metrics(df_cycle_i)

            features = {}
            for status_name, group in results.groupby("status"):
                features[f"{status_name}_avg_current"] = group["avg_current"].mean()
                features[f"{status_name}_std_current"] = group["std_current"].mean()
                features[f"{status_name}_avg_tension"] = group["avg_tension"].mean()
                features[f"{status_name}_std_tension"] = group["std_tension"].mean()
                features[f"{status_name}_duration"] = group["duration"].sum()

            features["SoH"] = max(df_cycle_i["capacity"].astype(float)) * 100 / 1.2
            features["cell_id"] = os.path.basename(os.path.dirname(checkup))
            all_cycles.append(features)

    return pd.DataFrame(all_cycles).fillna(0)


# %% [Cell 1: Imports and Functions]

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# %% [Cell 2: Load Dataset]
src_folder = "../../../../../../media/mods-pred/Datasets/Data_uconn_h5"
dataset = extract_cycles_features(src_folder)

# %% [Cell 3: Train-Test Split]
cells = dataset["cell_id"].unique()
train_cells, test_cells = train_test_split(cells, test_size=0.2, random_state=42)

train_df = dataset[dataset["cell_id"].isin(train_cells)]
test_df = dataset[dataset["cell_id"].isin(test_cells)]

X_train = train_df.drop(columns=["SoH", "cell_id"])
y_train = train_df["SoH"]
X_test = test_df.drop(columns=["SoH", "cell_id"])
y_test = test_df["SoH"]

# %% [Cell 4: Train Models and Evaluate]
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("=== Linear Regression ===")
print("R² test:", r2_score(y_test, y_pred_lin))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))

print("\n=== Random Forest ===")
print("R² test:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))

# %% [Cell 5: Plot Predictions]
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Linear Regression")
plt.xlabel("Real SoH (%)")
plt.ylabel("Predicted SoH (%)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Random Forest")
plt.xlabel("Real SoH (%)")
plt.ylabel("Predicted SoH (%)")
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
