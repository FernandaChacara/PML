
# # Petrignano Aquifer — Cross-validation and data leakage exercise
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPOSITORY/blob/main/petrignano_exercicio_revisado.ipynb)
# 
# ## Goal of the assignment
# 
# The goal is to predict the **response variable** `Depth_to_Groundwater_P25` for one month using the values of all variables from the two previous months.
# 
# Example: to predict March, the model uses January and February values.
# 
# This notebook compares two **cross-validation** strategies:
# 
# 1. Standard shuffled `KFold`
# 2. Sequential `TimeSeriesSplit`
# 
# The main question is whether standard KFold gives an over-optimistic result because of **data leakage** in a time-series problem.


# ## Required keywords used in this notebook
# 
# The assignment asks us to address these concepts in the video. They are also included as notebook notes:
# 
# - **response variable**
# - **predictors**
# - **imputation**
# - **cross-validation**
# - **folds**
# - **data leakage**
# - **hyper parameters**
# - **model selection**
# - **independent test data set**

# prompt: Create a complete Colab notebook for the Petrignano aquifer exercise.
# prompt: The notebook must tell a clear story with markdown notes and code sections.
# prompt: It must preprocess missing values, build lagged monthly predictors from the previous two months,
# prompt: compare KFold with TimeSeriesSplit, and use GridSearchCV for model selection.
# modification: I split the code into notebook steps so it is easier to explain in a screen-capture video.
# modification: I added graphs for visual interpretation, even though this exercise does not use a loss/epoch curve.
# modification: I added markdown notes with the keywords required by the assignment.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


# # 0. Organize the data
# 
# In this section, we load the dataset, sort observations chronologically and prepare the variables.
# 
# The **response variable** is:
# 
# `Depth_to_Groundwater_P25`
# 
# The **predictors** are all available variables from the two previous months.

# prompt: Load petrignanos.csv and prepare the date column for time-series modelling.
# modification: The code checks if the file and the target column exist, so the notebook fails with a clear message.

csv_path = "petrignanos.csv"

df = pd.read_csv(csv_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

# Keep only numeric variables for modelling.
df = df.apply(pd.to_numeric, errors="coerce")

target_col = "Depth_to_Groundwater_P25"

if target_col not in df.columns:
    raise ValueError(f"Column {target_col} was not found. Available columns are: {list(df.columns)}")

print("Dataset shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("Target variable:", target_col)

df.head()


# ## Imputation
# 
# The dataset may contain missing values and anomalous zeros.
# 
# For groundwater depth variables, zero values are suspicious because groundwater depth is measured as a distance from ground level. Therefore, zeros in `Depth_to_Groundwater` columns are treated as missing values.
# 
# The **imputation** strategy is:
# 
# 1. Replace anomalous zeros in groundwater columns with `NaN`.
# 2. Interpolate missing values over time.
# 3. Use forward fill and backward fill for remaining edge values.

# prompt: Treat missing values and anomalous zeros in the Petrignano dataset.
# modification: Rainfall zeros are preserved because a rainfall value of zero can be real.

groundwater_cols = [c for c in df.columns if "Depth_to_Groundwater" in c]

for col in groundwater_cols:
    df[col] = df[col].replace(0, np.nan)

missing_before = df.isna().sum().sum()

df = df.interpolate(method="time")
df = df.ffill().bfill()

missing_after = df.isna().sum().sum()

print("Total missing values before imputation:", missing_before)
print("Total missing values after imputation:", missing_after)


# ## Monthly aggregation
# 
# The assignment asks us to predict a month from the two previous months.
# 
# Therefore, the data is reorganized at monthly level.
# 
# - Rainfall variables are summed by month.
# - Other variables are averaged by month.

# prompt: Aggregate daily Petrignano data into monthly values.
# modification: Rainfall uses monthly sums; other variables use monthly means.

rainfall_cols = [c for c in df.columns if "Rainfall" in c]

agg_dict = {}
for col in df.columns:
    if col in rainfall_cols:
        agg_dict[col] = "sum"
    else:
        agg_dict[col] = "mean"

monthly = df.resample("MS").agg(agg_dict)
monthly = monthly.interpolate(method="time").ffill().bfill()

print("Monthly dataset shape:", monthly.shape)
monthly.head()


# ## Create lagged predictors
# 
# For each month, the model receives:
# 
# - all variables from one month before: `lag1`
# - all variables from two months before: `lag2`
# 
# For example, to predict March, the **predictors** are January and February values.
# 
# I also add two seasonal variables, `month_sin` and `month_cos`, because groundwater levels can have seasonal patterns.

# prompt: Create lagged predictors using the previous two months.
# modification: Add cyclical month variables to represent seasonality.

lagged_parts = []

for lag in [1, 2]:
    lagged = monthly.shift(lag)
    lagged.columns = [f"{col}_lag{lag}" for col in monthly.columns]
    lagged_parts.append(lagged)

X = pd.concat(lagged_parts, axis=1)

# Seasonality variables for the target month.
X["month_sin"] = np.sin(2 * np.pi * monthly.index.month / 12)
X["month_cos"] = np.cos(2 * np.pi * monthly.index.month / 12)

y = monthly[target_col].copy()

data_model = pd.concat([X, y.rename("response")], axis=1).dropna()

X = data_model.drop(columns=["response"])
y = data_model["response"]

print("Final modelling table shape:", data_model.shape)
print("Number of predictors:", X.shape[1])
print("Number of observations:", X.shape[0])

data_model.head()


# # 1. The hard split: independent test data set
# 
# The dataset is split chronologically:
# 
# - the first 80% is used for training and cross-validation
# - the last 20% is used as the **independent test data set**
# 
# This test set simulates future unseen data.

# prompt: Split the data chronologically into training and independent test sets.
# modification: shuffle=False is essential because this is a time-series problem.

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Training period:", X_train_full.index.min(), "to", X_train_full.index.max())
print("Test period:", X_test.index.min(), "to", X_test.index.max())
print("Training observations:", len(X_train_full))
print("Test observations:", len(X_test))


# # 2. Define the two cross-validation strategies
# 
# Here we compare two types of **cross-validation**:
# 
# ## Naive KFold
# 
# Standard `KFold` with shuffling creates random **folds**.  
# This can cause **data leakage**, because training folds may contain future observations while validation folds contain older observations.
# 
# ## TimeSeriesSplit
# 
# `TimeSeriesSplit` respects chronological order.  
# The model is always trained on the past and validated on later observations.

# prompt: Define shuffled KFold and TimeSeriesSplit for comparing cross-validation strategies.

cv_naive = KFold(n_splits=5, shuffle=True, random_state=42)
cv_temporal = TimeSeriesSplit(n_splits=5)

print("Naive KFold:", cv_naive)
print("Temporal split:", cv_temporal)


# # 3. Experiment with a fixed model
# 
# In this section, the same model is tested using the two cross-validation strategies.
# 
# The model is a `DecisionTreeRegressor` with `max_depth=10`.
# 
# Because this is a decision tree model, there is no **loss/epoch graph**. That type of graph is used for neural networks or iterative models trained over epochs. Here, the relevant comparison is between internal CV scores and the independent test score.

# prompt: Evaluate a fixed DecisionTreeRegressor with KFold and TimeSeriesSplit.
# modification: Use a pipeline with StandardScaler and DecisionTreeRegressor to follow the assignment structure.

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", DecisionTreeRegressor(max_depth=10, random_state=42))
])

scores_naive = cross_val_score(pipe, X_train_full, y_train_full, cv=cv_naive, scoring="r2")
scores_temporal = cross_val_score(pipe, X_train_full, y_train_full, cv=cv_temporal, scoring="r2")

print(f"Naive CV R2:    {scores_naive.mean():.4f} (+/- {scores_naive.std():.4f})")
print(f"Temporal CV R2: {scores_temporal.mean():.4f} (+/- {scores_temporal.std():.4f})")

pipe.fit(X_train_full, y_train_full)
final_test_r2 = r2_score(y_test, pipe.predict(X_test))

print(f"\nActual Test R2 on independent future data: {final_test_r2:.4f}")


# ## Visual comparison of fixed-model scores
# 
# This graph helps show whether the shuffled KFold estimate is more optimistic than the temporal estimate and the actual future test score.

# prompt: Plot the comparison between naive CV, temporal CV, and independent test R2.
# modification: The graph is added to support the screen-capture video explanation.

score_summary = pd.DataFrame({
    "Evaluation": ["Naive KFold CV", "TimeSeriesSplit CV", "Independent Test"],
    "R2": [scores_naive.mean(), scores_temporal.mean(), final_test_r2]
})

plt.figure(figsize=(8, 5))
plt.bar(score_summary["Evaluation"], score_summary["R2"])
plt.axhline(0, linestyle="--")
plt.ylabel("R2 score")
plt.title("Fixed model: CV scores versus independent test score")
plt.xticks(rotation=20)
plt.show()

score_summary


# # 4. Model selection and evaluation
# 
# Now we use `GridSearchCV` for **model selection**.
# 
# The goal is to choose the best **hyper parameters** for the decision tree.
# 
# The grid tests different values of:
# 
# - `max_depth`
# - `min_samples_leaf`
# - `min_samples_split`
# 
# This section is run twice:
# 
# 1. with naive KFold
# 2. with TimeSeriesSplit

# prompt: Complete the evaluate_model_selection function using GridSearchCV.
# modification: The function receives the cross-validation strategy as an argument to compare KFold and TimeSeriesSplit.

def evaluate_model_selection(X_train, y_train, X_test, y_test, cv_strategy, name):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    param_grid = {
        "regressor__max_depth": [2, 4, 6, 8, 10, 12, None],
        "regressor__min_samples_leaf": [1, 3, 5, 10],
        "regressor__min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="r2",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\n===== Results for: {name} =====")
    print(f"Best Parameters found: {grid.best_params_}")
    print(f"Internal CV Score (R2): {grid.best_score_:.4f}")
    print(f"Independent Test Score (R2): {test_r2:.4f}")
    print(f"Gap CV - Test: {grid.best_score_ - test_r2:.4f}")

    return grid, test_r2


# # 5. Run the comparison
# 
# The main comparison is the gap between:
# 
# - internal CV score
# - independent test score
# 
# A large gap suggests that the validation strategy may be too optimistic.

# prompt: Run model selection with both validation strategies and compare results.

result_naive, test_r2_naive = evaluate_model_selection(
    X_train_full, y_train_full, X_test, y_test, cv_naive, "Naive K-Fold"
)

result_temporal, test_r2_temporal = evaluate_model_selection(
    X_train_full, y_train_full, X_test, y_test, cv_temporal, "Temporal Split"
)


# ## Model selection comparison graph
# 
# This graph compares the internal model-selection score with the independent test score.
# 
# If the naive KFold score is much higher than the test score, this is evidence of an over-optimistic validation result caused by time-series **data leakage**.

# prompt: Plot the internal CV and independent test scores for both model-selection strategies.

selection_summary = pd.DataFrame({
    "Strategy": ["Naive KFold", "TimeSeriesSplit"],
    "Internal CV R2": [result_naive.best_score_, result_temporal.best_score_],
    "Independent Test R2": [test_r2_naive, test_r2_temporal]
})

ax = selection_summary.set_index("Strategy").plot(kind="bar", figsize=(8, 5))
plt.axhline(0, linestyle="--")
plt.ylabel("R2 score")
plt.title("Model selection: internal CV score versus independent test score")
plt.xticks(rotation=0)
plt.show()

selection_summary


# # 6. Final model predictions on the independent test data set
# 
# The final evaluation uses the best temporal model because it respects the chronological structure of the data.
# 
# This plot compares observed and predicted values in the future test period.

# prompt: Plot observed versus predicted values for the independent future test set.
# modification: Use the best TimeSeriesSplit model as the final model because it avoids time-series leakage.

best_temporal_model = result_temporal.best_estimator_
y_pred_temporal = best_temporal_model.predict(X_test)

pred_df = pd.DataFrame({
    "Observed": y_test,
    "Predicted": y_pred_temporal
}, index=y_test.index)

plt.figure(figsize=(10, 5))
plt.plot(pred_df.index, pred_df["Observed"], marker="o", label="Observed")
plt.plot(pred_df.index, pred_df["Predicted"], marker="o", label="Predicted")
plt.ylabel("Depth_to_Groundwater_P25")
plt.title("Independent test data set: observed versus predicted")
plt.legend()
plt.xticks(rotation=30)
plt.show()

pred_df.head()


# # Final interpretation
# 
# In the video, the most important point is not only the numerical value of each score, but the difference between them.
# 
# Expected interpretation:
# 
# - If `Naive KFold` gives a much higher internal CV score than the independent test score, it is probably too optimistic.
# - This happens because shuffled folds can mix past and future observations, creating **data leakage**.
# - `TimeSeriesSplit` is more appropriate for this time-series problem because it validates the model on future observations relative to the training fold.
# - The **independent test data set** remains the most important final evaluation because it represents unseen future data.


# # Suggested video structure
# 
# 1. Show the goal: predict the **response variable** `Depth_to_Groundwater_P25`.
# 2. Explain the **predictors**: all variables from the two previous months.
# 3. Explain **imputation** and monthly aggregation.
# 4. Explain the chronological hard split and the **independent test data set**.
# 5. Compare **cross-validation** strategies and explain **folds**.
# 6. Explain why shuffled KFold can create **data leakage**.
# 7. Show fixed-model results.
# 8. Explain **hyper parameters** and **model selection** with GridSearchCV.
# 9. Compare internal CV scores with independent test scores.
# 10. Conclude why TimeSeriesSplit is the safer evaluation strategy for this problem.
