import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# LOAD DATASET

df = pd.read_csv("data/housing.csv")

# HANDLE CATEGORICAL COLUMN

df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# FEATURES & TARGET

X = df.drop("median_house_value", axis=1)

y = df["median_house_value"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL TRAINING

model = DecisionTreeRegressor(
    # hyperparameters
    criterion="squared_error",
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)

# EVALUATION

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("MSE:", mse)

print("R2 Score:", r2)

# SAVE MODEL

with open("models/decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

# SAVE FEATURE COLUMNS

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model saved successfully.")
