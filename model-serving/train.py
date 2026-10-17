# train.py — train once, save one artifact
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ["Age", "EstimatedSalary"]   # the feature schema — order matters

df = pd.read_csv("Social_Network_Ads.csv")
X = df[FEATURES].values   # ndarray: order-based, same shape the API sends at serve time
y = df["Purchased"]

pipe = Pipeline([
    ("scale", StandardScaler()),      # step 1: scale the numbers
    ("clf", LogisticRegression()),    # step 2: the model
])
pipe.fit(X, y)

joblib.dump(pipe, "model.joblib")     # scaler + model together in one file
print("saved model.joblib")
