import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib

MODEL_DIR = "models"

def train_model(df, features, model_type="RandomForest"):
    """Train a ML model for directional prediction."""
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    elif model_type == "LightGBM":
        model = LGBMClassifier(n_estimators=200, max_depth=5)
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_type} directional accuracy: {acc:.4f}")

    # Save model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}/{model_type}_model.pkl")

    return model
