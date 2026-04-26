import os
import logging
import pandas as pd
import joblib
import yaml   # ✅ ADDED ONLY

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# LOAD PARAMS.YAML (ADDED ONLY)
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "params.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# =========================
# PATH SETUP (UNCHANGED)
# =========================

LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# =========================
# LOGGING SETUP (UNCHANGED)
# =========================

os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()

file_handler = logging.FileHandler(
    os.path.join(LOGS_DIR, "model_training.log")
)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================
# LOAD DATA (UNCHANGED)
# =========================

def load_data():
    logger.info("Loading train and test data from data/interim...")

    train_path = os.path.join(DATA_DIR, "interim", "train.csv")
    test_path = os.path.join(DATA_DIR, "interim", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    X_train = train_df.drop("Exited", axis=1)
    y_train = train_df["Exited"]

    X_test = test_df.drop("Exited", axis=1)
    y_test = test_df["Exited"]

    return X_train, X_test, y_train, y_test


# =========================
# TRAIN MODEL (ONLY CHANGE HERE)
# =========================

def train_model(X_train, y_train):

    logger.info("Initializing RandomForest model...")

    params = config["model_training"]   # ✅ ADDED

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],   # ✅ CHANGED
        max_depth=params["max_depth"],         # ✅ CHANGED
        random_state=params["random_state"],   # (now from YAML)
        n_jobs=-1
    )

    logger.info(
        f"Training model with n_estimators={params['n_estimators']}, max_depth={params['max_depth']}"
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model


# =========================
# EVALUATE MODEL (UNCHANGED)
# =========================

def evaluate_model(model, X_test, y_test):

    logger.info("Evaluating model...")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    logger.info(f"Accuracy: {acc}")
    logger.info("Classification Report:\n%s", classification_report(y_test, preds))

    return acc


# =========================
# SAVE MODEL (UNCHANGED except optional config use not needed)
# =========================

def save_model(model, model_name="random_forest.pkl"):

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, model_name)

    joblib.dump(model, model_path)

    logger.info(f"Model saved at {model_path}")


# =========================
# MAIN PIPELINE (UNCHANGED LOGIC)
# =========================

def main():

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, "random_forest.pkl")

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()