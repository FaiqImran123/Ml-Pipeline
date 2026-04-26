import os
import logging
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ----------------------------
# LOGGING SETUP (MODEL TRAINING)
# ----------------------------

logs_dir = "../logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

filehandler = logging.FileHandler(os.path.join(logs_dir, "model_training.log"))
filehandler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)


# ----------------------------
# LOAD DATA
# ----------------------------

def load_data():
    logger.info("Loading train and test data from data/interim...")

    train_path = "../data/interim/train.csv"
    test_path = "../data/interim/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    X_train = train_df.drop("Exited", axis=1)
    y_train = train_df["Exited"]

    X_test = test_df.drop("Exited", axis=1)
    y_test = test_df["Exited"]

    return X_train, X_test, y_train, y_test


# ----------------------------
# MODEL TRAINING FUNCTION
# ----------------------------

def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    logger.info("Initializing RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}")
    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model


# ----------------------------
# MODEL EVALUATION
# ----------------------------

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    logger.info(f"Accuracy: {acc}")

    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, preds))

    return acc


# ----------------------------
# SAVE MODEL
# ----------------------------

def save_model(model, model_name="random_forest.pkl"):
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_name)

    joblib.dump(model, model_path)

    logger.info(f"Model saved at {model_path}")


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(
        X_train,
        y_train,
        n_estimators=200,
        max_depth=10
    )

    evaluate_model(model, X_test, y_test)

    save_model(model, "random_forest.pkl")


if __name__ == "__main__":
    main()