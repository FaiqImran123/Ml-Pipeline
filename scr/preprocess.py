import numpy as np
import pandas as pd
import logging
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# =========================
# PATH SETUP (IMPORTANT FIX)
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")


# =========================
# LOGGING SETUP
# =========================

os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("data_pipeline")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "data_pipeline.log"))

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================
# GLOBAL ENCODERS
# =========================

label_encoder = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")


# =========================
# FIT ON TRAIN ONLY
# =========================

def fit_preprocess(df: pd.DataFrame):

    try:
        logger.info("Fitting preprocessors on TRAIN data...")

        # Encode Gender
        df["Gender"] = label_encoder.fit_transform(df["Gender"])

        # One-hot encode Geography
        geo = np.array(df["Geography"]).reshape(-1, 1)
        ohe.fit(geo)

        geo_encoded = pd.DataFrame(
            ohe.transform(geo),
            columns=ohe.get_feature_names_out(["Geography"])
        )

        df.drop(columns=["Geography"], inplace=True)
        df = pd.concat([df.reset_index(drop=True), geo_encoded], axis=1)

        logger.info("Train preprocessing completed (FIT DONE)")

        return df

    except Exception as e:
        logger.error("Error in fit_preprocess: %s", str(e))
        return None


# =========================
# TRANSFORM TEST ONLY
# =========================

def transform_preprocess(df: pd.DataFrame):

    try:
        logger.info("Transforming TEST data...")

        # Encode Gender
        df["Gender"] = label_encoder.transform(df["Gender"])

        # One-hot encode Geography
        geo = np.array(df["Geography"]).reshape(-1, 1)

        geo_encoded = pd.DataFrame(
            ohe.transform(geo),
            columns=ohe.get_feature_names_out(["Geography"])
        )

        df.drop(columns=["Geography"], inplace=True)
        df = pd.concat([df.reset_index(drop=True), geo_encoded], axis=1)

        logger.info("Test preprocessing completed")

        return df

    except Exception as e:
        logger.error("Error in transform_preprocess: %s", str(e))
        return None


# =========================
# SAVE INTERIM DATA
# =========================

def save_interim(train_data, test_data):

    try:
        logger.info("Saving interim data...")

        interim_dir = os.path.join(DATA_DIR, "interim")
        os.makedirs(interim_dir, exist_ok=True)

        train_path = os.path.join(interim_dir, "train.csv")
        test_path = os.path.join(interim_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Train saved at {train_path}")
        logger.info(f"Test saved at {test_path}")

    except Exception as e:
        logger.error("Error saving interim data: %s", str(e))


# =========================
# MAIN PIPELINE
# =========================

def main():

    logger.info("PIPELINE STARTED")

    # -------- Load RAW data --------
    try:
        raw_dir = os.path.join(DATA_DIR, "raw")

        train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(raw_dir, "test.csv"))

        logger.info("Raw data loaded successfully")

    except Exception as e:
        logger.error("Error loading raw data: %s", str(e))
        return

    # -------- FIT TRAIN --------
    train_df = fit_preprocess(train_df)

    if train_df is None:
        logger.error("Train preprocessing failed")
        return

    # -------- TRANSFORM TEST --------
    test_df = transform_preprocess(test_df)

    if test_df is None:
        logger.error("Test preprocessing failed")
        return

    # -------- SAVE INTERIM --------
    save_interim(train_df, test_df)

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()