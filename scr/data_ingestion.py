import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os


# ----------------------------
# PATH SETUP (IMPORTANT)
# ----------------------------

# Get project root (parent of src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ----------------------------
# LOGGING SETUP
# ----------------------------

os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

filehandler = logging.FileHandler(os.path.join(LOGS_DIR, "data_ingestion.log"))
filehandler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)


# ----------------------------
# LOAD DATA
# ----------------------------

def load_data(url: str):
    try:
        df = pd.read_csv(url)
        logger.info("Data loaded successfully from URL")
        return df

    except Exception as e:
        logger.error("Failed to load data from URL: %s", str(e))
        return None


# ----------------------------
# PREPROCESS
# ----------------------------

def preprocess_data(df: pd.DataFrame):
    try:
        logger.info("Starting preprocessing...")

        cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
        df = df.drop(columns=cols_to_drop)

        logger.info(f"Dropped columns: {cols_to_drop}")
        logger.info("Preprocessing completed successfully")

        return df

    except Exception as e:
        logger.error("Unexpected error in preprocessing: %s", str(e))
        return None


# ----------------------------
# SAVE DATA
# ----------------------------

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        logger.info("Starting data saving process...")

        raw_dir = os.path.join(DATA_DIR, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Train data saved at: {train_path}")
        logger.info(f"Test data saved at: {test_path}")
        logger.info("Data saving completed successfully")

    except Exception as e:
        logger.error("Error while saving data: %s", str(e))


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def main():

    url = "https://raw.githubusercontent.com/FaiqImran123/CustomerChurnPred/FaiqImran123/Churn_Modelling.csv"

    logger.info("Pipeline started")

    # Load
    df = load_data(url)
    if df is None:
        logger.error("Stopping pipeline: data loading failed")
        return

    # Preprocess
    df = preprocess_data(df)
    if df is None:
        logger.error("Stopping pipeline: preprocessing failed")
        return

    # Split
    try:
        train_data, test_data = train_test_split(
            df, test_size=0.2, random_state=42
        )
        logger.info("Train-test split completed")

    except Exception as e:
        logger.error("Error during train-test split: %s", str(e))
        return

    # Save
    save_data(train_data, test_data)

    logger.info("Pipeline completed successfully")


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    main()