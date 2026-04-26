import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
import yaml


# ----------------------------
# LOAD PARAMS.YAML
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "params.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ----------------------------
# PATH SETUP
# ----------------------------
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")


# ----------------------------
# LOGGING
# ----------------------------
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

if not logger.handlers:
    console_handler = logging.StreamHandler()
    filehandler = logging.FileHandler(os.path.join(LOGS_DIR, "data_ingestion.log"))

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    filehandler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(filehandler)


# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(url):
    return pd.read_csv(url)


# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_data(df, drop_cols):
    return df.drop(columns=drop_cols)


# ----------------------------
# SAVE DATA
# ----------------------------
def save_data(train_data, test_data, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)

    train_data.to_csv(os.path.join(raw_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(raw_dir, "test.csv"), index=False)


# ----------------------------
# MAIN
# ----------------------------
def main():

    url = config["data_ingestion"]["dataset_url"]
    test_size = config["data_ingestion"]["test_size"]
    random_state = config["data_ingestion"]["random_state"]

    drop_cols = config["preprocessing"]["drop_columns"]
    raw_dir = os.path.join(BASE_DIR, config["paths"]["raw_data"])

    df = load_data(url)

    df = preprocess_data(df, drop_cols)

    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    save_data(train_data, test_data, raw_dir)

    logger.info("Data ingestion completed")


if __name__ == "__main__":
    main()