import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os



logs_dir ="../logs"
os.makedirs(logs_dir, exist_ok=True)

# creating logging object
logger =logging.getLogger("data_ingestion")

# set level
logger.setLevel("DEBUG")

# handler
console_handler =logging.StreamHandler()
console_handler.setLevel("DEBUG")

filehandler =logging.FileHandler(os.path.join(logs_dir, "data_ingestion.log"))
filehandler.setLevel("DEBUG")

# formatter 
formatter =logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)



logger.addHandler(console_handler)
logger.addHandler(filehandler)

def load_data(url: str):
    try:
        df = pd.read_csv(url)
        logger.info("Data loaded successfully from URL")
        return df

    except Exception as e:
        logger.error("Failed to load data from URL: %s", str(e))
        return None
    

def preprocess_data(df: pd.DataFrame):

    try:
        logger.info("Starting preprocessing...")

        # columns to drop
        cols_to_drop = ["RowNumber", "CustomerId", "Surname"]

        df = df.drop(columns=cols_to_drop)

        logger.info("Dropped columns: %s", cols_to_drop)
        logger.info("Preprocessing completed successfully")

        return df

 

    except Exception as e:
        logger.error("Unexpected error in preprocessing: %s", str(e))
        return None
    


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_dir: str):

    try:
        logger.info("Starting data saving process...")

        # create raw directory
        raw_dir = os.path.join(save_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # file paths
        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

        # save files
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info("Train data saved at: %s", train_path)
        logger.info("Test data saved at: %s", test_path)
        logger.info("Data saving completed successfully")

    except Exception as e:
        logger.error("Error while saving data: %s", str(e))



def main():

    url = "https://raw.githubusercontent.com/FaiqImran123/CustomerChurnPred/FaiqImran123/Churn_Modelling.csv"

    logger.info("Pipeline started")

    # Step 1: Load data
    df = load_data(url)

    if df is None:
        logger.error("Pipeline stopped due to data loading failure")
        return

    # Step 2: Preprocess data
    df = preprocess_data(df)

    if df is None:
        logger.error("Pipeline stopped due to preprocessing failure")
        return

    # Step 3: Train-test split
    try:
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        logger.info("Train-test split completed")
    except Exception as e:
        logger.error("Error during train-test split: %s", str(e))
        return

    # Step 4: Save data
    save_data(train_data, test_data, save_dir="../data")

    logger.info("Pipeline completed successfully")


# Run pipeline
if __name__ == "__main__":
    main()