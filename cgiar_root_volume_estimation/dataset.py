from pathlib import Path
from typing import Tuple

# import typer
# from loguru import logger
# from tqdm import tqdm
import glob

import pandas as pd
import torch
from torch.utils.data import Dataset



# from cgiar_root_volume_estimation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()


# Metadata


# read train file


def read_csv_metadata(filepath: str,)->pd.DataFrame:
    """
    Helping function to read the metadata file
    """
    csv_file = pd.read_csv(filepath)

    return csv_file



class CGIAR_VOLUMNE_DATASET(Dataset):
    """
    A Dataset to read the Cgiar data
    """

    def __init__(self, input_folder: str,
                 input_file: str = "Train.csv")->None:
        """
        Read init
        """
        self.root_dataset = input_folder
        self.input_file = input_file

        # read sub datasets
        self.csv_file = read_csv_metadata(f"{self.root_dataset}/{input_file}")

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx: int)->Tuple[str,str]:

        return (self.csv_file["ID"][idx],
                f"{self.root_dataset}/zindi_cgiar_root_estimation/data/{self.input_file.split('.csv')[0].lower()}/{self.csv_file['FolderName'][idx]}/")
