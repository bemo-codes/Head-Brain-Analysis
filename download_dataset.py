#############################################################################
#
#   Importing required libraries
#
#############################################################################
from pathlib import Path
from urllib.request import urlretrieve

#############################################################################
#
#   Project constants
#
#############################################################################
DATA_URL = (
    "https://raw.githubusercontent.com/apekshar5/Machine-Learning/master/"
    "predict%20the%20brain%20weight/headbrain.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "headbrain.csv"

#############################################################################
#
#   Function name:      download_dataset
#   Input:              None
#   Description:        Downloads the public headbrain CSV and saves it beside
#                       the main Streamlit application.
#   Output:             headbrain.csv file
#   Date:               06/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def download_dataset():
    try:
        print("Downloading head-brain dataset...")
        urlretrieve(DATA_URL, OUTPUT_PATH)
        print(f"Dataset downloaded successfully: {OUTPUT_PATH}")
        print("Expected columns: Gender, Age Range, Head Size(cm^3), Brain Weight(grams)")
    except Exception as error:
        print(f"Dataset download failed: {error}")
        print("Download it manually from:")
        print("https://github.com/apekshar5/Machine-Learning/blob/master/predict%20the%20brain%20weight/headbrain.csv")
        print("Click Raw and save the file as headbrain.csv beside app.py.")


#############################################################################
#
#   Starter of main
#
#############################################################################
if __name__ == "__main__":
    download_dataset()
