import pandas as pd
import os

# data set function 
def load_dataset(path):

    file_extension = os.path.splitext(path)[1].lower()

    if file_extension == ".csv":
        df = pd.read_csv(path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    elif file_extension == ".json":
        df = pd.read_json(path)
    elif file_extension == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"File type not supported: {file_extension}")

    return df
