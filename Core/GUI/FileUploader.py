import os
import shutil
import pandas as pd
from tkinter import filedialog as fd

class FileUploader:
    def __init__(self, upload_folder="Uploads"):
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def upload_file(self):
        file_path = fd.askopenfilename(
            title="Select a Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.upload_folder, filename)
            shutil.copy(file_path, destination)

            try:
                # Read based on extension
                if filename.endswith(".csv"):
                    data = pd.read_csv(destination)
                elif filename.endswith(".xlsx"):
                    data = pd.read_excel(destination)
                elif filename.endswith(".json"):
                    data = pd.read_json(destination)
                elif filename.endswith(".txt"):
                    # Try to detect delimiter automatically
                    with open(destination, 'r') as f:
                        first_line = f.readline()
                        delimiter = ',' if ',' in first_line else '\t'
                    data = pd.read_csv(destination, delimiter=delimiter)
                else:
                    raise ValueError("Unsupported file type")

                return filename, destination, data

            except Exception as e:
                print(f"Error reading file: {e}")
                return filename, destination, None

        return None, None, None
