import customtkinter as ctk
import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DatasetSummaryApp(ctk.CTk):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.root.title("Dataset Summary - ML ALGONHUB")
        self.root.geometry("900x600")
        self.root.configure(fg_color="gray14")
        self.df = None
        self.create_widgets()

    def create_widgets(self):
        upperframe = ctk.CTkFrame(self.root, height=50, corner_radius=0, fg_color="gray14")
        upperframe.pack(fill="x")

        self.label_title = ctk.CTkLabel(self.root, text="Dataset Summary", font=("Arial", 24, "bold"), text_color="#2B5B6D")
        self.label_title.pack(pady=5)

        self.btn_load = ctk.CTkButton(self.root, text="Load CSV File", command=self.load_file, fg_color="#1E3A46", hover_color="#144870")
        self.btn_load.pack(pady=5)

        self.main_frame = ctk.CTkFrame(self.root, fg_color="gray14")
        self.main_frame.pack(pady=5, padx=5, fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(self.main_frame, fg_color="gray14")
        self.left_frame.pack(side="left", padx=5, fill="both", expand=True)

        self.label_sample = ctk.CTkLabel(self.left_frame, text="Name of the Dataset", font=("Arial", 12, "italic"), text_color="white")
        self.label_sample.pack(anchor="nw", padx=5, pady=2)

        self.sample_container = ctk.CTkFrame(self.left_frame, fg_color="gray14")
        self.sample_container.pack(pady=2, padx=2, fill="both", expand=True)

        self.sample_text = ctk.CTkTextbox(
            self.sample_container,
            height=300,
            width=400,
            fg_color="#1E3A46",
            text_color="white",
            wrap="none",
            border_width=1,
            border_color="#2B5B6D"
        )
        self.sample_text.pack(pady=2, padx=2, fill="both", expand=True)
        self.sample_text.insert("0.0", "sample of data")
        self.sample_text.configure(state="disabled")

        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color="gray14")
        self.right_frame.pack(side="right", padx=5, fill="both", expand=True)

        self.label_placeholder = ctk.CTkLabel(self.right_frame, text="", font=("Arial", 12, "italic"), text_color="white")
        self.label_placeholder.pack(anchor="nw", padx=5, pady=2)

        self.summary_text = ctk.CTkTextbox(
            self.right_frame,
            height=300,
            width=400,
            fg_color="#1E3A46",
            text_color="white",
            border_width=1,
            border_color="#2B5B6D"
        )
        self.summary_text.pack(pady=2, padx=2, fill="both", expand=True)
        self.summary_text.insert("0.0", "Overview: Dataset purpose and source.\n\n"
                                        "Shape: Rows and columns.\n\n"
                                        "Missing Values: Null count and % per column.\n\n"
                                        "Duplicates: Duplicated rows count.\n\n"
                                        "Data Types: Numerical, categorical, datetime, etc.\n\n"
                                        "Stats: Mean, median, std, min, max for numerical columns.\n\n"
                                        "Unique Values: Unique counts for categorical columns.\n\n"
                                        "Critical Columns: High missing values or key features.")
        self.summary_text.configure(state="disabled")

        self.button_frame = ctk.CTkFrame(self.right_frame, fg_color="gray14")
        self.button_frame.pack(pady=5, padx=2, fill="x")

        self.btn_back = ctk.CTkButton(
            self.button_frame,
            text="◀ Back",
            command=lambda: messagebox.showinfo("Back", "Back button pressed!"),
            fg_color="#1E3A46",
            hover_color="#144870",
            font=("Arial", 14),
            corner_radius=8,
            height=40
        )
        self.btn_back.pack(side="left", padx=5)

        self.btn_next_visualization = ctk.CTkButton(
            self.button_frame,
            text="Next ▶",
            command=self.next_visualization,
            fg_color="#1E3A46",
            hover_color="#144870",
            font=("Arial", 14, "bold"),
            corner_radius=8,
            height=40
        )
        self.btn_next_visualization.pack(side="right", padx=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path, encoding="utf-8")
                dataset_name = file_path.split("/")[-1]
                self.label_sample.configure(text=dataset_name)
                self.display_sample()
                self.display_summary()
                messagebox.showinfo("Success", "File loaded successfully!")
            except UnicodeDecodeError:
                try:
                    self.df = pd.read_csv(file_path, encoding="latin1")
                    dataset_name = file_path.split("/")[-1]
                    self.label_sample.configure(text=dataset_name)
                    self.display_sample()
                    self.display_summary()
                    messagebox.showinfo("Success", "File loaded successfully with latin1 encoding!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file: {e}")
                    self.df = None
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                self.df = None

    def display_sample(self):
        if self.df is None:
            return
        self.sample_text.configure(state="normal")
        self.sample_text.delete("0.0", "end")
        sample_data = self.df.head(30).to_string(index=False)  # غيرنا لـ 25 سطر
        self.sample_text.insert("0.0", sample_data)
        self.sample_text.configure(state="disabled")

    def display_summary(self):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("0.0", "end")

        overview = "Overview: Dataset purpose and source.\n\n"
        shape = f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns.\n\n"

        missing_values = "Missing Values:\n"
        missing_counts = self.df.isna().sum()
        total_rows = self.df.shape[0]
        for col, count in missing_counts.items():
            if count > 0:
                percentage = (count / total_rows) * 100
                missing_values += f"  {col}: {count} ({percentage:.2f}%)\n"

        duplicates = f"\nDuplicates: {self.df.duplicated().sum()} duplicated rows.\n\n"

        data_types = "Data Types:\n"
        for col, dtype in self.df.dtypes.items():
            data_types += f"  {col}: {dtype}\n"

        stats = "\nStats (Numerical Columns):\n"
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                stats += f"  {col}:\n"
                stats += f"    Mean: {self.df[col].mean():.2f}\n"
                stats += f"    Median: {self.df[col].median():.2f}\n"
                stats += f"    Std: {self.df[col].std():.2f}\n"
                stats += f"    Min: {self.df[col].min():.2f}\n"
                stats += f"    Max: {self.df[col].max():.2f}\n"

        unique_values = "\nUnique Values (Categorical Columns):\n"
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            unique_values += f"  {col}: {unique_count} unique values\n"

        critical_columns = "\nCritical Columns:\n"
        for col, count in missing_counts.items():
            percentage = (count / total_rows) * 100
            if percentage > 30:
                critical_columns += f"  {col}: High missing values ({percentage:.2f}%)\n"

        summary = overview + shape + missing_values + duplicates + data_types + stats + unique_values + critical_columns
        self.summary_text.insert("0.0", summary)
        self.summary_text.configure(state="disabled")

    def next_visualization(self):
        messagebox.showinfo("Next Visualization", "This feature is not implemented yet!")

if __name__ == "__main__":
    root = ctk.CTk()
    app = DatasetSummaryApp(root)
    root.mainloop()