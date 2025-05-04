import customtkinter as ctk
from tkinter import messagebox, simpledialog, ttk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from scipy import stats
import tkinter as tk
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class PrePage(ctk.CTk):
    def __init__(self, data, on_back_callback=None):
        super().__init__()
        self.title("Data Preprocessing Page")
        self.geometry('1024x720+250+50')
        self.configure(fg_color="gray14")
        self.original_df = data
        self.processed_df = self.original_df.copy() if self.original_df is not None else None
        self.on_back_callback = on_back_callback

        self.details = {
            "Simple Imputer": "mean, median, most_frequent, constant. Useful for numerical & categorical data.",
            "KNN Imputer": "Fills missing values based on nearest neighbors.",
            "IterativeImputer": "Fills missing values by predicting them using other features in the dataset.",
            "Label Encoder": "Converts categorical columns into numerical values.",
            "One Hot Encoder": "Converts categorical columns into multiple binary columns.",
            "IQR method": "Detects outliers using Interquartile Range.",
            "Z-score": "Detects outliers using standard deviation from mean.",
            "Remove Outliers": "Drops records that are considered outliers.",
            "Handle Duplicated": "Removes or flags duplicate records.",
            "Min-Max Normalization": "Scales between 0 and 1.",
            "Standard Scaler": "Scales using mean = 0 and std = 1.",
            "Log Transform": "Reduces skewness in data distribution."
        }

        self.create_widgets()
        if self.processed_df is not None:
            self.show_data_summary()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#1E3A46", foreground="white", rowheight=30, fieldbackground="#1E3A46")
        style.map("Treeview", background=[("selected", "#2B5B6D")], foreground=[("selected", "white")])
        style.configure("Treeview.Heading", background="gray14", foreground="white", font=('Arial', 12, 'bold'))

        tree_frame = ctk.CTkFrame(main_frame, fg_color="black")
        tree_frame.pack(side="left", fill="y")

        self.tree = ttk.Treeview(tree_frame)
        self.tree.heading("#0", text="Preprocessing Steps", anchor="w")

        step1 = self.tree.insert("", "end", text="Handle Missing Values", open=True)
        self.tree.insert(step1, "end", text="Simple Imputer")
        self.tree.insert(step1, "end", text="KNN Imputer")
        self.tree.insert(step1, "end", text="IterativeImputer")

        step2 = self.tree.insert("", "end", text="Data Encoding", open=True)
        self.tree.insert(step2, "end", text="Label Encoder")
        self.tree.insert(step2, "end", text="One Hot Encoder")

        step3 = self.tree.insert("", "end", text="Outlier Detection", open=True)
        self.tree.insert(step3, "end", text="IQR method")
        self.tree.insert(step3, "end", text="Z-score")
        self.tree.insert("", "end", text="Remove Outliers")
        self.tree.insert("", "end", text="Handle Duplicated")

        step4 = self.tree.insert("", "end", text="Normalization", open=True)
        self.tree.insert(step4, "end", text="Min-Max Normalization")
        self.tree.insert(step4, "end", text="Standard Scaler")

        self.tree.insert("", "end", text="Log Transform")
        self.tree.pack(side="left", fill="y")

        self.info = ctk.CTkTextbox(main_frame, width=400)
        self.info.pack(side="right", fill="both", expand=True)

        button_frame = ctk.CTkFrame(main_frame, fg_color="gray14")
        button_frame.pack(fill="x", pady=10)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)


    def on_select(self, event):
        if self.processed_df is None:
            self.info.delete("0.0", "end")
            self.info.insert("end", "No dataset available. Please ensure a dataset is loaded.")
            return
        selected = self.tree.item(self.tree.focus())["text"]
        self.info.delete("0.0", "end")
        self.info.insert("end", self.details.get(selected, "No description available."))
        try:
            self.apply_step(selected)
            if selected != "Simple Imputer":
                missing_values_numeric = self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum()
                missing_values_object = self.processed_df.select_dtypes(include=['object']).isnull().sum().sum()
                self.info.insert("end", f"\n\n✅ Operation applied successfully.\n"
                                        f"Missing Values (Numeric): {missing_values_numeric}\n"
                                        f"Missing Values (Categorical): {missing_values_object}")
        except Exception as e:
            self.info.insert("end", f"\n\n❌ Error: {e}")

    def show_data_summary(self):
        if self.processed_df is None:
            self.info.insert("0.0", "No dataset available.")
            return
        num_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        cat_cols = self.processed_df.select_dtypes(include=['object']).columns

        self.info.insert("end", f"\n\nSample Data (Numeric Columns):\n{self.processed_df[num_cols].head(10)}")
        self.info.insert("end", f"\n\n" + "_"*50)

        missing_values_count = self.processed_df.isnull().sum().sum()
        self.info.insert("end", f"\n\nTotal Missing Values: {missing_values_count}")
        self.info.insert("end", f"\n\n" + "_"*50)

        types = self.processed_df.dtypes
        self.info.insert("end", f"\n\nData Types:\n{types}")
        self.info.insert("end", f"\n\n" + "_"*50)

        duplicate_count = self.processed_df.duplicated().sum()
        self.info.insert("end", f"\n\nNumber of Duplicated Rows: {duplicate_count}")
        self.info.insert("end", f"\n\n" + "_"*50)

        if len(cat_cols) > 0:
            unique_counts = self.processed_df[cat_cols].nunique()
            self.info.insert("end", f"\n\nUnique Values (Categorical):\n{unique_counts}")
            self.info.insert("end", f"\n\n" + "_"*50)

    def apply_step(self, step_name):
        if self.processed_df is None:
            raise ValueError("No dataset available.")
        num_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        cat_cols = self.processed_df.select_dtypes(include=['object']).columns

        if step_name == "Simple Imputer":
            def on_confirm():
                strategy = strategy_var.get()
                try:
                    if len(num_cols) == 0 and strategy not in ["most_frequent", "constant"]:
                        messagebox.showerror("Error", "No numerical columns available for mean/median imputation.")
                        top.destroy()
                        return
                    if strategy == "constant":
                        fill_value = simpledialog.askstring("Constant Value", "Enter value:")
                        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                    else:
                        imputer = SimpleImputer(strategy=strategy)
                    if len(num_cols) > 0:
                        self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                    elif len(cat_cols) > 0 and strategy in ["most_frequent", "constant"]:
                        self.processed_df[cat_cols] = imputer.fit_transform(self.processed_df[cat_cols])
                    self.processed_df.to_csv("processed_data.csv", index=False)
                    messagebox.showinfo("Done", f"Imputation using '{strategy}' completed.")
                    top.destroy()
                    self.show_data_summary()
                except Exception as e:
                    messagebox.showerror("Error", str(e))
                    top.destroy()

            top = tk.Toplevel(self)
            top.title("Choose Imputer Strategy")
            tk.Label(top, text="Select Strategy:").pack(pady=5)
            strategy_var = tk.StringVar(value="mean")
            ttk.Combobox(top, textvariable=strategy_var, values=["mean", "median", "most_frequent", "constant"]).pack(pady=5)
            tk.Button(top, text="Apply", command=on_confirm).pack(pady=10)

        elif step_name == "KNN Imputer":
            if self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum() == 0:
                self.info.insert("end", "\n\nData doesn't have missing values.")
            elif len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for KNN Imputer.")
            else:
                imputer = KNNImputer(n_neighbors=2)
                self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

        elif step_name == "IterativeImputer":
            if self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum() == 0:
                self.info.insert("end", "\n\nData doesn't have missing values.")
            elif len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for IterativeImputer.")
            else:
                imputer = IterativeImputer()
                self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

        elif step_name == "Label Encoder":
            if len(cat_cols) == 0:
                self.info.insert("end", "\n\nNo categorical columns available for Label Encoder.")
            else:
                le = LabelEncoder()
                for col in cat_cols:
                    self.processed_df[col] = le.fit_transform(self.processed_df[col].astype(str))
                self.show_data_summary()

        elif step_name == "One Hot Encoder":
            if len(cat_cols) == 0:
                self.info.insert("end", "\n\nNo categorical columns available for One Hot Encoder.")
            else:
                self.processed_df = pd.get_dummies(self.processed_df, columns=cat_cols)
                self.show_data_summary()

        elif step_name == "Z-score":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for Z-score.")
            else:
                z_scores = np.abs(stats.zscore(self.processed_df[num_cols]))
                self.processed_df = self.processed_df[(z_scores < 3).all(axis=1)]
                self.show_data_summary()

        elif step_name == "IQR method":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for IQR method.")
            else:
                Q1 = self.processed_df[num_cols].quantile(0.25)
                Q3 = self.processed_df[num_cols].quantile(0.75)
                IQR = Q3 - Q1
                mask = ~((self.processed_df[num_cols] < (Q1 - 1.5 * IQR)) | (self.processed_df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                self.processed_df = self.processed_df[mask]
                self.show_data_summary()

        elif step_name == "Remove Outliers":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for outlier removal.")
            else:
                z_scores = np.abs(stats.zscore(self.processed_df[num_cols]))
                self.processed_df = self.processed_df[(z_scores < 3).all(axis=1)]
                self.show_data_summary()

        elif step_name == "Handle Duplicated":
            initial_rows = len(self.processed_df)
            self.processed_df = self.processed_df.drop_duplicates()
            removed_rows = initial_rows - len(self.processed_df)
            self.info.insert("end", f"\n\nRemoved {removed_rows} duplicated rows.")
            self.show_data_summary()

        elif step_name == "Min-Max Normalization":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for Min-Max Normalization.")
            else:
                scaler = MinMaxScaler()
                self.processed_df[num_cols] = scaler.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

        elif step_name == "Standard Scaler":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for Standard Scaler.")
            else:
                scaler = StandardScaler()
                self.processed_df[num_cols] = scaler.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

        elif step_name == "Log Transform":
            if len(num_cols) == 0:
                self.info.insert("end", "\n\nNo numerical columns available for Log Transform.")
            else:
                self.processed_df[num_cols] = self.processed_df[num_cols].apply(lambda x: np.log1p(x) if x.min() >= 0 else np.log1p(x - x.min() + 1))
                self.show_data_summary()

        if step_name != "Simple Imputer":
            self.processed_df.to_csv("processed_data.csv", index=False)