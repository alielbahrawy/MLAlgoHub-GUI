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

class SecondPage(ctk.CTk):
    def __init__(self, data):
        super().__init__()
        print("SecondPage initialized with data:", data)
        self.title("Data Preprocessing Page")
        self.geometry("1024x700")
        self.configure(fg_color="gray14")
        self.original_df = data
        self.processed_df = self.original_df.copy()

        self.details = {
            "Simple Imputer": "mean, median, most_frequent, constant. Useful for numerical & categorical data.",
            "KNN Imputer": "Fills missing values based on nearest neighbors.",
            "IterativeImputer": "fills missing values by predicting them using other features in the dataset.",
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

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

    def on_select(self, event):
        selected = self.tree.item(self.tree.focus())["text"]
        self.info.delete("0.0", "end")
        self.info.insert("end", self.details.get(selected, "No description available."))
        try:
            self.apply_step(selected)
            if selected != "Simple Imputer":
                missing_values_numeric = self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum()
                missing_values_object = self.processed_df.select_dtypes(include=['object']).isnull().sum().sum()
                
                self.info.insert("end", "\n\n✅ Operation applied successfully.")
                    
                    
        except Exception as e:
            self.info.insert("end", f"\n\n❌ Error: {e}")
            
    def show_data_summary(self):
        num_cols = self.processed_df.select_dtypes(include=[np.number]).columns

        # Show head of numeric data
        self.info.insert("end", f"\n\n{self.processed_df[num_cols].head(10)}")
        self.info.insert("end", f"\n\n" + "_"*50)
        
        # Count of missing values
        missing_values_count = self.processed_df[num_cols].isnull().sum().sum()
        self.info.insert("end", f"\n\n Total Missing Values(Numerics): {missing_values_count}")
        self.info.insert("end", f"\n\n" + "_"*50)
        
        # Data types
        types = self.processed_df.dtypes
        self.info.insert("end", f"\n\n Data Types:\n{types}")
        self.info.insert("end", f"\n\n" + "_"*50)
        
        # Duplicated rows
        duplicate_count = self.processed_df.duplicated().sum()
        self.info.insert("end", f"\n\n Number of Duplicated Rows: {duplicate_count}")
        self.info.insert("end", f"\n\n" + "_"*50)

            

    def apply_step(self, step_name):
        num_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        if step_name == "Simple Imputer":
            def on_confirm():
                strategy = strategy_var.get()
                try:
                    if strategy == "constant":
                        fill_value = simpledialog.askstring("Constant Value", "Enter value:")
                        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                    else:
                        imputer = SimpleImputer(strategy=strategy)
                    self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                    self.processed_df.to_csv("processed_data.csv", index=False)
                    messagebox.showinfo("Done", f"Imputation using '{strategy}' completed.")
                    top.destroy()
                except Exception as e:
                    messagebox.showerror("Error", str(e))

            top = tk.Toplevel(self)
            top.title("Choose Imputer Strategy")
            tk.Label(top, text="Select Strategy:").pack(pady=5)
            strategy_var = tk.StringVar(value="mean")
            ttk.Combobox(top, textvariable=strategy_var, values=["mean", "median", "most_frequent", "constant"]).pack(pady=5)
            tk.Button(top, text="Apply", command=on_confirm).pack(pady=10)

        elif step_name == "KNN Imputer":
            if self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum() == 0:
                self.info.insert("end", f"\n\ndata doesn't have a missing value")
            else:
        
                imputer = KNNImputer(n_neighbors=2)
                self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

                
                
        elif step_name == "IterativeImputer":
            if self.processed_df.select_dtypes(include=[np.number]).isnull().sum().sum()== 0:
                self.info.insert("end", f"\n\ndata doesn't have a missing value")
            else:
        
                imputer = IterativeImputer()
                self.processed_df[num_cols] = imputer.fit_transform(self.processed_df[num_cols])
                self.show_data_summary()

              
              
                
                

        elif step_name == "Label Encoder":
            le = LabelEncoder()
            for col in self.processed_df.select_dtypes(include=['object']).columns:
                self.processed_df[col] = le.fit_transform(self.processed_df[col].astype(str))
            self.show_data_summary()


        elif step_name == "One Hot Encoder":
            self.processed_df = pd.get_dummies(self.processed_df)
            self.show_data_summary()


        elif step_name == "Z-score":
            z_scores = np.abs(stats.zscore(self.processed_df[num_cols]))
            self.processed_df = self.processed_df[(z_scores < 3).all(axis=1)]
            self.show_data_summary()

        elif step_name == "IQR method":
            Q1 = self.processed_df[num_cols].quantile(0.25)
            Q3 = self.processed_df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.processed_df[num_cols] < (Q1 - 1.5 * IQR)) | (self.processed_df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            self.processed_df = self.processed_df[mask]
            self.show_data_summary()

        elif step_name == "Remove Outliers":
            z_scores = np.abs(stats.zscore(self.processed_df[num_cols]))
            self.processed_df = self.processed_df[(z_scores < 3).all(axis=1)]
            self.show_data_summary()
 
        elif step_name == "Handle Duplicated":
            self.processed_df = self.processed_df.drop_duplicates()
            self.show_data_summary()

           
        elif step_name == "Min-Max Normalization":
            scaler = MinMaxScaler()
            self.processed_df[num_cols] = scaler.fit_transform(self.processed_df[num_cols])
            self.show_data_summary()
 
        elif step_name == "Standard Scaler":
            scaler = StandardScaler()
            self.processed_df[num_cols] = scaler.fit_transform(self.processed_df[num_cols])
            self.show_data_summary()


        elif step_name == "Log Transform":
            self.processed_df = self.processed_df.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)
            self.show_data_summary()

            
        if step_name != "Simple Imputer":
            self.processed_df.to_csv("processed_data.csv", index=False)
            
