import customtkinter as ctk
from tkinter import messagebox, StringVar
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
from tkinter import IntVar

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PrePage(ctk.CTk):
    def __init__(self, data=None, on_back_callback=None, on_next_callback=None):
        super().__init__()

        self.title("ML AlgoHub - Data Preprocessing")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.data = data
        self.processed_df = data.copy() if data is not None else None
        self.on_back_callback = on_back_callback
        self.on_next_callback = on_next_callback
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numeric_impute_vars = {}  # For selecting numeric columns to impute
        self.categorical_impute_vars = {}  # For selecting categorical columns to impute
        self.normalize_column_vars = {}  # For selecting columns to normalize
        self.numeric_select_all_var = IntVar(value=0)  # For "All" checkbox in numeric imputation
        self.categorical_select_all_var = IntVar(value=0)  # For "All" checkbox in categorical imputation

        self.create_gui()

    def create_gui(self):
        # Top Header Frame
        top_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60)
        top_frame.pack(side="top", fill="x")
        ctk.CTkLabel(top_frame, text="Data Preprocessing", font=("Arial", 28, "bold"), text_color="#00b7eb").pack(pady=10)

        # Main Body Frame
        body_frame = ctk.CTkFrame(self)
        body_frame.pack(fill="both", expand=True)

        # Sidebar with Scrollable Frame
        sidebar_container = ctk.CTkFrame(body_frame, width=300)
        sidebar_container.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar = ctk.CTkScrollableFrame(sidebar_container, width=300, height=600, fg_color="#1a1a1a")
        self.sidebar.pack(fill="both", expand=True)

        # 1. Handle Missing Values - Numeric
        ctk.CTkLabel(self.sidebar, text="Handle Missing Values (Numeric):", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.numeric_missing_strategy = StringVar(value="Simple Imputer")
        self.numeric_imputer_combobox = ctk.CTkComboBox(
            self.sidebar, variable=self.numeric_missing_strategy, values=["Simple Imputer", "KNN", "Iterative"], state="readonly", width=200
        )
        self.numeric_imputer_combobox.pack(pady=5)
        self.simple_imputer_strategy = StringVar(value="Mean")
        self.simple_imputer_combobox = ctk.CTkComboBox(
            self.sidebar, variable=self.simple_imputer_strategy, values=["Mean", "Median"], state="readonly", width=200
        )
        self.simple_imputer_combobox.pack(pady=5)
        self.simple_imputer_combobox.configure(state="disabled")  # Initially disabled
        self.numeric_missing_strategy.trace_add("write", self.toggle_simple_imputer_combobox)
        ctk.CTkLabel(self.sidebar, text="Select Numeric Columns:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        ctk.CTkCheckBox(self.sidebar, text="Select All", variable=self.numeric_select_all_var, command=self.toggle_numeric_select_all).pack(anchor="w", padx=5, pady=2)
        self.numeric_impute_frame = ctk.CTkScrollableFrame(self.sidebar, height=100, fg_color="#2b2b2b")
        self.numeric_impute_frame.pack(pady=5, fill="x")
        self.update_numeric_impute_checkboxes()
        ctk.CTkButton(self.sidebar, text="Impute Numeric Missing Values", command=self.impute_numeric_missing, fg_color="#1E3A46").pack(pady=5)

        # 1. Handle Missing Values - Categorical
        ctk.CTkLabel(self.sidebar, text="Handle Missing Values (Categorical):", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.categorical_missing_strategy = StringVar(value="Mode")
        ctk.CTkComboBox(
            self.sidebar, variable=self.categorical_missing_strategy, values=["Mode"], state="readonly", width=200
        ).pack(pady=5)
        ctk.CTkLabel(self.sidebar, text="Select Categorical Columns:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        ctk.CTkCheckBox(self.sidebar, text="Select All", variable=self.categorical_select_all_var, command=self.toggle_categorical_select_all).pack(anchor="w", padx=5, pady=2)
        self.categorical_impute_frame = ctk.CTkScrollableFrame(self.sidebar, height=100, fg_color="#2b2b2b")
        self.categorical_impute_frame.pack(pady=5, fill="x")
        self.update_categorical_impute_checkboxes()
        ctk.CTkButton(self.sidebar, text="Impute Categorical Missing Values", command=self.impute_categorical_missing, fg_color="#1E3A46").pack(pady=5)

        # 1. Handle Missing Values - Drop Rows
        ctk.CTkLabel(self.sidebar, text="Drop Rows with Missing Values:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.drop_rows_strategy = StringVar(value="All")
        ctk.CTkComboBox(
            self.sidebar, variable=self.drop_rows_strategy, values=["All", "More than 50%"], state="readonly", width=200
        ).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Drop Rows", command=self.drop_missing_rows, fg_color="#1E3A46").pack(pady=5)

        # 2. Handle Duplicates
        ctk.CTkLabel(self.sidebar, text="Handle Duplicates:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        ctk.CTkButton(self.sidebar, text="Check Duplicates", command=self.check_duplicates, fg_color="#1E3A46").pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Remove Duplicates", command=self.remove_duplicates, fg_color="#1E3A46").pack(pady=5)

        # 3. Encoding Section
        ctk.CTkLabel(self.sidebar, text="Encode Categorical Data:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        ctk.CTkLabel(self.sidebar, text="Select Target for Label Encoding:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.label_encode_target_var = StringVar()
        self.label_encode_target_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.label_encode_target_var, values=self.get_categorical_columns(), state="readonly", width=200
        )
        self.label_encode_target_dropdown.pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Label Encode Target", command=self.label_encode_target, fg_color="#1E3A46").pack(pady=5)

        # One-Hot Encoding for Selected Column
        ctk.CTkLabel(self.sidebar, text="Select Column for One-Hot Encoding:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.one_hot_column_var = StringVar()
        self.one_hot_column_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.one_hot_column_var, values=self.get_categorical_columns(), state="readonly", width=200
        )
        self.one_hot_column_dropdown.pack(pady=5)
        ctk.CTkButton(self.sidebar, text="One-Hot Encode Selected Column", command=self.one_hot_encode_selected, fg_color="#1E3A46").pack(pady=5)

        # 4. Normalization Section
        ctk.CTkLabel(self.sidebar, text="Normalize Numeric Features:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.normalize_strategy = StringVar(value="StandardScaler")
        ctk.CTkComboBox(
            self.sidebar, variable=self.normalize_strategy, values=["StandardScaler", "MinMaxScaler", "None"], state="readonly", width=200
        ).pack(pady=5)
        ctk.CTkLabel(self.sidebar, text="Select Columns to Normalize:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.normalize_columns_frame = ctk.CTkScrollableFrame(self.sidebar, height=100, fg_color="#2b2b2b")
        self.normalize_columns_frame.pack(pady=5, fill="x")
        self.update_normalize_columns_checkboxes()
        ctk.CTkButton(self.sidebar, text="Apply Normalization", command=self.normalize_features, fg_color="#1E3A46").pack(pady=5)

        # 5. Outlier Handling Section
        ctk.CTkLabel(self.sidebar, text="Handle Outliers:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.outlier_strategy = StringVar(value="IQR")
        ctk.CTkComboBox(
            self.sidebar, variable=self.outlier_strategy, values=["IQR", "Z-Score", "Winsorize", "None"], state="readonly", width=200
        ).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Apply Outlier Handling", command=self.handle_outliers, fg_color="#1E3A46").pack(pady=5)

        # 6. Log Transform for Skewness Section
        ctk.CTkLabel(self.sidebar, text="Log Transform for Skewness:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        ctk.CTkLabel(self.sidebar, text="Select Numeric Column:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.log_transform_var = StringVar()
        self.log_transform_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.log_transform_var, values=self.get_numeric_columns(), state="readonly", width=200
        )
        self.log_transform_dropdown.pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Apply Log Transform", command=self.apply_log_transform, fg_color="#1E3A46").pack(pady=5)

        # 7. PCA Section
        ctk.CTkLabel(self.sidebar, text="PCA Dimensionality Reduction:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.pca_n_components_var = StringVar(value="2")
        ctk.CTkEntry(self.sidebar, textvariable=self.pca_n_components_var, placeholder_text="Number of components", width=200).pack(pady=5)
        ctk.CTkButton(self.sidebar, text="Apply PCA", command=self.apply_pca, fg_color="#1E3A46").pack(pady=5)

        # Navigation Buttons
        ctk.CTkButton(self.sidebar, text="Back to Summary", command=self.go_back, fg_color="#ff4d4d").pack(pady=(20, 5))
        ctk.CTkButton(self.sidebar, text="Next: Visualization", command=self.go_to_next, fg_color="green").pack(pady=5)

        # Main Area for Data Preview
        self.main_area = ctk.CTkFrame(body_frame, fg_color="#2b2b2b")
        self.main_area.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Data Preview
        self.preview_label = ctk.CTkLabel(self.main_area, text="Data Preview", font=("Arial", 16, "bold"))
        self.preview_label.pack(pady=(10, 5))
        self.preview_text = ctk.CTkTextbox(self.main_area, height=400, width=800, font=("Courier", 12), state="disabled")
        self.preview_text.pack(pady=5)
        ctk.CTkButton(self.main_area, text="Refresh Preview", command=self.show_data_summary, fg_color="#00b7eb").pack(pady=5)

        # Initial Data Preview
        self.show_data_summary()

    def toggle_simple_imputer_combobox(self, *args):
        if self.numeric_missing_strategy.get() == "Simple Imputer":
            self.simple_imputer_combobox.configure(state="readonly")
        else:
            self.simple_imputer_combobox.configure(state="disabled")

    def toggle_numeric_select_all(self):
        select_all = self.numeric_select_all_var.get()
        for var in self.numeric_impute_vars.values():
            var.set(select_all)
    
    def toggle_categorical_select_all(self):
        select_all = self.categorical_select_all_var.get()
        for var in self.categorical_impute_vars.values():
            var.set(select_all)

    def get_columns(self):
        return list(self.processed_df.columns) if self.processed_df is not None else []

    def get_numeric_columns(self):
        if self.processed_df is None:
            return []
        return list(self.processed_df.select_dtypes(include=[np.number]).columns)

    def get_categorical_columns(self):
        if self.processed_df is None:
            return []
        return list(self.processed_df.select_dtypes(include=['object']).columns)

    def update_numeric_impute_checkboxes(self):
        for widget in self.numeric_impute_frame.winfo_children():
            widget.destroy()
        self.numeric_impute_vars.clear()
        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            ctk.CTkLabel(self.numeric_impute_frame, text="No numeric columns available", font=("Arial", 11)).pack(anchor="w", padx=5, pady=2)
            return
        for col in numeric_cols:
            var = IntVar(value=0)
            self.numeric_impute_vars[col] = var
            ctk.CTkCheckBox(self.numeric_impute_frame, text=col, variable=var).pack(anchor="w", padx=5, pady=2)
        # Update "Select All" checkbox state based on current selections
        self.numeric_select_all_var.set(0)

    def update_categorical_impute_checkboxes(self):
        for widget in self.categorical_impute_frame.winfo_children():
            widget.destroy()
        self.categorical_impute_vars.clear()
        categorical_cols = self.get_categorical_columns()
        if not categorical_cols:
            ctk.CTkLabel(self.categorical_impute_frame, text="No categorical columns available", font=("Arial", 11)).pack(anchor="w", padx=5, pady=2)
            return
        for col in categorical_cols:
            var = IntVar(value=0)
            self.categorical_impute_vars[col] = var
            ctk.CTkCheckBox(self.categorical_impute_frame, text=col, variable=var).pack(anchor="w", padx=5, pady=2)
        # Update "Select All" checkbox state based on current selections
        self.categorical_select_all_var.set(0)

    def update_normalize_columns_checkboxes(self):
        for widget in self.normalize_columns_frame.winfo_children():
            widget.destroy()
        self.normalize_column_vars.clear()
        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            ctk.CTkLabel(self.normalize_columns_frame, text="No numeric columns available", font=("Arial", 11)).pack(anchor="w", padx=5, pady=2)
            return
        for col in numeric_cols:
            var = IntVar(value=1)
            self.normalize_column_vars[col] = var
            ctk.CTkCheckBox(self.normalize_columns_frame, text=col, variable=var).pack(anchor="w", padx=5, pady=2)

    def show_data_summary(self):
        self.preview_text.configure(state="normal")  # Temporarily enable to update text
        self.preview_text.delete("1.0", "end")
        if self.processed_df is None or self.processed_df.empty:
            self.preview_text.insert("1.0", "No data loaded.")
            self.preview_text.configure(state="disabled")
            return
        preview = "First 5 Rows:\n" + str(self.processed_df.head()) + "\n\n"
        preview += "Data Types:\n" + str(self.processed_df.dtypes)
        self.preview_text.insert("1.0", preview)
        self.preview_text.configure(state="disabled")
        self.label_encode_target_dropdown.configure(values=self.get_categorical_columns())
        self.one_hot_column_dropdown.configure(values=self.get_categorical_columns())
        self.log_transform_dropdown.configure(values=self.get_numeric_columns())
        self.update_numeric_impute_checkboxes()
        self.update_categorical_impute_checkboxes()
        self.update_normalize_columns_checkboxes()

    def impute_numeric_missing(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        strategy = self.numeric_missing_strategy.get()
        selected_cols = [col for col, var in self.numeric_impute_vars.items() if var.get()]
        if not selected_cols:
            messagebox.showerror("Error", "No numeric columns selected for imputation.")
            return
        try:
            if strategy == "Simple Imputer":
                sub_strategy = self.simple_imputer_strategy.get().lower()
                imputer = SimpleImputer(strategy=sub_strategy)
            elif strategy == "KNN":
                imputer = KNNImputer(n_neighbors=5)
            elif strategy == "Iterative":
                imputer = IterativeImputer(max_iter=10, random_state=0)
            self.processed_df[selected_cols] = imputer.fit_transform(self.processed_df[selected_cols])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Imputed missing numeric values using {strategy}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute numeric missing values: {str(e)}")

    def impute_categorical_missing(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        strategy = self.categorical_missing_strategy.get()
        selected_cols = [col for col, var in self.categorical_impute_vars.items() if var.get()]
        if not selected_cols:
            messagebox.showerror("Error", "No categorical columns selected for imputation.")
            return
        try:
            if strategy == "Mode":
                imputer = SimpleImputer(strategy="most_frequent")
            # elif strategy == "Constant":
            #     imputer = SimpleImputer(strategy="constant", fill_value="Missing")
            self.processed_df[selected_cols] = imputer.fit_transform(self.processed_df[selected_cols])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Imputed missing categorical values using {strategy}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute categorical missing values: {str(e)}")

    def drop_missing_rows(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        try:
            initial_rows = len(self.processed_df)
            strategy = self.drop_rows_strategy.get()
            if strategy == "All":
                self.processed_df = self.processed_df.dropna()
            elif strategy == "More than 50%":
                thresh = int(self.processed_df.shape[1] * 0.5)  # At least 50% non-NA values
                self.processed_df = self.processed_df.dropna(thresh=thresh)
            removed_rows = initial_rows - len(self.processed_df)
            self.show_data_summary()
            messagebox.showinfo("Success", f"Dropped {removed_rows} rows with missing values using {strategy} strategy.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to drop rows with missing values: {str(e)}")

    def check_duplicates(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        try:
            duplicate_count = len(self.processed_df[self.processed_df.duplicated()])
            messagebox.showinfo("Duplicates", f"Found {duplicate_count} duplicate rows.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check duplicates: {str(e)}")

    def remove_duplicates(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        try:
            initial_rows = len(self.processed_df)
            self.processed_df = self.processed_df.drop_duplicates()
            removed_rows = initial_rows - len(self.processed_df)
            self.show_data_summary()
            messagebox.showinfo("Success", f"Removed {removed_rows} duplicate rows.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove duplicates: {str(e)}")

    def label_encode_target(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        target = self.label_encode_target_var.get()
        if not target or target not in self.processed_df.columns:
            messagebox.showerror("Error", "Please select a target column.")
            return
        try:
            self.processed_df[target] = self.label_encoder.fit_transform(self.processed_df[target])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Label encoded target column: {target}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to label encode {target}: {str(e)}")

    def one_hot_encode_selected(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        selected_col = self.one_hot_column_var.get()
        if not selected_col or selected_col not in self.processed_df.columns:
            messagebox.showerror("Error", "Please select a column for one-hot encoding.")
            return
        try:
            encoded_data = self.one_hot_encoder.fit_transform(self.processed_df[[selected_col]])
            encoded_cols = self.one_hot_encoder.get_feature_names_out([selected_col])
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=self.processed_df.index)
            self.processed_df = pd.concat([self.processed_df.drop(selected_col, axis=1), encoded_df], axis=1)
            self.show_data_summary()
            messagebox.showinfo("Success", f"One-hot encoded column: {selected_col}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to one-hot encode {selected_col}: {str(e)}")

    def normalize_features(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        if self.processed_df.isna().any().any():
            messagebox.showwarning("Warning", "Dataset contains missing values. Handle them before normalization.")
            return
        strategy = self.normalize_strategy.get()
        if strategy == "None":
            messagebox.showinfo("Info", "No normalization applied.")
            return
        selected_cols = [col for col, var in self.normalize_column_vars.items() if var.get()]
        if not selected_cols:
            messagebox.showerror("Error", "No columns selected for normalization.")
            return
        try:
            if strategy == "StandardScaler":
                scaler = StandardScaler()
            else:  # MinMaxScaler
                scaler = MinMaxScaler()
            self.processed_df[selected_cols] = scaler.fit_transform(self.processed_df[selected_cols])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Normalized selected features using {strategy}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to normalize features: {str(e)}")

    def handle_outliers(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        if self.processed_df.isna().any().any():
            messagebox.showwarning("Warning", "Dataset contains missing values. Handle them before outlier detection.")
            return
        strategy = self.outlier_strategy.get()
        if strategy == "None":
            messagebox.showinfo("Info", "No outlier handling applied.")
            return
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        if numeric_cols.size == 0:
            messagebox.showinfo("Info", "No numeric features for outlier handling.")
            return
        try:
            if strategy == "IQR":
                Q1 = self.processed_df[numeric_cols].quantile(0.25)
                Q3 = self.processed_df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                self.processed_df = self.processed_df[~((self.processed_df[numeric_cols] < (Q1 - 1.5 * IQR)) | (self.processed_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
            elif strategy == "Z-Score":
                z_scores = np.abs((self.processed_df[numeric_cols] - self.processed_df[numeric_cols].mean()) / self.processed_df[numeric_cols].std())
                self.processed_df = self.processed_df[(z_scores < 3).all(axis=1)]
            elif strategy == "Winsorize":
                for col in numeric_cols:
                    self.processed_df[col] = winsorize(self.processed_df[col], limits=[0.05, 0.05])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Handled outliers using {strategy} method.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle outliers: {str(e)}")

    def apply_log_transform(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        if self.processed_df.isna().any().any():
            messagebox.showwarning("Warning", "Dataset contains missing values. Handle them before log transformation.")
            return
        selected_col = self.log_transform_var.get()
        if not selected_col or selected_col not in self.processed_df.columns:
            messagebox.showerror("Error", "Please select a numeric column for log transformation.")
            return
        try:
            if not pd.api.types.is_numeric_dtype(self.processed_df[selected_col]):
                messagebox.showerror("Error", "Log transform requires a numeric column.")
                return
            if (self.processed_df[selected_col] <= 0).any():
                messagebox.showerror("Error", "Log transform requires all values to be positive. Consider shifting the data.")
                return
            self.processed_df[f"{selected_col}_log"] = np.log(self.processed_df[selected_col])
            self.show_data_summary()
            messagebox.showinfo("Success", f"Applied log transform to {selected_col}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply log transform: {str(e)}")

    def apply_pca(self):
        if self.processed_df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        if self.processed_df.isna().any().any():
            messagebox.showwarning("Warning", "Dataset contains missing values. Handle them before PCA.")
            return
        numeric_cols = self.get_numeric_columns()
        if not numeric_cols:
            messagebox.showinfo("Info", "No numeric features for PCA.")
            return
        try:
            n_components = int(self.pca_n_components_var.get())
            if n_components <= 0 or n_components > len(numeric_cols):
                messagebox.showerror("Error", f"Number of components must be between 1 and {len(numeric_cols)}.")
                return
            pca = PCA(n_components=n_components)
            X = self.processed_df[numeric_cols]
            pca_result = pca.fit_transform(X)
            pca_cols = [f"PC{i+1}" for i in range(n_components)]
            pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=self.processed_df.index)
            self.processed_df = pd.concat([self.processed_df.drop(numeric_cols, axis=1), pca_df], axis=1)
            self.show_data_summary()
            messagebox.showinfo("Success", f"Applied PCA with {n_components} components.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of components: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply PCA: {str(e)}")

    def go_back(self):
        if self.on_back_callback:
            self.on_back_callback(self.processed_df)
        self.destroy()

    def go_to_next(self):
        if self.processed_df is None or self.processed_df.empty:
            messagebox.showerror("Error", "No dataset to process.")
            return
        if self.on_next_callback:
            self.on_next_callback(self.processed_df)
        self.destroy()

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    app = PrePage(data=df)
    app.mainloop()