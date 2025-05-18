import customtkinter as ctk
from tkinter import messagebox, StringVar
import pandas as pd
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DatasetSummaryApp(ctk.CTk):
    def __init__(self, data=None, on_next_callback=None):
        super().__init__()

        self.title("ML AlgoHub - Dataset Summary")
        self.geometry("1200x750+250+50")
        self.configure(fg_color="#2b2b2b")  # Dark gray background
        self.data = data
        self.on_next_callback = on_next_callback
        self.preview_rows = 5  # Default number of rows to display
        self.preview_rows_var = StringVar(value="5")

        # Debugging: Check if data is loaded properly
        print("Data loaded into DatasetSummaryApp:", self.data)

        self.create_gui()

    def create_gui(self):
        # Top Header Frame
        top_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60)
        top_frame.pack(side="top", fill="x")
        ctk.CTkLabel(top_frame, text="Dataset Summary", font=("Arial", 28, "bold"), text_color="#00b7eb").pack(pady=10)

        # Main Body Frame with Vertical Scroll
        body_frame = ctk.CTkScrollableFrame(self, fg_color="#2b2b2b")
        body_frame.pack(fill="both", expand=True)

        # Main Area
        self.main_area = ctk.CTkFrame(body_frame, fg_color="#2b2b2b")
        self.main_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Display Summary
        self.display_summary()

        # Bottom Buttons Frame
        buttons_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        buttons_frame.pack(side="bottom", fill="x", pady=10)
        ctk.CTkButton(buttons_frame, text="Refresh Summary", command=self.display_summary, fg_color="#00b7eb").pack(side="left", padx=10)
        ctk.CTkButton(buttons_frame, text="Next: Preprocessing", command=self.go_to_next, fg_color="green").pack(side="right", padx=10)

    def create_table(self, frame, data, header_color="#3a4971", row_colors=("#2b2b2b", "#1e1e1e")):
        # Debugging: Check the data being passed to the table
        print("Data passed to create_table:", data)

        if data is None or data.empty:
            ctk.CTkLabel(frame, text="No data available to display in the table.", font=("Arial", 12), text_color="white").pack(pady=5)
            return

        # Use scrollable frame for horizontal scrolling
        table_frame = ctk.CTkScrollableFrame(frame, fg_color="#2b2b2b", orientation="horizontal")
        table_frame.pack(fill="both", expand=True, pady=5, padx=5)

        # Headers
        for col_idx, col_name in enumerate(data.columns):
            header_label = ctk.CTkLabel(
                table_frame, text=str(col_name), font=("Arial", 12, "bold"), fg_color=header_color,
                text_color="white", width=150, height=30, corner_radius=5, anchor="center", wraplength=150
            )
            header_label.grid(row=0, column=col_idx, padx=1, pady=1, sticky="nsew")

        # Rows
        for row_idx in range(len(data)):
            for col_idx, value in enumerate(data.iloc[row_idx]):
                cell_label = ctk.CTkLabel(
                    table_frame, text=str(value), font=("Arial", 11), fg_color=row_colors[row_idx % 2],
                    text_color="white", width=150, height=25, corner_radius=3, anchor="w", wraplength=150
                )
                cell_label.grid(row=row_idx + 1, column=col_idx, padx=1, pady=1, sticky="nsew")

        # Configure grid to allow resizing and scrolling
        for i in range(len(data.columns)):
            table_frame.grid_columnconfigure(i, weight=1, uniform="col")
        for i in range(len(data) + 1):
            table_frame.grid_rowconfigure(i, weight=1)

    def display_summary(self):
        # Clear previous content
        for widget in self.main_area.winfo_children():
            widget.destroy()

        if self.data is None or self.data.empty:
            ctk.CTkLabel(self.main_area, text="No dataset loaded.", font=("Arial", 16), text_color="white").pack(pady=10)
            return

        # Section 1: Dataset Preview with Table
        preview_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        preview_frame.pack(fill="both", expand=True, pady=(10, 5), padx=10)
        ctk.CTkLabel(
            preview_frame, text="Dataset Preview", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        # Row Selection Dropdown
        row_select_frame = ctk.CTkFrame(preview_frame, fg_color="#2b2b2b")
        row_select_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(row_select_frame, text="Show Rows:", font=("Arial", 12), text_color="white").pack(side="left", padx=5)
        ctk.CTkComboBox(
            row_select_frame, variable=self.preview_rows_var, values=["5", "10", "20"],
            state="readonly", width=80, command=self.update_preview_rows
        ).pack(side="left", padx=5)
        # Display Table
        preview_data = self.data.head(int(self.preview_rows_var.get()))
        # Debugging: Check the preview data
        print("Preview Data:", preview_data)
        self.create_table(preview_frame, preview_data)

        # Separator
        ctk.CTkFrame(self.main_area, height=2, fg_color="#00b7eb").pack(fill="x", pady=5)

        # Section 2: Data Types
        dtype_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        dtype_frame.pack(fill="x", pady=(10, 5), padx=10)
        ctk.CTkLabel(
            dtype_frame, text="Data Types", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        dtype_text = ctk.CTkTextbox(dtype_frame, height=80, width=800, font=("Arial", 12), fg_color="#1e1e1e", text_color="white")
        dtype_text.pack(pady=5, fill="x")
        dtypes = self.data.dtypes.astype(str).to_string()
        dtype_text.insert("1.0", dtypes)
        if self.data.dtypes.nunique() > 1:
            ctk.CTkLabel(
                dtype_frame, text="*Note: Dataset contains mixed data types.", font=("Arial", 12),
                text_color="#ff4d4d", fg_color="#2b2b2b"
            ).pack(pady=5)

        # Separator
        ctk.CTkFrame(self.main_area, height=2, fg_color="#00b7eb").pack(fill="x", pady=5)

        # Section 3: Basic Statistics
        stats_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        stats_frame.pack(fill="x", pady=(10, 5), padx=10)
        ctk.CTkLabel(
            stats_frame, text="Basic Statistics", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        stats_text = ctk.CTkTextbox(stats_frame, height=120, width=800, font=("Arial", 12), fg_color="#1e1e1e", text_color="white")
        stats_text.pack(pady=5, fill="x")
        numeric_stats = self.data.describe().to_string()
        stats_text.insert("1.0", numeric_stats)
        ctk.CTkLabel(
            stats_frame, text="*Note: Statistics apply only to numeric columns.", font=("Arial", 12),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=5)

        # Separator
        ctk.CTkFrame(self.main_area, height=2, fg_color="#00b7eb").pack(fill="x", pady=5)

        # Section 4: Missing Values
        missing_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        missing_frame.pack(fill="x", pady=(10, 5), padx=10)
        ctk.CTkLabel(
            missing_frame, text="Missing Values", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        missing_text = ctk.CTkTextbox(missing_frame, height=80, width=800, font=("Arial", 12), fg_color="#1e1e1e", text_color="white")
        missing_text.pack(pady=5, fill="x")
        missing = self.data.isnull().sum()
        missing_percent = (self.data.isnull().sum() / len(self.data) * 100).round(2)
        missing_summary = pd.DataFrame({"Count": missing, "Percentage (%)": missing_percent})
        missing_text.insert("1.0", missing_summary.to_string())
        if missing.sum() > 0:
            ctk.CTkLabel(
                missing_frame, text="*Warning: Dataset contains missing values. Consider handling them in preprocessing.",
                font=("Arial", 12), text_color="#ff4d4d", fg_color="#2b2b2b"
            ).pack(pady=5)

        # Separator
        ctk.CTkFrame(self.main_area, height=2, fg_color="#00b7eb").pack(fill="x", pady=5)

        # Section 5: Value Distribution
        dist_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        dist_frame.pack(fill="x", pady=(10, 5), padx=10)
        ctk.CTkLabel(
            dist_frame, text="Value Distribution", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        dist_text = ctk.CTkTextbox(dist_frame, height=80, width=800, font=("Arial", 12), fg_color="#1e1e1e", text_color="white")
        dist_text.pack(pady=5, fill="x")
        unique_counts = self.data.nunique()
        value_dist = pd.DataFrame({
            "Unique Values": unique_counts,
            "Most Frequent (%)": [self.data[col].value_counts(normalize=True).iloc[0] * 100 for col in self.data.columns]
        })
        dist_text.insert("1.0", value_dist.to_string())
        if any(unique_counts == 1):
            ctk.CTkLabel(
                dist_frame, text="*Warning: Some columns have only one unique value (potential constant feature).",
                font=("Arial", 12), text_color="#ff4d4d", fg_color="#2b2b2b"
            ).pack(pady=5)

        # Separator
        ctk.CTkFrame(self.main_area, height=2, fg_color="#00b7eb").pack(fill="x", pady=5)

        # Section 6: Potential Outliers
        outlier_frame = ctk.CTkFrame(self.main_area, fg_color="#2b2b2b")
        outlier_frame.pack(fill="x", pady=(10, 5), padx=10)
        ctk.CTkLabel(
            outlier_frame, text="Potential Outliers", font=("Arial", 16, "bold"),
            text_color="#00b7eb", fg_color="#2b2b2b"
        ).pack(pady=(5, 5))
        outlier_text = ctk.CTkTextbox(outlier_frame, height=80, width=800, font=("Arial", 12), fg_color="#1e1e1e", text_color="white")
        outlier_text.pack(pady=5, fill="x")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if numeric_cols.size > 0:
            Q1 = self.data[numeric_cols].quantile(0.25)
            Q3 = self.data[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.data[numeric_cols] < lower_bound) | (self.data[numeric_cols] > upper_bound)).sum()
            outlier_summary = pd.DataFrame({"Outlier Count": outliers})
            outlier_text.insert("1.0", outlier_summary.to_string())
            if outliers.sum() > 0:
                ctk.CTkLabel(
                    outlier_frame, text="*Warning: Potential outliers detected. Consider handling them in preprocessing.",
                    font=("Arial", 12), text_color="#ff4d4d", fg_color="#2b2b2b"
                ).pack(pady=5)
        else:
            outlier_text.insert("1.0", "No numeric columns for outlier detection.")

    def update_preview_rows(self, *args):
        self.preview_rows = int(self.preview_rows_var.get())
        self.display_summary()

    def go_to_next(self):
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No dataset to process.")
            return
        if self.on_next_callback:
            self.on_next_callback(self.data)
        self.destroy()

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    app = DatasetSummaryApp(data=df)
    app.mainloop()