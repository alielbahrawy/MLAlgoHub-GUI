import customtkinter as ctk
from tkinter import messagebox, StringVar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VisualizationPage(ctk.CTk):
    def __init__(self, data=None, on_next_callback=None, on_back_callback=None):
        super().__init__()

        self.title("ML AlgoHub - Data Visualization")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.data = data
        self.on_next_callback = on_next_callback
        self.on_back_callback = on_back_callback
        self.plots = []  # To store plot canvases for clearing

        self.create_gui()

    def create_gui(self):
        # Top Header Frame
        top_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60)
        top_frame.pack(side="top", fill="x")
        ctk.CTkLabel(top_frame, text="Data Visualization", font=("Arial", 28, "bold"), text_color="#00b7eb").pack(pady=10)

        # Main Body Frame
        body_frame = ctk.CTkFrame(self)
        body_frame.pack(fill="both", expand=True)

        # Sidebar with Scrollable Frame
        sidebar_container = ctk.CTkFrame(body_frame, width=300)
        sidebar_container.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar = ctk.CTkScrollableFrame(sidebar_container, width=300, height=600, fg_color="#1a1a1a")
        self.sidebar.pack(fill="both", expand=True)

        # Visualization Options
        ctk.CTkLabel(self.sidebar, text="Select Visualization Type:", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        self.plot_type_var = StringVar(value="Histogram")
        self.plot_type_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.plot_type_var, values=["Histogram", "Scatter Plot", "Box Plot", "Bar Plot", "Count Plot"],
            state="readonly", width=200, command=self.update_column_options
        )
        self.plot_type_dropdown.pack(pady=5)

        # Column Selection for X-axis
        ctk.CTkLabel(self.sidebar, text="Select X Column:", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.x_column_var = StringVar()
        self.x_column_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.x_column_var, values=self.get_columns(), state="readonly", width=200
        )
        self.x_column_dropdown.pack(pady=5)

        # Column Selection for Y-axis (used in Scatter Plot, Box Plot, etc.)
        ctk.CTkLabel(self.sidebar, text="Select Y Column (Optional):", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.y_column_var = StringVar()
        self.y_column_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.y_column_var, values=["None"] + self.get_columns(), state="readonly", width=200
        )
        self.y_column_dropdown.pack(pady=5)
        self.y_column_var.set("None")

        # Hue Selection for Grouping (used in Scatter Plot, Box Plot, etc.)
        ctk.CTkLabel(self.sidebar, text="Select Hue (Optional):", font=("Arial", 11)).pack(pady=(5, 2), anchor="w")
        self.hue_var = StringVar()
        self.hue_dropdown = ctk.CTkComboBox(
            self.sidebar, variable=self.hue_var, values=["None"] + self.get_columns(), state="readonly", width=200
        )
        self.hue_dropdown.pack(pady=5)
        self.hue_var.set("None")

        # Plot Button
        ctk.CTkButton(self.sidebar, text="Generate Plot", command=self.generate_plot, fg_color="#1E3A46").pack(pady=10)

        # Navigation Buttons
        ctk.CTkButton(self.sidebar, text="Back to Preprocessing", command=self.go_back, fg_color="#ff4d4d").pack(pady=(20, 5))
        ctk.CTkButton(self.sidebar, text="Next: Models", command=self.go_to_models, fg_color="green").pack(pady=5)

        # Main Area for Plots
        self.plot_area = ctk.CTkScrollableFrame(body_frame, fg_color="#2b2b2b")
        self.plot_area.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Initial Message
        self.plot_label = ctk.CTkLabel(self.plot_area, text="Select options and generate a plot", font=("Arial", 16, "bold"))
        self.plot_label.pack(pady=(10, 5))

        # Initial update of column options
        self.update_column_options(None)

    def get_columns(self):
        return list(self.data.columns) if self.data is not None else []

    def update_column_options(self, _):
        plot_type = self.plot_type_var.get()
        self.x_column_dropdown.configure(values=self.get_columns())
        self.y_column_dropdown.configure(values=["None"] + self.get_columns())
        self.hue_dropdown.configure(values=["None"] + self.get_columns())
        self.y_column_var.set("None")
        self.hue_var.set("None")
        # Keep Y and Hue dropdowns enabled for all plot types, including Histogram
        self.y_column_dropdown.configure(state="readonly")
        self.hue_dropdown.configure(state="readonly")

    def generate_plot(self):
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        plot_type = self.plot_type_var.get()
        x_col = self.x_column_var.get()
        y_col = self.y_column_var.get() if self.y_column_var.get() != "None" else None
        hue = self.hue_var.get() if self.hue_var.get() != "None" else None

        if not x_col:
            messagebox.showerror("Error", "Please select an X column.")
            return

        # Clear previous plots
        for plot in self.plots:
            plot.get_tk_widget().destroy()
        self.plots.clear()
        self.plot_label.destroy()

        # Create a new figure
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background')

        try:
            if plot_type == "Histogram":
                # Validate X and Y (if provided) are numeric
                if not pd.api.types.is_numeric_dtype(self.data[x_col]):
                    messagebox.showerror("Error", "Histogram X column must be numeric.")
                    plt.close(fig)
                    return
                if y_col and not pd.api.types.is_numeric_dtype(self.data[y_col]):
                    messagebox.showerror("Error", "Histogram Y column must be numeric.")
                    plt.close(fig)
                    return
                # Plot histogram with optional Y and Hue
                sns.histplot(data=self.data, x=x_col, y=y_col, hue=hue, ax=ax, palette="deep")
                title = f"Histogram of {x_col}"
                if y_col:
                    title += f" vs {y_col}"
                if hue:
                    title += f" (Hue: {hue})"
                ax.set_title(title, color="#00b7eb")

            elif plot_type == "Scatter Plot":
                if y_col is None:
                    messagebox.showerror("Error", "Scatter Plot requires a Y column.")
                    plt.close(fig)
                    return
                if not (pd.api.types.is_numeric_dtype(self.data[x_col]) and pd.api.types.is_numeric_dtype(self.data[y_col])):
                    messagebox.showerror("Error", "Scatter Plot requires numeric X and Y columns.")
                    plt.close(fig)
                    return
                sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue, ax=ax, palette="deep")
                ax.set_title(f"Scatter Plot of {x_col} vs {y_col}", color="#00b7eb")

            elif plot_type == "Box Plot":
                if y_col and pd.api.types.is_numeric_dtype(self.data[y_col]):
                    sns.boxplot(data=self.data, x=x_col, y=y_col, hue=hue, ax=ax, palette="deep")
                    ax.set_title(f"Box Plot of {x_col} vs {y_col}", color="#00b7eb")
                else:
                    sns.boxplot(data=self.data, x=x_col, ax=ax, palette="deep")
                    ax.set_title(f"Box Plot of {x_col}", color="#00b7eb")

            elif plot_type == "Bar Plot":
                if y_col and pd.api.types.is_numeric_dtype(self.data[y_col]):
                    sns.barplot(data=self.data, x=x_col, y=y_col, hue=hue, ax=ax, palette="deep")
                    ax.set_title(f"Bar Plot of {x_col} vs {y_col}", color="#00b7eb")
                else:
                    messagebox.showerror("Error", "Bar Plot requires a numeric Y column.")
                    plt.close(fig)
                    return

            elif plot_type == "Count Plot":
                sns.countplot(data=self.data, x=x_col, hue=hue, ax=ax, palette="deep")
                ax.set_title(f"Count Plot of {x_col}", color="#00b7eb")

            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10)
            self.plots.append(canvas)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {str(e)}")
            plt.close(fig)

    def go_back(self):
        if self.on_back_callback:
            self.on_back_callback(self.data)
        self.destroy()

    def go_to_models(self):
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No dataset to process.")
            return
        if self.on_next_callback:
            self.on_next_callback(self.data)
        self.destroy()

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    app = VisualizationPage(data=df)
    app.mainloop()