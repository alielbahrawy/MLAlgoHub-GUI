# import customtkinter as ctk
# from tkinter import messagebox
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from ydata_profiling import ProfileReport
# import seaborn as sns

# class VisualizationPage(ctk.CTk):
#     def __init__(self, data, on_next_callback=None):
#         super().__init__()
#         self.title("Dataset Visualization")
#         self.geometry('1024x720+250+50')
#         self.configure(fg_color="#2b2b2b")  # Dark Mode background
#         self.df = data
#         self.chart_frames = []
#         self.chart_canvases = []
#         self.on_next_callback = on_next_callback

#         self.create_header_frame()
#         self.create_main_frame()
#         self.display_data()  # Display charts and insights automatically

#     def create_header_frame(self):
#         self.header_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=50)
#         self.header_frame.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)
#         self.header_frame.pack_propagate(False)

#         # Placeholder for Logo
#         self.logo_label = ctk.CTkLabel(
#             self.header_frame,
#             text="ML ALGOHUB",
#             font=("Arial", 16, "bold"),
#             text_color="#00b7eb",
#             fg_color="#1a1a1a"
#         )
#         self.logo_label.pack(side=ctk.LEFT, padx=10)

#         # Title
#         title_label = ctk.CTkLabel(
#             self.header_frame,
#             text="Dataset Visualization",
#             font=("Arial", 24, "bold"),
#             text_color="#00b7eb",
#             fg_color="#1a1a1a"
#         )
#         title_label.pack(expand=True)

#     def create_main_frame(self):
#         self.main_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
#         self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
#         self.main_frame.pack_propagate(False)

#         # Split into left and right frames
#         self.eda_left_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=10, width=int(1200 * 0.65))
#         self.eda_left_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=False, padx=5, pady=5)
#         self.eda_left_frame.pack_propagate(False)

#         self.eda_right_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=10, width=int(1200 * 0.35))
#         self.eda_right_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=False, padx=5, pady=5)
#         self.eda_right_frame.pack_propagate(False)

#         # Left frame: Charts
#         charts_label = ctk.CTkLabel(
#             self.eda_left_frame,
#             text="Important Charts: Histogram, Heatmap, Bar Chart, Pair plot, else",
#             font=("Arial", 14, "bold"),
#             text_color="#ffffff"
#         )
#         charts_label.pack(anchor="w", padx=10, pady=5)

#         # Create a grid for chart placeholders (2 rows, 3 columns, but we use 5 slots)
#         self.chart_grid_frame = ctk.CTkFrame(self.eda_left_frame, fg_color="#333333")
#         self.chart_grid_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

#         self.chart_frames = []
#         for i in range(5):
#             chart_frame = ctk.CTkFrame(
#                 self.chart_grid_frame,
#                 fg_color="#444444",
#                 corner_radius=10,
#                 height=150
#             )
#             row = i // 3
#             col = i % 3
#             chart_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
#             chart_frame.pack_propagate(False)

#             placeholder_label = ctk.CTkLabel(chart_frame, text="Image here", font=("Arial", 12), fg_color="#444444", text_color="#aaaaaa")
#             placeholder_label.pack(expand=True)

#             self.chart_frames.append(chart_frame)

#         self.chart_grid_frame.grid_rowconfigure(0, weight=1)
#         self.chart_grid_frame.grid_rowconfigure(1, weight=1)
#         self.chart_grid_frame.grid_columnconfigure(0, weight=1)
#         self.chart_grid_frame.grid_columnconfigure(1, weight=1)
#         self.chart_grid_frame.grid_columnconfigure(2, weight=1)

#         # Right frame: Insights and Dropdown
#         insights_label = ctk.CTkLabel(
#             self.eda_right_frame,
#             text="Insights text "
#             "\n(correlations ... " \
#             "\nalert from ydata Profiling)",
#             font=("Arial", 14, "bold"),
#             text_color="#ffffff"
#         )
#         insights_label.pack(anchor="w", padx=10, pady=5)

#         self.insights_text = ctk.CTkTextbox(self.eda_right_frame, height=300, wrap="word", fg_color="#444444", corner_radius=10, text_color="#ffffff")
#         self.insights_text.pack(fill="x", padx=10, pady=5)
#         self.insights_text.insert("end", "Insights will be displayed here after loading the dataset.")

#         # Dropdown for chart generation
#         dropdown_frame = ctk.CTkFrame(self.eda_right_frame, fg_color="#333333")
#         dropdown_frame.pack(fill="x", padx=10, pady=5)

#         dropdown_label = ctk.CTkLabel(dropdown_frame, text="Dropdown to", font=("Arial", 12), text_color="#ffffff")
#         dropdown_label.pack(side=ctk.LEFT, padx=(0, 5))

#         self.chart_dropdown_var = ctk.StringVar()
#         self.chart_dropdown = ctk.CTkComboBox(
#             dropdown_frame,
#             values=["Make bar chart relation between two columns", "Make histogram for one column"],
#             variable=self.chart_dropdown_var,
#             state="readonly",
#             corner_radius=10,
#             width=250,
#             text_color="#ffffff",
#             fg_color="#444444",
#             button_color="#555555"
#         )
#         self.chart_dropdown.pack(side=ctk.LEFT, pady=5)

#         # Button to generate chart based on dropdown selection
#         self.generate_chart_button = ctk.CTkButton(
#             dropdown_frame,
#             text="Generate Chart",
#             command=self.generate_chart,
#             fg_color="#00b7eb",
#             text_color="#ffffff",
#             font=("Arial", 12),
#             corner_radius=10
#         )
#         self.generate_chart_button.pack(side=ctk.LEFT, padx=5)

#         # Button to go to the Preprocessing Page
#         next_button = ctk.CTkButton(
#             self.eda_right_frame,
#             text="Next: Preprocessing",
#             command=self.go_to_preprocessing,
#             fg_color="#00b7eb",
#             text_color="#ffffff",
#             font=("Arial", 14, "bold"),
#             corner_radius=10
#         )
#         next_button.pack(anchor="w", padx=10, pady=10)

#     def display_data(self):
#         for canvas in self.chart_canvases:
#             canvas.get_tk_widget().destroy()
#         self.chart_canvases = []

#         if self.df is not None:
#             profile = ProfileReport(self.df, explorative=True, minimal=True)
#             insights = profile.get_description()
#             self.insights_text.delete("1.0", "end")
#             self.insights_text.insert("end", str(insights))

#             self.generate_default_charts()

#     def generate_default_charts(self):
#         num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
#         if len(num_cols) > 0:
#             fig, ax = plt.subplots(figsize=(3, 2))
#             self.df[num_cols[0]].hist(ax=ax, color="#00b7eb")
#             ax.set_title(f"Histogram of {num_cols[0]}", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[0])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#         if len(num_cols) > 1:
#             fig, ax = plt.subplots(figsize=(3, 2))
#             sns.heatmap(self.df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
#             ax.set_title("Correlation Heatmap", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[1])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#         cat_cols = self.df.select_dtypes(include=['object']).columns
#         if len(cat_cols) > 0 and len(num_cols) > 0:
#             fig, ax = plt.subplots(figsize=(3, 2))
#             self.df.groupby(cat_cols[0])[num_cols[0]].mean().plot(kind="bar", ax=ax, color="#00b7eb")
#             ax.set_title(f"Bar Chart: {cat_cols[0]} vs {num_cols[0]}", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[2])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#         if len(num_cols) >= 2:
#             fig, ax = plt.subplots(figsize=(3, 2))
#             ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], color="#00b7eb")
#             ax.set_xlabel(num_cols[0], color="#ffffff")
#             ax.set_ylabel(num_cols[1], color="#ffffff")
#             ax.set_title("Pair Plot", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[3])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#     def generate_chart(self):
#         if self.df is None:
#             messagebox.showwarning("Warning", "No dataset available.")
#             return

#         chart_type = self.chart_dropdown_var.get()
#         if chart_type == "Make bar chart relation between two columns":
#             cat_cols = self.df.select_dtypes(include=['object']).columns
#             num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
#             if len(cat_cols) < 1 or len(num_cols) < 1:
#                 messagebox.showerror("Error", "Dataset must have at least one categorical and one numerical column.")
#                 return

#             col1, col2 = cat_cols[0], num_cols[0]
#             fig, ax = plt.subplots(figsize=(3, 2))
#             self.df.groupby(col1)[col2].mean().plot(kind="bar", ax=ax, color="#00b7eb")
#             ax.set_title(f"Bar Chart: {col1} vs {col2}", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[4])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#         elif chart_type == "Make histogram for one column":
#             num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
#             if len(num_cols) < 1:
#                 messagebox.showerror("Error", "No numerical columns available for histogram.")
#                 return

#             col = num_cols[0]
#             fig, ax = plt.subplots(figsize=(3, 2))
#             self.df[col].hist(ax=ax, color="#00b7eb")
#             ax.set_title(f"Histogram of {col}", color="#ffffff")
#             ax.tick_params(colors="#ffffff")
#             ax.set_facecolor("#333333")
#             fig.set_facecolor("#444444")
#             canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[4])
#             canvas.draw()
#             canvas.get_tk_widget().pack(fill="both", expand=True)
#             self.chart_canvases.append(canvas)

#     def go_to_preprocessing(self):
#         if self.on_next_callback:
#             self.destroy()
#             self.on_next_callback(self.df)



import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ydata_profiling import ProfileReport
import seaborn as sns
import os
import sys
from PIL import Image as PILImage  # Renamed to avoid naming conflicts

class VisualizationPage(ctk.CTk):
    def __init__(self, data, on_next_callback=None):
        super().__init__()
        self.title("Dataset Visualization")
        self.geometry('1024x720+250+50')
        self.configure(fg_color="#2b2b2b")  # Dark Mode background
        self.df = data
        self.chart_frames = []
        self.chart_canvases = []
        self.on_next_callback = on_next_callback

        self.create_header_frame()
        self.create_main_frame()
        self.display_data()  # Display charts and insights automatically

    def get_resource_path(self, relative_path):
        """Get the absolute path to a resource, works for dev and PyInstaller."""
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def create_header_frame(self):
        self.header_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", height=50)
        self.header_frame.pack(side=ctk.TOP, fill=ctk.X, padx=5, pady=5)
        self.header_frame.pack_propagate(False)

        # Logo
        try:
            logo_path = self.get_resource_path("Resources/LogoIcon.png")
            img_r = PILImage.open(logo_path)
            img_1 = ctk.CTkImage(img_r, size=(170, 70))  # Adjusted size to fit header
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                text="",  # No text, only image
                image=img_1,
                fg_color="#1a1a1a"
            )
            self.logo_label.pack(side=ctk.LEFT, padx=10)
        except Exception as e:
            # Fallback to text if logo fails to load
            self.logo_label = ctk.CTkLabel(
                self.header_frame,
                text="ML ALGOHUB",
                font=("Arial", 16, "bold"),
                text_color="#00b7eb",
                fg_color="#1a1a1a"
            )
            self.logo_label.pack(side=ctk.LEFT, padx=10)
            print(f"Failed to load logo: {e}")

        # Title
        title_label = ctk.CTkLabel(
            self.header_frame,
            text="Dataset Visualization",
            font=("Arial", 24, "bold"),
            text_color="#00b7eb",
            fg_color="#1a1a1a"
        )
        title_label.pack(expand=True)

    def create_main_frame(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
        self.main_frame.pack_propagate(False)

        # Split into left and right frames
        self.eda_left_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=10, width=int(1200 * 0.65))
        self.eda_left_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=False, padx=5, pady=5)
        self.eda_left_frame.pack_propagate(False)

        self.eda_right_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=10, width=int(1200 * 0.35))
        self.eda_right_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=False, padx=5, pady=5)
        self.eda_right_frame.pack_propagate(False)

        # Left frame: Charts
        charts_label = ctk.CTkLabel(
            self.eda_left_frame,
            text="Important Charts: Histogram, Heatmap, Bar Chart, Pair plot, else",
            font=("Arial", 14, "bold"),
            text_color="#ffffff"
        )
        charts_label.pack(anchor="w", padx=10, pady=5)

        # Create a grid for chart placeholders (2 rows, 3 columns, but we use 5 slots)
        self.chart_grid_frame = ctk.CTkFrame(self.eda_left_frame, fg_color="#333333")
        self.chart_grid_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        self.chart_frames = []
        for i in range(5):
            chart_frame = ctk.CTkFrame(
                self.chart_grid_frame,
                fg_color="#444444",
                corner_radius=10,
                height=150
            )
            row = i // 3
            col = i % 3
            chart_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            chart_frame.pack_propagate(False)

            placeholder_label = ctk.CTkLabel(chart_frame, text="Image here", font=("Arial", 12), fg_color="#444444", text_color="#aaaaaa")
            placeholder_label.pack(expand=True)

            self.chart_frames.append(chart_frame)

        self.chart_grid_frame.grid_rowconfigure(0, weight=1)
        self.chart_grid_frame.grid_rowconfigure(1, weight=1)
        self.chart_grid_frame.grid_columnconfigure(0, weight=1)
        self.chart_grid_frame.grid_columnconfigure(1, weight=1)
        self.chart_grid_frame.grid_columnconfigure(2, weight=1)

        # Right frame: Insights and Dropdown
        insights_label = ctk.CTkLabel(
            self.eda_right_frame,
            text="Insights text "
            "\n(correlations ... " \
            "\nalert from ydata Profiling)",
            font=("Arial", 14, "bold"),
            text_color="#ffffff"
        )
        insights_label.pack(anchor="w", padx=10, pady=5)

        self.insights_text = ctk.CTkTextbox(self.eda_right_frame, height=300, wrap="word", fg_color="#444444", corner_radius=10, text_color="#ffffff")
        self.insights_text.pack(fill="x", padx=10, pady=5)
        self.insights_text.insert("end", "Insights will be displayed here after loading the dataset.")

        # Dropdown for chart generation
        dropdown_frame = ctk.CTkFrame(self.eda_right_frame, fg_color="#333333")
        dropdown_frame.pack(fill="x", padx=10, pady=5)

        dropdown_label = ctk.CTkLabel(dropdown_frame, text="Dropdown to", font=("Arial", 12), text_color="#ffffff")
        dropdown_label.pack(side=ctk.LEFT, padx=(0, 5))

        self.chart_dropdown_var = ctk.StringVar()
        self.chart_dropdown = ctk.CTkComboBox(
            dropdown_frame,
            values=["Make bar chart relation between two columns", "Make histogram for one column"],
            variable=self.chart_dropdown_var,
            state="readonly",
            corner_radius=10,
            width=250,
            text_color="#ffffff",
            fg_color="#444444",
            button_color="#555555"
        )
        self.chart_dropdown.pack(side=ctk.LEFT, pady=5)

        # Button to generate chart based on dropdown selection
        self.generate_chart_button = ctk.CTkButton(
            dropdown_frame,
            text="Generate Chart",
            command=self.generate_chart,
            fg_color="#00b7eb",
            text_color="#ffffff",
            font=("Arial", 12),
            corner_radius=10
        )
        self.generate_chart_button.pack(side=ctk.LEFT, padx=5)

        # Button to go to the Preprocessing Page
        next_button = ctk.CTkButton(
            self.eda_right_frame,
            text="Next: Preprocessing",
            command=self.go_to_preprocessing,
            fg_color="#00b7eb",
            text_color="#ffffff",
            font=("Arial", 14, "bold"),
            corner_radius=10
        )
        next_button.pack(anchor="w", padx=10, pady=10)

    def display_data(self):
        for canvas in self.chart_canvases:
            canvas.get_tk_widget().destroy()
        self.chart_canvases = []

        if self.df is not None:
            profile = ProfileReport(self.df, explorative=True, minimal=True)
            insights = profile.get_description()
            self.insights_text.delete("1.0", "end")
            self.insights_text.insert("end", str(insights))

            self.generate_default_charts()

    def generate_default_charts(self):
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            fig, ax = plt.subplots(figsize=(3, 2))
            self.df[num_cols[0]].hist(ax=ax, color="#00b7eb")
            ax.set_title(f"Histogram of {num_cols[0]}", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[0])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.heatmap(self.df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[1])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0 and len(num_cols) > 0:
            fig, ax = plt.subplots(figsize=(3, 2))
            self.df.groupby(cat_cols[0])[num_cols[0]].mean().plot(kind="bar", ax=ax, color="#00b7eb")
            ax.set_title(f"Bar Chart: {cat_cols[0]} vs {num_cols[0]}", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[2])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.scatter(self.df[num_cols[0]], self.df[num_cols[1]], color="#00b7eb")
            ax.set_xlabel(num_cols[0], color="#ffffff")
            ax.set_ylabel(num_cols[1], color="#ffffff")
            ax.set_title("Pair Plot", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[3])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

    def generate_chart(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No dataset available.")
            return

        chart_type = self.chart_dropdown_var.get()
        if chart_type == "Make bar chart relation between two columns":
            cat_cols = self.df.select_dtypes(include=['object']).columns
            num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            if len(cat_cols) < 1 or len(num_cols) < 1:
                messagebox.showerror("Error", "Dataset must have at least one categorical and one numerical column.")
                return

            col1, col2 = cat_cols[0], num_cols[0]
            fig, ax = plt.subplots(figsize=(3, 2))
            self.df.groupby(col1)[col2].mean().plot(kind="bar", ax=ax, color="#00b7eb")
            ax.set_title(f"Bar Chart: {col1} vs {col2}", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[4])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

        elif chart_type == "Make histogram for one column":
            num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) < 1:
                messagebox.showerror("Error", "No numerical columns available for histogram.")
                return

            col = num_cols[0]
            fig, ax = plt.subplots(figsize=(3, 2))
            self.df[col].hist(ax=ax, color="#00b7eb")
            ax.set_title(f"Histogram of {col}", color="#ffffff")
            ax.tick_params(colors="#ffffff")
            ax.set_facecolor("#333333")
            fig.set_facecolor("#444444")
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frames[4])
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.chart_canvases.append(canvas)

    def go_to_preprocessing(self):
        if self.on_next_callback:
            self.destroy()
            self.on_next_callback(self.df)

