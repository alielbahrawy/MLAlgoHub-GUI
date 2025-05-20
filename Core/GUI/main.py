import customtkinter as ctk
import sys
import os
from App import App
from Dataset_Summary import DatasetSummaryApp
from PreprocessingPage import PrePage
from VisualizationPage import VisualizationPage
from Data_Models import MLModel
from tkinter import messagebox

def get_resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def open_dataset_summary(data):
    app.destroy()
    dataset_summary = DatasetSummaryApp(data=data, on_next_callback=lambda df: open_preprocessing(dataset_summary, df))
    dataset_summary.mainloop()

def open_preprocessing(dataset_summary, data):
    dataset_summary.destroy()
    preprocessing_page = PrePage(
        data=data,
        on_back_callback=lambda df: open_dataset_summary_from_preprocessing(preprocessing_page, df),
        on_next_callback=lambda df: open_visualization(preprocessing_page, df)
    )
    preprocessing_page.mainloop()

def open_dataset_summary_from_preprocessing(preprocessing_page, data):
    preprocessing_page.destroy()
    dataset_summary = DatasetSummaryApp(data=data, on_next_callback=lambda df: open_preprocessing(dataset_summary, df))
    dataset_summary.mainloop()

def open_visualization(preprocessing_page, data):
    preprocessing_page.destroy()
    visualization_page = VisualizationPage(
        data=data,
        on_next_callback=lambda df: open_data_models(visualization_page, df),
        on_back_callback=lambda df: open_preprocessing(visualization_page, df)
    )
    visualization_page.mainloop()

def open_data_models(visualization_page, data):
    visualization_page.destroy()
    data_models = MLModel(
        processed_df=data,
        on_back_callback=lambda df: open_visualization(data_models, df)
    )
    data_models.mainloop()

def main():
    ctk.set_default_color_theme(get_resource_path("Resources/breeze.json"))
    ctk.set_appearance_mode("Dark")
    
    global app
    app = App(on_done_callback=open_dataset_summary)
    app.mainloop()

if __name__ == '__main__':
    main()