import customtkinter as ctk
import pandas as pd
import PIL.Image
import os
import sys
from tkinter import messagebox 
from customtkinter import CTk, CTkButton, CTkLabel
from FileUploader import FileUploader

def get_resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class App(ctk.CTk):
    def __init__(self, on_done_callback=None):
        super().__init__()
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        self.title("ML AlgoHub")
        self.uploader = FileUploader()
        self.uploaded_data = None
        self.uploaded_path = None
        self.on_done_callback = on_done_callback
        self.create_gui()

    def create_gui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        upperframe = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="gray14")
        upperframe.grid(row=0, column=0, sticky='ew')

        upperframe.grid_rowconfigure(0, weight=1)
        upperframe.grid_columnconfigure((0, 1, 2, 3), weight=1)

        logo_path = get_resource_path("Resources/LogoIcon.png")
        img_r = PIL.Image.open(logo_path)
        img_1 = ctk.CTkImage(img_r, size=(170, 70))
        img1 = ctk.CTkLabel(upperframe, text=" ", image=img_1)
        img1.grid(row=0, column=0, padx=20, pady=20, sticky='w')

        text2 = ctk.CTkLabel(upperframe, text="Home", font=("arial", 15, 'bold'), text_color="#2B5B6D")
        text2.grid(row=0, column=1, padx=10, pady=30)

        about_us_btn = ctk.CTkButton(
            upperframe, 
            text="About Us", 
            font=("arial", 13, 'bold'),
            fg_color="transparent",
            hover_color="#1E3A46",
            command=self.show_about_us
        )
        about_us_btn.grid(row=0, column=2, padx=10, pady=30)

        contact_us_btn = ctk.CTkButton(
            upperframe, 
            text="Contact Us", 
            font=("arial", 13, 'bold'),
            fg_color="transparent",
            hover_color="#1E3A46",
            command=self.show_contact_us
        )
        contact_us_btn.grid(row=0, column=3, padx=10, pady=30)

        center_wrapper = ctk.CTkFrame(self, fg_color="transparent")
        center_wrapper.grid(row=1, column=0, sticky='nsew')
        center_wrapper.grid_rowconfigure(0, weight=1)
        center_wrapper.grid_columnconfigure(0, weight=1)

        centerframe = ctk.CTkFrame(center_wrapper, width=900, height=600, fg_color="gray14")
        centerframe.grid(row=0, column=0)
        centerframe.grid_propagate(False)

        centerframe.grid_rowconfigure((0, 1, 2, 3), weight=1)
        centerframe.grid_columnconfigure(0, weight=1)

        img_2 = ctk.CTkImage(img_r, size=(500, 200))
        img2 = ctk.CTkLabel(centerframe, text=" ", image=img_2)
        img2.grid(row=0, column=0, padx=20, pady=20)

        upload_btn = ctk.CTkButton(centerframe, text="Upload CSV", fg_color="#1E3A46", command=self.upload_file)
        upload_btn.grid(row=1, column=0, pady=(1, 0))

        self.label = CTkLabel(centerframe, text=" ")
        self.label.grid(row=2, column=0, pady=(0, 1), sticky='n')

        center_button = ctk.CTkButton(centerframe, text="Done", fg_color="#1E3A46", command=self.done)
        center_button.grid(row=3, column=0, sticky='n')

    def show_about_us(self):
        messagebox.showinfo(
            "About Us",
            "We are a dedicated team of students from the Faculty of Artificial Intelligence at Menoufia University."
            "This project, ML AlgoHub, was developed as part of our Machine Learning course requirements during the "
            "second semester of our second academic year..\nOur team consists of five passionate members:\n"
            "Ali Elbahrawy\nMohamed Alaa\nMaryam Hassan\nAmira Tallat and Shrouk Kabeel\n(two gentlemen and three ladies)."
            "We poured our hearts into creating this application, aiming to make machine learning accessible and intuitive."
            "Thank you for exploring our work—we hope it inspires and supports your data science journey.."
        )

    def show_contact_us(self):
        messagebox.showinfo(
            "Contact Us",
            "Reach out to our team:\n\n"
            "Ali Elbahrawy: ali.fathy.ali20@gmail.com\n"
            "Mohamed Alaa: Mohamed.Aalaa5@ai.menofia.edu.eg\n"
            "Maryam Hassan: marryam.hassan10@gmail.com\n"
            "Amira Tallat: amiratallat063@gmail.com\n"
            "Shrouk Kabeel: shroukkabeel12345@gmail.com"
        )

    def upload_file(self):
        filename, destination, data = self.uploader.upload_file()
        if filename:
            self.uploaded_path = destination
            self.uploaded_data = data
            self.label.configure(text=f"Uploaded: {filename}")

    def done(self):
        if self.uploaded_data is not None and len(self.uploaded_data) > 0:
            if self.on_done_callback:
                self.on_done_callback(self.uploaded_data)
        else:
            messagebox.showerror("Error", "Please upload a file first.")