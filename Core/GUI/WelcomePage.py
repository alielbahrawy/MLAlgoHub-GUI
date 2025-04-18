from functools import partial
import PIL.Image
import customtkinter as ctk
import PIL
from collections import defaultdict
import os
import shutil
import pandas as pd
from tkinter import filedialog as fd
from customtkinter import CTk, CTkButton, CTkLabel

ctk.set_default_color_theme(r"breeze.json")
ctk.set_appearance_mode("Dark")

class SecondPage(CTk):
    def __init__(self, data):
        super().__init__()
        self.title("New GUI Page - Data Loaded")
        self.geometry("900x600")

        # Just a placeholder layout - you can add widgets here freely
        title = CTkLabel(self, text="Welcome to Page 2!", font=("Arial", 24, "bold"))
        title.pack(pady=40)

        preview = CTkLabel(self, text="Preview of your CSV file:", font=("Arial", 16))
        preview.pack()

        # Display part of DataFrame
        df_preview = CTkLabel(self, text=str(data.head()), font=("Consolas", 12), justify="left")
        df_preview.pack(pady=20)

        # Exit or Back button
        exit_btn = CTkButton(self, text="Close", command=self.destroy)
        exit_btn.pack(pady=20)



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry('1024x720')
        self.title("ML AlgoHub")
        self.upload_folder = "uploads"
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # This will store the DataFrame or file path
        self.uploaded_data = None
        self.uploaded_path = None

        # Configure layout
        self.grid_rowconfigure(0, weight=0)  # upperframe
        self.grid_rowconfigure(1, weight=1)  # center wrapper
        self.grid_columnconfigure(0, weight=1)

        # Upper frame
        self.upperframe = ctk.CTkFrame(self, height=50 ,corner_radius=0,fg_color="gray14" )
        self.upperframe.grid(row=0, column=0, sticky='ew')
        
        self.upperframe.grid_rowconfigure(0,weight=1)
        self.upperframe.grid_columnconfigure(0,weight=1)
        self.upperframe.grid_columnconfigure(1,weight=1)
        self.upperframe.grid_columnconfigure(2,weight=1)
        self.upperframe.grid_columnconfigure(3,weight=1)
        
        self.img_1=PIL.Image.open("LogoIcon.png")
        self.img_1=ctk.CTkImage(self.img_1,size=(170,70))
        self.img1=ctk.CTkLabel(self.upperframe , text=" ",image=self.img_1)
        self.img1.grid(row=0,column=0 ,padx=20, pady=20,sticky='w')
        
        self.text2=ctk.CTkLabel(self.upperframe , text="Home",font=("arial",15,'bold'),text_color="#2B5B6D")
        self.text2.grid(row=0,column=1,padx=10 , pady=30)
        
        self.text3=ctk.CTkLabel(self.upperframe , text="About Us",font=("arial",13,'bold'))
        self.text3.grid(row=0,column=2,padx=10 , pady=30)
        
        self.text4=ctk.CTkLabel(self.upperframe , text="Contact Us",font=("arial",13,'bold') )
        self.text4.grid(row=0,column=3,padx=10 , pady=30)

        # Center wrapper (fills space)
        self.center_wrapper = ctk.CTkFrame(self, fg_color="transparent")
        self.center_wrapper.grid(row=1, column=0, sticky='nsew')
        self.center_wrapper.grid_rowconfigure(0, weight=1)
        self.center_wrapper.grid_columnconfigure(0, weight=1)

                # Actual centerframe (smaller)
        self.centerframe = ctk.CTkFrame(self.center_wrapper, width=900, height=600 ,fg_color="gray14")
        self.centerframe.grid(row=0, column=0)
        self.centerframe.grid_propagate(False)  # Prevent resize to children

        # Configure 3 rows in centerframe
        self.centerframe.grid_rowconfigure((0, 1, 2,3), weight=1)  # All rows same height
        self.centerframe.grid_columnconfigure(0, weight=1)       # Center column
        
        #logo
        self.img_2=PIL.Image.open("LogoIcon.png")
        self.img_2=ctk.CTkImage(self.img_2,size=(500,200))
        self.img2=ctk.CTkLabel(self.centerframe , text=" ",image=self.img_2)
        self.img2.grid(row=0,column=0 ,padx=20, pady=20,)
        
        #upload file
        
       # Upload file button and label inside centerframe
        self.upload_btn = ctk.CTkButton(self.centerframe, text="Upload CSV", fg_color="#1E3A46", command=self.upload_file)
        self.upload_btn.grid(row=1, column=0, pady=(1,0))  # Upload button

        self.label = CTkLabel(self.centerframe, text=" ")
        self.label.grid(row=2, column=0, pady=(0, 1),sticky='n')  # Label right under button

# Done button in the next row
        self.center_button = ctk.CTkButton(self.centerframe, text="Done", fg_color="#1E3A46", command=self.use_uploaded_data)
        self.center_button.grid(row=3, column=0,sticky='n')

    def upload_file(self):
        file_path = fd.askopenfilename(
            title="Select a CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.upload_folder, filename)
            shutil.copy(file_path, destination)

            # Store file path and data
            self.uploaded_path = destination
            self.uploaded_data = pd.read_csv(destination)

            self.label.configure(text=f"Uploaded: {filename}")    
    def use_uploaded_data(self):
        if self.uploaded_data is not None:
            self.destroy()  # close current window
            new_gui = SecondPage(self.uploaded_data)
            new_gui.mainloop()
        else:
            self.label.configure(text="No file uploaded yet.")
        
if __name__ == '__main__':
    app = App()
    app.mainloop()
