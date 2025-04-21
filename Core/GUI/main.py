import customtkinter as ctk
import sys
import os
from App import App
from PreprocessingPage import SecondPage
from tkinter import messagebox

def get_resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Callback function to be called when 'Done' is pressed
def on_done(data):
    print("Data received:", data)  
    app.destroy()
    second_page = SecondPage(data)
    second_page.mainloop()

def main():
    # Set theme and appearance
    ctk.set_default_color_theme(get_resource_path("Resources/breeze.json"))
    ctk.set_appearance_mode("Dark")
    
    # Create and run the main application
    global app
    app = App(on_done_callback=on_done)
    app.mainloop()

if __name__ == '__main__':
    main()                        