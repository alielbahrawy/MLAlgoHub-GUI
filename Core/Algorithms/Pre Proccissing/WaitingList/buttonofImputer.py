import tkinter as tk
from tkinter import filedialog
from data_loader import load_dataset
from KNNImputer import Knn_imputer
from Iterativeimputer import Iterative_imputer
from SimpleImputer import simple_imputer_object, simple_imputer_numerical

# متغير عالمي لتخزين البيانات
df_data = None

def upload_file():
    global df_data
    file_path = filedialog.askopenfilename(
        title="اختار ملف البيانات",
        filetypes=[("All Files", "*.*"), ("CSV Files", "*.csv"),
                   ("Excel Files", "*.xls;*.xlsx"), ("JSON Files", "*.json"),
                   ("Parquet Files", "*.parquet")]
    )

    if file_path:
        print(f"تم اختيار الملف: {file_path}")
        df_data = load_dataset(file_path)

        label.config(text="تم تحميل البيانات بنجاح! اختر العملية التي تريد تنفيذها:")

        knn_button.pack(pady=10)
        iterative_button.pack(pady=10)
        simple_button.pack(pady=10)  # زر Simple Imputer

def apply_knn():
    global df_data
    if df_data is not None:
        df_imputed = Knn_imputer(df_data)
        print("البيانات بعد تطبيق KNN Imputer:")
        print(df_imputed.head())

def apply_iterative():
    global df_data
    if df_data is not None:
        df_imputed = Iterative_imputer(df_data)
        print("البيانات بعد تطبيق Iterative Imputer:")
        print(df_imputed.head())

def apply_simple_imputer():
    global df_data
    if df_data is not None:
        df_data = simple_imputer_numerical(df_data)
        df_data = simple_imputer_object(df_data)
        print("البيانات بعد تطبيق Simple Imputer:")
        print(df_data.head())

# واجهة المستخدم
root = tk.Tk()
root.title("تحميل وتطبيق المعالج")

label = tk.Label(root, text="اختر ملف البيانات لتبدأ")
label.pack(pady=20)

upload_button = tk.Button(root, text="تحميل ملف البيانات", command=upload_file)
upload_button.pack(pady=20)

knn_button = tk.Button(root, text="تطبيق KNN Imputer", command=apply_knn)
iterative_button = tk.Button(root, text="تطبيق Iterative Imputer", command=apply_iterative)
simple_button = tk.Button(root, text="تطبيق Simple Imputer", command=apply_simple_imputer)

root.mainloop()
