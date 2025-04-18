import tkinter as tk
from tkinter import filedialog
from data_loader import load_dataset  # استيراد دالة تحميل البيانات
from Iterativeimputer import Iterative_imputer


# دالة لتحديد ملف وتحميله باستخدام ملف data_loader
def upload_file():
    file_path = filedialog.askopenfilename(
        title="اختار ملف البيانات",
        filetypes=[("All Files", "*.*"), ("CSV Files", "*.csv"), ("Excel Files", "*.xls;*.xlsx"),
                   ("JSON Files", "*.json"), ("Parquet Files", "*.parquet")]
    )

    if file_path:  # إذا تم اختيار ملف
        print(f"تم اختيار الملف: {file_path}")
        # تحميل البيانات باستخدام دالة load_dataset
        df_data = load_dataset(file_path)

        # تطبيق Iterative Imputer على البيانات
        df_imputed = Iterative_imputer(df_data)

        # عرض البيانات بعد تطبيق Iterative Imputer
        print("البيانات بعد تطبيق Iterative Imputer:")
        print(df_imputed.head())


# إعداد واجهة المستخدم باستخدام Tkinter
root = tk.Tk()
root.title("تحميل وتطبيق Iterative Imputer")

# إضافة زر لتحميل الملف
upload_button = tk.Button(root, text="تحميل ملف البيانات", command=upload_file)
upload_button.pack(pady=20)

# تشغيل الواجهة
root.mainloop()
