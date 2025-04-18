import tkinter as tk
from tkinter import filedialog
from data_loader import load_dataset
from KNNImputer import Knn_imputer  # استيراد دالة KNN Imputer من الملف الخاص بها
from Iterativeimputer import Iterative_imputer  # استيراد دالة Iterative Imputer من الملف الخاص بها

# متغير عالمي لتخزين البيانات
df_data = None

# دالة لتحميل البيانات وعرض الأزرار لاختيار العملية
def upload_file():
    global df_data  # استخدام المتغير العالمي
    file_path = filedialog.askopenfilename(
        title="اختار ملف البيانات",
        filetypes=[("All Files", "*.*"), ("CSV Files", "*.csv"),
                   ("Excel Files", "*.xls;*.xlsx"), ("JSON Files", "*.json"),
                   ("Parquet Files", "*.parquet")]
    )

    if file_path:
        print(f"تم اختيار الملف: {file_path}")
        df_data = load_dataset(file_path)  # تحميل البيانات في المتغير العالمي

        # رسالة تأكيد تحميل البيانات
        label.config(text="تم تحميل البيانات بنجاح! اختر العملية التي تريد تنفيذها:")

        # إظهار الأزرار لاختيار العملية
        knn_button.pack(pady=10)
        iterative_button.pack(pady=10)

# دالة لتطبيق KNN Imputer
def apply_knn():
    global df_data  # استخدام المتغير العالمي
    if df_data is not None:
        df_imputed = Knn_imputer(df_data)  # استخدام الدالة من ملف KNNImputer
        print("البيانات بعد تطبيق KNN Imputer:")
        print(df_imputed.head())

# دالة لتطبيق Iterative Imputer
def apply_iterative():
    global df_data  # استخدام المتغير العالمي
    if df_data is not None:
        df_imputed = Iterative_imputer(df_data)  # استخدام الدالة من ملف Iterativeimputer
        print("البيانات بعد تطبيق Iterative Imputer:")
        print(df_imputed.head())

# إعداد واجهة المستخدم باستخدام Tkinter
root = tk.Tk()
root.title("تحميل وتطبيق المعالج")

# إعداد رسالة لشرح ما يحدث
label = tk.Label(root, text="اختر ملف البيانات لتبدأ")
label.pack(pady=20)

# إضافة زر لتحميل الملف
upload_button = tk.Button(root, text="تحميل ملف البيانات", command=upload_file)
upload_button.pack(pady=20)

# إضافة زر لتطبيق KNN Imputer
knn_button = tk.Button(root, text="تطبيق KNN Imputer", command=apply_knn)

# إضافة زر لتطبيق Iterative Imputer
iterative_button = tk.Button(root, text="تطبيق Iterative Imputer", command=apply_iterative)

# تشغيل الواجهة
root.mainloop()
