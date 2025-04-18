import tkinter as tk
from tkinter import filedialog
from data_loader import load_dataset  # استيراد دالة تحميل البيانات
from sklearn.impute import KNNImputer
from data_loader import load_dataset
from KNNImputer import Knn_imputer
  # استيراد دالة KNNImputer


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

        # استدعاء دالة KNN_imputer
        df_imputed = Knn_imputer(df_data)
        # عرض البيانات بعد المعالجة
        print("البيانات بعد تطبيق KNN Imputer:")
        print(df_imputed.head())


# إعداد واجهة المستخدم باستخدام Tkinter
root = tk.Tk()
root.title("تحميل وتطبيق KNN Imputer")

# إضافة زر لتحميل الملف
upload_button = tk.Button(root, text="تحميل ملف البيانات", command=upload_file)
upload_button.pack(pady=20)

# تشغيل الواجهة
root.mainloop()
