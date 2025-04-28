import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from data_loader import load_dataset
from encoding_functions import (
    get_non_numeric_columns,
    apply_label_encoding,
    apply_onehot_encoding,
    predict_ordinal_columns  # نفترض إن دي موجودة كميزة إضافية اختيارية
)

# متغير عالمي لتخزين البيانات
df_data = None
non_numeric_cols = []
label_vars = {}
predicted_ordinal_cols = []


# تحميل البيانات
def upload_file():
    global df_data, non_numeric_cols, label_vars
    file_path = filedialog.askopenfilename(
        title="اختر ملف البيانات",
        filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xls;*.xlsx")]
    )
    if file_path:
        try:
            df_data = load_dataset(file_path)
            non_numeric_cols = get_non_numeric_columns(df_data)
            label_vars = {col: tk.BooleanVar() for col in non_numeric_cols}
            show_column_selection()
            show_data_preview(df_data)
            messagebox.showinfo("تم التحميل", "تم تحميل البيانات بنجاح!")
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تحميل البيانات:\n{e}")


# عرض جزء من البيانات في واجهة المستخدم
def show_data_preview(df):
    text_box.delete("1.0", tk.END)
    preview = df.head().to_string(index=False)
    text_box.insert(tk.END, f"عرض جزء من البيانات:\n{preview}\n\nالأعمدة غير الرقمية:\n")
    for col in non_numeric_cols:
        text_box.insert(tk.END, f"- {col}\n")


# عرض اختيار الأعمدة
def show_column_selection():
    for widget in column_frame.winfo_children():
        widget.destroy()

    tk.Label(column_frame, text="اختر الأعمدة التي تريد ترميزها باستخدام Label Encoding:").pack(anchor="w")

    # عرض الأعمدة غير الرقمية مع checkboxes
    for col in non_numeric_cols:
        tk.Checkbutton(column_frame, text=col, variable=label_vars[col]).pack(anchor="w")

    tk.Button(column_frame, text="تحديد الكل (غير رقمي)", command=select_all).pack(side="left", padx=5, pady=5)
    tk.Button(column_frame, text="إلغاء التحديد", command=deselect_all).pack(side="left", padx=5)
    tk.Button(column_frame, text="تجربة توقع الأعمدة المرتبة (اختياري)", command=suggest_ordinal).pack(side="left",
                                                                                                       padx=5)
    encode_button.pack(pady=10)


# تحديد الكل (الغير رقمي)
def select_all():
    for var in label_vars.values():
        var.set(True)


# إلغاء التحديد
def deselect_all():
    for var in label_vars.values():
        var.set(False)


# ميزة توقع الأعمدة المرتبة باستخدام NLP (اختياري)
def suggest_ordinal():
    global predicted_ordinal_cols
    if df_data is not None:
        suggested = predict_ordinal_columns(df_data[non_numeric_cols])
        predicted_ordinal_cols = suggested
        # عرض الأعمدة التي تم التنبؤ بها أولاً
        for col in suggested:
            if col in label_vars:
                label_vars[col].set(True)

        # عرض الأعمدة المتوقعة
        messagebox.showinfo("اقتراحات", f"تم تحديد الأعمدة المقترحة: {', '.join(suggested)}")

    # أضف زر لتعديل الاختيارات
    show_column_selection()


# تطبيق الترميز
def apply_encoding():
    global df_data
    if df_data is None:
        messagebox.showwarning("تنبيه", "يرجى تحميل البيانات أولًا.")
        return

    # تحديد الأعمدة بناءً على اختيار المستخدم
    selected_label_cols = [col for col, var in label_vars.items() if var.get()]

    if selected_label_cols:
        label_cols = selected_label_cols
        onehot_cols = list(set(non_numeric_cols) - set(label_cols))
    else:
        label_cols = [col for col in non_numeric_cols if df_data[col].nunique() <= 10]
        onehot_cols = list(set(non_numeric_cols) - set(label_cols))

    df_encoded = apply_label_encoding(df_data.copy(), label_cols)
    df_encoded = apply_onehot_encoding(df_encoded, onehot_cols)

    text_box.delete("1.0", tk.END)
    text_box.insert(tk.END, df_encoded.head().to_string(index=False))

    save = messagebox.askyesno("تم الترميز", "تم الترميز بنجاح. هل ترغب في حفظ الملف؟")
    if save:
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path:
            df_encoded.to_csv(save_path, index=False)
            messagebox.showinfo("تم الحفظ", "تم حفظ الملف بنجاح.")


# زر إعادة التعيين
def reset():
    global df_data, non_numeric_cols, label_vars, predicted_ordinal_cols
    df_data = None
    non_numeric_cols = []
    label_vars = {}
    predicted_ordinal_cols = []

    # إعادة تعيين واجهة المستخدم
    text_box.delete("1.0", tk.END)
    label.config(text="اختر ملف البيانات لتبدأ")
    upload_btn.config(state="normal")
    encode_button.config(state="disabled")
    for widget in column_frame.winfo_children():
        widget.destroy()


# واجهة المستخدم
root = tk.Tk()
root.title("Smart Encoding GUI")
root.geometry("850x700")

frame = tk.Frame(root)
frame.pack(pady=10)

upload_btn = tk.Button(frame, text="تحميل ملف البيانات", command=upload_file)
upload_btn.pack(pady=5)

column_frame = tk.Frame(root)
column_frame.pack(pady=10)

encode_button = tk.Button(root, text="تطبيق Encoding", command=apply_encoding, state="disabled")

reset_button = tk.Button(root, text="إعادة تعيين", command=reset)
reset_button.pack(pady=10)

text_box = tk.Text(root, wrap="none", height=25, width=100)
text_box.pack(pady=10)

root.mainloop()
