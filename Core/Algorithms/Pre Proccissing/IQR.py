import pandas as pd
import numpy as np

def handle_outliers_iqr(df):
    """
    معالجة القيم الشاذة (Outliers) باستخدام طريقة IQR.
    يتم استبدال القيم الشاذة بالـ Median لكل عمود رقمي.
    تُرجع DataFrame جديدة وجدول بعدد القيم الشاذة قبل وبعد المعالجة.
    """

    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    outlier_summary = pd.DataFrame(columns=["Before", "After"])

    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # تحديد القيم الشاذة
        outlier_indices = df_processed[col].index[
            (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        ]
        count_before = len(outlier_indices)

        # استبدالها بالـ median
        median_val = df_processed[col].median()
        df_processed.loc[outlier_indices, col] = median_val

        # إعادة الفحص بعد المعالجة
        count_after = len(df_processed[
            (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        ])

        outlier_summary.loc[col] = [count_before, count_after]

    print("\n✅ عدد القيم الشاذة قبل وبعد المعالجة بطريقة IQR:")
    print(outlier_summary)

    return df_processed, outlier_summary

from data_loader import load_dataset
df = load_dataset(r"C:\Users\TUF\Documents\My Data Sets\patients.xlsx")
df_cleaned, summary = handle_outliers_iqr(df)
