import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_zscore(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}

    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_indices = df[col].index[z_scores > 3]
        count_outliers = len(outlier_indices)
        outlier_summary[col] = {
            'count': count_outliers,
            'indices': outlier_indices.tolist()
        }
        print(f"Column: {col}, Outliers: {count_outliers}")




    return  outlier_summary


