import pandas as pd
import numpy as np

def detect_outliers_iqr(df):
  
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = df[col].index[
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ]
        count_outliers = len(outlier_indices)


        outlier_summary[col] = {
            'count': count_outliers,
            'indices': outlier_indices.tolist()
        }


    return  outlier_summary


