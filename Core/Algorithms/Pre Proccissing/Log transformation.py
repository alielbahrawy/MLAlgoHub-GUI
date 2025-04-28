import pandas as pd
import numpy as np

def handle_outliers_log(data):
    column_num=data.select_dtypes(include='number').columns.tolist()

    data[column_num] = np.log1p(data[column_num])
    return data