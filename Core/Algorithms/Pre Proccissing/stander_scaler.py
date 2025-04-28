from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def standardize(data):
    scaler = StandardScaler()
    column_num=data.select_dtypes(include='number').columns.tolist()
    data[column_num] = scaler.fit_transform(data[column_num])
    
    return data