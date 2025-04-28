from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def min_max_normalize(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    column_num=data.select_dtypes(include='number').columns.tolist()
    data[column_num] = scaler.fit_transform(data[column_num])

    return data

