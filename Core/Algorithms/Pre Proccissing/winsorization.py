from scipy.stats.mstats import winsorize
import pandas as pd
def apply_winsorization(data,  lower_limit=0.05, upper_limit=0.05):
    data = data.copy()
    columns = data.select_dtypes(include='number').columns.tolist()

    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            winsorized_col = winsorize(data[col], limits=(lower_limit, upper_limit))
            data[col] = winsorized_col
    return data

