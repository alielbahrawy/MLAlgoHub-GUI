from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

# Iterative imputer to handel missing value
def Iterative_imputer(df):
    # select numerical column 
    numeric_df = df.select_dtypes(include=['number'])

    # show data if doesn't have a missing value
    if numeric_df.isnull().sum().sum() == 0:
        print("data doesn't have a missing value ")
        return df

    # do Iterative Imputer
    imputer = IterativeImputer()
    imputed_data = imputer.fit_transform(numeric_df)

    # convert result to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)
    df[numeric_df.columns] = imputed_df

    print(" It was completed apply IterativeImputer on data set. ")
    return df
