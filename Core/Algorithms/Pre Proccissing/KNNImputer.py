from sklearn.impute import KNNImputer
import pandas as pd


# KNN imputer to handel missing value
def Knn_imputer (df):

    # select numerical column 
    numeric_df = df.select_dtypes(include=['number'])

    # show data if doesn't have a missing value
    if numeric_df.isnull().sum().sum() == 0:
        print("data doesn't have a missing value ")
        return df

    # do KNN Imputer
    imputer = KNNImputer(n_neighbors=2)
    imputed_data = imputer.fit_transform(numeric_df)

    # convert result to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)

    # change old column with new column after filling 
    df[numeric_df.columns] = imputed_df

    print(" It was completed apply KNNImputer on data set. ")
    return df  