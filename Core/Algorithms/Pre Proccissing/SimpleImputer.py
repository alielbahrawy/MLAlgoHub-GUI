from sklearn.impute import SimpleImputer
import pandas as pd


# simple imputer to handel missing value

def simple_imputer_numerical(df):
    # select numerical column 
    numeric_df = df.select_dtypes(include=['number'])

    # show data if doesn't have a missing value
    if numeric_df.isnull().sum().sum() == 0:
        print("data doesn't have a missing value ")
        return df
    
    # do simple Imputer
    imputer=SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numeric_df)

    # convert result to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)
    df[numeric_df.columns] = imputed_df

    print(" It was completed apply simpleImputer on data set. ")
    return df


def simple_imputer_object(df):
    # select numerical column 
    category_df = df.select_dtypes(include=['object'])

    # show data if doesn't have a missing value
    if category_df.isnull().sum().sum() == 0:
        print("data doesn't have a missing value ")
        return df
    
    # do simple Imputer
    imputer=SimpleImputer(strategy='most_frequent')
    imputed_data=imputer.fit_transform(category_df)

    # convert result to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=category_df.columns, index=category_df.index)
    df[category_df.columns] = imputed_df

    print(" It was completed apply simpleImputer on data set. ")
    return df





