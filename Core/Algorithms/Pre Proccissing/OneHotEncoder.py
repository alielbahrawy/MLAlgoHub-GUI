from sklearn.preprocessing import OneHotEncoder
import pandas as pd
def suggest_encoding_columns(df,  threshold_onehot=3, threshold_binary=10):
    onehot_encoding_columns = []
    cat_cols = df.select_dtypes(include=['object',"category"])

    for col in cat_cols.columns:
        n_unique = df[col].nunique()

        if n_unique <= threshold_onehot:
            onehot_encoding_columns.append(col)
        elif n_unique <= threshold_binary:
            onehot_encoding_columns.append(col)

    return onehot_encoding_columns 
def OneHot_Encoder(df):
    encoder = OneHotEncoder(sparse_output=False)
    cat_cols = suggest_encoding_columns(df)
    encoded_cols = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(cat_cols))
    df = df.drop(cat_cols, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df
 
  
             
