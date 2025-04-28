from sklearn.preprocessing import LabelEncoder
def suggest_encoding_columns(df,  threshold_binary=10):
    label_encoding_columns = []
    cat_cols = df.select_dtypes(include=['object',"category"])

    for col in cat_cols.columns:
        n_unique = df[col].nunique()

        if n_unique > threshold_binary:
            label_encoding_columns.append(col)

    return label_encoding_columns   
             
def label_encoder(df):
    encoder = LabelEncoder()
    for col in suggest_encoding_columns(df):
        df[col] = encoder.fit_transform(df[col])
