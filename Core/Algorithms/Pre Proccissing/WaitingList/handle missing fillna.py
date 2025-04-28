def handle_missing_values(df, method):
    if method == 'drop':
        df = df.dropna()
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'mode':
        for column in df.columns:
            df[column] = df[column].fillna(df[column].mode().iloc[0])
    return df
