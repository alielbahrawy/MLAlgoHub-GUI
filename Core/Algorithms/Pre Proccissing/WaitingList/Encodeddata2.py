import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_non_numeric_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def convert_to_category(df):
    non_numeric_cols = get_non_numeric_columns(df)
    for col in non_numeric_cols:
        df[col] = df[col].astype('category')


def get_eligible_encoding_columns(df, columns=None, max_unique=20):
    """
    تُرجع الأعمدة غير الرقمية (أو subset منها لو تم تمريرها)
    بشرط أن تحتوي على عدد فئات مناسب للترميز.
    """
    convert_to_category(df)
    all_non_numeric = get_non_numeric_columns(df)

    if columns:
        target_cols = [col for col in columns if col in all_non_numeric]
    else:
        target_cols = all_non_numeric

    eligible_cols = []

    for col in target_cols:
        if df[col].nunique() <= max_unique:
            eligible_cols.append(col)

    return eligible_cols


def apply_label_encoding(df, selected_columns=None, columns=None, max_unique=20):
    global df_label_encoded
    df_label_encoded = df.copy()
    convert_to_category(df_label_encoded)

    cols = get_eligible_encoding_columns(df_label_encoded, columns, max_unique)

    if selected_columns:
        cols = [col for col in cols if col in selected_columns]

    for col in cols:
        le = LabelEncoder()
        df_label_encoded[col] = le.fit_transform(df_label_encoded[col].astype(str))


def apply_onehot_encoding(df, selected_columns=None, columns=None, max_unique=10):
    global df_onehot_encoded
    df_onehot_encoded = df.copy()
    convert_to_category(df_onehot_encoded)

    cols = get_eligible_encoding_columns(df_onehot_encoded, columns, max_unique)

    if selected_columns:
        cols = [col for col in cols if col in selected_columns]

    for col in cols:
        df_onehot_encoded = pd.get_dummies(df_onehot_encoded, columns=[col])


def predict_ordinal_columns(df, max_unique=10):
    convert_to_category(df)
    ordinal_columns = []
    for col in get_non_numeric_columns(df):
        if df[col].nunique() <= max_unique:
            ordinal_columns.append(col)
    return ordinal_columns
df = pd.read_csv(r"C:\Users\TUF\Documents\My Data Sets\Spotify_2024_Global_Streaming_Data.csv")

columns = ["Contract ", "InternetService ","SeniorCitizen"]

apply_label_encoding(df, columns=columns, max_unique=20)
print("🔹 Label Encoding بعد تمرير columns:")
print(df_label_encoded.head())

apply_label_encoding(df, selected_columns=["Contract "], max_unique=20)
print("\n🔹 Label Encoding بعد تمرير selected_columns:")
print(df_label_encoded.head())

apply_onehot_encoding(df, selected_columns=[
    "Churn", "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "StreamingMovies", "StreamingTV", "TechSupport",
    "OnlineBackup", "OnlineSecurity", "MultipleLines", "DeviceProtection", "PaymentMethod"
], max_unique=10)
print("\n🔹 نتيجة One-Hot Encoding:")
print(df_onehot_encoded.head())
