import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
def apply_pca(data, n_components=2):
    
    numeric_cols = data.select_dtypes(include='number').columns
    X = data[numeric_cols]

 

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)

    print(f"Explained Variance Ratio (first 10): {pca.explained_variance_ratio_[:10]}")
    return pca_df, pca