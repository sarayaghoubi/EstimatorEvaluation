import polars as pl
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, api_url=None, file_path=None):
        """
        Initialize the DataProcessor with the file path to the Parquet file.
        :param api_url: the address where we can read train data from
        :param file_path:
        """
        self.api_url = api_url
        self.file_path = file_path
        self.data = None

    def read_parquet(self):
        """
        Reads the Parquet file into a Polars DataFrame.
        """
        try:
            self.data = pl.read_parquet(self.file_path)
            print(f"Data loaded successfully with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error reading the Parquet file: {e}")

    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, encode categorical variables, etc.
        """
        if self.data is None:
            print("No data to process. Please read the Parquet file first.")
            return

        # Handling missing values by filling with the median for numerical columns
        numeric_columns = self.data.select(pl.col(pl.Float64) | pl.col(pl.Int64)).columns
        for col in numeric_columns:
            median_value = self.data[col].median()
            self.data = self.data.with_column(
                pl.col(col).fill_none(median_value)
            )

        # Encode categorical variables using one-hot encoding
        categorical_columns = self.data.select(pl.col(pl.Categorical)).columns
        self.data = self.data.to_dummies(columns=categorical_columns)

        print("Preprocessing completed.")

    def find_correlations(self, threshold=0.5):
        """
        Find correlations in the dataset and return a DataFrame with pairs that exceed the threshold.
        """
        if self.data is None:
            print("No data to process. Please read the Parquet file first.")
            return None

        # Compute the correlation matrix
        corr_matrix = self.data.to_pandas().corr().abs()

        # Extract pairs with correlation above the threshold
        corr_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]
        high_corr_pairs = corr_pairs[corr_pairs["Correlation"] > threshold]

        return pl.DataFrame(high_corr_pairs)

    def extract_features(self, target_column, method="pca", k=5):
        """
        Extract features using a specified method: 'pca', 'mutual_info'.
        """
        if self.data is None:
            print("No data to process. Please read the Parquet file first.")
            return None

        if target_column not in self.data.columns:
            print(f"Target column '{target_column}' not found in data.")
            return None

        X = self.data.drop(target_column).to_numpy()
        y = self.data[target_column].to_numpy()

        if method == "pca":
            # Standardizing the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA
            pca = PCA(n_components=k)
            X_pca = pca.fit_transform(X_scaled)
            print(f"Explained variance by each principal component: {pca.explained_variance_ratio_}")
            return pl.DataFrame(X_pca)

        elif method == "mutual_info":
            # Apply Mutual Information
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_new = selector.fit_transform(X, y)
            selected_features = self.data.drop(target_column).columns[selector.get_support(indices=True)]
            print(f"Selected features based on Mutual Information: {selected_features}")
            return pl.DataFrame(X_new, columns=selected_features)

        else:
            print("Invalid method. Choose from 'pca' or 'mutual_info'.")
            return None

# Example usage:
# processor = DataProcessor('your_data_file.parquet')
# processor.read_parquet()
# processor.preprocess_data()
# correlations = processor.find_correlations(threshold=0.7)
# features = processor.extract_features(target_column='target', method='pca', k=3)
