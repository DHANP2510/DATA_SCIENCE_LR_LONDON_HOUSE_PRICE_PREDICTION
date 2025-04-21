import pandas as pd

class DataCleaning:
    def __init__(self, data):
        self.data = data.copy()

    def show_info(self):
        """Display basic information about the dataset."""
        print("Data Info:")
        print(self.data.info())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nDuplicate Rows:", self.data.duplicated().sum())

    def drop_duplicates(self):
        """Remove duplicate rows."""
        before = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        after = self.data.shape[0]
        print(f"Removed {before - after} duplicate rows.")

    def fill_missing(self, strategy='mean', columns=None):
        """
        Fill missing values in specified columns using a strategy.
        strategy: 'mean', 'median', 'mode', or a specific value
        columns: list of column names to apply the strategy
        """
        if columns is None:
            columns = self.data.columns

        for col in columns:
            if self.data[col].isnull().sum() > 0:
                if strategy == 'mean':
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
                elif strategy == 'median':
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                elif strategy == 'mode':
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                else:
                    self.data[col] = self.data[col].fillna(strategy)
                print(f"Filled missing values in '{col}' using strategy: {strategy}")

    def remove_outliers(self, columns=None):
        """
        Remove outliers from specified numerical columns using the IQR method.

        The IQR (Interquartile Range) is calculated as Q3 - Q1.
        Any data point outside the range:
            [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
        is considered an outlier and will be removed.

        columns: list of column names to apply outlier removal.
                 If None, all numerical columns will be used.

        """
        if columns is None:
            columns = self.data.select_dtypes(include=['number']).columns

        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            before = self.data.shape[0]
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            after = self.data.shape[0]
            print(f"Removed {before - after} outliers from '{col}'.")

    def convert_data_types(self, conversions):
        """
        Convert column data types.
        conversions: dictionary where keys are column names and values are target data types
        Example: {'price': 'float', 'date': 'datetime'}
        """
        for col, dtype in conversions.items():
            try:
                if dtype == 'datetime':
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                else:
                    self.data[col] = self.data[col].astype(dtype)
                print(f"Converted '{col}' to {dtype}.")
            except Exception as e:
                print(f"Failed to convert '{col}' to {dtype}. Error: {e}")

    def get_clean_data(self):
        """Return the cleaned DataFrame."""
        return self.data

