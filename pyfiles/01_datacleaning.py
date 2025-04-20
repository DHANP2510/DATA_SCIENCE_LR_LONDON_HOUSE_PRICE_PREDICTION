import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataCleaning:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.cleaned_data = None

    def load_data(self):
        """Loads the data from the CSV file"""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded from {self.data_path}")

    def handle_missing_values(self):
        """Handles missing values using SimpleImputer"""
        # Create imputer to fill missing values
        imputer = SimpleImputer(strategy='mean')  # You can change strategy based on needs
        self.df.iloc[:, :] = imputer.fit_transform(self.df)
        print("Missing values handled")

    def handle_outliers(self):
        """Handles outliers in the dataset"""
        # For simplicity, we'll use IQR method for outlier detection and removal
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        print("Outliers handled using IQR method")

    def standardize_data(self):
        """Standardize numerical features"""
        scaler = StandardScaler()
        self.df[self.df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(self.df.select_dtypes(include=[np.number]))
        print("Data standardized")

    def save_cleaned_data(self, output_path):
        """Save the cleaned dataset to a pickle file"""
        self.cleaned_data = self.df
        with open(output_path, 'wb') as file:
            pickle.dump(self.cleaned_data, file)
        print(f"Cleaned data saved to {output_path}")
        
    def execute_cleaning(self):
        """Execute all cleaning steps"""
        self.load_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.standardize_data()

# Example usage:
#if __name__ == "__main__":
    # Path to your raw dataset
    #data_path = 'path/to/your/raw_data.csv'
    #output_path = 'path/to/save/cleaned_data.pkl'
    
    #cleaner = DataCleaning(data_path)
    #cleaner.execute_cleaning()
    $cleaner.save_cleaned_data(output_path)
