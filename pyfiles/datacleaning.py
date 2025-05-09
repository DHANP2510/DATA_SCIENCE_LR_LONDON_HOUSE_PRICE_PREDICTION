from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings
import sys
import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, dataframe, bins=10):
        # Initialize the class with a dataframe and a number of bins for categorization.
        # 'bins' is used to decide how many ranges we split continuous variables like latitude and longitude into.
        self.df = dataframe.copy()
        self.bins = bins
        self.mean_encoders = {}  # Store mean encodings for categorical features
        self.global_means = {}  # Store global mean values for features
        self.bin_edges = {}  # Store bin edges for latitude and longitude
        self.label_encoders = {}  # Store label encoders for categorizing continuous variables
        self.energy_encoder = None  # Store the encoder for the energy rating
        self.dp = None  # Store the deterministic process for creating time-based features
        self.future_time_features = None  # Store future time-based features for prediction

    def show_info(self):
        # Show the shape of the dataframe, data types, and missing values for a quick overview of the dataset.
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def extract_incode(self):
        # Extract the "incode" from the postcode (part of the postcode after the space) for geographic analysis.
        try:
            self.df['incode'] = self.df['postcode'].apply(
                lambda x: x.split(" ")[1] if isinstance(x, str) and " " in x else None
            )
        except Exception as e:
            print("Error while extracting incode:", e)

    def impute_with_mode(self):
        # Impute missing values with the most frequent value (mode) for specific columns.
        cols_to_impute = [
            'bathrooms', 'bedrooms', 'floorAreaSqM',
            'livingRooms', 'tenure', 'propertyType', 'currentEnergyRating'
        ]
        for col in cols_to_impute:
            if col in self.df.columns:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    # If mode is found, fill missing values with the mode.
                    self.df[col].fillna(mode_val[0], inplace=True)
                    print(f"Imputed '{col}' with mode: {mode_val[0]}")
                else:
                    print(f"Could not compute mode for '{col}' — column may be empty.")
            else:
                print(f"Column '{col}' not found in DataFrame.")

    def generate_time_features(self, fit=True, future_steps=12):
        # Generate time-based features like year, month, and seasonal effects.
        
        # Step 1: Create a datetime column from 'sale_year' and 'sale_month'.
        self.df['temp_date'] = pd.to_datetime(
            self.df['sale_year'].astype(str) + '-' + self.df['sale_month'].astype(str) + '-01',
            errors='coerce'
        )
        
        # Step 2: Drop rows where the date is invalid (e.g., NaT values after conversion).
        self.df = self.df.dropna(subset=['temp_date'])
        
        # Step 3: Sort the dataframe by 'temp_date' to ensure time-based features are in order.
        self.df = self.df.sort_values('temp_date')
        
        # Step 4: Set 'temp_date' as the index, which will help in time-series analysis.
        self.df.set_index('temp_date', inplace=True)
        
        # Ensure that the index (date) is unique; remove duplicates if needed.
        self.df = self.df[~self.df.index.duplicated(keep='first')]  # Remove duplicates
        
        # Step 5: Extract year, month, and day as separate columns from the datetime index.
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        
        # Optional: Create seasonal features like sine and cosine transformations to model yearly seasonality.
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Step 6: Generate polynomial features for interaction terms between year, month, and day.
        if fit:
            poly = PolynomialFeatures(degree=2)  # Polynomial features up to degree 2 (you can increase this)
            time_features = poly.fit_transform(self.df[['year', 'month', 'day']])  # Apply to year, month, day
            
            # Generate names for the polynomial features (e.g., year^2, year*month)
            feature_names = ['1', 'year', 'month', 'day', 'year^2', 'year*month', 'year*day', 'month^2', 'month*day', 'day^2']
            
            # Convert the resulting array back to a DataFrame with appropriate column names.
            time_features_df = pd.DataFrame(time_features, columns=feature_names, index=self.df.index)
            
            # Step 7: Merge the new time features with the original DataFrame.
            self.df = pd.concat([self.df, time_features_df], axis=1)
        
        # Optional: Reset the index to have 'temp_date' as a normal column again.
        self.df.reset_index(inplace=True)
    
        return self.df

    def encode_features(self, target_col='price', fit=True):
        # Encode categorical features and create meaningful transformations for them (like mean encoding).
        df = self.df.copy()

        if fit:
            # Step 1: Mean Encoding for categorical columns (e.g., 'postcode', 'outcode', etc.).
            cat_features = ['postcode', 'outcode', 'tenure', 'propertyType']
            for col in cat_features:
                # For each categorical feature, calculate the mean value of the target (e.g., 'price').
                group_mean = df.groupby(col)[target_col].mean()
                self.mean_encoders[col] = group_mean
                self.global_means[col] = group_mean.mean()

            # Step 2: Bin continuous features (latitude and longitude) into categories.
            for col in ['latitude', 'longitude']:
                # Divide the continuous feature into 'bins' (ranges) and label them.
                edges = pd.cut(df[col], bins=self.bins, retbins=True)[1]
                labels = pd.cut(df[col], bins=edges, include_lowest=True)
                le = LabelEncoder().fit(labels)  # Fit a LabelEncoder to convert the labels to integers.
                self.bin_edges[col] = edges
                self.label_encoders[col] = le

            # Step 3: Encode the energy rating (currentEnergyRating) as an ordinal feature (A-G scale).
            self.energy_encoder = OrdinalEncoder(
                categories=[['G', 'F', 'E', 'D', 'C', 'B', 'A']]
            ).fit(df[['currentEnergyRating']])

        # Step 4: Apply the mean encodings to the categorical features.
        for col in self.mean_encoders:
            df[col] = df[col].map(self.mean_encoders[col])
            df[col] = df[col].fillna(self.global_means[col])  # Fill any missing values with the global mean.

        # Step 5: Apply label encoding to latitude and longitude bins.
        for col in ['latitude', 'longitude']:
            bin_col = f'{col}Bins'  # New column to store binned data.
            df[bin_col] = pd.cut(df[col], bins=self.bin_edges[col], include_lowest=True)
            df[bin_col] = self.label_encoders[col].transform(df[bin_col])  # Convert the bins to numerical labels.

        # Step 6: Apply the energy rating encoding.
        df['currentEnergyRating'] = self.energy_encoder.transform(df[['currentEnergyRating']])
        
        # Step 7: Save the updated dataframe with encoded features.
        self.df = df

#####################################################################################################################
# How to use this file
# train_df = pd.read_csv("train.csv")
# test_df = pd.read_csv("test.csv")


# # TRAINING
# train_cleaner = DataCleaner(train_df)
# train_cleaner.extract_incode()
# train_cleaner.impute_with_mode()
# train_cleaner.generate_time_features(fit=True, future_steps=12)
# train_cleaner.encode_features(target_col='price', fit=True)

# train_data_final = train_cleaner.df
# future_features = train_cleaner.future_time_features  # for forecasting test

# # TESTING
# test_cleaner = DataCleaner(test_df)
# test_cleaner.dp = train_cleaner.dp  # share fitted DP
# test_cleaner.mean_encoders = train_cleaner.mean_encoders
# test_cleaner.global_means = train_cleaner.global_means
# test_cleaner.bin_edges = train_cleaner.bin_edges
# test_cleaner.label_encoders = train_cleaner.label_encoders
# test_cleaner.energy_encoder = train_cleaner.energy_encoder

# test_cleaner.extract_incode()
# test_cleaner.impute_with_mode()
# test_cleaner.generate_time_features(fit=False)  # use same DP
# test_cleaner.encode_features(fit=False)

# test_data_final = test_cleaner.df

