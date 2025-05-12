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

    def generate_time_features(self, index, fit=True):                
        dp_features = DeterministicProcess(
            index = index,
            constant = True,
            seasonal = True,
            order = 12,
            drop = True,
            additional_terms = [CalendarFourier(freq="QE", order=4)],
        )
            
        if fit:
            tmp_test = dp_features.out_of_sample(steps = 7 + 1)
            self.df = self.df.join(tmp_test, how = 'left')
        else: 
            tmp_feature = dp_features.in_sample()
            self.df = self.df.join(tmp_feature, how = 'left')

    
    def encode_features(self, target_col='price', fit=True):
        # Encode categorical features and create meaningful transformations for them (like mean encoding).
        df = self.df.copy()

        if fit:
            # Step 1: Mean Encoding for categorical columns (e.g., 'postcode', 'outcode', etc.).
            cat_features = ['postcode', 'outcode', 'tenure', 'propertyType', 'incode']
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


# Steps to use this class:
# train_df = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv")
# test_df = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv")
# train_df['sale_date'] = pd.to_datetime({
#     'year': df['sale_year'],
#     'month': df['sale_month'],
#     'day': 1
# })

# test_df['sale_date'] = pd.to_datetime({
#     'year': df['sale_year'],
#     'month': df['sale_month'],
#     'day': 1
# })

# sale_dates = pd.to_datetime(train_df['sale_date'].sort_values().unique())
# sale_index = pd.date_range(start=sale_dates.min(), end=sale_dates.max(), freq='MS')  # 'MS' = Month Start

# # TRAINING
# train_cleaner = DataCleaner(train_df)
# train_cleaner.extract_incode()
# train_cleaner.impute_with_mode()
# train_cleaner.generate_time_features(index=sale_index, fit=False)
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
# test_cleaner.generate_time_features(index=sale_index, fit=True)  # use same DP
# test_cleaner.encode_features(fit=False)

# test_data_final = test_cleaner.df
