"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and initial transformations
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Titanic dataset
    """
    
    def __init__(self):
        self.age_imputer = None
        self.fare_imputer = None
        self.embarked_mode = None
        
    def load_data(self, train_path, test_path):
        """
        Load training and test datasets
        """
        logger.info("Loading datasets...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    
    def handle_missing_values(self, df, is_train=True):
        """
        Handle missing values in the dataset
        
        Strategy:
        - Age: Median imputation grouped by Pclass and Sex
        - Embarked: Mode imputation
        - Fare: Median imputation
        - Cabin: Create new feature indicating cabin presence
        """
        logger.info("Handling missing values...")
        df = df.copy()
        
        # Age imputation - use grouped median
        if is_train:
            age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()
            self.age_medians = age_medians
        
        for (pclass, sex), median_age in self.age_medians.items():
            mask = (df['Pclass'] == pclass) & (df['Sex'] == sex) & (df['Age'].isna())
            df.loc[mask, 'Age'] = median_age
        
        # Fill remaining Age nulls with overall median
        if df['Age'].isna().any():
            df = df.fillna({'Age': df['Age'].median()})
        
        # Embarked imputation
        if is_train:
            self.embarked_mode = df['Embarked'].mode()[0]
        df = df.fillna({'Embarked': self.embarked_mode})
        
        # Fare imputation
        if is_train:
            self.fare_median = df['Fare'].median()
        df = df.fillna({'Fare': self.fare_median})
        
        # Cabin - create binary feature
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        
        logger.info(f"Missing values after preprocessing:\n{df.isnull().sum()}")
        return df
    
    def basic_cleaning(self, df):
        """
        Perform basic data cleaning
        """
        df = df.copy()
        
        # Remove duplicates if any
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape:
            logger.info(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        # Ensure consistent data types
        df['Pclass'] = df['Pclass'].astype(int)
        df['SibSp'] = df['SibSp'].astype(int)
        df['Parch'] = df['Parch'].astype(int)
        
        return df
    
    def preprocess(self, train_df, test_df):
        """
        Complete preprocessing pipeline
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Basic cleaning
        train_df = self.basic_cleaning(train_df)
        test_df = self.basic_cleaning(test_df)
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df, is_train=True)
        test_df = self.handle_missing_values(test_df, is_train=False)
        
        logger.info("Preprocessing completed successfully!")
        return train_df, test_df
    
    def save_processed_data(self, train_df, test_df, train_path, test_path):
        """
        Save preprocessed data
        """
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved processed data to {train_path} and {test_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load data
    train_df, test_df = preprocessor.load_data(
        "../data/raw/train.csv",
        "../data/raw/test.csv"
    )
    
    # Preprocess
    train_processed, test_processed = preprocessor.preprocess(train_df, test_df)
    
    # Save
    preprocessor.save_processed_data(
        train_processed, 
        test_processed,
        "../data/processed/train_processed.csv",
        "../data/processed/test_processed.csv"
    )
