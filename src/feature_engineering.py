"""
Feature Engineering Module
Creates advanced features to improve model performance
"""

import pandas as pd
import numpy as np
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for Titanic dataset
    """
    
    def __init__(self):
        self.title_mapping = None
        self.categorical_columns = {}
        self.all_categories = {}
        
    def extract_title(self, df):
        """
        Extract titles from passenger names
        """
        logger.info("Extracting titles from names...")
        df = df.copy()
        
        # Extract title using regex
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Map rare titles to common ones
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }
        
        df['Title'] = df['Title'].map(title_mapping)
        df = df.fillna({'Title': 'Rare'})
        
        logger.info(f"Title distribution:\n{df['Title'].value_counts()}")
        return df
    
    def create_family_features(self, df):
        """
        Create family-related features
        """
        logger.info("Creating family features...")
        df = df.copy()
        
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Is alone
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Family size category
        df['FamilySize_Category'] = pd.cut(
            df['FamilySize'], 
            bins=[0, 1, 4, 20], 
            labels=['Alone', 'Small', 'Large']
        )
        
        logger.info(f"Family size distribution:\n{df['FamilySize'].value_counts().sort_index()}")
        return df
    
    def create_age_features(self, df):
        """
        Create age-related features
        """
        logger.info("Creating age features...")
        df = df.copy()
        
        # Age bins
        df['Age_Category'] = pd.cut(
            df['Age'], 
            bins=[0, 12, 18, 35, 60, 100], 
            labels=['Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior']
        )
        
        # Is child
        df['Is_Child'] = (df['Age'] < 18).astype(int)
        
        return df
    
    def create_fare_features(self, df):
        """
        Create fare-related features
        """
        logger.info("Creating fare features...")
        df = df.copy()
        
        # Fare per person (accounting for family tickets)
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        
        # Fare bins - handle small datasets and duplicate values
        try:
            # Only attempt binning if we have enough unique values and samples
            n_unique = df['Fare'].nunique()
            if len(df) >= 10 and n_unique >= 4:
                df['Fare_Category'] = pd.qcut(
                    df['Fare'], 
                    q=4, 
                    labels=['Low', 'Medium', 'High', 'Very_High'],
                    duplicates='drop'
                )
            else:
                # For small datasets or few unique values, assign based on percentile
                fare_val = df['Fare'].iloc[0] if len(df) == 1 else df['Fare'].median()
                if fare_val < 10:
                    df['Fare_Category'] = 'Low'
                elif fare_val < 30:
                    df['Fare_Category'] = 'Medium'
                elif fare_val < 100:
                    df['Fare_Category'] = 'High'
                else:
                    df['Fare_Category'] = 'Very_High'
        except (ValueError, Exception):
            # Fallback: assign based on fare value
            df['Fare_Category'] = 'Medium'
        
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features
        """
        logger.info("Creating interaction features...")
        df = df.copy()
        
        # Class and Sex interaction
        df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex'].astype(str)
        
        # Age and Class interaction
        df['Age_Class'] = df['Age'] * df['Pclass']
        
        # Fare and Class interaction
        df['Fare_Class'] = df['Fare'] / (df['Pclass'] + 1)
        
        return df
    
    def create_deck_feature(self, df):
        """
        Extract deck from cabin number
        """
        logger.info("Extracting deck from cabin...")
        df = df.copy()
        
        # Extract first letter of cabin as deck
        df['Deck'] = df['Cabin'].astype(str).str[0]
        df = df.fillna({'Deck': 'Unknown'})
        
        return df
    
    def encode_categorical_features(self, df, is_train=True):
        """
        Encode categorical variables
        """
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        # Binary encoding for Sex
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # One-hot encoding for categorical features
        categorical_features = ['Embarked', 'Title', 'FamilySize_Category', 
                               'Age_Category', 'Fare_Category', 'Deck']
        
        for feature in categorical_features:
            if feature in df.columns:
                if is_train:
                    # Store unique categories from training data
                    self.all_categories[feature] = df[feature].unique().tolist()
                
                # Get dummies
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
                
                # If not training, ensure all expected columns exist
                if not is_train and feature in self.all_categories:
                    for category in self.all_categories[feature]:
                        col_name = f"{feature}_{category}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                    # Keep only expected columns in correct order
                    expected_cols = [f"{feature}_{cat}" for cat in self.all_categories[feature]]
                    dummies = dummies[[col for col in expected_cols if col in dummies.columns]]
                
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def engineer_features(self, train_df, test_df):
        """
        Complete feature engineering pipeline
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Extract titles
        train_df = self.extract_title(train_df)
        test_df = self.extract_title(test_df)
        
        # Create family features
        train_df = self.create_family_features(train_df)
        test_df = self.create_family_features(test_df)
        
        # Create age features
        train_df = self.create_age_features(train_df)
        test_df = self.create_age_features(test_df)
        
        # Create fare features
        train_df = self.create_fare_features(train_df)
        test_df = self.create_fare_features(test_df)
        
        # Create interaction features
        train_df = self.create_interaction_features(train_df)
        test_df = self.create_interaction_features(test_df)
        
        # Extract deck
        train_df = self.create_deck_feature(train_df)
        test_df = self.create_deck_feature(test_df)
        
        # Encode categorical features
        train_df = self.encode_categorical_features(train_df, is_train=True)
        test_df = self.encode_categorical_features(test_df, is_train=False)
        
        logger.info("Feature engineering completed successfully!")
        logger.info(f"Final train shape: {train_df.shape}, test shape: {test_df.shape}")
        
        return train_df, test_df
    
    def get_feature_columns(self, df):
        """
        Get list of feature columns for model training
        """
        # Columns to exclude
        exclude_cols = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 
                       'FamilySize_Category', 'Age_Category', 'Fare_Category', 
                       'Title', 'Embarked', 'Deck', 'Pclass_Sex']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    import joblib
    
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data(
        "../data/raw/train.csv",
        "../data/raw/test.csv"
    )
    
    train_processed, test_processed = preprocessor.preprocess(train_df, test_df)
    
    engineer = FeatureEngineer()
    train_featured, test_featured = engineer.engineer_features(
        train_processed, test_processed
    )
    
    # Save
    train_featured.to_csv("../data/processed/train_featured.csv", index=False)
    test_featured.to_csv("../data/processed/test_featured.csv", index=False)
    
    # Save the engineer instance with category mappings
    joblib.dump(engineer, "../models/feature_engineer.pkl")
    print("Saved feature engineer with category mappings to ../models/feature_engineer.pkl")
