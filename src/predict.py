"""
Prediction Module
Handles predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TitanicPredictor:
    """
    Make predictions using trained model
    """
    
    def __init__(self, model_path, feature_cols_path):
        """
        Load trained model and feature columns
        """
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(feature_cols_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using {len(self.feature_cols)} features")
    
    def prepare_features(self, df):
        """
        Prepare features for prediction
        """
        # Add missing columns with 0 values
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features used in training, in the correct order
        X = df[self.feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        
        return X
    
    def predict(self, df):
        """
        Make predictions
        """
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_with_details(self, df):
        """
        Make predictions with additional details
        """
        predictions, probabilities = self.predict(df)
        
        results_df = df[['PassengerId', 'Name', 'Sex', 'Age', 'Pclass']].copy()
        results_df['Survived'] = predictions
        results_df['Survival_Probability'] = probabilities[:, 1]
        results_df['Death_Probability'] = probabilities[:, 0]
        
        return results_df
    
    def filter_predictions(self, results_df, filters):
        """
        Filter predictions based on user criteria
        
        filters = {
            'pclass': [1, 2, 3],
            'sex': ['male', 'female'],
            'age_min': 0,
            'age_max': 100,
            'survived': [0, 1]
        }
        """
        filtered_df = results_df.copy()
        
        if 'pclass' in filters and filters['pclass']:
            filtered_df = filtered_df[filtered_df['Pclass'].isin(filters['pclass'])]
        
        if 'sex' in filters and filters['sex']:
            # Convert back from encoded values if needed
            filtered_df = filtered_df[filtered_df['Sex'].isin(filters['sex'])]
        
        if 'age_min' in filters:
            filtered_df = filtered_df[filtered_df['Age'] >= filters['age_min']]
        
        if 'age_max' in filters:
            filtered_df = filtered_df[filtered_df['Age'] <= filters['age_max']]
        
        if 'survived' in filters and filters['survived'] is not None:
            filtered_df = filtered_df[filtered_df['Survived'].isin(filters['survived'])]
        
        return filtered_df
    
    def sort_predictions(self, results_df, sort_by='Survival_Probability', ascending=False):
        """
        Sort predictions
        """
        return results_df.sort_values(by=sort_by, ascending=ascending)


def main():
    """
    Example prediction pipeline
    """
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and preprocess test data
    preprocessor = DataPreprocessor()
    _, test_df = preprocessor.load_data("../data/raw/train.csv", "../data/raw/test.csv")
    
    # Load preprocessing artifacts if they exist
    try:
        train_df = pd.read_csv("../data/processed/train_processed.csv")
        preprocessor.age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
        preprocessor.embarked_mode = train_df['Embarked'].mode()[0]
        preprocessor.fare_median = train_df['Fare'].median()
    except:
        pass
    
    test_processed = preprocessor.handle_missing_values(test_df, is_train=False)
    
    # Feature engineering
    engineer = FeatureEngineer()
    train_df = pd.read_csv("../data/processed/train_featured.csv")
    _, test_featured = engineer.engineer_features(train_df, test_processed)
    
    # Make predictions
    predictor = TitanicPredictor(
        "../models/best_model.pkl",
        "../models/feature_columns.pkl"
    )
    
    results = predictor.predict_with_details(test_featured)
    
    # Save predictions
    submission = results[['PassengerId', 'Survived']]
    submission.to_csv("../data/processed/predictions.csv", index=False)
    
    logger.info(f"Predictions saved for {len(results)} passengers")
    logger.info(f"\nSurvival rate: {results['Survived'].mean():.2%}")
    
    return results


if __name__ == "__main__":
    results = main()
