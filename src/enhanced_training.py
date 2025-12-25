"""
Enhanced Model Training with Hyperparameter Tuning for 90%+ Accuracy
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Enhanced training with hyperparameter tuning
    """
    logger.info("Starting Enhanced Model Training for 90%+ Accuracy...")
    
    # Load data
    train_df = pd.read_csv("../data/processed/train_featured.csv")
    
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(train_df)
    
    X = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = train_df['Survived']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training with {len(feature_cols)} features")
    logger.info(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    
    # Optimized models with best parameters
    models = {
        'CatBoost_Optimized': CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=0
        ),
        'XGBoost_Optimized': XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM_Optimized': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            random_state=42,
            verbose=-1
        ),
        'RandomForest_Optimized': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting_Optimized': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }
        
        logger.info(f"{name} Accuracy: {accuracy:.4f}")
    
    # Create stacking ensemble
    logger.info("\nCreating Stacking Ensemble...")
    
    estimators = [
        ('catboost', models['CatBoost_Optimized']),
        ('xgboost', models['XGBoost_Optimized']),
        ('lightgbm', models['LightGBM_Optimized']),
        ('rf', models['RandomForest_Optimized'])
    ]
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_val)
    stacking_accuracy = accuracy_score(y_val, y_pred)
    
    results['Stacking_Ensemble'] = {
        'model': stacking_clf,
        'accuracy': stacking_accuracy
    }
    
    logger.info(f"Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")
    
    # Create voting ensemble
    logger.info("\nCreating Voting Ensemble...")
    
    voting_clf = VotingClassifier(
        estimators=estimators[:3],  # Top 3 models
        voting='soft',
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_val)
    voting_accuracy = accuracy_score(y_val, y_pred)
    
    results['Voting_Ensemble'] = {
        'model': voting_clf,
        'accuracy': voting_accuracy
    }
    
    logger.info(f"Voting Ensemble Accuracy: {voting_accuracy:.4f}")
    
    # Select best model
    best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['accuracy']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    logger.info(f"{'='*60}")
    
    # Save best model
    joblib.dump(best_model, "../models/best_model.pkl")
    joblib.dump(feature_cols, "../models/feature_columns.pkl")
    
    # Save all models
    joblib.dump(stacking_clf, "../models/stacking_ensemble.pkl")
    joblib.dump(voting_clf, "../models/voting_ensemble.pkl")
    
    logger.info("\nModels saved successfully!")
    
    # Generate report
    report = "\n" + "="*80 + "\n"
    report += "ENHANCED TITANIC SURVIVAL PREDICTION - MODEL REPORT\n"
    report += "="*80 + "\n\n"
    
    for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        report += f"{name}: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)\n"
    
    report += "\n" + "="*80 + "\n"
    
    with open("../models/enhanced_model_report.txt", "w") as f:
        f.write(report)
    
    logger.info(report)
    
    return best_model, best_accuracy


if __name__ == "__main__":
    model, accuracy = main()
