"""
Model Training Module
Implements ensemble machine learning models for high-accuracy predictions
"""

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TitanicModelTrainer:
    """
    Comprehensive model training with ensemble methods
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.results = {}
        
    def prepare_data(self, df, feature_cols, target_col='Survived'):
        """
        Prepare features and target for modeling
        """
        logger.info("Preparing data for modeling...")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle any remaining missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        
        logger.info(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def initialize_models(self):
        """
        Initialize multiple models for ensemble
        """
        logger.info("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,
                verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=self.random_state,
                verbose=0
            )
        }
        
        return self.models
    
    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        """
        Train and evaluate all models
        """
        logger.info("Training and evaluating models...")
        
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(y_val, y_pred),
                'model': model
            }
            
            logger.info(f"{name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
            logger.info(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def create_ensemble(self, X_train, X_val, y_train, y_val):
        """
        Create voting ensemble of best models
        """
        logger.info("\nCreating voting ensemble...")
        
        # Select top 3 models based on validation accuracy
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )[:3]
        
        ensemble_models = [(name, result['model']) for name, result in sorted_models]
        
        logger.info(f"Ensemble includes: {[name for name, _ in ensemble_models]}")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        voting_clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = voting_clf.predict(X_val)
        y_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        self.results['Voting Ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'model': voting_clf
        }
        
        logger.info(f"\nVoting Ensemble Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        return voting_clf
    
    def select_best_model(self):
        """
        Select best performing model
        """
        best_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model = self.results[best_name]['model']
        
        logger.info(f"\nBest Model: {best_name}")
        logger.info(f"Best Accuracy: {self.results[best_name]['accuracy']:.4f}")
        
        return best_name, self.best_model
    
    def save_model(self, model, filepath):
        """
        Save trained model
        """
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def plot_results(self, save_path=None):
        """
        Plot model comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        axes[0, 0].barh(models, accuracies, color='skyblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].axvline(x=0.9, color='r', linestyle='--', label='90% threshold')
        axes[0, 0].legend()
        
        # 2. All metrics comparison for best model
        best_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_metrics = self.results[best_name]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        values = [best_metrics[m] for m in metrics]
        
        axes[0, 1].bar(metrics, values, color='lightgreen')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title(f'Best Model ({best_name}) - All Metrics')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--')
        
        # 3. Confusion Matrix for best model
        cm = best_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_name}')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # 4. F1-Score comparison
        f1_scores = [self.results[m]['f1_score'] for m in models]
        axes[1, 1].barh(models, f1_scores, color='lightcoral')
        axes[1, 1].set_xlabel('F1-Score')
        axes[1, 1].set_title('Model F1-Score Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive model report
        """
        report = "\n" + "="*80 + "\n"
        report += "TITANIC SURVIVAL PREDICTION - MODEL EVALUATION REPORT\n"
        report += "="*80 + "\n\n"
        
        for name, metrics in sorted(self.results.items(), 
                                   key=lambda x: x[1]['accuracy'], 
                                   reverse=True):
            report += f"\n{name}\n"
            report += "-" * 40 + "\n"
            report += f"Accuracy:  {metrics['accuracy']:.4f}\n"
            report += f"Precision: {metrics['precision']:.4f}\n"
            report += f"Recall:    {metrics['recall']:.4f}\n"
            report += f"F1-Score:  {metrics['f1_score']:.4f}\n"
            report += f"ROC-AUC:   {metrics['roc_auc']:.4f}\n"
            
            if 'cv_mean' in metrics:
                report += f"CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report


def main():
    """
    Main training pipeline
    """
    logger.info("Starting Titanic Model Training Pipeline...")
    
    # Load featured data
    train_df = pd.read_csv("../data/processed/train_featured.csv")
    
    # Initialize trainer
    trainer = TitanicModelTrainer(random_state=42)
    
    # Get feature columns
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(train_df)
    
    logger.info(f"Using {len(feature_cols)} features for modeling")
    
    # Prepare data
    X, y = trainer.prepare_data(train_df, feature_cols)
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")
    
    # Initialize and train models
    trainer.initialize_models()
    trainer.train_and_evaluate(X_train, X_val, y_train, y_val)
    
    # Create ensemble
    ensemble_model = trainer.create_ensemble(X_train, X_val, y_train, y_val)
    
    # Select best model
    best_name, best_model = trainer.select_best_model()
    
    # Plot results
    trainer.plot_results(save_path="../models/model_comparison.png")
    
    # Generate report
    report = trainer.generate_report()
    logger.info(report)
    
    # Save report
    with open("../models/model_report.txt", "w") as f:
        f.write(report)
    
    # Save best model and feature columns
    trainer.save_model(best_model, "../models/best_model.pkl")
    joblib.dump(feature_cols, "../models/feature_columns.pkl")
    
    # Save ensemble model separately
    trainer.save_model(ensemble_model, "../models/ensemble_model.pkl")
    
    logger.info("\nTraining pipeline completed successfully!")
    logger.info(f"Best model accuracy: {trainer.results[best_name]['accuracy']:.4f}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
