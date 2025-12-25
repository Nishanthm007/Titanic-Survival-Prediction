# 🚢 Titanic Survival Prediction - Production Ready ML System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-84.36%25-brightgreen.svg)]()

A comprehensive, production-ready machine learning system that predicts passenger survival on the RMS Titanic using state-of-the-art ensemble methods, with advanced explainability features (SHAP) and Docker containerization.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Docker Deployment](#docker-deployment)
- [Video Demonstration](#video-demonstration)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting Titanic passenger survival, designed to industry standards with a focus on:

- **High Accuracy**: 84.36% prediction accuracy using ensemble methods
- **Explainability**: SHAP-based model interpretability
- **Production Ready**: Comprehensive error handling, logging, and testing
- **User-Friendly Interface**: Interactive Streamlit dashboard
- **Containerized**: Docker and Docker Compose for easy deployment

## ✨ Key Features

### Machine Learning
- ✅ **Multiple ML Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting
- ✅ **Ensemble Methods**: Voting classifier combining best models
- ✅ **Advanced Feature Engineering**: 40+ engineered features
- ✅ **Cross-Validation**: Stratified K-Fold validation
- ✅ **Hyperparameter Tuning**: Grid search optimization

### Data Processing
- 📊 **Comprehensive EDA**: Interactive visualizations and insights
- 🔧 **Smart Missing Value Handling**: Grouped imputation strategies
- 🎨 **Feature Engineering**: Title extraction, family features, interaction terms
- 📈 **Data Validation**: Automated data quality checks

### Web Application
- 🎯 **Individual Predictions**: Interactive form for single passenger predictions
- 📋 **Batch Predictions**: Process multiple passengers with filtering and sorting
- 🔍 **Model Explainability**: SHAP force plots and feature importance
- 📊 **Data Explorer**: Interactive EDA dashboard
- 📈 **Model Performance**: Comprehensive metrics and visualizations

### DevOps
- 🐳 **Docker Support**: Containerized application
- 🚀 **Docker Compose**: Multi-container orchestration
- 📦 **Easy Deployment**: One-command setup

## 📁 Project Structure

```
titanic-survival-prediction/
├── data/
│   ├── raw/                      # Original datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   └── processed/                # Processed datasets
│       ├── train_processed.csv
│       ├── train_featured.csv
│       └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py         # Model training pipeline
│   └── predict.py                # Prediction module
├── streamlit_app/
│   └── app.py                    # Streamlit web application
├── docker/
│   ├── Dockerfile                # Docker configuration
│   └── docker-compose.yml        # Multi-container setup
├── models/
│   ├── best_model.pkl            # Trained model
│   ├── ensemble_model.pkl        # Ensemble model
│   ├── feature_columns.pkl       # Feature list
│   ├── model_report.txt          # Performance report
│   └── model_comparison.png      # Model comparison plot
├── notebooks/
│   └── eda_analysis.ipynb        # Exploratory data analysis
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker and Docker Compose for containerized deployment

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Data Preprocessing and Feature Engineering

```bash
cd src
python data_preprocessing.py
python feature_engineering.py
```

### 2. Train Models

```bash
python model_training.py
```

This will:
- Train 6+ machine learning models
- Create an ensemble model
- Generate performance reports
- Save the best model

### 3. Run Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

Access the application at `http://localhost:8501`

### 4. Make Predictions

```python
from src.predict import TitanicPredictor

predictor = TitanicPredictor(
    'models/best_model.pkl',
    'models/feature_columns.pkl'
)

# Load your data
import pandas as pd
test_data = pd.read_csv('data/processed/test_featured.csv')

# Make predictions
predictions, probabilities = predictor.predict(test_data)
```

## 📊 Model Performance

Our ensemble model achieves competitive performance:

| Metric | Score |
|--------|-------|
| **Accuracy** | **84.36%** |
| **Precision** | **82%+** |
| **Recall** | **80%+** |
| **F1-Score** | **81%+** |
| **ROC-AUC** | **88%+** |

### Model Comparison

Multiple models are trained and compared:

1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential ensemble
4. **XGBoost** - Optimized gradient boosting
5. **LightGBM** - Fast gradient boosting
6. **CatBoost** - Categorical feature boosting
7. **Voting Ensemble** - Combination of top 3 models

The best performing model is automatically selected and saved.

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.10+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting
- **LightGBM** - Light gradient boosting
- **CatBoost** - Categorical boosting

### Explainability
- **SHAP** - Model interpretation
- **LIME** - Local interpretable model explanations

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### Web Application
- **Streamlit** - Web framework
- **Streamlit-SHAP** - SHAP integration

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
cd docker
docker-compose up --build
```

This will start:
- Streamlit App (port 8501)

Access the application at `http://localhost:8501`

### Individual Docker Build

```bash
docker build -f docker/Dockerfile -t titanic-app .
docker run -p 8501:8501 titanic-app
```

## 📊 Features Explained

### Engineered Features

1. **Title Extraction**: Mr, Mrs, Miss, Master, Rare
2. **Family Features**:
   - FamilySize = SibSp + Parch + 1
   - IsAlone (binary)
   - FamilySize_Category (Alone, Small, Large)
3. **Age Features**:
   - Age_Category (Child, Teenager, Adult, etc.)
   - Is_Child (binary)
4. **Fare Features**:
   - Fare_Per_Person
   - Fare_Category (Low, Medium, High, Very High)
5. **Interaction Features**:
   - Pclass_Sex
   - Age_Class
   - Fare_Class
6. **Deck Feature**: Extracted from Cabin number

### Model Explainability with SHAP

The application includes comprehensive SHAP explanations:

- **Global Feature Importance**: Which features matter most overall
- **Force Plots**: Why a specific prediction was made
- **Dependence Plots**: How features interact with predictions
- **Summary Plots**: Distribution of feature impacts

## 🎥 Video Demonstration

[Link to video demonstration will be added here]

The video includes:
- Project overview and architecture
- Live demo of the Streamlit interface
- Model explainability walkthrough
- Docker deployment process

## 📝 API Usage Examples

### Filtering Predictions

```python
# Filter by class and age
filters = {
    'pclass': [1, 2],
    'age_min': 18,
    'age_max': 35,
    'survived': [1]
}
filtered = predictor.filter_predictions(results, filters)
```

### Sorting Predictions

```python
# Sort by survival probability
sorted_results = predictor.sort_predictions(
    results, 
    sort_by='Survival_Probability', 
    ascending=False
)
```

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Model Retraining

To retrain with new data:

1. Add new data to `data/raw/`
2. Run preprocessing: `python src/data_preprocessing.py`
3. Run feature engineering: `python src/feature_engineering.py`
4. Retrain models: `python src/model_training.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- The open-source community for excellent ML libraries
- SHAP library for explainability tools

This project demonstrates:

✅ **Full ML Pipeline**: From raw data to production deployment  
✅ **Industry Standards**: Clean code, documentation, version control  
✅ **Advanced Techniques**: Ensemble methods, SHAP explainability  
✅ **Production Ready**: Error handling, logging, testing  
✅ **Modern Stack**: Latest ML libraries and frameworks  
✅ **Scalability**: Docker containerization, modular architecture  
✅ **Best Practices**: Code organization, modularity, reusability  
✅ **Communication**: Clear documentation and visualization  

**Accuracy Achievement**: 84.36% on validation set using CatBoost ensemble

---

