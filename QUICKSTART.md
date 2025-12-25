# Quick Start Guide - Titanic Survival Prediction

## 🚀 Quick Setup (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing and training
cd src
python data_preprocessing.py
python feature_engineering.py
python model_training.py

# 3. Launch Streamlit app
cd ..
streamlit run streamlit_app/app.py
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preprocessing
cd src
python data_preprocessing.py

# 3. Run feature engineering
python feature_engineering.py

# 4. Train models
python model_training.py

# 5. Launch app
cd ..
streamlit run streamlit_app/app.py
```

## 📊 What You Get

After running the setup, you'll have:

✅ **Preprocessed Data** in `data/processed/`  
✅ **Trained Models** (84%+ accuracy) in `models/`  
✅ **Performance Report** in `models/model_report.txt`  
✅ **Streamlit App** ready at `http://localhost:8501`

## 🎯 Using the Application

### 1. Individual Predictions
- Navigate to "Make Predictions" tab
- Fill in passenger details
- Get instant survival prediction with probability

### 2. Batch Predictions
- Go to "Batch Predictions" tab
- Apply filters (class, age, gender)
- Sort results by various metrics
- Download predictions as CSV

### 3. Model Explainability
- View "Model Explainability" tab
- See feature importance rankings
- Understand individual predictions with SHAP
- Explore feature dependencies

### 4. Data Explorer
- Explore "Data Explorer" tab
- View interactive visualizations
- Analyze survival patterns
- Check statistical summaries

## 🔥 Advanced Features

### Docker Deployment

```bash
# Build and run all services
cd docker
docker-compose up --build

# Access app at http://localhost:8501
```

## 📈 Expected Results

| Model | Accuracy |
|-------|----------|
| CatBoost | 84.36% |
| Voting Ensemble | 83.80% |
| Logistic Regression | 83.24% |
| Gradient Boosting | 82.12% |
| LightGBM | 80.45% |

## 🆘 Troubleshooting

### Issue: Module not found
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: Model file not found
**Solution**: 
```bash
python src/model_training.py
```

### Issue: Streamlit port already in use
**Solution**:
```bash
streamlit run streamlit_app/app.py --server.port=8502
```

## 📝 Key Files

- `src/model_training.py` - Train models
- `streamlit_app/app.py` - Web interface
- `models/model_report.txt` - Performance metrics
- `README.md` - Complete documentation

## 💡 Tips

1. **First Time**: Use automated setup with `python setup_and_run.py`
2. **Model Updates**: Just run `python src/model_training.py`
3. **New Data**: Add to `data/raw/` and rerun preprocessing
4. **Deployment**: Use Docker Compose for production

## 🎓 Learning Path

1. ✅ Run automated setup
2. ✅ Explore Streamlit interface
3. ✅ Review model performance
4. ✅ Understand SHAP explanations
5. ✅ Try individual predictions
6. ✅ (Advanced) Deploy with Docker

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review code comments in `src/` directory
- Open an issue on GitHub

---

**Time to first prediction**: < 5 minutes  
**Accuracy achieved**: 84.36%  
**Production ready**: ✅

Happy Predicting! 🚢
