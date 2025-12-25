# 🚀 Quick Command Reference

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 🎯 Running the Pipeline

### Data Processing
```bash
cd src
python data_preprocessing.py
python feature_engineering.py
```

### Model Training
```bash
# Standard training
python model_training.py

# Enhanced training (recommended)
python enhanced_training.py
```

### Make Predictions
```bash
python predict.py
```

## 🌐 Launch Streamlit App

```bash
# From project root
streamlit run streamlit_app/app.py

# Custom port
streamlit run streamlit_app/app.py --server.port=8502
```

## 🔄 Kafka Streaming

### Start Kafka Services (Docker)
```bash
cd docker
docker-compose up -d zookeeper kafka
```

### Run Consumer (Terminal 1)
```bash
python kafka_streaming/consumer.py
```

### Run Producer (Terminal 2)
```bash
python kafka_streaming/producer.py
```

## 🐳 Docker Commands

### Build and Run All Services
```bash
cd docker
docker-compose up --build
```

### Run Only Streamlit
```bash
docker build -f docker/Dockerfile -t titanic-app .
docker run -p 8501:8501 titanic-app
```

### Stop Services
```bash
docker-compose down
```

## 📊 Quick Tests

### Check Model Performance
```bash
cat models/model_report.txt
```

### View Model Comparison
```bash
# Open in image viewer
models/model_comparison.png
```

## 🛠️ Troubleshooting

### Reinstall Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Cache
```bash
# Remove pycache
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove Streamlit cache
rm -rf streamlit_app/.streamlit
```

### Reset Kafka
```bash
docker-compose down -v
docker-compose up -d
```

## 📁 File Locations

- **Data**: `data/raw/`, `data/processed/`
- **Models**: `models/*.pkl`
- **Reports**: `models/model_report.txt`
- **Source Code**: `src/`
- **App**: `streamlit_app/app.py`
- **Kafka**: `kafka_streaming/`

## 🎓 Development Workflow

1. **Process Data**
   ```bash
   cd src
   python data_preprocessing.py
   python feature_engineering.py
   ```

2. **Train Models**
   ```bash
   python model_training.py
   ```

3. **Test App**
   ```bash
   cd ..
   streamlit run streamlit_app/app.py
   ```

4. **Deploy**
   ```bash
   cd docker
   docker-compose up --build
   ```

## 🔍 Useful Commands

```bash
# Check Python version
python --version

# List installed packages
pip list

# Show Streamlit version
streamlit version

# Check port usage (Windows)
netstat -ano | findstr :8501

# Check Docker containers
docker ps

# View Docker logs
docker logs <container-id>
```

## ⚡ One-Liner Shortcuts

```bash
# Complete pipeline
cd src && python data_preprocessing.py && python feature_engineering.py && python model_training.py && cd .. && streamlit run streamlit_app/app.py

# Quick app launch
streamlit run streamlit_app/app.py

# Docker quick start
cd docker && docker-compose up -d
```

## 📝 Notes

- Always run commands from correct directory
- Ensure virtual environment is activated
- Check logs if errors occur
- Use `--help` flag for more options

## 🆘 Get Help

```bash
# Streamlit help
streamlit --help

# Python module help
python -m <module> --help

# Docker Compose help
docker-compose --help
```

---

**Quick Access**: Keep this file handy for common commands!
