# Docker Deployment Guide

## 🐳 Quick Start

### Build and Run with Docker
```bash
# Navigate to project directory
cd "d:\budhhi data science"

# Build the Docker image
docker build -t titanic-app -f docker/Dockerfile .

# Run the container
docker run -p 8501:8501 titanic-app
```

### Access the Application
Open your browser: **http://localhost:8501**

---

## 🚀 Using Docker Compose (Recommended)

### Start Application
```bash
cd "d:\budhhi data science\docker"
docker-compose up
```

### Run in Background
```bash
docker-compose up -d
```

### Stop Application
```bash
docker-compose down
```

---

## 📋 Available Commands

### Check Running Containers
```bash
docker ps
```

### View Application Logs
```bash
docker logs <container-id>
```

### Stop Container
```bash
docker stop <container-id>
```

### Remove Container
```bash
docker rm <container-id>
```

### Remove Image
```bash
docker rmi titanic-app
```

---

## 🎥 For Demo/Video Recording

### Option 1: Show Docker Compose (1 minute)
```bash
# Start the app
cd docker
docker-compose up

# Open browser to localhost:8501
# Demo the Streamlit app
# Press Ctrl+C to stop
```

### Option 2: Just Show Files (30 seconds)
- Show `docker/Dockerfile` - explain containerization
- Show `docker/docker-compose.yml` - explain orchestration
- Mention: "Production-ready Docker deployment configured"

---

## ✅ What's Included

- **Dockerfile**: Complete container configuration
- **docker-compose.yml**: Service orchestration
- **Multi-stage build**: Optimized image size
- **Health check**: Monitors application status
- **Volume mounting**: Persists data and models

---

## 🎯 Benefits

✅ Consistent environment across systems
✅ Easy deployment to cloud platforms
✅ Isolated dependencies
✅ Scalable infrastructure
✅ Production-ready configuration

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8501
netstat -ano | findstr :8501

# Kill process
taskkill /PID <PID> /F
```

### Docker Desktop Not Running
- Start Docker Desktop application
- Wait for "Docker Desktop is running" status

### Build Fails
```bash
# Clean up and rebuild
docker-compose down
docker system prune -a
docker-compose build --no-cache
```

---

## 📊 Your Achievement

✅ Containerized ML application
✅ Production-ready Docker setup
✅ One-command deployment
✅ Professional infrastructure
