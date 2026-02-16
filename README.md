# Hiring Intelligence Dashboard

**AI-powered hiring platform** that analyzes GitHub and Codeforces data using RandomForest, GradientBoosting, and DecisionTree models to identify top candidates.

## ✨ Features

- **Manager View**: 4 ML model predictions (GitHub + Codeforces)
- **HR View**: Final shortlist + 1-click candidate emails
- **Candidate View**: Self-service status checker


## 🚀 Start

**Run** streamlit run app.py

## ✨ Code Structure 

code/

app.py - Main Streamlit UI

.streamlit/secrets.toml - Email credentials 

config.py - Configuration

utils/ - Business logic

email.py - SMTP email sender

data_processor.py - Data aggregation

models/ - ML model wrappers

all_models.py - All 4 models combined

👥 Team
Simran, Ishwari & Sakshi


