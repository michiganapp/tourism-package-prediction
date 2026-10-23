# Tourism Package Prediction - MLOps Project

## Business Context
"Visit with Us," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

## Project Objective
As an MLOps Engineer, the responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.

The pipeline includes:
- Data cleaning, preprocessing, and transformation
- Model building, training, and evaluation
- Experiment tracking with MLflow
- Deployment-ready model artifact
- CI/CD integration via GitHub Actions
- Streamlit Community Cloud deployment

## Dataset
- **Source:** `tourism_project/data/tourism.csv`
- **Features:** 20 customer and interaction attributes (TypeofContact, CityTier, Occupation, Gender, etc.)
- **Target:** `ProdTaken` (0 = No purchase, 1 = Purchase)
- **Size:** 4,128 customers, 19.3% purchase rate
- **Key insight:** Class-imbalanced binary classification problem

## Model Information
- **Algorithm:** XGBoost Classifier with hyperparameter tuning
- **Best F1-Score:** 0.8155 on untouched 20% test set
- **ROC-AUC:** 0.9585
- **PR-AUC:** 0.8716
- **Model artifact:** `tourism_project/deployment/model.joblib` (scikit-learn Pipeline with preprocessing and classifier)

## Critical Thinking Questions
1. **Why `stratify=y` in train_test_split?** Buyers are only 19.3% of the data. Stratification keeps that ratio in both splits, ensuring stable F1, recall, and PR-AUC metrics.
2. **Why is F1-Score better than Accuracy?** With only 19.3% buyers, a "predict everyone stays" model scores 80.7% accuracy while catching zero buyers. F1 focuses on the minority buyer class.
3. **How does GitHub Actions workflow keep the model up to date?** Every push to `main` triggers the pipeline: `register-dataset` validates data, `data-prep` builds fresh splits, and `model-training` retrains, logs to MLflow, and commits the new model with `[skip ci]`. Streamlit Community Cloud auto-redeploys when the model commits.

## MLOps Pipeline (GitHub Actions)
Three-job workflow triggered on `main` push:
1. **register-dataset** - Validates schema, prints summary, uploads artifact
2. **data-prep** - Downloads artifact, cleans data, creates train/test splits, uploads splits
3. **model-training** - Downloads splits, trains model (comparison + GridSearchCV), logs to MLflow, commits model.joblib

## Streamlit Deployment
- **App URL:** Deployed on Streamlit Community Cloud
- **Entry point:** `tourism_project/deployment/app.py`
- **Features:** Interactive customer profile form, prediction probability, threshold slider
- **Python version:** 3.12

## Project Structure
```
tourism-package-prediction/
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   └── train.py
│   └── deployment/
│       ├── app.py
│       ├── model.joblib
│       └── requirements.txt
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── reports/
│   ├── model_comparison.csv
│   └── model_metrics.json
├── tourism_project/requirements.txt
└── Tourism_Package_Prediction_MLOps.html (submission notebook)
```

## Setup & Local Running
1. Set `GITHUB_USERNAME` in notebook Configuration cell
2. Run Cell 52 to push to GitHub and trigger pipeline
3. Deploy to Streamlit Community Cloud (Python 3.12)
4. Access app at generated URL

## License
This project is for educational/MLOps practice purposes.