# Tourism Package Prediction - MLOps Project

[![Streamlit App](https://tourism-package-prediction-z56fr4vrewa2jpk4bpsfdx.streamlit.app/badge)](https://tourism-package-prediction-z56fr4vrewa2jpk4bpsfdx.streamlit.app)

## Project Overview
Executed Jupyter notebook with end-to-end MLOps pipeline for predicting Wellness Tourism Package purchase.

## Repository Structure
- `tourism_project/` - Data, scripts, model artifact
- `.github/workflows/pipeline.yml` - 3-job CI/CD pipeline
- `reports/` - Model comparison & metrics
- `Tourism_Package_Prediction_MLOps.html` - Executed notebook submission

## Model Trained
- **Algorithm:** XGBoost
- **F1-Score:** 0.8155 | **ROC-AUC:** 0.9585 | **PR-AUC:** 0.8716
- **Critical Questions:** Q1-Q3 answered in Conclusion

## Streamlit Deployment
Live app: https://tourism-package-prediction-z56fr4vrewa2jpk4bpsfdx.streamlit.app

## Setup & Run
```bash
git remote add origin https://github.com/michiganapp/tourism-package-prediction.git
git add .
git commit -m "Initial MLOps tourism package prediction project"
git push -u origin main
```