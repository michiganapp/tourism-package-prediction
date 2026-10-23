from google.colab import files

# All the details, code, and instructions compiled into one string
project_details = """
================================================================================
TOURISM MLOPS PROJECT - COMPLETE REFERENCE & CODE GUIDE
================================================================================

INSTRUCTIONS:
1. Copy the code blocks below into the corresponding empty cells in your 
   Learner_Template_Notebook.
2. Ensure you have uploaded 'tourism.csv' to 'tourism_project/data/'.
3. Run the cells sequentially.
4. Push to GitHub, let the Actions workflow complete, and deploy to Streamlit.

================================================================================
1. DATA REGISTRATION (tourism_project/model_building/data_register.py)
================================================================================
import pandas as pd
import os

data_path = 'tourism_project/data/tourism.csv'
df = pd.read_csv(data_path)

print("--- Data Registration Summary ---")
print(f"Dataset successfully loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing Values:\n{df.isnull().sum()}")
print("---------------------------------")

================================================================================
2. DATA PREPARATION (tourism_project/model_building/prep.py)
================================================================================
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('tourism_project/data/tourism.csv')
df = df.drop(columns=['CustomerID'])
df = df.dropna(subset=['ProdTaken'])

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)
print("Data preparation complete. Splits saved.")

================================================================================
3. MODEL TRAINING (tourism_project/model_building/train.py)
================================================================================
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

X_train = pd.read_csv('Xtrain.csv')
X_test = pd.read_csv('Xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze()
y_test = pd.read_csv('ytest.csv').squeeze()

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.01, 0.1]
}

mlflow.set_experiment("Tourism_MLOps_Experiment")

with mlflow.start_run():
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.sklearn.log_model(best_model, "best_model")
    
    joblib.dump(best_model, 'tourism_project/deployment/model.joblib')
    print(f"Model trained! Best F1-Score: {f1_score(y_test, y_pred):.4f}")

================================================================================
4. STREAMLIT APP (tourism_project/deployment/app.py)
================================================================================
import streamlit as st
import pandas as pd
import joblib

st.title("Wellness Tourism Package Predictor")
st.write("Enter customer details to predict purchase likelihood.")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")
model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Self Employed"])
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
with col2:
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
    num_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=3)
    num_children = st.number_input("Number of Children Visiting", min_value=0, value=0)
    num_trips = st.number_input("Average Number of Trips", min_value=0, value=3)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Owns a Car?", [0, 1])
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

input_data = {
    'Age': [age], 'TypeofContact': [type_of_contact], 'CityTier': [city_tier],
    'DurationOfPitch': [10.0], 'Occupation': [occupation], 'Gender': [gender], 
    'NumberOfPersonVisiting': [num_visiting], 'NumberOfFollowups': [3.0], 
    'ProductPitched': [product_pitched], 'PreferredPropertyStar': [preferred_star],
    'MaritalStatus': [marital_status], 'NumberOfTrips': [num_trips], 'Passport': [passport],
    'PitchSatisfactionScore': [pitch_score], 'OwnCar': [own_car], 
    'NumberOfChildrenVisiting': [num_children], 'Designation': [designation], 
    'MonthlyIncome': [monthly_income]
}
input_df = pd.DataFrame(input_data)

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    if prediction[0] == 1:
        st.success(f"High Likelihood of Purchase! (Confidence: {probability:.2%})")
    else:
        st.warning(f"Low Likelihood of Purchase. (Confidence: {1-probability:.2%})")

================================================================================
5. REQUIREMENTS FILES
================================================================================
# File: tourism_project/deployment/requirements.txt
streamlit==1.28.0
pandas==2.1.3
scikit-learn==1.3.2
xgboost==2.0.2
joblib==1.3.2

# File: tourism_project/requirements.txt
pandas==2.1.3
scikit-learn==1.3.2
xgboost==2.0.2
joblib==1.3.2
mlflow==2.9.2

================================================================================
6. GITHUB ACTIONS WORKFLOW (.github/workflows/pipeline.yml)
================================================================================
name: Tourism Project Pipeline

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: write

jobs:
  register-dataset:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Dependencies
        run: pip install -r tourism_project/requirements.txt
      - name: Register Dataset
        run: python tourism_project/model_building/data_register.py
      - name: Upload Registered Dataset
        uses: actions/upload-artifact@v4
        with:
          name: registered-data
          path: tourism_project/data/tourism.csv

  data-prep:
    needs: register-dataset
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Dependencies
        run: pip install -r tourism_project/requirements.txt
      - name: Download Registered Data
        uses: actions/download-artifact@v4
        with:
          name: registered-data
          path: tourism_project/data/
      - name: Run Data Preparation
        run: python tourism_project/model_building/prep.py
      - name: Upload Train Test Splits
        uses: actions/upload-artifact@v4
        with:
          name: data-splits
          path: |
            Xtrain.csv
            Xtest.csv
            ytrain.csv
            ytest.csv

  model-traning:
    needs: data-prep
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Dependencies
        run: pip install -r tourism_project/requirements.txt
      - name: Download Train Test Splits
        uses: actions/download-artifact@v4
        with:
          name: data-splits
      - name: Start MLflow Server
        run: |
          nohup mlflow ui --host 0.0.0.0 --port 5000 &
          sleep 5
      - name: Model Building
        run: python tourism_project/model_building/train.py
      - name: Commit Trained Model
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add tourism_project/deployment/model.joblib
          git commit -m "Add trained model [skip ci]" || echo "No changes to commit"
          git push

================================================================================
TA INSIGHTS FOR YOUR FINAL REPORT:
Before submitting, make sure you can answer these critical thinking questions 
in your notebook's markdown cells:
1. Why did we use 'stratify=y' in the train_test_split? (Hint: Think about the 
   ratio of buyers vs. non-buyers).
2. Why is F1-Score a better metric than Accuracy for this specific dataset? 
   (Hint: Look at the distribution of the 'ProdTaken' column).
3. How does the GitHub Actions workflow ensure that the model is automatically 
   updated if the underlying data changes?
================================================================================
"""

# Save the text file
filename = "Tourism_MLOps_Project_Details.txt"
with open(filename, "w") as file:
    file.write(project_details)

# Trigger the download to your local computer
files.download(filename)
print(f"✅ Successfully downloaded {filename}! Check your browser's downloads folder.")