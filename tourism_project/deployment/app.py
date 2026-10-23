"""Streamlit front-end for the Wellness Tourism Package propensity model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model.joblib")

FEATURE_ORDER = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="✈️", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "model.joblib was not found next to app.py. Run the training job first "
            "(or wait for the GitHub Actions pipeline to commit the model)."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("✈️ Wellness Tourism Package Predictor")
st.caption(
    "Estimate how likely a customer is to purchase the Wellness Tourism Package "
    "before the sales team contacts them."
)

with st.sidebar:
    st.header("About this app")
    st.write(
        "The model is an XGBoost classifier trained on 4,128 past customers. "
        "It combines customer profile, trip history and sales-interaction data."
    )
    st.write("**Deployed by:** GitHub Actions → Streamlit Community Cloud")
    threshold = st.slider(
        "Decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower the threshold to contact more customers (higher recall), raise it "
             "to call only the strongest prospects (higher precision).",
    )

left, right = st.columns(2)

with left:
    st.subheader("Customer profile")
    age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    st.caption(
        "Designation is not requested: it is perfectly correlated with the product "
        "pitched in the training data, so the model uses ProductPitched only."
    )
    monthly_income = st.number_input(
        "Monthly income", min_value=1000, max_value=100000, value=22000, step=500
    )
    city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
    preferred_star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
    passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])

with right:
    st.subheader("Trip and interaction details")
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    number_visiting = st.number_input(
        "Number of persons visiting", min_value=1, max_value=5, value=3, step=1
    )
    number_children = st.number_input(
        "Number of children visiting", min_value=0, max_value=3, value=0, step=1
    )
    number_trips = st.number_input(
        "Average trips per year", min_value=0, max_value=25, value=3, step=1
    )
    followups = st.number_input(
        "Number of follow-ups", min_value=1, max_value=6, value=3, step=1
    )
    duration = st.number_input(
        "Duration of pitch (minutes)", min_value=1.0, max_value=130.0, value=15.0, step=1.0
    )
    satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_visiting,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_children,
    "MonthlyIncome": monthly_income,
}])[FEATURE_ORDER]

st.divider()
predict_clicked = st.button("Predict purchase likelihood", type="primary", use_container_width=True)

if predict_clicked:
    probability = float(model.predict_proba(input_df)[0, 1])
    st.metric("Predicted purchase probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))

    if probability >= threshold:
        st.success(
            f"High propensity: contact this customer. The estimated probability "
            f"({probability:.1%}) is above the {threshold:.0%} threshold."
        )
    else:
        st.warning(
            f"Low propensity: deprioritise this customer. The estimated probability "
            f"({probability:.1%}) is below the {threshold:.0%} threshold."
        )

    with st.expander("Input summary"):
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

st.caption(
    "This is a decision-support tool. Predictions are estimates from historical data "
    "and should be combined with the sales team's judgement."
)
