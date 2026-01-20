import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("insurance.csv")


le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])

X = df.drop("charges", axis=1)
y = df["charges"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")


st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#2C3E50;">💼 Insurance Cost Prediction App</h1>
        <p style="font-size:18px;color:#7F8C8D;">
            Enter your details and get an instant prediction of your health insurance cost.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(" 🧍 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.5)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

st.markdown("### 🚬 Lifestyle")

smoker = st.selectbox("Are you a smoker?", ["yes", "no"])

st.markdown("### 🌍 Region")

region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encode user input
sex_encoded = le_sex.transform([sex])[0]
smoker_encoded = le_smoker.transform([smoker])[0]
region_encoded = le_region.transform([region])[0]

# Prepare input for model
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex_encoded],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker_encoded],
    "region": [region_encoded]
})


st.markdown("---")
if st.button("🔮 Predict Insurance Cost"):
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Insurance Cost: **${prediction:,.2f}**")

    
    if smoker == "yes":
        st.warning("⚠ Smoking significantly increases insurance cost.")
    if bmi > 30:
        st.info("ℹ Higher BMI can increase medical risk factors.")



