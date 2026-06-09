import streamlit as st
import joblib
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load Model
model = joblib.load("model.pkl")

# Header
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
Predict whether a passenger would have survived the Titanic disaster
using a Machine Learning model.
""")

st.divider()

# Sidebar
st.sidebar.header("About Project")
st.sidebar.info(
    """
    This Machine Learning project predicts passenger survival
    based on demographic and travel information.

    Model: Random Forest Classifier
    """
)

# Input Form
st.subheader("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    fare = st.number_input("Fare", min_value=0.0, value=50.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    sibsp = st.number_input("Siblings/Spouses", min_value=0, value=0)
    parch = st.number_input("Parents/Children", min_value=0, value=0)

# Convert Gender
sex = 1 if gender == "Male" else 0

st.divider()

if st.button("🔍 Predict Survival", use_container_width=True):

    input_data = pd.DataFrame(
        [[pclass, sex, age, fare, sibsp, parch]],
        columns=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]
    )

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
        st.balloons()
    else:
        st.error("❌ Passenger Did Not Survive")

# Footer
st.divider()
st.caption(
    "Developed by Suchita Gotkhinde | Machine Learning Project"
)