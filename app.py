import streamlit as st
import joblib

model = joblib.load('model.pkl')

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class",[1,2,3])
sex = st.selectbox("Gender",[0,1])
age = st.number_input("Age")
fare = st.number_input("Fare")
sibsp = st.number_input("Siblings/Spouses")
parch = st.number_input("Parents/Children")

if st.button("Predict"):

    prediction = model.predict(
        [[pclass,sex,age,fare,sibsp,parch]]
    )

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")