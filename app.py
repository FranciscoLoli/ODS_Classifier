import streamlit as st
import joblib


model = joblib.load("modelo_ods.pkl")

st.title("Clasificador de ODS")

texto = st.text_area("Ingrese el texto")

if st.button("Predecir"):

    pred = model.predict([texto])

    st.write("ODS predicho:", pred[0])