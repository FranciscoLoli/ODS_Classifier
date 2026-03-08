import streamlit as st
import joblib
from preprocess import text_preprocess


model = joblib.load("modelo_ods.pkl")
ODS_MAP = {
1: "Fin de la pobreza",
2: "Hambre cero",
3: "Salud y bienestar",
4: "Educación de calidad",
5: "Igualdad de género",
6: "Agua limpia y saneamiento",
7: "Energía asequible y no contaminante",
8: "Trabajo decente y crecimiento económico",
9: "Industria, innovación e infraestructura",
10: "Reducción de las desigualdades",
11: "Ciudades y comunidades sostenibles",
12: "Producción y consumo responsables",
13: "Acción por el clima",
14: "Vida submarina",
15: "Vida de ecosistemas terrestres",
16: "Paz, justicia e instituciones sólidas",
17: "Alianzas para lograr los objetivos"
}

st.title("Clasificador de ODS")

texto = st.text_area("Ingrese el texto")

if st.button("Predecir"):

    pred = model.predict([texto])[0]

    st.write("ODS:", pred)  
    st.write("Significado:", ODS_MAP.get(pred, "ODS desconocido"))