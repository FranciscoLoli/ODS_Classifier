import streamlit as st
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")
tokenizer = RegexpTokenizer(r'\w+')

def text_preprocess(text):
    
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in spanish_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

model = joblib.load("modelo_ods.pkl")

st.title("Clasificador de ODS")

texto = st.text_area("Ingrese el texto")

if st.button("Predecir"):

    pred = model.predict([texto])

    st.write("ODS predicho:", pred[0])