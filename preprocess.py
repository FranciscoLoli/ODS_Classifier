import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download("stopwords")
nltk.download("punkt")

spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")
tokenizer = RegexpTokenizer(r'\w+')

def text_preprocess(text):
    
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in spanish_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)