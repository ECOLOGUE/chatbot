import streamlit as st
import string
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Télécharger UNE SEULE FOIS
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

load_nltk()

# Charger le texte
@st.cache_data
def load_data():
    with open("ia.txt", "r", encoding="utf-8") as f:
        return f.read()

text = load_data()
sentences = sent_tokenize(text)

# Stopwords chargés UNE fois
stop_words = set(stopwords.words('french'))

# PREPROCESSING
def preprocess(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [w for w in words if w not in string.punctuation]
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

processed_sentences = [preprocess(s) for s in sentences]

# TF-IDF calculé UNE fois
@st.cache_resource
def build_vectorizer(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer(processed_sentences)

# SIMILARITÉ
def get_most_relevant_sentence(user_input):
    user_input = preprocess(user_input)

    user_vec = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)

    index = similarity_scores.argmax()
    score = similarity_scores[0][index]

    if score < 0.1:
        return "Désolé, je n'ai pas trouvé de réponse pertinente."
    else:
        return sentences[index]

# INTERFACE STREAMLIT
def main():
    st.title("🤖 Chatbot sur l'Intelligence Artificielle")
    st.write("Posez une question sur l'intelligence artificielle.")

    user_input = st.text_input("Votre question :")

    if user_input:
        response = get_most_relevant_sentence(user_input)
        st.success(response)

if __name__ == "__main__":
    main()