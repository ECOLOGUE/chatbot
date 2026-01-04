import streamlit as st
import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Chargement du texte
with open("ia.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = sent_tokenize(text)

# 1. PREPROCESSING
def preprocess(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    words = [w for w in words if w not in string.punctuation]
    stop_words = stopwords.words('french')
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

processed_sentences = [preprocess(s) for s in sentences]

# 2. SIMILARITÉ
def get_most_relevant_sentence(user_input):
    user_input = preprocess(user_input)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences + [user_input])

    similarity_scores = cosine_similarity(
        tfidf_matrix[-1],
        tfidf_matrix[:-1]
    )

    index = similarity_scores.argmax()
    score = similarity_scores[0][index]

    if score < 0.1:
        return "Désolé, je n'ai pas trouvé de réponse pertinente."
    else:
        return sentences[index]

# 3. CHATBOT
def chatbot(user_input):
    return get_most_relevant_sentence(user_input)

# 4. INTERFACE STREAMLIT
def main():
    st.title("🤖 Chatbot sur l'Intelligence Artificielle")
    st.write("Posez une question sur l'intelligence artificielle.")

    user_input = st.text_input("Votre question :")

    if user_input:
        response = chatbot(user_input)
        st.success(response)

if __name__ == "__main__":
    main()
